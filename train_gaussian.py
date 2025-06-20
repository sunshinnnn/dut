import time
import numpy as np
import os
import logging
from datasets.domeImageDataset import DomeImageDataset

from models.network import Discriminator
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F

import pdb
import cv2
import os.path as osp
from icecream import ic

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from utils.general_utils import resizeImg, direction_to_quaternion, saveCSV
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from kaolin.metrics.pointcloud import chamfer_distance
from kaolin.metrics.trianglemesh import point_to_mesh_distance, uniform_laplacian_smoothing

from configs.my_config import ConfigDUT as config
from models.unet import UNet
from models.utils import index_feat, makePath, save_ply, seed_everything
from scene.gaussian_model import GaussianModel
from tools.metric_tools import my_lpips
from models.loss import l1_loss, ssim, IDMRFLoss, gan_loss, PSNR, SSIM, l2_loss, rot_quat_loss
from gaussian_renderer import render as render_gaussian
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, BasicPointCloud

class Mock_PipelineParams:
    def __init__(self):
        """
           Nothing but a hack
        """
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0

class Trainer:
    def __init__(self, cfg_file, split='train'):
        self.cfg = cfg_file
        self.device = "cuda:0"

        self.is_white_background = self.cfg.dataset.is_white_background
        if self.is_white_background:
            self.background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device)
        else:
            self.background = torch.tensor([0., 0., 0.], dtype=torch.float32).to(self.device)

        if split=='train':
            self.train_set = DomeImageDataset(self.cfg, split='train', warmup= True)
            self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                           num_workers=self.cfg.batch_size * 16, pin_memory=True)
            self.train_iterator = iter(self.train_loader)
        else:
            self.test_set = DomeImageDataset(self.cfg, split='test')
            self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False,
                                           num_workers=16, pin_memory=False)
            self.test_iterator = iter(self.test_loader)

        self.sh_degree = self.cfg.gaussian.sh_degree
        self.inputDim = 6 if self.cfg.withNormal else 3
        self.model = UNet(self.inputDim, 11 + 3* (self.sh_degree+1)**2).to(self.device)

        print("input dim: {}".format(self.inputDim))

        self.gaussians = GaussianModel(sh_degree = self.sh_degree)
        self.mocked_pipeline = Mock_PipelineParams()

        self.lr = self.cfg.gaussian.lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.cfg.wdecay, eps=1e-8)

        self.updated_loss_dict = self.make_tensorboard_settings()

        self.total_steps = 0
        self.num_steps = self.cfg.gaussian.num_steps
        self.warmup_steps = self.cfg.gaussian.warmup_steps

        self.model.cuda()

        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.train()

        if split == 'train':
            self.faces = self.train_set.faces.cuda()
            self.uvs = self.train_set.uvs.cuda()
            self.facesuv = self.train_set.facesuv.cuda()
            self.uvs_unique = self.train_set.uvs_unique.cuda()
            self.uv_idx = self.train_set.uv_idx.cuda()
            self.uv_idy = self.train_set.uv_idy.cuda()
            self.canoNormal = self.train_set.canoNormal.data.cpu().numpy()

            self.quat_cano = direction_to_quaternion(self.canoNormal)
            self.quat_cano = torch.FloatTensor(self.quat_cano).cuda()
            self.quat_cano = torch.nn.functional.normalize(self.quat_cano, dim = -1)
        else:
            self.faces = self.test_set.faces.cuda()
            self.uvs = self.test_set.uvs.cuda()
            self.facesuv = self.test_set.facesuv.cuda()
            self.uvs_unique = self.test_set.uvs_unique.cuda()
            self.uv_idx = self.test_set.uv_idx.cuda()
            self.uv_idy = self.test_set.uv_idy.cuda()
            self.canoNormal = self.test_set.canoNormal.data.cpu().numpy()    #.cuda()

            self.quat_cano = direction_to_quaternion(self.canoNormal)
            self.quat_cano = torch.FloatTensor(self.quat_cano).cuda()
            self.quat_cano = torch.nn.functional.normalize(self.quat_cano, dim = -1)
        self.quat_reg = torch.zeros([self.canoNormal.shape[0], 4]).float().cuda()
        self.quat_reg[:, 0] = 1.0

        self.mrfloss = IDMRFLoss()
        self.weightColor = self.cfg.weightColor
        self.weightSSIM = self.cfg.weightSSIM
        self.weightMRF = self.cfg.weightMRF
        self.weightReg = self.cfg.weightReg

        if split == 'train':
            self.writer = SummaryWriter(log_dir=self.cfg.record.logs_path)

    def create_ref_gaussian(self, ret_dict):


        pt_color = (ret_dict["inputs"][0,:3].permute(1,2,0)[self.uv_idx.cpu(),self.uv_idy.cpu(),:] + 1) /2
        pt_pos = ret_dict['canoPoints'][0]
        pt_normal = ret_dict['canoNormals'][0]

        ret_ref_gaussian = GaussianModel(
            sh_degree=self.sh_degree
        )
        # pdb.set_trace()
        ret_ref_pcd = BasicPointCloud(
            points=pt_pos[:, :].data.cpu().numpy().copy(),
            normals=pt_normal[:, :].data.cpu().numpy().copy(),
            colors=pt_color.data.cpu().numpy().copy()
        )

        ret_ref_gaussian.create_from_pcd(
            ret_ref_pcd, 0.0
        )

        ret_ref_gaussian._features_rest = ret_ref_gaussian._features_rest * 0.

        return ret_ref_gaussian

    def update_learning_rate(self):

        start_constant_lr = self.cfg.lrDecayStep #650000
        alpha = 0.1 #0.5 #0.1  # self.learning_rate_alpha

        if self.total_steps < self.warmup_steps:
            learning_factor = (self.total_steps + 1) / self.warmup_steps
            weighted_learing_rate = self.lr * learning_factor
        else:
            progress = (
                               min(self.total_steps, start_constant_lr + 2000) - self.warmup_steps
                       ) / (
                               min(self.num_steps, start_constant_lr + 2000) - self.warmup_steps
                       )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

            if self.cfg.withStepDecay:
                weighted_learing_rate = self.lr * 1.0 #alpha #1.0
            else:
                weighted_learing_rate = self.lr * learning_factor



        if self.total_steps > start_constant_lr:
            weighted_learing_rate = alpha * self.lr

        for g in self.optimizer.param_groups:
            g['lr'] = weighted_learing_rate

        return

    def profile_loss(self, loss_dict):

        fin_loss_dict = {}

        for each_key in self.updated_loss_dict['loss_dict']:
            if each_key in loss_dict:
                fin_loss_dict[each_key] = loss_dict[each_key]
            else:
                fin_loss_dict[each_key] = 0.

        for each_key in fin_loss_dict:
            self.writer.add_scalar(each_key, fin_loss_dict[each_key], self.total_steps)

        return

    def make_tensorboard_settings(self):
        ret_dict = {}

        loss_dict = {
            'loss': True,
            'color': True,
            'ssim': True,
            'lr': True,
            'dc': True,
            'rest': True,
            'rot': True,
            'op': True,
            'del': True,
            'sc': True,
            'mrf_loss': True
        }

        ret_dict['loss_dict'] = loss_dict

        print("#################tensorboard settings###################")
        ic(ret_dict)
        print("########################################################")
        return ret_dict

    def train(self):

        self.update_learning_rate()
        pbar = tqdm(range(self.total_steps, self.num_steps))
        criterion = torch.nn.MSELoss()
        for _ in pbar:
            self.update_learning_rate()

            if self.total_steps <= self.warmup_steps:

                st0 = time.time()
                data = self.fetch_data(phase='train', warmup=True)
                st1 = time.time()
                full_feats = self.model(data["inputs"])[0]
                full_feats = full_feats[:, self.uv_idx, self.uv_idy].permute(1, 0).contiguous()

                st2 = time.time()
                _delta = full_feats[:,8:11]
                _scaling = full_feats[:, :3]
                if self.cfg.withRelaRot:
                    _rotation_delta = full_feats[:, 3:7]
                else:
                    _rotation = full_feats[:, 3:7]
                _opacity = full_feats[:, 7].unsqueeze(1)
                _features_dc = full_feats[:, 11:14].reshape(-1, 1, 3)
                _features_rest = full_feats[:, 14:].reshape(_delta.shape[0], -1, 3)

                st3 = time.time()

                ret_ref_gaussian = self.create_ref_gaussian(data)
                st4 = time.time()

                features_dc_loss = criterion(_features_dc, ret_ref_gaussian._features_dc)
                features_rest_loss = criterion(_features_rest, ret_ref_gaussian._features_rest)
                scaling_loss = criterion(_scaling, ret_ref_gaussian._scaling)
                if self.cfg.withRelaRot:
                    rotation_loss = criterion(_rotation_delta, self.quat_reg)
                else:
                    rotation_loss = criterion(_rotation, ret_ref_gaussian._rotation)
                opacity_loss = criterion(_opacity, ret_ref_gaussian._opacity)
                delta_loss = criterion(_delta, torch.zeros_like(_delta))

                final_loss = features_dc_loss + features_rest_loss + scaling_loss + rotation_loss + opacity_loss * 0.1 + delta_loss
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()
                st5 = time.time()

                post_fix_dict = {
                    'loss': final_loss.cpu().item(),
                    'dc': features_dc_loss.cpu().item(),
                    'rest': features_rest_loss.cpu().item(),
                    'rot': rotation_loss.cpu().item(),
                    'op': opacity_loss.cpu().item(),
                    'del': delta_loss.cpu().item(),
                    'sc': scaling_loss.cpu().item()
                }

                pbar.set_postfix(post_fix_dict)
            else:
                data = self.fetch_data(phase='train', warmup=False)

                full_feats = self.model(data["inputs"])[0]
                full_feats = full_feats[:, self.uv_idx, self.uv_idy].permute(1, 0).contiguous()

                canoDelta = full_feats[:, 8:11]
                canoPoints = data["canoPoints"][0] + canoDelta
                self.gaussians._xyz = canoPoints
                if self.cfg.withPostScale:
                    canoPostScale = data["canoPostScale"][0]
                    self.gaussians._scaling = full_feats[:, :3] + torch.log(canoPostScale)
                else:
                    self.gaussians._scaling = full_feats[:, :3]
                if self.cfg.withRelaRot:
                    self.gaussians._rotation = quaternion_multiply( full_feats[:, 3:7], self.quat_cano)
                else:
                    self.gaussians._rotation = full_feats[:, 3:7]
                self.gaussians._opacity = full_feats[:, 7].unsqueeze(1)
                self.gaussians._features_dc = full_feats[:, 11:14].reshape(-1, 1, 3)
                self.gaussians._features_rest = full_feats[:, 14:].reshape(canoPoints.shape[0], -1, 3)

                imgScale = float(data['imgScale'][0])
                T_fw = data["canoTransformations"][0]
                K = data['K'].to(self.device)[0]
                K[:2,:3] *= imgScale
                E = data['E'].to(self.device)[0]
                H = int(data['H'][0] * imgScale )  # .to(self.device) #[0]
                W = int(data['W'][0] * imgScale )  # .to(self.device) #[0]
                gt_img = data['img'][0].permute(2,0,1).to(self.device)
                mask_np = data['mask'][0].cpu().numpy().astype(np.uint8)
                render_pkg = render_gaussian(K, E, H, W, T_fw, self.gaussians, self.mocked_pipeline, self.background)
                image = render_pkg["render"]#.permute(1,2,0)
                color_loss = l1_loss(image, gt_img)

                x, y, w, h = cv2.boundingRect(mask_np)
                img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
                img_gt = gt_img[:, y:y + h, x:x + w].unsqueeze(0)

                if h < 11 or w < 11:
                    if h < 7 or w < 7:
                        if h < 5 or w < 5:
                            ssim_loss = torch.Tensor([1.0]).cuda()
                        else:
                            ssim_loss = ssim(img_pred, img_gt, window_size=5)
                    else:
                        ssim_loss = ssim(img_pred, img_gt, window_size=7)
                else:
                    ssim_loss = ssim(img_pred, img_gt)

                # IDMRF_PATCH_SIZE = 256
                IDMRF_PATCH_SIZE = 512
                if h > 0:
                    mrf_loss = self.mrfloss(F.upsample(img_pred, size=(IDMRF_PATCH_SIZE // 2, IDMRF_PATCH_SIZE // 2), mode='bilinear', align_corners=True),  F.upsample(img_gt, size=(IDMRF_PATCH_SIZE // 2, IDMRF_PATCH_SIZE // 2), mode='bilinear', align_corners=True) )
                else:
                    mrf_loss = torch.Tensor([0.0]).cuda()

                reg_loss = l2_loss(canoDelta,0)
                final_loss = self.weightColor * color_loss + self.weightSSIM * (1.0 - ssim_loss) + self.weightMRF * mrf_loss + self.weightReg * reg_loss

                if self.cfg.withRotReg:
                    if self.cfg.withRelaRot:
                        rotation_loss = rot_quat_loss(full_feats[:, 3:7], self.quat_reg)
                    else:
                        rotation_loss = rot_quat_loss(full_feats[:, 3:7], self.quat_cano)
                    final_loss = final_loss + 0.005 * rotation_loss

                if self.cfg.weightChamfer > 0:
                    out_verts_world = (T_fw[..., :3, :3] @ canoPoints[..., None]).squeeze(-1) + T_fw[..., :3, 3]
                    final_loss = self.cfg.weightChamfer * chamfer_distance(out_verts_world[None], data['pointclouds']).mean()

                self.optimizer.zero_grad()
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                post_fix_dict = {
                    'loss': final_loss.cpu().item(),
                    'mrf_loss': mrf_loss.cpu().item(),
                    'fIdx': data['fIdx'][0].cpu().item(),
                    'color': color_loss.cpu().item(),
                    'ssim': ssim_loss.cpu().item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                pbar.set_postfix(post_fix_dict)

            if (self.total_steps % 10 == 0):
                self.profile_loss(post_fix_dict)

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq1 == 0 and self.total_steps < self.warmup_steps:
                self.save_ckpt(save_path=Path('%s/%s_%s.pth' % (cfg.record.ckpt_path, cfg.name, self.total_steps)), show_log=False)

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq2 == 0 and self.total_steps > self.warmup_steps:
                self.save_ckpt(save_path=Path('%s/%s_%s.pth' % (cfg.record.ckpt_path, cfg.name, self.total_steps)), show_log=False)

            if self.total_steps == 1970000 or self.total_steps == 680000:
                self.save_ckpt(save_path=Path('%s/%s_%s.pth' % (cfg.record.ckpt_path, cfg.name, self.total_steps)), show_log=False)

            if self.total_steps % 100 ==0 and self.total_steps > self.warmup_steps:
                image_np = image.permute(1,2,0).data.cpu().numpy()
                makePath(osp.join(cfg.record.debug_path, 'render'))
                cv2.imwrite(osp.join(cfg.record.debug_path, 'render', '{}_{}_{}.jpg'.format(str(self.total_steps).zfill(6), str(data['cIdx'][0].cpu().item()).zfill(3), str(data['fIdx'][0].cpu().item()))), image_np[:,:,::-1] * 255.0)

            if self.total_steps == 500 and self.total_steps < self.warmup_steps:
                makePath(osp.join(cfg.record.debug_path, 'gaussian'))
                ret_ref_gaussian.save_ply(osp.join(cfg.record.debug_path, 'gaussian', 'ref_{}.ply'.format(str(self.total_steps).zfill(6))))

            if self.total_steps > self.warmup_steps and self.total_steps % 1000 ==0:
                makePath(osp.join(cfg.record.debug_path, 'gaussian'))
                self.gaussians.save_ply(
                    osp.join(cfg.record.debug_path, 'gaussian', 'pred_{}.ply'.format(str(self.total_steps).zfill(6))))

            self.total_steps += 1

        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_test(self):
        print(f"Doing testing ...")
        torch.cuda.empty_cache()
        metrics = []
        self.len_test = int(len(self.test_loader))
        self.model.eval()

        criterion_psnr = PSNR()
        criterion_ssim = SSIM()
        criterion_lpips = my_lpips()

        metrics_name = []
        metrics_psnr = []
        metrics_ssim = []
        metrics_lpips = []
        for idx in range(self.len_test):
            data = self.fetch_data(phase='test')
            with torch.no_grad():
                full_feats = self.model(data["inputs"])[0]
                full_feats = full_feats[:, self.uv_idx, self.uv_idy].permute(1, 0).contiguous()
                canoDelta = full_feats[:, 8:11]
                canoPoints = data["canoPoints"][0] + canoDelta
                self.gaussians._xyz = canoPoints

                if self.cfg.withPostScale:
                    canoPostScale = data["canoPostScale"][0]
                    self.gaussians._scaling = full_feats[:, :3] + torch.log(canoPostScale)
                else:
                    self.gaussians._scaling = full_feats[:, :3]

                if self.cfg.withRelaRot:
                    self.gaussians._rotation = quaternion_multiply( full_feats[:, 3:7], self.quat_cano)
                else:
                    self.gaussians._rotation = full_feats[:, 3:7]
                self.gaussians._opacity = full_feats[:, 7].unsqueeze(1)
                self.gaussians._features_dc = full_feats[:, 11:14].reshape(-1, 1, 3)
                self.gaussians._features_rest = full_feats[:, 14:].reshape(canoPoints.shape[0], -1, 3)

                imgScale = self.cfg.imgScale
                T_fw = data["canoTransformations"][0]
                K = data['K'].to(self.device)[0]
                K[:2,:3] *= imgScale
                E = data['E'].to(self.device)[0]
                H = int(data['H'][0] * imgScale )  # .to(self.device) #[0]
                W = int(data['W'][0] * imgScale )  # .to(self.device) #[0]
                gt_img = data['img'][0].permute(2,0,1).to(self.device)
                mask_np = data['mask'][0].cpu().numpy().astype(np.uint8)

                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_np = cv2.erode(mask_np, element)
                maskGT_float = mask_np.astype(np.float32)[:, :, None]
                maskGT_float = torch.Tensor(maskGT_float).cuda().permute(2,0,1)
                gt_img *= maskGT_float

                render_pkg = render_gaussian(K, E, H, W, T_fw, self.gaussians, self.mocked_pipeline, self.background)
                image = render_pkg["render"].clone()#.permute(1,2,0)
                image *= maskGT_float

                x, y, w, h = cv2.boundingRect(mask_np)
                img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
                img_gt = gt_img[:, y:y + h, x:x + w].unsqueeze(0)

                psnr_ = criterion_psnr(image, gt_img)
                ssim_ = ssim(img_pred, img_gt)
                lpips_ = criterion_lpips.forward_tensor(img_pred, img_gt)

                print("fIdx-cIdx: {}-{} | PSNR: {:3.4f}  | SSIM: {:3.4f} | LPIPS: {:3.4f}".format(data['fIdx'][0].item(), data['cIdx'][0].item(), psnr_.item(), ssim_.item(), lpips_.item() * 1.0))
                metrics_psnr.append(psnr_)
                metrics_ssim.append(ssim_)
                metrics_lpips.append(lpips_)
                metrics_name.append('{}-{}'.format(  data['fIdx'][0].item(), data['cIdx'][0].item()   ))

                if self.cfg.saveDebug:
                    image_np = render_pkg["render"].permute(1,2,0).data.cpu().numpy()
                    makePath(osp.join(cfg.record.debug_path, 'render_test'))

                    cv2.imwrite(osp.join(cfg.record.debug_path, 'render_test', '{}_{}.png'.format( \
                        str(data['fIdx'][0].cpu().item()).zfill(6), str(data['cIdx'][0].cpu().item()).zfill(3) )),\
                                image_np[:,:,::-1] * 255.0,
                                # [cv2.IMWRITE_PNG_COMPRESSION, 0]
                                )

        print("Mean PSNR: {:3.4f}".format(torch.mean(torch.stack(metrics_psnr, 0 ))))
        print("Mean SSIM: {:3.4f}".format(torch.mean(torch.stack(metrics_ssim, 0 ))))
        print("Mean LPIPS: {:3.4f}".format(1 * torch.mean(torch.stack(metrics_lpips, 0 ))))
        print("This is just for reference, please run ./evaluations/00_eval_dut.py for image evaluation implemented in the main paper.")

        if self.cfg.saveCSV:
            csvData = {
                'fIdx-cIdx': metrics_name,
                'PSNR': [float(item) for item in metrics_psnr],
                'SSIM': [float(item) for item in metrics_ssim],
                'LPIPS': [float(item) for item in metrics_lpips],
            }
            saveCSV(osp.join("./experiments/CSV", cfg.exp_name + '_{}'.format(self.cfg.imgScale) + '.csv'), csvData)



    def fetch_data(self, phase, warmup=False):
        if phase == 'train':
            try:
                if warmup:
                    if self.train_set.warmup:
                        data = next(self.train_iterator)
                    else:
                        self.train_set.warmup = True
                        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                                       num_workers=self.cfg.batch_size * 16, pin_memory=True)
                        self.train_iterator = iter(self.train_loader)
                        data = next(self.train_iterator)
                else:
                    if not self.train_set.warmup:
                        data = next(self.train_iterator)
                    else:
                        self.train_set.warmup = False
                        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                                       num_workers=self.cfg.batch_size * 16, pin_memory=True)
                        self.train_iterator = iter(self.train_loader)
                        data = next(self.train_iterator)
            except:
                if warmup:
                    self.train_set.warmup = True
                    self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                                   num_workers=self.cfg.batch_size * 16, pin_memory=True)
                    self.train_iterator = iter(self.train_loader)
                    data = next(self.train_iterator)
                else:
                    self.train_set.warmup = False
                    self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                                   num_workers=self.cfg.batch_size * 16, pin_memory=True)
                    self.train_iterator = iter(self.train_loader)
                    data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)
        elif phase == 'test':
            try:
                data = next(self.test_iterator)
            except:
                self.test_iterator = iter(self.test_loader)
                data = next(self.test_iterator)

        for item in data.keys():
            data[item] = data[item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)

        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.optimizer.load_state_dict(ckpt['optimizer'])

            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--outDir', type=str, default='')
    parser.add_argument('--expName', type=str, default='')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--lowRes', action='store_true')
    parser.add_argument('--with2D', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--withRelaRot', action='store_true')
    parser.add_argument('--withPostScale', action='store_true')
    parser.add_argument('--level', type=int, default=-1)
    parser.add_argument('--saveDebug', action='store_true')
    parser.add_argument('--saveCSV', action='store_true')
    parser.add_argument('--imgScale', type=float, default=-1.0)
    parser.add_argument('--numStep', type=int, default=-1)
    parser.add_argument('--weightColor', type=float, default=-1.0)
    parser.add_argument('--weightChamfer', type=float, default=-1.0)
    parser.add_argument('--weightSSIM', type=float, default=-1.0)
    parser.add_argument('--weightReg', type=float, default=-1.0)
    parser.add_argument('--weightMRF', type=float, default=-1.0)
    parser.add_argument('--addPostScale', type=float, default=-1.0)
    parser.add_argument('--texResCano', type=int, default=-1)
    parser.add_argument('--texResGau', type=int, default=-1)
    parser.add_argument('--withRotReg', action='store_true')
    parser.add_argument('--withDeformNormal', action='store_true')
    parser.add_argument('--withStepDecay', action='store_true')
    parser.add_argument('--lrDecayStep', type=int, default=2000000)
    parser.add_argument('--lr', type=float, default=-1.0)
    parser.add_argument('--lossFreq2', type=int, default=-1)
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--deltaType', type=str, default='xyz')
    parser.add_argument('--noRGBInput', action='store_true')
    parser.add_argument('--withOccAug', action='store_true')
    parser.add_argument('--sparseMotion', action='store_true')
    args = parser.parse_args()

    cfg = config()
    cfg.load(args.config)
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    if args.deltaType == 'xyz':
        cfg.deltaType = 'xyz'
    elif args.deltaType == 'xyz_nodeform':
        cfg.deltaType = 'xyz_nodeform'
    elif args.deltaType == 'xyz_norgb':
        cfg.deltaType = 'xyz_norgb'
    elif args.deltaType == 'xyz_nonormal':
        cfg.deltaType = 'xyz_nonormal'
    else:
        print("Unspported deltaType: {}".format(args.deltaType))
        exit()
    if len(args.expName) > 0:
        cfg.exp_name =  '%s_%s%s_%s%s%s' % (cfg.name + '_gaussian'+'_{}'.format(args.expName) + '_'+cfg.deltaType, str(dt.month).zfill(2), str(dt.day).zfill(2) , str(dt.hour).zfill(2), str(dt.minute).zfill(2), str(dt.second).zfill(2))
    else:
        cfg.exp_name =  '%s_%s%s_%s%s%s' % (cfg.name + '_gaussian' +'_' + cfg.deltaType, str(dt.month).zfill(2), str(dt.day).zfill(2) , str(dt.hour).zfill(2), str(dt.minute).zfill(2), str(dt.second).zfill(2))
    cfg.record.ckpt_path = "%s/%s/ckpt" % (cfg.outDir ,cfg.exp_name)
    cfg.record.logs_path = "%s/%s" % (cfg.outDir ,cfg.exp_name)
    cfg.record.file_path = "%s/%s/file" % (cfg.outDir ,cfg.exp_name)
    cfg.record.debug_path = "%s/%s/debug" % (cfg.outDir ,cfg.exp_name)
    if len(args.ckpt) >0:
        cfg.restore_ckpt = args.ckpt
    if args.numStep > 0:
        cfg.gaussian.num_steps = args.numStep
    if args.texResCano >0:
        cfg.dataset.texResCano = args.texResCano
    if args.texResGau >0:
        cfg.dataset.texResGau = args.texResGau
    if args.weightColor > -1:
        cfg.weightColor = args.weightColor
    if args.weightSSIM > -1:
        cfg.weightSSIM = args.weightSSIM
    if args.weightMRF > -1:
        cfg.weightMRF = args.weightMRF
    if args.weightChamfer > -1:
        cfg.weightChamfer = args.weightChamfer
    if args.weightReg > -1:
        cfg.weightReg = args.weightReg
    if args.noRGBInput:
        cfg.noRGBInput = args.noRGBInput

    if args.sparseMotion:
        cfg.sparseMotion  = args.sparseMotion
    if args.withOccAug:
        cfg.withOccAug = args.withOccAug

    if args.lr >-1:
        cfg.gaussian.lr = args.lr

    if args.level > -1:
        cfg.level = args.level


    if args.lossFreq2 > -1:
        cfg.record.loss_freq2 = args.lossFreq2

    cfg.saveDebug = args.saveDebug
    cfg.saveCSV = args.saveCSV
    if args.imgScale > 0.0:
        cfg.imgScale = args.imgScale
    cfg.gaussian.with2D = args.with2D
    cfg.withRotReg = args.withRotReg
    cfg.withRelaRot = args.withRelaRot
    cfg.withPostScale = args.withPostScale
    cfg.lrDecayStep = args.lrDecayStep
    cfg.withStepDecay = args.withStepDecay
    cfg.withDeformNormal = args.withDeformNormal

    if args.addPostScale > -1:
        cfg.addPostScale = args.addPostScale

    cfg.freeze()

    if args.split=='train' or (args.split=='test' and args.saveDebug):
        for path in [cfg.record.ckpt_path, cfg.record.logs_path, cfg.record.file_path]:
            Path(path).mkdir(exist_ok=True, parents=True)

        with open(osp.join(osp.dirname(cfg.record.debug_path), 'config_updated.yaml'), "w") as f:
            f.write(cfg.dump())
    seed_everything(1314)


    print("config: {}".format(args.config))
    print("withOccAug: {}".format(cfg.withOccAug))
    print("deltaType: {}".format(cfg.deltaType))
    print('texResCano: {}'.format(cfg.dataset.texResCano))
    print('texResGeo: {} | texResGau: {}'.format(cfg.dataset.texResGeo, cfg.dataset.texResGau))
    print('weightColor: {} | weightSSIM: {} | weightMRF: {} | weightChamfer: {} | weightReg: {}'.format(cfg.weightColor,\
                                                                                        cfg.weightSSIM, cfg.weightMRF, \
                                                                                        cfg.weightChamfer, cfg.weightReg))
    print('numStep: {}'.format(cfg.gaussian.num_steps))
    print('withPostScale: {}'.format(cfg.withPostScale))
    print('lrDecayStep: {}'.format(cfg.lrDecayStep))
    print("addPostScale: {}".format(cfg.addPostScale))
    print("noRGBInput: {} | noNormalInput: {}".format(cfg.noRGBInput, cfg.noNormalInput))

    trainer = Trainer(cfg, split=args.split)
    if args.split=='train':
        trainer.train()
    elif args.split=='test':
        trainer.run_test()
