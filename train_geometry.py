import os
import os.path as osp
import pdb
import cv2
import logging

from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kaolin.metrics.pointcloud import chamfer_distance
# from chamfer_distance import ChamferDistance
from pytorch3d.transforms import axis_angle_to_matrix

from configs.my_config import ConfigDUT as config
from datasets.domeDataset import DomeDataset
from models.network import Regresser2
from models.train_recoder import Logger, file_backup
from models.loss import l1_loss, l2_loss, compute_laplacian_loss, compute_normal_consistency
from models.utils import index_feat, makePath, save_ply, setup_logger, seed_everything
from utils.geometry_utils import transform_pos_batch, compute_edge_length, find_edges_with_two_faces, compute_self_intersection


class Trainer:
    def __init__(self, cfg_file, split='train'):
        self.cfg = cfg_file
        global noDeform, handScale, noEval
        self.handScale = handScale
        self.noEval = noEval

        self.inputDim = 6 #if self.cfg.withNormal else 3
        self.model = Regresser2(self.cfg, rgb_dim =  self.inputDim)
        self.train_set = DomeDataset(self.cfg, split='train')

        self.uv_idx = self.train_set.uv_idx.cuda()
        self.uv_idy = self.train_set.uv_idy.cuda()
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size * 8, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.len_train = int(len(self.train_loader))

        self.lr = self.cfg.lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.cfg.wdecay, eps=1e-8)

        if self.noEval:
            pass
        else:
            self.val_set = DomeDataset(self.cfg, split='test')
            self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
            self.val_iterator = iter(self.val_loader)
            self.len_val = int(len(self.val_loader))
        self.logger, _ = setup_logger(osp.dirname(self.cfg.record.debug_path), save=True)


        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.train()

        self.faces = self.train_set.faces.cuda()
        self.faces_np = self.train_set.faces.data.cpu().numpy().astype(np.int64)
        self.uvs = self.train_set.uvs.cuda()
        self.facesuv = self.train_set.facesuv.cuda()
        self.uvs_unique = self.train_set.uvs_unique.cuda()
        self.handMask = self.train_set.eg.character.handMask
        self.numDof = self.train_set.eg.character.motion_base.shape[1]

        self.edge2vert, self.edge2face  = find_edges_with_two_faces(self.faces)

        self.writer = SummaryWriter(log_dir=self.cfg.record.logs_path)
        # self.compute_chamfer = ChamferDistance()


    def compute_rotation_matrix_within_cone(self, delta_angle, max_angle=30 / 180 * np.pi):
        delta_zeros = torch.zeros_like(delta_angle)[:, :, :1]
        Rs = axis_angle_to_matrix(torch.cat([delta_angle[:, :, :1], delta_zeros, delta_angle[:, :, 1:]], dim=-1))
        return Rs

    def rotate_normal_within_cone(self, delta_angle, normal, max_angle=30 / 180 * np.pi):
        Rs = self.compute_rotation_matrix_within_cone(delta_angle, max_angle=max_angle)
        return torch.matmul(Rs, normal[:,:,:,None])[:,:,:,0]

    def train(self):
        # pbar = tqdm(range(self.total_steps, self.cfg.num_steps))
        pbar = range(self.total_steps, self.cfg.num_steps)
        self.update_learning_rate()

        for _ in pbar:
            self.update_learning_rate()
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            delta_temp = self.model(data["inputs"])
            delta_temp = index_feat(delta_temp, (self.uvs_unique.transpose(0,1)[None].repeat(delta_temp.shape[0],1,1) -0.5)*2)[:,:].transpose(1, 2)

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq2 == 0:
                self.save_ckpt(save_path=Path('%s/%s_%s.pth' % (cfg.record.ckpt_path, cfg.name, str(self.total_steps) )), show_log=False)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)

            out_verts = data['verts'] + delta_temp
            out_verts_world = transform_pos_batch(data['T_fw'], out_verts)
            chamfer_loss = chamfer_distance(out_verts_world, data['pointclouds']).mean()

            if self.cfg.worldLap:
                lap_loss = compute_laplacian_loss(out_verts_world, self.faces.to(torch.long))
            else:
                lap_loss = compute_laplacian_loss(out_verts, self.faces.to(torch.long))
            iso_loss = l2_loss(compute_edge_length(out_verts,  self.faces.to(torch.long)), compute_edge_length(data['verts'],  self.faces.to(torch.long)))
            nmlCons_loss = compute_normal_consistency(out_verts, self.faces, self.edge2face)

            loss = chamfer_loss + self.cfg.weightLap * lap_loss + self.cfg.weightIso  * iso_loss + self.cfg.weightNmlCons * nmlCons_loss
            if self.total_steps % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.total_steps)
                self.writer.add_scalar('train/chamfer_loss', chamfer_loss.item(), self.total_steps)
                self.writer.add_scalar('train/lap_loss', lap_loss.item(), self.total_steps)
                self.writer.add_scalar('train/iso_loss', iso_loss.item(), self.total_steps)
                self.writer.add_scalar('train/nmlCons_loss', nmlCons_loss.item(), self.total_steps)

            loss_stats = "Iter: {} | Loss: {:.4f} | Chamfer: {:.4f} | Lap: {:4f} | Iso: {:.4f} | NMLCons: {:.4f}".format(
                self.total_steps, loss.item() * 1000.0, chamfer_loss.item()* 1000.0, lap_loss.item()* 1000.0, iso_loss.item()* 1000.0, nmlCons_loss.item()* 1000.0
            )
            self.logger.info(loss_stats)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if not self.noEval:
                if  self.total_steps>0 and self.total_steps< 60001 and self.total_steps % 10000 == 0:
                    self.model.eval()
                    self.run_eval()
                    self.model.train()
                elif self.total_steps > 60001  and self.total_steps % self.cfg.record.eval_freq == 0:
                    self.model.eval()
                    self.run_eval()
                    self.model.train()

            self.total_steps += 1

        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))


    def run_eval(self, split=None):
        print(f"Doing validation ...")
        torch.cuda.empty_cache()
        metricsCD = []
        metricsSR = []
        metricsSI = []
        global noDeform
        global saveDebug
        self.model.eval()

        for idx in tqdm(range(self.len_val)):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                delta_temp = self.model(data["inputs"])
                delta_temp = index_feat(delta_temp,
                                   (self.uvs_unique.transpose(0, 1)[None].repeat(delta_temp.shape[0], 1, 1) - 0.5) * 2)[:,
                             :].transpose(1, 2)

                if noDeform:
                    out_verts = data['verts']
                else:
                    if self.total_steps ==0:
                        out_verts = data['verts']
                    else:
                        out_verts = data['verts'] + delta_temp

                out_verts_world = transform_pos_batch(data['T_fw'], out_verts)
                lap = compute_laplacian_loss(out_verts_world, self.faces.to(torch.long)).item() * 100000.0
                cd = chamfer_distance(out_verts_world, data['pointclouds']).item() * 1000.0
                si = compute_self_intersection(out_verts_world[0].data.cpu().numpy(), self.faces_np)

                metricsCD.append(cd)
                metricsSR.append(lap)
                metricsSI.append(si)

                if saveDebug and split == 'test':
                    self.writer.add_scalar('cd', cd, idx)
                    self.writer.add_scalar('sr', lap, idx)
                    self.writer.add_scalar('si', si, idx)


        self.logger.info("Iter: {} | CD: {} | SR: {} | SI: {}".format(int(self.total_steps), \
                                                                      np.mean(metricsCD), np.mean(metricsSR), np.mean(metricsSI)))
        self.writer.add_scalar('val/cd', np.mean(metricsCD), self.total_steps)
        self.writer.add_scalar('val/sr', np.mean(metricsSR), self.total_steps)
        self.writer.add_scalar('val/si', np.mean(metricsSI), self.total_steps)


    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

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

    def update_learning_rate(self):

        start_constant_lr = self.cfg.lrDecayStep #650000
        alpha = 1.0 #0.1 #0.5 #0.1  # self.learning_rate_alpha

        weighted_learing_rate = self.lr

        if self.total_steps > start_constant_lr:
            weighted_learing_rate = alpha * self.lr

        for g in self.optimizer.param_groups:
            g['lr'] = weighted_learing_rate

        return
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--lrDecayStep', type=int, default=2000000)
    parser.add_argument('--saveDebug', action='store_true')
    parser.add_argument('--expName', type=str, default='')
    parser.add_argument('--weightLap', type=float, default=-1.0)
    parser.add_argument('--weightIso', type=float, default=-1.0)
    parser.add_argument('--weightNmlCons', type=float, default=-1.0)
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--texResGeo', type=int, default=-1)
    parser.add_argument('--evalStep', type=int, default=-1)
    parser.add_argument('--noRGBInput', action='store_true')
    parser.add_argument('--noNormalInput', action='store_true')
    parser.add_argument('--worldLap', action='store_true')
    parser.add_argument('--noDeform', action='store_true')
    parser.add_argument('--noEval', action='store_true')
    parser.add_argument('--handScale', type=float, default=1.0)
    parser.add_argument('--withOccAug', action='store_true')

    args = parser.parse_args()

    cfg = config()
    cfg.load(args.config)
    cfg = cfg.get_cfg()

    global noDeform, handScale, noEval, saveDebug
    noDeform = args.noDeform
    handScale = args.handScale
    noEval = args.noEval
    saveDebug = args.saveDebug

    cfg.defrost()
    dt = datetime.today()
    if len(args.expName) > 0:
        cfg.exp_name =  '%s_%s%s_%s%s%s' % (cfg.name  +"_xyz" + '_geometry'+'_{}'.format(args.expName), str(dt.month).zfill(2), str(dt.day).zfill(2) , str(dt.hour).zfill(2), str(dt.minute).zfill(2), str(dt.second).zfill(2))
    else:
        cfg.exp_name =  '%s_%s%s_%s%s%s' % (cfg.name  +"_xyz" + '_geometry', str(dt.month).zfill(2), str(dt.day).zfill(2) , str(dt.hour).zfill(2), str(dt.minute).zfill(2), str(dt.second).zfill(2))
    cfg.record.ckpt_path = "%s/%s/ckpt" % (cfg.outDir ,cfg.exp_name)
    cfg.record.logs_path = "%s/%s" % (cfg.outDir ,cfg.exp_name)
    cfg.record.file_path = "%s/%s/file" % (cfg.outDir ,cfg.exp_name)
    cfg.record.debug_path = "%s/%s/debug" % (cfg.outDir ,cfg.exp_name)
    cfg.restore_ckpt = args.ckpt
    cfg.lrDecayStep = args.lrDecayStep

    if args.weightLap > -1:
        cfg.weightLap = args.weightLap
    if args.weightIso > -1:
        cfg.weightIso = args.weightIso
    if args.weightNmlCons > -1:
        cfg.weightNmlCons = args.weightNmlCons
    if args.texResGeo > 0:
        cfg.dataset.texResGeo = args.texResGeo
    if args.evalStep>0:
        cfg.record.eval_freq = args.evalStep
    if args.noRGBInput:
        cfg.noRGBInput = args.noRGBInput
    if args.noNormalInput:
        cfg.noNormalInput = args.noNormalInput
    if args.withOccAug:
        cfg.withOccAug = args.withOccAug
    cfg.worldLap = args.worldLap
    cfg.freeze()

    if args.split=='train' or (args.split=='test' and args.saveDebug):
        for path in [cfg.record.ckpt_path, cfg.record.logs_path, cfg.record.file_path]:
            Path(path).mkdir(exist_ok=True, parents=True)

        with open(osp.join(osp.dirname(cfg.record.debug_path), 'config_updated.yaml'), "w") as f:
            f.write(cfg.dump())

    seed_everything(1314)


    print('Subject: {}'.format(cfg.dataset.subject))
    print("withOccAug: {}".format(cfg.withOccAug))
    print('Cond views: {}'.format(cfg.dataset.condCameraIdxs))
    print('lrDecayStep: {}'.format(cfg.lrDecayStep))
    print("worldLap: {}".format(cfg.worldLap))
    print("handScale: {}".format(handScale))
    print("noRGBInput: {} | noNormalInput: {}".format(cfg.noRGBInput, cfg.noNormalInput))
    print('texResGeo: {}'.format(cfg.dataset.texResGeo))
    print('weightLap: {} | weightIso: {} | weightNmlCons: {}'.format(cfg.weightLap, cfg.weightIso, cfg.weightNmlCons))

    print('noDeform: {}'.format(noDeform))

    trainer = Trainer(cfg, split=args.split)
    if args.split=='train':
        trainer.train()
    elif args.split=='test':
        trainer.run_eval(split='test')
