"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2024-07-30
"""
import sys
from sys import platform
import os

import numpy as np
import logging
import math

from datasets.domeDatasetInfer import DomeDatasetInfer

from models.network import Regresser2
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from models.train_recoder import Logger, file_backup
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from configs.my_config import ConfigDUT as config

from models.unet import UNet
from models.loss import l1_loss
import pdb
import cv2
import os.path as osp

from models.utils import index_feat, makePath, save_ply, seed_everything
from torch.utils.tensorboard import SummaryWriter
import time
import nvdiffrast.torch as dr

from cuda_projection import projection_fw as projectPointsCuda
from gaussian_renderer import render as render_gaussian

from scene.gaussian_model import GaussianModel
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import pdb
from simple_knn._C import distCUDA2
from pytorch3d.transforms import axis_angle_to_matrix
from utils.sh_utils import RGB2SH, SH2RGB

from kaolin.metrics.trianglemesh import point_to_mesh_distance, uniform_laplacian_smoothing, uniform_laplacian
from kaolin.metrics.pointcloud import chamfer_distance
from kornia.morphology import erosion, dilation

from tools.socket_tools import BaseSocketClient, Config, read_smpl, read_keypoints3d, encode_image_opencv
from tools.config_tools import load_config_with_default
from tools.skel_tools import Skeleton, loadMotion, saveMotion, load_model, load_ddc_param, saveMeshes
from tools.omni_tools import makePath, mask_to_rect, resizeImg, checkPlatformDir
from tools.mesh_tools import save_ply, compute_normal_torch
from utils.render_utils import render_uv, render_depth_batch
from utils.geometry_utils import transform_pos_batch, compute_edge_length, find_vertex_to_face_index
from utils.graphics_utils import focal2fov, getProjectionMatrix_refine


class Mock_PipelineParams:
    def __init__(self):
        """
           Nothing but a hack
        """
        self.convert_SHs_python = False
        self.compute_cov3D_python = False

class Trainer:
    def __init__(self, cfg_file, split='train'):
        self.cfg = cfg_file

        global saveTexFirst, saveTexSecond, sendData, saveDebug, sendImage, handScale, vis, demo
        self.saveTexFirst = saveTexFirst
        self.saveTexSecond = saveTexSecond
        self.sendData = sendData
        self.saveDebug = saveDebug
        self.sendImage = sendImage
        self.handScale = handScale
        self.vis = vis
        self.demo = demo

        self.device = "cuda:0"
        self.is_white_background = self.cfg.dataset.is_white_background
        if self.is_white_background:
            self.background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device)
        else:
            self.background = torch.tensor([0., 0., 0.], dtype=torch.float32).to(self.device)

        self.model_geometry = Regresser2(self.cfg, rgb_dim=6)
        self.inputDim = 6 # if self.cfg.withNormal else 3
        self.sh_degree = self.cfg.gaussian.sh_degree
        self.model_gaussian = UNet(self.inputDim, 11 + 3* (self.sh_degree+1)**2 ).to(self.device)
        self.split = split

        self.condCameraIdxs = self.cfg.dataset.condCameraIdxs
        self.numCondCam = len(self.condCameraIdxs)

        self.val_set = DomeDatasetInfer(self.cfg, split=split, vis=vis, demo=args.demo)
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
        self.val_iterator = iter(self.val_loader)
        self.len_val = int(len(self.val_set))
        self.texResCano = self.cfg.dataset.texResCano
        self.texResGeo = self.cfg.dataset.texResGeo
        self.texResGau = self.cfg.dataset.texResGau


        use_opengl = False
        self.glctx0 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.glctx1 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.glctx2 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.glctx3 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.glctx4 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.glctx5 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.glctx6 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.glctx7 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.glctx8 = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()



        self.total_steps = 0
        self.model_geometry.cuda()
        self.model_gaussian.cuda()
        if len(self.cfg.ckpt1) > 0:
            self.load_ckpt1(self.cfg.ckpt1, load_optimizer=False)
        if len(self.cfg.ckpt2) > 0:
            self.load_ckpt2(self.cfg.ckpt2, load_optimizer=False)
        self.model_geometry.eval()
        self.model_gaussian.eval()

        self.eg = self.val_set.eg
        self.faces = self.val_set.faces.cuda()
        self.uvs = self.val_set.uvs.cuda()
        self.facesuv = self.val_set.facesuv.cuda()
        self.uvs_unique = self.val_set.uvs_unique.cuda()
        self.verts0 = self.val_set.verts0.cuda()
        self.normal_cano = self.val_set.normal_cano.cuda()

        self.canoMask = render_uv(self.glctx0, self.uvs, self.facesuv, [self.texResGau, self.texResGau], torch.ones([self.eg.character.numVert, 1]).cuda(), self.faces)[0, :, :, 0]
        self.uv_idx, self.uv_idy = torch.where(self.canoMask > 0)
        self.gaussians = GaussianModel(sh_degree = self.sh_degree)
        self.mocked_pipeline = Mock_PipelineParams()

        self.edgeLengthCano = compute_edge_length(self.verts0[None], self.faces)[0]
        self.vert2face = find_vertex_to_face_index(self.verts0.shape[0], self.faces)

        self.handMask = self.eg.character.handMask
        self.numDof = self.eg.character.motion_base.shape[1]


        kernel = torch.ones(3, 3).cuda()
        canoMaskErosion = erosion(self.canoMask.clone()[None, None], kernel)#[0,0]
        canoMaskDilation = dilation(self.canoMask.clone()[None, None], kernel)#[0,0]
        canoMaskBoundary = canoMaskDilation - canoMaskErosion
        mask_bound_temp = index_feat(canoMaskBoundary, (self.uvs_unique.transpose(0, 1)[None].repeat(canoMaskBoundary.shape[0], 1, 1) - 0.5) * 2)[:, :].transpose(1, 2)
        mask_bound_temp = mask_bound_temp.reshape(-1,1)
        self.boudMask = torch.where(mask_bound_temp > 0.5, 1.0, 0.0)
        self.L = uniform_laplacian(self.verts0.shape[0], self.faces.to(torch.long))

    def run_eval(self):
        print(f"Doing validation ...")
        torch.cuda.empty_cache()
        metrics = []

        if self.sendData:
            host = 'XXX.XXX.XXX.XXX'
            port = 9999
            client = BaseSocketClient(host, port)

        if self.sendImage:
            host = 'XXX.XXX.XXX.XXX'
            port = 9998
            client = BaseSocketClient(host, port)

        frameList = []
        vertLBSList = []
        vertDeformedList = []
        metrics = []

        for idx in tqdm(range(self.len_val)):
            data = next(self.val_iterator)


            fIdx = data['fIdx'][0]
            camPoss = data['camPoss'][0].cuda()
            H = int(data['H'][0])
            W = int(data['W'][0])
            Ks = data['Ks'][0].cuda()
            Es = data['Es'][0].cuda()
            Ps = data['Ps'][0].cuda()

            H_render = int(data['H_render'][0])
            W_render = int(data['W_render'][0])
            K_render = data['K_render'][0].cuda()
            E_render = data['E_render'][0].cuda()

            imgs = data['imgs'][0].cuda()
            mtx = data['mtx'][0].cuda()
            motion = data['motion'].cuda()

            st = time.time()
            with torch.no_grad():
                verts, T_fw = self.eg.character.forward_cuda(motion, returnTransformation=True)
                # verts, T_fw = self.eg.character.forward(data['motion'].cuda(), returnTransformation=True)
                T_fw[:, :, :3, 3] /= 1000.0
                ed = time.time() - st
                # print(ed * 1000.0)
                print("--------Stage I------------")
                print("forward kinematic: {}".format(ed * 1000.0))

                # tempMotion = data['motion'].clone()
                # tempMotion[:,6:] = 0.0
                # verts2, T_fw2 = self.eg.character.forward(tempMotion.cuda(), returnTransformation=True)

                rootTransform = self.eg.character.skeleton.jointGlobalTransformations[:, 6, :, :].data
                rootTransform[:, :3, 3] /= 1000.0
                T_fw_noroot_R = torch.matmul(rootTransform[:, :3, :3].transpose(1, 2)[:, None], T_fw[:, :, :3, :3])  # B,3,3   x B,N,3,3
                normalWorld_noroot = torch.matmul(T_fw_noroot_R, self.normal_cano[None, :, :, None])[0, :, :, 0]  # T,N,3,1

                st_local = time.time()
                edgeLengthCur = compute_edge_length(verts, self.faces)[0]
                postScale_temp, _ = torch.max((edgeLengthCur / self.edgeLengthCano)[self.vert2face], dim=1)
                postScale_temp *= 1.0 #1.25 #1.0
                postScale_temp = torch.clamp_min(postScale_temp, 1.0).reshape(-1,1)
                print("fIdx: {} - {}".format(fIdx, postScale_temp.max()))
                ed = time.time() - st_local
                print("[local] compute postscale: {}".format(ed * 1000.0))

                st_local = time.time()
                normals_ = compute_normal_torch(verts.clone(), self.faces)[0]  # N,3
                # normals__ = normals_.clone()
                ed = time.time() - st_local
                print("[local] compute normal: {}".format(ed * 1000.0))

                temp_pos = torch.cat([verts[0] / 1000.0, normals_, normalWorld_noroot], -1)
                ed = time.time() - st_local
                print("[local] concatenate: {}".format(ed * 1000.0))
                render_temp = render_uv(self.glctx1, self.uvs, self.facesuv, [self.texResCano, self.texResCano], temp_pos, self.faces)
                ed = time.time() - st_local
                print("render: {}".format(ed * 1000.0))
                pos_uv = render_temp[:, :, :, :3]
                normal_uv = render_temp[:, :, :, 3:6]
                normal_uv_noroot = render_temp[:, :, :, 6:9]
                ed = time.time() - st_local
                # print(ed * 1000.0)
                print("render uv normal and position: {}".format(ed * 1000.0))
                ed = time.time() - st
                print("render uv normal and position: {}".format(ed * 1000.0))
                # pdb.set_trace()
                rayDir = torch.nn.functional.normalize(-(pos_uv - camPoss[:, None, None]), dim=-1)
                temp = (rayDir * normal_uv).sum(-1)
                # ed = time.time() - st
                vis_uv = temp > 0.17
                verts_image_space = projectPointsCuda(pos_uv.view(1, self.texResCano * self.texResCano, 3), Ps = Ps[None], H=H, W=W)
                ed = time.time() - st
                # print(ed * 1000.0)
                # print()
                print("project points: {}".format(ed * 1000.0))

                depth = render_depth_batch(self.glctx2, mtx, verts / 1000.0, self.faces, resolutions=[H, W]).permute(0, 3, 1, 2)
                temp = index_feat(imgs, verts_image_space[0, :, :, :2].transpose(1, 2)).reshape(self.numCondCam, 4, self.texResCano,
                                                                                                self.texResCano)
                depth_uv = verts_image_space[0, :, :, 2].reshape(self.numCondCam, self.texResCano, self.texResCano)
                temp_depth = index_feat(depth, verts_image_space[0, :, :, :2].transpose(1, 2)).reshape(self.numCondCam, self.texResCano,
                                                                                                       self.texResCano)

                vis_uv_depth = torch.where(torch.abs(depth_uv - temp_depth) < 0.02, 1, 0)
                vis_uv_new = vis_uv * temp[:, 3, :, :] * vis_uv_depth
                temp = temp[:, :3, :, :] * vis_uv_new[:, None]
                vis_uv_sum = vis_uv_new.sum(0)
                vis_uv_sum = torch.where(vis_uv_sum > 0, 1 / vis_uv_sum, 0)
                temp2 = temp.sum(0) * vis_uv_sum[None]

                if self.saveTexFirst:
                    if args.texRes == 1024:
                        makePath(self.val_set.texDir+'_1k')
                        cv2.imwrite(osp.join(osp.join(self.val_set.texDir+'_1k', str(int(fIdx)).zfill(6)) + '.jpg'), \
                                (temp2.permute(1, 2, 0)).data.cpu().numpy()[::-1, :, ::-1] * 255.0)
                    else:
                        makePath(self.val_set.texDir)
                        cv2.imwrite(osp.join(osp.join(self.val_set.texDir, str(int(fIdx)).zfill(6)) + '.jpg'), \
                                (temp2.permute(1, 2, 0)).data.cpu().numpy()[::-1, :, ::-1] * 255.0)
                    continue


                temp2 = temp2 * 2 -1
                temp2 = torch.cat([temp2, normal_uv_noroot[0].permute(2, 0, 1)], 0)

                ed = time.time() - st
                # print(ed * 1000.0)
                print("render depth and get uv unprojection map: {}".format(ed * 1000.0))

                temp2 = T.Resize((self.texResGeo,self.texResGeo), antialias=True)(temp2)

                if self.cfg.noRGBInput:
                    temp2[:3] *= 0.0
                if self.cfg.noNormalInput:
                    temp2[3:6] *= 0.0

                delta_temp = self.model_geometry(temp2[None])
                # torch.cuda.synchronize()
                ed = time.time() - st
                print("geometry infer: {}".format(ed * 1000.0))

                delta_temp = index_feat(delta_temp,
                                   (self.uvs_unique.transpose(0, 1)[None].repeat(delta_temp.shape[0], 1, 1) - 0.5) * 2)[
                             :, :].transpose(1, 2)

                if 'nodeform' in self.cfg.deltaType:
                    delta_temp = torch.zeros_like(delta_temp)


                out_verts = self.verts0[None] / 1000.0 + delta_temp

                if False:
                    out_verts = torch.matmul(self.L, out_verts) * self.boudMask[None] + out_verts

                out_verts_world = (torch.matmul(T_fw[:,:,:3,:3], out_verts[:,:,:,None]) + T_fw[:,:,:3,3:])[:,:,:,0].data
                # out_verts_world2 = (torch.matmul(T_fw2[:, :, :3, :3], out_verts[:, :, :, None]) + T_fw2[:, :, :3, 3:])[:, :, :, 0].data
                ed = time.time() - st
                print("geometry transformation: {}".format(ed * 1000.0))
                print()

                if self.sendData:

                    data0 = {
                        'id': 0,
                        'vertices': out_verts_world[0].data.cpu().numpy()
                        # 'vertices': out_verts[0].data.cpu().numpy()
                    }
                    data1 = {
                        'id': 1,
                        'vertices': verts[0].data.cpu().numpy()/1000.0 + np.array([[1.0,0,0]])
                    }
                    print(data1['vertices'].shape)
                    datas = [data0, data1]
                    client.send_vertices(datas)
                    continue

                print("--------Stage II------------")
                st_local = time.time()
                normals_ = compute_normal_torch(out_verts_world, self.faces)[0]  # N,3
                ed = time.time() - st_local
                print("compute normal: {}".format(ed * 1000.0))
                temp_pos = torch.cat([out_verts_world[0] / 1.0, normals_], -1)
                ed = time.time() - st_local
                print("concatenate: {}".format(ed * 1000.0))

                render_temp = render_uv(self.glctx3, self.uvs, self.facesuv, [self.texResCano, self.texResCano], temp_pos, self.faces)
                ed = time.time() - st_local
                print("render: {}".format(ed * 1000.0))
                pos_uv = render_temp[:, :, :, :3]
                normal_uv = render_temp[:, :, :, 3:6]
                ed = time.time() - st_local
                print("render uv normal and position: {}".format(ed * 1000.0))
                ed = time.time() - st
                print("render uv normal and position: {}".format(ed * 1000.0))

                rayDir = torch.nn.functional.normalize(-(pos_uv - camPoss[:, None, None]), dim=-1)
                temp = (rayDir * normal_uv).sum(-1)
                if self.cfg.removeNormalMap:
                    vis_uv = temp > 0.17
                else:
                    vis_uv = temp > 0.17
                verts_image_space = projectPointsCuda(pos_uv.view(1, self.texResCano * self.texResCano, 3), Ps = Ps[None], H=H, W=W)
                ed = time.time() - st
                print("project points: {}".format(ed * 1000.0))

                depth = render_depth_batch(self.glctx4, mtx, out_verts_world / 1.0, self.faces, resolutions=[H, W]).permute(0, 3, 1, 2)
                temp = index_feat(imgs, verts_image_space[0, :, :, :2].transpose(1, 2)).reshape(self.numCondCam, 4, self.texResCano,
                                                                                                self.texResCano)
                depth_uv = verts_image_space[0, :, :, 2].reshape(self.numCondCam, self.texResCano, self.texResCano)
                temp_depth = index_feat(depth, verts_image_space[0, :, :, :2].transpose(1, 2)).reshape(self.numCondCam, self.texResCano,
                                                                                                       self.texResCano)

                vis_uv_depth = torch.where(torch.abs(depth_uv - temp_depth) < 0.02, 1, 0)
                if self.cfg.removeNormalMap:
                    vis_uv_new = temp[:, 3, :, :] * vis_uv_depth
                else:
                    vis_uv_new = vis_uv * temp[:, 3, :, :] * vis_uv_depth

                temp = temp[:, :3, :, :] * vis_uv_new[:, None]
                vis_uv_sum = vis_uv_new.sum(0)
                vis_uv_sum = torch.where(vis_uv_sum > 0, 1 / vis_uv_sum, 0)
                temp2 = temp.sum(0) * vis_uv_sum[None]

                if self.saveTexSecond:
                    displacments = delta_temp
                    makePath(self.val_set.dispDir)
                    np.save(osp.join(self.val_set.dispDir, '{}.npy'.format(  str(int(data["fIdx"][0])).zfill(6) ) ), displacments[0].data.cpu().numpy())

                    if args.texRes == 1024:
                        makePath(self.val_set.texDirDeformed+'_1k')
                        cv2.imwrite(osp.join(osp.join(self.val_set.texDirDeformed+'_1k', str(int(fIdx)).zfill(6)) + '.jpg'), \
                                (temp2.permute(1, 2, 0)).data.cpu().numpy()[::-1, :, ::-1] * 255.0)
                    else:
                        makePath(self.val_set.texDirDeformed)
                        cv2.imwrite(osp.join(osp.join(self.val_set.texDirDeformed, str(int(fIdx)).zfill(6)) + '.jpg'), \
                                (temp2.permute(1, 2, 0)).data.cpu().numpy()[::-1, :, ::-1] * 255.0)
                    continue



                # pdb.set_trace()
                temp2 = temp2 * 2 -1
                ed = time.time() - st
                print("render depth and get uv unprojection map: {}".format(ed * 1000.0))

                verts_temp = torch.cat([out_verts[0], postScale_temp, T_fw[0].reshape(-1, 16).data], -1)

                render_temp = render_uv(self.glctx5, self.uvs, self.facesuv, [self.texResGau, self.texResGau], verts_temp, self.faces)[0]

                canoPoints = render_temp[self.uv_idx, self.uv_idy, :3]
                canoPostScale = render_temp[self.uv_idx, self.uv_idy, 3:4]
                canoTransformations = render_temp[self.uv_idx, self.uv_idy, 4:].reshape([-1, 4, 4])
                ed = time.time() - st
                print("render pos & transform map: {}".format(ed * 1000.0))

                temp2 = torch.cat([temp2, normal_uv_noroot[0].permute(2,0,1)],0)
                if self.cfg.noRGBInput:
                    temp2[:3] *= 0.0

                temp2 = T.Resize((self.texResGau, self.texResGau), antialias=True)(temp2)
                # torch.cuda.synchronize()
                full_feats = self.model_gaussian(temp2[None])[0]

                full_feats = full_feats[:, self.uv_idx, self.uv_idy].permute(1, 0).contiguous()
                ed = time.time() - st
                print("infer gaussian: {}".format(ed * 1000.0))

                canoDelta = full_feats[:, 8:11]
                canoPoints = canoPoints + canoDelta
                self.gaussians._xyz = canoPoints
                self.gaussians._scaling = full_feats[:, :3] + torch.log(canoPostScale)
                self.gaussians._rotation = full_feats[:, 3:7]
                self.gaussians._opacity = full_feats[:, 7].unsqueeze(1)
                self.gaussians._features_dc = full_feats[:, 11:14].reshape(-1, 1, 3)
                self.gaussians._features_rest = full_feats[:, 14:].reshape(canoPoints.shape[0], -1, 3)

                render_pkg = render_gaussian(K_render, E_render, H_render, W_render, canoTransformations, self.gaussians, self.mocked_pipeline, self.background)
                ed = time.time() - st
                print("render gaussian: {}".format(ed * 1000.0))
                print()

                if self.saveDebug:
                    if args.demo:
                        outRenderDir = checkPlatformDir('./outputs/demo/dut_demo_{}_gau{}'.format(self.cfg.dataset.subject, self.texResGau))
                    else:
                        outRenderDir = checkPlatformDir('Y:/HOIMOCAP5/nobackup/debug/dut_{}_{}'.format(self.cfg.dataset.subject, self.split))
                    makePath(outRenderDir)
                    fIdx = idx
                    cv2.imwrite(osp.join(outRenderDir, str(int(fIdx)).zfill(6) + '.png'),
                                (render_pkg["render"].permute(1, 2, 0)).data.cpu().numpy()[:, :, ::-1] * 255.0,
                                # [cv2.IMWRITE_PNG_COMPRESSION, 0]
                                )
                    continue

                imgOut = ((render_pkg["render"].permute(1, 2, 0)).data.cpu().numpy()[:, :, ::-1] * 255.0).astype(np.uint8)
                if self.sendImage:
                    dataImg0 = {
                        'id': 0,
                        'images': encode_image_opencv(imgOut)
                    }
                    datas = [dataImg0]
                    client.send_images(datas)


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

    def load_ckpt1(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        print(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model_geometry.load_state_dict(ckpt['network'], strict=strict)
        print(f"Parameter loading done")

    def load_ckpt2(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        print(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model_gaussian.load_state_dict(ckpt['network'], strict=strict)
        print(f"Parameter loading done")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt1', type=str, default='')
    parser.add_argument('--ckpt2', type=str, default='')
    parser.add_argument('--texRes', type=int, default=-1)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--removeNormalMap', action='store_true')
    parser.add_argument('--imgScale', type=float, default=-1.0)
    parser.add_argument('--saveTexFirst', action='store_true')
    parser.add_argument('--saveTexSecond', action='store_true')
    parser.add_argument('--sendData', action='store_true')
    parser.add_argument('--sendImage', action='store_true')
    parser.add_argument('--saveDebug', action='store_true')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--texResGeo', type=int, default=-1)
    parser.add_argument('--texResGau', type=int, default=-1)
    parser.add_argument('--deltaType', type=str, default='xyz')
    parser.add_argument('--handScale', type=float, default=1.0)
    parser.add_argument('--noNormalInput', action='store_true')
    parser.add_argument('--noRGBInput', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--demo', action='store_true')



    args = parser.parse_args()

    cfg = config()
    cfg.load(args.config)
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.exp_name =  '%s_%s%s_%s%s%s' % (cfg.name + '_gaussian', str(dt.month).zfill(2), str(dt.day).zfill(2) , str(dt.hour).zfill(2), str(dt.minute).zfill(2), str(dt.second).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.record.debug_path = "experiments/%s/debug" % cfg.exp_name

    cfg.ckpt1 = args.ckpt1
    cfg.ckpt2 = args.ckpt2

    global saveTexFirst, saveTexSecond, sendData, saveDebug, sendImage, handScale, vis, demo
    saveTexFirst = args.saveTexFirst
    saveTexSecond = args.saveTexSecond
    sendData = args.sendData
    saveDebug = args.saveDebug
    sendImage = args.sendImage
    handScale = args.handScale
    vis = args.vis
    demo = args.demo

    if args.noRGBInput:
        cfg.noRGBInput = args.noRGBInput
    if args.noNormalInput:
        cfg.noNormalInput = args.noNormalInput
    if args.sparse:
        cfg.sparse = args.sparse

    if args.texRes > 0:
        cfg.dataset.texRes = args.texRes
    if args.imgScale > 0.0:
        cfg.imgScale = args.imgScale
    if args.texResGeo > 0:
        cfg.dataset.texResGeo = args.texResGeo
    if args.texResGau > 0:
        cfg.dataset.texResGau = args.texResGau

    cfg.removeNormalMap = args.removeNormalMap
    cfg.deltaType = args.deltaType
    cfg.freeze()

    print("vis: {}".format(vis))
    print("split: {}".format(args.split))
    print('texResGeo: {} | texResGau: {}'.format(cfg.dataset.texResGeo, cfg.dataset.texResGau))
    print('Cond views: {}'.format(cfg.dataset.condCameraIdxs))
    print("deltaType: {}".format(cfg.deltaType))
    print("noRGBInput: {} | noNormalInput: {}".format(cfg.noRGBInput, cfg.noNormalInput))
    if 'nodeform' in cfg.deltaType:
        print("We don't perform deformation, so it's single unprojection.")
    else:
        print("We perform deformation on template, so it's double unprojection.")
    seed_everything(1314)

    trainer = Trainer(cfg, split = args.split)
    trainer.run_eval()
