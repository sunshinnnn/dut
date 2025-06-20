"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2024-07-30
"""
from torch.utils.data import Dataset

import numpy as np
import os
from PIL import Image
import cv2
import torch
from pathlib import Path
import logging
import json
from tqdm import tqdm
import os.path as osp
import glob
import pdb
import random

import sys
from sys import platform
import nvdiffrast.torch as dr
import time

from tools.config_tools import load_config_with_default
from tools.skel_tools import Skeleton, loadMotion, saveMotion, load_model, load_ddc_param
from tools.omni_tools import makePath, mask_to_rect, resizeImg, checkPlatformDir
from tools.mesh_tools import save_ply, compute_normal_torch
from tools.cam_tools import projectPoints, load_camera_param, load_camera_param_sparse
from tools.pyrender_tools import get_extrinc_from_sphere
from utils.render_utils import render_uv, getProjectionMatrix_refine_batch
from utils.geometry_utils import transform_pos_batch, assign_unique_uvs

class DomeDatasetInfer(Dataset):
    def __init__(self, cfg, split='train', vis=False, demo=False):
        self.cfg = cfg
        self.split = split

        self.activeCameraIdxs = self.cfg.dataset.activeCameraIdxs
        self.condCameraIdxs = self.cfg.dataset.condCameraIdxs
        self.numCondCam = len(self.condCameraIdxs)
        self.subject = self.cfg.dataset.subject
        self.testCameraIdxs   = self.cfg.dataset.testCameraIdxs

        self.baseDir = self.cfg.dataset.baseDir
        self.camIdxs = self.cfg.dataset.condCameraIdxs
        self.deltaType = self.cfg.deltaType

        self.vis = vis
        self.sparse = self.cfg.sparse
        self.demo = demo

        self.texResGeo = self.cfg.dataset.texResGeo
        self.texResGau = self.cfg.dataset.texResGau
        self.texResCano = self.cfg.dataset.texResCano

        use_opengl = False
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        print('debug0')
        if self.split == 'training':
            if self.subject == 'Subject0022':
                # self.fIdxs = np.arange(110, 25000, 10).tolist()
                self.fIdxs = [17000]
            elif self.subject == 'Subject0003':
                self.fIdxs = np.arange(100, 18000, 10).tolist()
            elif self.subject == 'Subject0000':
                self.fIdxs = np.arange(100, 26000, 10).tolist()
            elif self.subject == 'Subject2618':
                self.fIdxs = np.arange(100, 3500, 3).tolist()
            elif self.subject == 'Subject0724':
                self.fIdxs = np.arange(100, 27000, 10).tolist()
            elif self.subject == 'Subject0066':
                self.fIdxs = np.arange(100, 27500, 10).tolist()
        elif self.split == 'testing':
            if self.subject == 'Subject0022':
                if self.vis:
                    self.fIdxs = np.arange(6900, 7400, 1).tolist()
                    # self.fIdxs = np.arange(7230, 7300, 1).tolist()
                else:
                    self.fIdxs = np.arange(110, 7200, 10).tolist()
            elif self.subject == 'Subject0003':
                if self.vis:
                    self.fIdxs = np.arange(1500, 2000, 1).tolist()
                    # self.fIdxs = np.arange(2300, 2800, 1).tolist()
                    # self.fIdxs = np.arange(710, 710+500, 1).tolist()
                else:
                    self.fIdxs = np.arange(100, 7000, 10).tolist()
            elif self.subject == 'Subject0000':
                self.fIdxs = np.arange(100, 9000, 10).tolist()
            elif self.subject == 'Subject2618':
                if self.vis:
                    # self.fIdxs = np.arange(3500, 5000, 1).tolist()
                    self.fIdxs = np.arange(4100, 4100+500, 1).tolist()
                    # self.fIdxs = np.arange(3700, 3700+500, 1).tolist()
                else:
                    self.fIdxs = np.arange(3500, 5000, 3).tolist()
            elif self.subject == 'Subject0724':
                if self.vis:
                    # self.fIdxs = np.arange(3600, 3750, 1).tolist()
                    # self.fIdxs = np.arange(3920, 4420, 1).tolist()
                    self.fIdxs = np.arange(1420, 1420 + 500, 1).tolist()

                else:
                    self.fIdxs = np.arange(100, 8000, 10).tolist()
            elif self.subject == 'Subject0066':
                if self.vis:
                    self.fIdxs = np.arange(3800, 3800+500, 1).tolist()
                else:
                    self.fIdxs = np.arange(100, 8000, 10).tolist()

        elif self.split == 'testing_ood':
            if self.subject == 'Subject0724':
                if self.vis:
                    # self.fIdxs = np.arange(3600, 3750, 1).tolist()
                    # self.fIdxs = np.arange(4700, 5000, 1).tolist()
                    # self.fIdxs = np.arange(2450, 2700, 1).tolist()
                    # self.fIdxs = np.arange(2450, 2650, 1).tolist()
                    # self.fIdxs = [2607] * 180
                    # self.fIdxs = [2609] * 180
                    self.fIdxs = np.arange(2450, 2700, 1).tolist()
                    # self.fIdxs = np.arange(3470, 3820, 1).tolist()
                    # self.fIdxs = np.arange(3470, 3850, 1).tolist()
                    # self.fIdxs = np.arange(7530, 7600, 1).tolist()
                    # self.fIdxs = np.arange(7050, 7600, 1).tolist()
                    # self.fIdxs = np.arange(5400, 6300, 1).tolist()
                else:
                    self.fIdxs = np.arange(100, 8000, 10).tolist()

        if self.demo:
            self.sparse = True
            self.subject = 'Subject0724'
            self.split = 'testing'
            self.vis = True
            self.fIdxs = np.arange(3920, 4420, 1).tolist()
            self.baseDir = "./datas/demo"

        if self.vis:
            self.H_render = 2048
            self.W_render = 2048
            K_ = np.eye(4)
            K_[0, 0] = K_[1, 1] = 2048  # 960 * H / 540
            K_[0, 2] = self.W_render / 2
            K_[1, 2] = self.H_render / 2
            self.Ks_render = torch.Tensor(K_).float()

            Eall = []
            elevations = 10 #10 #30 #10 #30 #10
            rotate_step = 3.0 #1.0 #2 #1.0 #2
            rotate_shift = -90 #120 #60 #0 #-90 #0 #0 #-90 #60 #120#-120 #180 #-90 #0 #180 #-90 #90 #-90 #0 #-90 #0 #-90 #0 #180 #0 #-80 #80 #0  # 180
            rotate_scale = 1.0 #1.5 #2.0 #1.0
            distance = 3.2 #5 #3.2 #5 #3
            azimuths = (np.arange(0, 360, rotate_step)).tolist()

            if self.subject == 'Subject2618':
                rotate_center = [0, 1.2, 0]
            elif self.subject == 'Subject0003':
                rotate_center = [0.0, 1.2, 1.0]
            else:
                if self.split == 'testing_live':
                    rotate_center = [0, 1.2, 0]
                elif self.split == 'testing_light':
                    rotate_center = [0.5, 1.2, 0.5]
                else:
                    rotate_center = [2.5, 1.2, 0]

            for j in range(len(azimuths)):
                E_ = get_extrinc_from_sphere(distance=distance, elevation=elevations,
                                             azimuth=azimuths[j] * rotate_scale - rotate_shift,
                                             t_shift=rotate_center)  # E c2w
                E_ = np.linalg.inv(E_)
                Eall.append(E_)

            self.Es_render = torch.Tensor(Eall).float()


        if self.sparse:
            self.imgDir = imgDir = checkPlatformDir('{}/{}/tight/{}/recon_neus2_sparse/imgs'.format(self.baseDir, self.subject, self.split))
            if self.subject == 'Subject2618':
                self.imgDir = imgDir = checkPlatformDir('{}/{}/tight/{}/recon_neus2_sparse/imgs'.format(self.baseDir, self.subject, 'training'))
            if self.subject == 'Subject0066':
                self.imgDir = imgDir = checkPlatformDir(
                    '{}/{}/loose/{}/recon_neus2_sparse/imgs'.format(self.baseDir, self.subject, self.split))
        else:
            self.imgDir = imgDir = checkPlatformDir('{}/{}/tight/{}/recon_neus2/imgs'.format(self.baseDir, self.subject, self.split))
            if self.subject == 'Subject2618':
                self.imgDir = imgDir = checkPlatformDir('{}/{}/tight/{}/recon_neus2/imgs'.format(self.baseDir, self.subject, 'training'))
            if self.subject == 'Subject0066':
                self.imgDir = imgDir = checkPlatformDir(
                    '{}/{}/loose/{}/recon_neus2/imgs'.format(self.baseDir, self.subject, self.split))

        self.texDir = texDir = checkPlatformDir('{}/{}/tight/{}/partial_textures2_{}view'.format(self.baseDir, self.subject, self.split, self.numCondCam))
        if self.subject == 'Subject2618':
            self.texDir = texDir = checkPlatformDir('{}/{}/tight/{}/partial_textures2_{}view'.format(self.baseDir, self.subject, "training", self.numCondCam))
        if self.subject == 'Subject0066':
            self.texDir = texDir = checkPlatformDir(
                '{}/{}/loose/{}/partial_textures2_{}view'.format(self.baseDir, self.subject, self.split,
                                                                 self.numCondCam))

        self.texDirDeformed = texDirDeformed = checkPlatformDir(
            '{}/{}/tight/{}/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject, self.split, self.deltaType, self.numCondCam))
        if self.subject == 'Subject2618':
            self.texDirDeformed = texDirDeformed = checkPlatformDir(
                '{}/{}/tight/{}/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject,
                                                                                  'training', self.deltaType, self.numCondCam))
        if self.subject == 'Subject0066':
            self.texDirDeformed = texDirDeformed = checkPlatformDir(
                '{}/{}/loose/{}/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject,
                                                                                      self.split, self.deltaType,
                                                                                      self.numCondCam))

        self.pcDir = pcDir = checkPlatformDir('{}/{}/tight/{}/pointclouds'.format(self.baseDir, self.subject, self.split))
        if self.subject == 'Subject2618':
            self.pcDir = pcDir = checkPlatformDir(
                '{}/{}/tight/{}/pointclouds'.format(self.baseDir, self.subject, "training"))

        if self.subject == 'Subject0022' or self.subject == 'Subject0000' or self.subject == 'Subject0724':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}/motions/107dof.motion'.format(self.baseDir, self.subject, self.split))
        elif self.subject == 'Subject0003':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}/motions/54dof.motion'.format(self.baseDir, self.subject, self.split))
        elif self.subject == 'Subject2618':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}/motions/159dof.motion'.format(self.baseDir, self.subject, "training"))
        elif self.subject == 'Subject0066':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/loose/{}/motions/107dof.motion'.format(self.baseDir, self.subject, self.split))

        self.dispDir = dispDir = checkPlatformDir('{}/{}/tight/{}/deltaNew_{}_{}view/displacements'.format(self.baseDir,\
                                                                                                  self.subject, self.split, self.deltaType, self.numCondCam))
        if self.subject == 'Subject2618':
            self.dispDir = dispDir = checkPlatformDir(
                '{}/{}/tight/{}/deltaNew_{}_{}view/displacements'.format(self.baseDir, self.subject, 'training', self.deltaType, self.numCondCam))
        if self.subject == 'Subject0066':
            self.dispDir = dispDir = checkPlatformDir(
                '{}/{}/loose/{}/deltaNew_{}_{}view/displacements'.format(self.baseDir, \
                                                                         self.subject, self.split, self.deltaType,
                                                                         self.numCondCam))
        self.motions = loadMotion(motionDir, returnTensor=True)

        default_path = osp.join(checkPlatformDir(self.cfg.rootDir), 'config/ddc_configs/default.yaml')
        config_path = osp.join(checkPlatformDir(self.cfg.rootDir), 'config/ddc_configs/s{}_smooth_eg.yaml'.format(str(int(self.subject[-4:]))) )
        cfgs = load_config_with_default(default_path=default_path, path=config_path, log=False)
        self.eg = load_model(cfgs, useCuda=True, device=None, useDQ=False)
        self.uvs = torch.from_numpy(self.eg.character.uvs.astype(np.float32))
        self.faces = torch.from_numpy(self.eg.character.faces.astype(np.int32))
        self.facesuv = torch.from_numpy(self.eg.character.facesuv.astype(np.int32))
        self.uvs_unique = assign_unique_uvs(self.eg.character.verts0, self.faces, self.uvs, self.facesuv)
        self.normal_cano = compute_normal_torch(self.eg.character.verts0[None], self.faces)[0].cpu()
        self.verts0 = self.eg.character.verts0.cpu()


    def get_item(self, index):
        dataDict = {}

        fIdx_i = index
        fIdx = self.fIdxs[fIdx_i]
        dataDir = checkPlatformDir(osp.join(self.imgDir, str(fIdx).zfill(6)))
        print(dataDir)
        cam_path = sorted(glob.glob(f"{dataDir}/*.json"))[0]

        if self.sparse:
            img_lists = []
            for cIdx in self.camIdxs:
                img_lists.append(
                    osp.join(dataDir, 'image_c_{}_f_{}.png'.format(str(cIdx).zfill(3), str(fIdx).zfill(6)))
                )

        else:
            img_lists = sorted(glob.glob(f"{dataDir}/*.png"))

        if self.sparse:
            if self.split == 'testing_light':
                Ks, Es, H, W = load_camera_param_sparse(cam_path, self.camIdxs)  # T, C, 4, 4
                Ks = torch.Tensor(Ks).float()
                Es = torch.Tensor(Es).float()
                Ks[:,:2,:] *= 0.5
                H = int(H*0.5)
                W = int(W*0.5)
                camPoss = torch.inverse(Es)[:, :3, 3]
                mtx = getProjectionMatrix_refine_batch(Ks, H, W) @ Es
                Ps = torch.matmul(Ks, Es)

                imgs = []
                for idx in range(len(self.camIdxs)):
                    temp_img = cv2.imread(img_lists[idx], -1)
                    img = temp_img[:, :, [2, 1, 0, 3]]
                    img = resizeImg(img, scale = 0.5)
                    img = (img[..., :] / 255).astype(np.float32)
                    img[:, :, 3] = np.where(img[:, :, 3] > 0.5, 1.0, 0.0)
                    imgs.append(torch.Tensor(img))
                imgs = torch.stack(imgs).permute(0, 3, 1, 2)
            else:
                Ks, Es, H, W = load_camera_param_sparse(cam_path, self.camIdxs)  # T, C, 4, 4
                Ks = torch.Tensor(Ks).float()
                Es = torch.Tensor(Es).float()
                camPoss = torch.inverse(Es)[:, :3, 3]
                mtx = getProjectionMatrix_refine_batch(Ks, H, W) @ Es
                Ps = torch.matmul(Ks, Es)

                imgs = []
                for idx in range(len(self.camIdxs)):
                    temp_img = cv2.imread(img_lists[idx], -1)
                    img = temp_img[:, :, [2, 1, 0, 3]]
                    img = (img[..., :] / 255).astype(np.float32)
                    img[:, :, 3] = np.where(img[:, :, 3] > 0.5, 1.0, 0.0)
                    imgs.append(torch.Tensor(img))
                imgs = torch.stack(imgs).permute(0, 3, 1, 2)
        else:
            Ks, Es, H, W = load_camera_param(cam_path, self.camIdxs)  # T, C, 4, 4
            Ks = torch.Tensor(Ks).float()
            Es = torch.Tensor(Es).float()
            camPoss = torch.inverse(Es)[:, :3, 3]
            mtx = getProjectionMatrix_refine_batch(Ks, H, W) @ Es
            Ps = torch.matmul(Ks, Es)

            imgs = []
            for idx in self.camIdxs:
                temp_img = cv2.imread(img_lists[idx], -1)
                img = temp_img[:, :, [2, 1, 0, 3]]
                img = (img[..., :] / 255).astype(np.float32)
                img[:, :, 3] = np.where(img[:, :, 3] > 0.5, 1.0, 0.0)
                imgs.append(torch.Tensor(img))
            imgs = torch.stack(imgs).permute(0, 3, 1, 2)

        dataDict['fIdx'] = fIdx # torch.Tensor(fIdx)
        dataDict['motion'] = self.motions[fIdx]
        dataDict['imgs'] = imgs
        dataDict['camPoss'] = camPoss
        dataDict['mtx'] = mtx
        #
        dataDict['Ks'] = Ks
        dataDict['Es'] = Es
        dataDict['Ps'] = Ps
        dataDict['H'] = H
        dataDict['W'] = W

        if self.vis:
            tempVisIdx = index % len(self.Es_render)
            dataDict['K_render'] = self.Ks_render
            dataDict['E_render'] = self.Es_render[tempVisIdx]
            dataDict['H_render'] = self.H_render
            dataDict['W_render'] = self.W_render
        else:
            Ks, Es, H, W = load_camera_param(cam_path, [self.testCameraIdxs[0]])  # T, C, 4, 4
            Ks = torch.Tensor(Ks).float()
            Es = torch.Tensor(Es).float()
            dataDict['K_render'] = Ks[0]
            dataDict['E_render'] = Es[0]
            dataDict['H_render'] = H
            dataDict['W_render'] = W

        return dataDict

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.fIdxs)
