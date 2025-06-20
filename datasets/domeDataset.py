"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2024-07-22
"""
import os
import os.path as osp
import sys
from sys import platform

import cv2
from PIL import Image
from pathlib import Path
import logging
import json
from tqdm import tqdm
import glob
import pdb
import random
import copy

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

import nvdiffrast.torch as dr
from utils.render_utils import render_uv
from utils.geometry_utils import assign_unique_uvs
from models.utils import index_feat
from tools.mesh_tools import save_ply, compute_normal_torch, load_pointcoud
from tools.omni_tools import makePath, mask_to_rect, resizeImg, checkPlatformDir, listSub
from tools.config_tools import load_config_with_default
from tools.skel_tools import Skeleton, loadMotion, saveMotion, load_model, load_ddc_param

class DomeDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.activeCameraIdxs = self.cfg.dataset.activeCameraIdxs
        self.condCameraIdxs = self.cfg.dataset.condCameraIdxs
        self.numCondCam = len(self.condCameraIdxs)
        self.subject = self.cfg.dataset.subject
        self.texResCano = self.cfg.dataset.texResCano
        self.texResGeo = self.cfg.dataset.texResGeo
        self.baseDir = self.cfg.dataset.baseDir

        self.noRGBInput = self.cfg.noRGBInput
        self.noNormalInput = self.cfg.noNormalInput


        if self.split == 'train':
            if self.subject == 'Subject0022':
                self.fIdxs = np.arange(110, 25000, 10).tolist()
            elif self.subject == 'Subject0003':
                self.fIdxs = np.arange(100, 18000, 10).tolist()
                self.fIdxs = listSub(self.fIdxs, [3920, 4070])
            elif self.subject == 'Subject0000':
                self.fIdxs = np.arange(100, 26000, 10).tolist()
            elif self.subject == 'Subject2618':
                self.fIdxs = np.arange(100, 3500, 3).tolist()
                self.fIdxs = listSub(self.fIdxs, [681, 846])
            elif self.subject == 'Subject0724':
                self.fIdxs = np.arange(100, 27000, 10).tolist()
                self.fIdxs = listSub(self.fIdxs, [4950, 5020, 5040, 5120, 8530, 15540,18150, 18160, 18170, 18180, 18190, 18200, 18210, 18220, 18230, \
                                                  18240, 18250, 18260, 18270, 18280, 18290])

            elif self.subject == 'Subject0066':
                self.fIdxs = np.arange(100, 27500, 10).tolist()
        else:
            if self.subject == 'Subject0022':
                self.fIdxs = np.arange(110, 7200, 10).tolist()
            elif self.subject == 'Subject0003':
                self.fIdxs = np.arange(100, 7000, 10).tolist()
            elif self.subject == 'Subject0000':
                self.fIdxs = np.arange(100, 9000, 10).tolist()
            elif self.subject == 'Subject2618':
                self.fIdxs = np.arange(3500, 5000, 3).tolist()
            elif self.subject == 'Subject0066':
                self.fIdxs = np.arange(100, 8000, 10).tolist()
            elif self.subject == 'Subject0724':
                self.fIdxs = np.arange(100, 8000, 10).tolist()

        use_opengl = False
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

        self.imgDir = imgDir = checkPlatformDir('{}/{}/tight/{}ing/recon_neus2/imgs'.format(self.baseDir, self.subject, self.split))
        if self.subject == 'Subject2618':
            self.imgDir = imgDir = checkPlatformDir('{}/{}/tight/{}ing/recon_neus2/imgs'.format(self.baseDir, self.subject, 'train'))
        if self.subject == 'Subject0066':
            self.imgDir = imgDir = checkPlatformDir(
                '{}/{}/loose/{}ing/recon_neus2/imgs'.format(self.baseDir, self.subject, self.split))

        self.texDir = texDir = checkPlatformDir('{}/{}/tight/{}ing/partial_textures2_{}view'.format(self.baseDir, self.subject, self.split, self.numCondCam))
        if self.subject == 'Subject2618':
            self.texDir = texDir = checkPlatformDir('{}/{}/tight/{}ing/partial_textures2_{}view'.format(self.baseDir, self.subject, "train", self.numCondCam))
        if self.subject == 'Subject0066':
            self.texDir = texDir = checkPlatformDir(
                '{}/{}/loose/{}ing/partial_textures2_{}view'.format(self.baseDir, self.subject, self.split,
                                                                    self.numCondCam))

        self.texDirDeformed = texDirDeformed = checkPlatformDir('{}/{}/tight/{}ing/partial_textures2_deformed'.format(self.baseDir, self.subject, self.split))
        if self.subject == 'Subject2618':
            self.texDirDeformed = texDirDeformed = checkPlatformDir(
                '{}/{}/tight/{}ing/partial_textures2_deformed'.format(self.baseDir, self.subject, "train"))
        if self.subject == 'Subject0066':
            self.texDirDeformed = texDirDeformed = checkPlatformDir(
                '{}/{}/loose/{}ing/partial_textures2_deformed'.format(self.baseDir, self.subject, self.split))

        self.pcDir = pcDir = checkPlatformDir('{}/{}/tight/{}ing/pointclouds'.format(self.baseDir, self.subject, self.split))
        if self.subject == 'Subject2618':
            self.pcDir = pcDir = checkPlatformDir(
                '{}/{}/tight/{}ing/pointclouds'.format(self.baseDir, self.subject, "train"))
        if self.subject == 'Subject0066':
            self.pcDir = pcDir = checkPlatformDir(
                '{}/{}/loose/{}ing/pointclouds'.format(self.baseDir, self.subject, self.split))

        if self.subject == 'Subject0022' or self.subject == 'Subject0000' or self.subject == 'Subject0724':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}ing/motions/107dof.motion'.format(self.baseDir, self.subject, self.split))
        elif self.subject == 'Subject0003':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}ing/motions/54dof.motion'.format(self.baseDir, self.subject, self.split))
        elif self.subject == 'Subject2618':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}ing/motions/159dof.motion'.format(self.baseDir, self.subject, "train"))
        elif self.subject == 'Subject0066':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/loose/{}ing/motions/107dof.motion'.format(self.baseDir, self.subject, self.split))

        self.motions = loadMotion(motionDir, returnTensor=True)
        default_path = osp.join(checkPlatformDir(self.cfg.rootDir), 'config/ddc_configs/default.yaml')
        config_path = osp.join(checkPlatformDir(self.cfg.rootDir), 'config/ddc_configs/s{}_smooth_eg.yaml'.format(str(int(self.subject[-4:]))) )
        cfgs = load_config_with_default(default_path=default_path, path=config_path, log=False)
        self.eg = load_model(cfgs, useCuda=False, device=None, useDQ=False)

        self.verts0 = self.eg.character.verts0.cuda()
        self.uvs = torch.from_numpy(self.eg.character.uvs.astype(np.float32)).cuda()
        self.faces = torch.from_numpy(self.eg.character.faces.astype(np.int32)).cuda()
        self.facesuv = torch.from_numpy(self.eg.character.facesuv.astype(np.int32)).cuda()
        self.uvs_unique = assign_unique_uvs(self.eg.character.verts0, self.faces, self.uvs, self.facesuv).cuda()

        self.verts_all, self.T_fw  = self.eg.character.forward(self.motions[self.fIdxs], useDQ=False, returnTransformation=True)
        self.T_fw[:,:,:3,3] /= 1000.0
        self.normal_cano = compute_normal_torch(self.eg.character.verts0[None], self.faces.cpu())[0]

        rootTransform = self.eg.character.skeleton.jointGlobalTransformations[:, 6, :, :].data
        rootTransform[:, :3, 3] /= 1000.0
        self.T_fw_noroot_R = torch.matmul(rootTransform[:, :3, :3].transpose(1, 2)[:, None],
                                          self.T_fw[:, :, :3, :3])  # B,3,3   x B,N,3,3
        normalWorld_noroot = torch.matmul(self.T_fw_noroot_R, self.normal_cano[None, :, :, None])[:, :, :, 0]  # T,N,3,1

        self.canoMask = render_uv(self.glctx, self.uvs, self.facesuv, [self.texResGeo, self.texResGeo], torch.ones([self.eg.character.numVert,1]).cuda(), self.faces)[0,:,:,0]
        self.uv_idx, self.uv_idy = torch.where(self.canoMask > 0)
        self.worldNormalMapList = []
        for idx, fIdx in tqdm(enumerate(self.fIdxs)):
            verts_temp = normalWorld_noroot[idx]
            render_temp = render_uv(self.glctx, self.uvs, self.facesuv, [self.texResCano, self.texResCano], verts_temp.cuda(), self.faces)[0].cpu()
            self.worldNormalMapList.append(render_temp[:, :, 0:3])

    def get_item(self, index):
        dataDict = {}
        fIdx_i = index
        fIdx = self.fIdxs[fIdx_i]

        if self.cfg.withOccAug:
            texDirTemp = copy.deepcopy(self.texDir)
            possibility = torch.rand(1)[0]
            if possibility < 0.3:
                texDirTemp = texDirTemp.replace('4view', '1view')
            else:
                pass
            img = cv2.imread(osp.join(texDirTemp, str(fIdx).zfill(6) + '.jpg'), -1)[::-1,:,::-1]
        else:
            img = cv2.imread(osp.join(self.texDir, str(fIdx).zfill(6) + '.jpg'), -1)[::-1,:,::-1]
        img = 2 * (img.astype(np.float32) / 255.0 ) - 1.0
        img = torch.Tensor(img).permute(2,0,1) # 3,H,W
        img = torch.cat([img, self.worldNormalMapList[fIdx_i].permute(2,0,1)],0)
        img = T.Resize((self.texResGeo, self.texResGeo), antialias=True)(img)
        if self.noRGBInput:
            img[:3] *= 0.0
        if self.noNormalInput:
            img[3:6] *= 0.0
        dataDict['inputs'] = img
        dataDict['verts'] = self.eg.character.verts0  / 1000.0
        dataDict['normals'] = self.normal_cano
        dataDict['T_fw'] = self.T_fw[fIdx_i]
        dataDict['fIdx'] = fIdx # torch.Tensor(fIdx)
        dataDict['pointclouds'] = torch.Tensor(load_pointcoud(osp.join(self.pcDir, "depthMap_{}.obj".format(fIdx))))

        return dataDict

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.fIdxs)
