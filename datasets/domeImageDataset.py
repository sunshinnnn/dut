"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2024-07-30
"""
import os
import os.path as osp
import sys
from sys import platform

from PIL import Image
import cv2

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
from torch.utils.data import Dataset
import torchvision.transforms as T

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from kornia.geometry.conversions import quaternion_to_rotation_matrix, Rt_to_matrix4x4, rotation_matrix_to_quaternion
import nvdiffrast.torch as dr

from utils.render_utils import render_uv
from utils.geometry_utils import compute_edge_length, find_vertex_to_face_index, assign_unique_uvs
from tools.cam_tools import load_camera_param
from tools.mesh_tools import compute_normal_torch, load_pointcoud
from tools.omni_tools import makePath, mask_to_rect, resizeImg, checkPlatformDir, listSub
from tools.skel_tools import Skeleton, loadMotion, saveMotion, load_model, load_ddc_param
from tools.config_tools import load_config_with_default

class DomeImageDataset(Dataset):
    def __init__(self, cfg, split='train', warmup=False):
        self.cfg = cfg
        self.split = split
        self.warmup = warmup
        self.texResGau = self.cfg.dataset.texResGau
        self.texResCano = self.cfg.dataset.texResCano # // 2
        self.cIdxs = self.activeCameraIdxs = self.cfg.dataset.activeCameraIdxs
        self.condCameraIdxs = self.cfg.dataset.condCameraIdxs
        self.numCondCam = len(self.condCameraIdxs)
        self.noRGBInput = self.cfg.noRGBInput
        self.sparseMotion = self.cfg.sparseMotion


        self.condCameraIdxs = self.cfg.dataset.condCameraIdxs
        self.numCondCam = len(self.condCameraIdxs)
        if split=='test':
            self.cIdxs = self.activeCameraIdxs = self.cfg.dataset.testCameraIdxs
        self.subject = self.cfg.dataset.subject
        self.imgScale = cfg.imgScale
        self.withNormal = cfg.withNormal

        self.weightChamfer = cfg.weightChamfer
        self.baseDir = self.cfg.dataset.baseDir
        self.camIdxs = self.cfg.dataset.condCameraIdxs
        self.deltaType = self.cfg.deltaType

        use_opengl = False
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        print('debug0')

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
            elif self.subject == 'Subject0066':
                self.fIdxs = np.arange(100, 27500, 10).tolist()

        else:
            if self.subject == 'Subject0022':
                self.fIdxs = np.arange(110, 7200, 10).tolist()
            elif self.subject == 'Subject0003':
                self.fIdxs = np.arange(100, 7000, 10).tolist()
                # self.fIdxs = np.arange(100, 150, 10).tolist()
            elif self.subject == 'Subject0000':
                self.fIdxs = np.arange(100, 9000, 10).tolist()
            elif self.subject == 'Subject2618':
                self.fIdxs = np.arange(3500, 5000, 3).tolist()
            elif self.subject == 'Subject0724':
                self.fIdxs = np.arange(100, 8000, 10).tolist()
                # self.fIdxs = np.arange(5000, 6000, 10).tolist()
            elif self.subject == 'Subject0066':
                self.fIdxs = np.arange(100, 8000, 10).tolist()

        self.imgDir = imgDir = checkPlatformDir('{}/{}/tight/{}ing/recon_neus2/imgs'.format(self.baseDir, self.subject, self.split))
        if self.subject == 'Subject2618':
            self.imgDir = imgDir = checkPlatformDir('{}/{}/tight/{}ing/recon_neus2/imgs'.format(self.baseDir, self.subject, 'train'))
        if self.subject == 'Subject0066':
            self.imgDir = imgDir = checkPlatformDir(
                '{}/{}/loose/{}ing/recon_neus2/imgs'.format(self.baseDir, self.subject, self.split))

        self.texDirDeformed = texDirDeformed = checkPlatformDir(
            '{}/{}/tight/{}ing/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject, self.split, self.deltaType, self.numCondCam))
        if self.subject == 'Subject2618':
            self.texDirDeformed = texDirDeformed = checkPlatformDir(
                '{}/{}/tight/{}ing/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject,
                                                                                  'train', self.deltaType, self.numCondCam))
        if self.subject == 'Subject0066':
            self.texDirDeformed = texDirDeformed = checkPlatformDir(
                '{}/{}/loose/{}ing/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject,
                                                                                         self.split, self.deltaType,
                                                                                         self.numCondCam))
        self.texDirDeformed = texDirDeformed = checkPlatformDir(
            '{}/{}/tight/{}ing/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject, self.split, self.deltaType, self.numCondCam))
        if self.subject == 'Subject2618':
            self.texDirDeformed = texDirDeformed = checkPlatformDir(
                '{}/{}/tight/{}ing/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject,
                                                                                  'train', self.deltaType, self.numCondCam))
        if self.subject == 'Subject0066':
            self.texDirDeformed = texDirDeformed = checkPlatformDir(
                '{}/{}/loose/{}ing/deltaNew_{}_{}view/partial_textures2_deformed'.format(self.baseDir, self.subject,
                                                                                         self.split, self.deltaType,
                                                                                         self.numCondCam))

        self.pcDir = pcDir = checkPlatformDir('{}/{}/tight/{}ing/pointclouds'.format(self.baseDir, self.subject, self.split))
        if self.subject == 'Subject2618':
            self.pcDir = pcDir = checkPlatformDir(
                '{}/{}/tight/{}ing/pointclouds'.format(self.baseDir, self.subject, "train"))
        if self.subject == 'Subject0022' or self.subject == 'Subject0000' or self.subject == 'Subject0724':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}ing/motions/107dof.motion'.format(self.baseDir, self.subject, self.split))
        elif self.subject == 'Subject0003':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}ing/motions/54dof.motion'.format(self.baseDir, self.subject, self.split))
        elif self.subject == 'Subject2618':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/tight/{}ing/motions/159dof.motion'.format(self.baseDir, self.subject, "train"))
        elif self.subject == 'Subject0066':
            self.motionDir = motionDir = checkPlatformDir('{}/{}/loose/{}ing/motions/107dof.motion'.format(self.baseDir, self.subject, self.split))

        # import pdb
        # pdb.set_trace()
        if self.sparseMotion:
            self.motionDir = self.motionDir.replace("motions", "motionsSparse")
            print()
            print(self.motionDir)
            print()

        self.dispDir = dispDir = checkPlatformDir('{}/{}/tight/{}ing/deltaNew_{}_{}view/displacements'.format(self.baseDir,\
                                                                                                  self.subject, self.split, self.deltaType, self.numCondCam))
        if self.subject == 'Subject0066':
            self.dispDir = dispDir = checkPlatformDir(
                '{}/{}/loose/{}ing/deltaNew_{}_{}view/displacements'.format(self.baseDir, \
                                                                            self.subject, self.split, self.deltaType,
                                                                            self.numCondCam))
        if self.subject == 'Subject2618':
            self.dispDir = dispDir = checkPlatformDir(
                '{}/{}/tight/{}ing/deltaNew_{}_{}view/displacements'.format(self.baseDir, self.subject, 'train', self.deltaType, self.numCondCam))
        self.motions = loadMotion(motionDir, returnTensor=True)

        self.motions_noroot = self.motions.clone()
        self.motions_noroot[:,:6] = 0.0
        default_path = osp.join(checkPlatformDir(self.cfg.rootDir), 'config/ddc_configs/default.yaml')
        config_path = osp.join(checkPlatformDir(self.cfg.rootDir), 'config/ddc_configs/s{}_smooth_eg.yaml'.format(str(int(self.subject[-4:]))) )
        cfgs = load_config_with_default(default_path=default_path, path=config_path, log=False)
        self.eg = load_model(cfgs, useCuda=False, device=None, useDQ=False)
        self.verts0 = self.eg.character.verts0.cuda()
        self.uvs = torch.from_numpy(self.eg.character.uvs.astype(np.float32)).cuda()
        self.faces = torch.from_numpy(self.eg.character.faces.astype(np.int32)).cuda()
        self.facesuv = torch.from_numpy(self.eg.character.facesuv.astype(np.int32)).cuda()
        self.uvs_unique = assign_unique_uvs(self.eg.character.verts0, self.faces, self.uvs, self.facesuv).cuda()
        self.normal = compute_normal_torch(self.eg.character.verts0[None], self.faces.cpu())[0] # N,3
        self.edgeLengthCano = compute_edge_length(self.verts0[None], self.faces)[0]
        self.vert2face = find_vertex_to_face_index(self.verts0.shape[0], self.faces)
        self.verts_all, self.T_fw  = self.eg.character.forward(self.motions[self.fIdxs], useDQ=False, returnTransformation=True)
        self.T_fw[:,:,:3,3] /= 1000.0

        self.displacements = []
        for fIdx in self.fIdxs:
            self.displacements.append( torch.FloatTensor(np.load(osp.join(self.dispDir, "{}.npy".format(str(fIdx).zfill(6))))) ) #/ 1000.0 )
        self.displacements = torch.stack(self.displacements, 0)

        vertsCano = self.eg.character.verts0[None] / 1000.0
        vertsCanoDeformed = self.eg.character.verts0[None] / 1000.0 + self.displacements #T,N,3

        if self.cfg.withDeformNormal:
            self.normal_deformed = compute_normal_torch(vertsCanoDeformed, self.faces.cpu()) #[0] # N,3
            rootTransform = self.eg.character.skeleton.jointGlobalTransformations[:,6,:,:].data
            rootTransform[:,:3,3] /= 1000.0
            self.T_fw_noroot_R = torch.matmul(rootTransform[:,:3,:3].transpose(1,2)[:,None], self.T_fw[:,:,:3,:3])         #B,3,3   x B,N,3,3
            normalWorld_noroot = torch.matmul(self.T_fw_noroot_R, self.normal_deformed[:,:,:,None])[:,:,:,0] # T,N,3,1
        else:
            rootTransform = self.eg.character.skeleton.jointGlobalTransformations[:,6,:,:].data
            rootTransform[:,:3,3] /= 1000.0
            self.T_fw_noroot_R = torch.matmul(rootTransform[:,:3,:3].transpose(1,2)[:,None], self.T_fw[:,:,:3,:3])         #B,3,3   x B,N,3,3
            normalWorld_noroot = torch.matmul(self.T_fw_noroot_R, self.normal[None,:,:,None])[:,:,:,0] # T,N,3,1

        self.img_lists = []
        self.cam_path_lists = []
        for fIdx in self.fIdxs:
            self.img_lists.append( sorted(glob.glob(f"{self.imgDir}/{str(fIdx).zfill(6)}/*.png")) )
            self.cam_path_lists.append(sorted(glob.glob(f"{self.imgDir}/{str(fIdx).zfill(6)}/*.json"))[0])
        Ks, Es, H, W = load_camera_param(self.cam_path_lists, scale=1000.0) # T, C, 4, 4
        self.H, self.W = H, W
        self.Ks, self.Es = torch.Tensor(Ks), torch.Tensor(Es)
        print('debug1')


        self.canoMask = render_uv(self.glctx, self.uvs, self.facesuv, [self.texResGau, self.texResGau], torch.ones([self.eg.character.numVert,1]).cuda(), self.faces)[0,:,:,0]
        self.uv_idx, self.uv_idy = torch.where(self.canoMask > 0)
        self.uv_idx_cpu, self.uv_idy_cpu = self.uv_idx.cpu(), self.uv_idy.cpu()
        print('debug2')

        render_temp = render_uv(self.glctx, self.uvs, self.facesuv, [self.texResGau, self.texResGau], self.normal.cuda(), self.faces)[0].cpu()
        self.canoNormal = render_temp[self.uv_idx_cpu, self.uv_idy_cpu, :]

        self.canoPointList = []
        self.canoTransformationList = []
        self.worldNormalMapList = []
        self.canoPostScaleList = []

        for idx, fIdx in tqdm(enumerate(self.fIdxs)):
            edgeLengthCur = compute_edge_length(self.verts_all[idx][None].cuda(), self.faces)[0]
            postScale_temp, _ = torch.max((edgeLengthCur / self.edgeLengthCano)[self.vert2face], dim=1)
            postScale_temp *= self.cfg.addPostScale
            postScale_temp = torch.clamp_min(postScale_temp, 1.0).reshape(-1, 1).cpu()

            verts_temp = torch.cat([vertsCanoDeformed[idx], postScale_temp, self.T_fw[idx].reshape(-1,16)], -1)
            verts_temp2 = normalWorld_noroot[idx]
            render_temp = render_uv(self.glctx, self.uvs, self.facesuv, [self.texResGau, self.texResGau], verts_temp.cuda(), self.faces)[0].cpu()
            render_temp2 = render_uv(self.glctx, self.uvs, self.facesuv, [self.texResCano, self.texResCano], verts_temp2.cuda(), self.faces)[0].cpu()

            self.canoPointList.append(render_temp[self.uv_idx_cpu, self.uv_idy_cpu, :3])
            self.canoPostScaleList.append(render_temp[self.uv_idx_cpu, self.uv_idy_cpu, 3:4])
            self.canoTransformationList.append(render_temp[self.uv_idx_cpu, self.uv_idy_cpu, 4:].reshape([-1, 4, 4]))
            self.worldNormalMapList.append(render_temp2[:, :, 0:3])




    def get_item(self, index):
        dataDict = {}
        fIdx_i = index // len(self.cIdxs)
        cIdx_i = index % len(self.cIdxs)
        fIdx = self.fIdxs[fIdx_i]
        cIdx = self.cIdxs[cIdx_i]

        if self.imgScale > 0:
            imgScale = self.imgScale
        else:
            imgScale = 1.0


        img = cv2.imread(osp.join(self.texDirDeformed, str(fIdx).zfill(6) + '.jpg'), -1)[::-1,:,::-1]
        img = 2 * (img.astype(np.float32) / 255.0 ) - 1.0
        img = torch.Tensor(img).permute(2,0,1) # 3,H,W
        if self.texResGau == self.texResCano:
            img = torch.cat([img, self.worldNormalMapList[fIdx_i].permute(2,0,1)],0)
        else:
            img = torch.cat([img, self.worldNormalMapList[fIdx_i].permute(2,0,1)],0)
            img = T.Resize((self.texResGau, self.texResGau), antialias=True)(img)
        if self.noRGBInput:
            img[:3] *= 0.0
        dataDict['inputs'] = img
        temp_img = cv2.imread(self.img_lists[fIdx_i][cIdx], -1)
        if self.imgScale != 1.0:
            temp_img = resizeImg(temp_img, scale=self.imgScale)
        img = temp_img[:, :, [2, 1, 0]]
        img = (img[..., :] / 255).astype(np.float32)
        mask = (temp_img[:, :, 3] /255.0).astype(np.float32)

        dataDict['fIdx'] = fIdx
        dataDict['cIdx'] = cIdx

        dataDict['img'] = torch.FloatTensor(img)
        dataDict['mask'] = torch.FloatTensor(mask)

        dataDict['canoPoints'] = self.canoPointList[fIdx_i]
        dataDict['canoTransformations'] = self.canoTransformationList[fIdx_i]
        dataDict['canoPostScale'] = self.canoPostScaleList[fIdx_i]
        dataDict['canoNormals'] = self.canoNormal #self.canoNormalList[fIdx_i]
        dataDict['T_fw'] = self.T_fw[fIdx_i]

        dataDict['imgScale'] = torch.Tensor([imgScale]).float() #torch.FloatTensor(imgScale)
        dataDict['H'] = self.H
        dataDict['W'] = self.W
        dataDict['K'] = self.Ks[fIdx_i, cIdx]
        dataDict['E'] = self.Es[fIdx_i, cIdx]

        if self.weightChamfer > 0:
            dataDict['pointclouds'] = torch.Tensor(load_pointcoud(osp.join(self.pcDir, "depthMap_{}.obj".format(fIdx))))

        return dataDict

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        if self.warmup:
            return len(self.cIdxs) * int(len(self.fIdxs) * 0.1)
        else:
            return len(self.cIdxs) * len(self.fIdxs)
