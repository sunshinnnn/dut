"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2024-11-01
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

# from diff_gaussian_rasterization_mip import GaussianRasterizationSettingsMip, GaussianRasterizerMip
from diff_gaussian_rasterization_mip import GaussianRasterizationSettings as GaussianRasterizationSettingsMip
from diff_gaussian_rasterization_mip import GaussianRasterizer as GaussianRasterizerMip
from cuda_projection import projection_fw as projectPointsCuda
from gaussian_renderer import render as render_gaussian
# from gaussian_renderer import render_fast as render_gaussian
# from  gaussian_renderer import render_gsplat as render_gaussian

from scene.gaussian_model import GaussianModel
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import pdb
from simple_knn._C import distCUDA2
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_multiply
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

        global saveTexFirst, saveTexSecond, sendData, saveDebug, sendImage, handScale, vis
        self.saveTexFirst = saveTexFirst
        self.saveTexSecond = saveTexSecond
        self.sendData = sendData
        self.saveDebug = saveDebug
        self.sendImage = sendImage
        self.handScale = handScale
        self.vis = vis

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

        self.val_set = DomeDatasetInfer(self.cfg, split=split, vis=vis)
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
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

        self.subpixel_offset = None
        self.kernel_size = 0.1

    def run_eval(self):
        print(f"Doing validation ...")
        torch.cuda.empty_cache()

        frameList = []
        vertLBSList = []
        vertDeformedList = []
        metrics = []


        forwardKinamaticTimeList = []
        renderPosNMLTimeList = []
        projectionPointTimeList = []
        getUnprojectionTimeList = []
        geoInferTimeList = []
        geoTransmationTimeList = []


        renderPosNMLTimeList2 = []
        projectionPointTimeList2 = []
        getUnprojectionTimeList2 = []
        getGauInitMapTimeList2 = []
        gauInferTimeList = []
        gauRenderTimeList = []

        for idx in tqdm(range(self.len_val)):
            data = next(self.val_iterator)
            fIdx = data['fIdx'][0]
            camPoss = data['camPoss'][0].cuda()
            H = int(data['H'][0])
            W = int(data['W'][0])
            Ks = data['Ks'][0].cuda()
            Es = data['Es'][0].cuda()
            Ps = data['Ps'][0].cuda()

            imgScale = self.cfg.imgScale
            H_render = int(data['H_render'][0] * imgScale)
            W_render = int(data['W_render'][0] * imgScale)
            K_render = data['K_render'][0].cuda()
            E_render = data['E_render'][0].cuda()
            K_render[:2, :3] *= imgScale


            if self.subpixel_offset is None:
                self.subpixel_offset = torch.zeros((H_render, W_render, 2), dtype=torch.float32, device="cuda")
            imgs = data['imgs'][0].cuda()
            mtx = data['mtx'][0].cuda()
            motion = data['motion'].cuda()

            st = time.time()
            with torch.no_grad():
                verts, T_fw = self.eg.character.forward_cuda(motion, returnTransformation=True)
                T_fw[:, :, :3, 3] /= 1000.0
                ed = time.time() - st
                # print(ed * 1000.0)
                print("--------Stage I------------")
                print("forward kinematic: {}".format(ed * 1000.0))
                if idx >= 10:
                    forwardKinamaticTimeList.append(ed * 1000.0)

                rootTransform = self.eg.character.skeleton.jointGlobalTransformations[:, 6, :, :].data
                rootTransform[:, :3, 3] /= 1000.0
                T_fw_noroot_R = torch.matmul(rootTransform[:, :3, :3].transpose(1, 2)[:, None], T_fw[:, :, :3, :3])  # B,3,3   x B,N,3,3
                normalWorld_noroot = torch.matmul(T_fw_noroot_R, self.normal_cano[None, :, :, None])[0, :, :, 0]  # T,N,3,1

                st_local = time.time()
                edgeLengthCur = compute_edge_length(verts, self.faces)[0]
                postScale_temp, _ = torch.max((edgeLengthCur / self.edgeLengthCano)[self.vert2face], dim=1)
                postScale_temp *= 1.0 #1.25 #1.0
                postScale_temp = torch.clamp_min(postScale_temp, 1.0).reshape(-1,1)
                ed = time.time() - st_local
                print("[local] compute postscale: {}".format(ed * 1000.0))

                st_local = time.time()
                normals_ = compute_normal_torch(verts, self.faces)[0]  # N,3
                ed = time.time() - st_local
                print("[local] compute normal: {}".format(ed * 1000.0))

                temp_pos = torch.cat([verts[0] / 1000.0, normals_, normalWorld_noroot], -1)
                ed = time.time() - st_local
                print("[local] concatenate: {}".format(ed * 1000.0))
                render_temp = render_uv(self.glctx1, self.uvs, self.facesuv, [self.texResCano, self.texResCano], temp_pos, self.faces)
                pos_uv = render_temp[:, :, :, :3]
                normal_uv = render_temp[:, :, :, 3:6]
                normal_uv_noroot = render_temp[:, :, :, 6:9]
                ed = time.time() - st_local
                print("[local] render uv normal and position: {}".format(ed * 1000.0))

                ed = time.time() - st
                print("render uv normal and position: {}".format(ed * 1000.0))
                if idx >= 10:
                    renderPosNMLTimeList.append(ed * 1000.0)

                rayDir = torch.nn.functional.normalize(-(pos_uv - camPoss[:, None, None]), dim=-1)
                temp = (rayDir * normal_uv).sum(-1)
                vis_uv = temp > 0.17
                verts_image_space = projectPointsCuda(pos_uv.view(1, self.texResCano * self.texResCano, 3), Ps = Ps[None], H=H, W=W)
                ed = time.time() - st
                print("project points: {}".format(ed * 1000.0))
                if idx >= 10:
                    projectionPointTimeList.append(ed * 1000.0)

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

                temp2 = temp2 * 2 -1
                temp2 = torch.cat([temp2, normal_uv_noroot[0].permute(2, 0, 1)], 0)

                ed = time.time() - st
                print("render depth and get uv unprojection map: {}".format(ed * 1000.0))
                if idx >= 10:
                    getUnprojectionTimeList.append(ed * 1000.0)

                temp2 = T.Resize((self.texResGeo,self.texResGeo), antialias=True)(temp2)
                delta_temp = self.model_geometry(temp2[None])
                torch.cuda.synchronize()
                ed = time.time() - st
                print("geometry infer: {}".format(ed * 1000.0))
                if idx >= 10:
                    geoInferTimeList.append(ed * 1000.0)

                delta_temp = index_feat(delta_temp,
                                   (self.uvs_unique.transpose(0, 1)[None].repeat(delta_temp.shape[0], 1, 1) - 0.5) * 2)[
                             :, :].transpose(1, 2)
                out_verts = self.verts0[None] / 1000.0 + delta_temp

                if False:
                    out_verts = torch.matmul(self.L, out_verts) * self.boudMask[None] + out_verts

                out_verts_world = (torch.matmul(T_fw[:,:,:3,:3], out_verts[:,:,:,None]) + T_fw[:,:,:3,3:])[:,:,:,0].data
                # out_verts_world2 = (torch.matmul(T_fw2[:, :, :3, :3], out_verts[:, :, :, None]) + T_fw2[:, :, :3, 3:])[:, :, :, 0].data
                ed = time.time() - st
                print("geometry transformation: {}".format(ed * 1000.0))
                print()
                if idx >= 10:
                    geoTransmationTimeList.append(ed * 1000.0)

                print("--------Stage II------------")
                st_local = time.time()
                normals_ = compute_normal_torch(out_verts_world, self.faces)[0]  # N,3
                ed = time.time() - st_local
                print("[local] compute normal: {}".format(ed * 1000.0))
                temp_pos = torch.cat([out_verts_world[0] / 1.0, normals_], -1)
                ed = time.time() - st_local
                print("[local] concatenate: {}".format(ed * 1000.0))

                render_temp = render_uv(self.glctx3, self.uvs, self.facesuv, [self.texResCano, self.texResCano], temp_pos, self.faces)
                pos_uv = render_temp[:, :, :, :3]
                normal_uv = render_temp[:, :, :, 3:6]
                ed = time.time() - st_local
                print("[local] render uv normal and position: {}".format(ed * 1000.0))
                ed = time.time() - st
                print("render uv normal and position: {}".format(ed * 1000.0))
                if idx >= 10:
                    renderPosNMLTimeList2.append(ed * 1000.0)

                rayDir = torch.nn.functional.normalize(-(pos_uv - camPoss[:, None, None]), dim=-1)
                temp = (rayDir * normal_uv).sum(-1)
                if self.cfg.removeNormalMap:
                    vis_uv = temp > 0.17
                else:
                    vis_uv = temp > 0.17
                verts_image_space = projectPointsCuda(pos_uv.view(1, self.texResCano * self.texResCano, 3), Ps = Ps[None], H=H, W=W)
                ed = time.time() - st
                print("project points: {}".format(ed * 1000.0))
                if idx >= 10:
                    projectionPointTimeList2.append(ed * 1000.0)

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

                temp2 = temp2 * 2 -1
                ed = time.time() - st
                print("render depth and get uv unprojection map: {}".format(ed * 1000.0))
                if idx >= 10:
                    getUnprojectionTimeList2.append(ed * 1000.0)

                verts_temp = torch.cat([out_verts[0], postScale_temp, T_fw[0].reshape(-1, 16).data], -1)
                render_temp = render_uv(self.glctx5, self.uvs, self.facesuv, [self.texResGau, self.texResGau], verts_temp, self.faces)[0]

                canoPoints = render_temp[self.uv_idx, self.uv_idy, :3]
                canoPostScale = render_temp[self.uv_idx, self.uv_idy, 3:4]
                canoTransformations = render_temp[self.uv_idx, self.uv_idy, 4:].reshape([-1, 4, 4])
                ed = time.time() - st
                print("render pos & transform of Gaussian: {}".format(ed * 1000.0))
                if idx >= 10:
                    getGauInitMapTimeList2.append(ed * 1000.0)

                temp2 = torch.cat([temp2, normal_uv_noroot[0].permute(2,0,1)],0)
                temp2 = T.Resize((self.texResGau, self.texResGau), antialias=True)(temp2)
                full_feats = self.model_gaussian(temp2[None])[0]
                torch.cuda.synchronize()
                full_feats = full_feats[:, self.uv_idx, self.uv_idy].permute(1, 0).contiguous()
                ed = time.time() - st
                print("infer gaussian: {}".format(ed * 1000.0))
                if idx >= 10:
                    gauInferTimeList.append(ed * 1000.0)

                canoDelta = full_feats[:, 8:11]
                canoPoints = canoPoints + canoDelta
                canoPoints = (canoTransformations[..., :3, :3] @ canoPoints[..., None]).squeeze(-1) + canoTransformations[..., :3, 3]

                focalX, focalY = K_render[0, 0], K_render[1, 1]
                FovX, FovY = focal2fov(focalX, W_render), focal2fov(focalY, H_render)
                projection_matrix = getProjectionMatrix_refine(K_render, H_render, W_render, 0.01, 10.0).transpose(0, 1)
                world_view_transform = E_render.transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                tanfovx, tanfovy = math.tan(FovX * 0.5), math.tan(FovY * 0.5)

                raster_settings = GaussianRasterizationSettingsMip(
                    image_height=int(H_render),
                    image_width=int(W_render),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    kernel_size=self.kernel_size,
                    subpixel_offset=self.subpixel_offset,
                    bg=self.background,
                    scale_modifier=1.0,
                    viewmatrix=world_view_transform,
                    projmatrix=full_proj_transform,
                    sh_degree=self.gaussians.active_sh_degree,
                    campos=camera_center,
                    prefiltered=False,
                    debug=False
                )

                rasterizer = GaussianRasterizerMip(raster_settings=raster_settings)
                img, radii = rasterizer(
                    means3D=canoPoints,
                    means2D=None,
                    shs=torch.cat((full_feats[:, 11:14].reshape(-1, 1, 3), full_feats[:, 14:].reshape(canoPoints.shape[0], -1, 3)), dim=1),
                    colors_precomp=None,
                    opacities=self.gaussians.opacity_activation(full_feats[:, 7].unsqueeze(1)),
                    scales=self.gaussians.scaling_activation(full_feats[:, :3] + torch.log(canoPostScale)),
                    rotations=self.gaussians.rotation_activation(quaternion_multiply(matrix_to_quaternion(canoTransformations[..., :3, :3]),
                                            full_feats[:, 3:7])),
                    cov3D_precomp=None
                )

                ed = time.time() - st
                print("render gaussian: {}".format(ed * 1000.0))
                if idx >= 10:
                    gauRenderTimeList.append(ed * 1000.0)
                print()
                if idx == 109:
                    break

        mean_forwardKinamaticTime = np.mean(forwardKinamaticTimeList)
        mean_renderPosNMLTime = np.mean(renderPosNMLTimeList)
        mean_projectionPointTime = np.mean(projectionPointTimeList)
        mean_getUnprojectionTime = np.mean(getUnprojectionTimeList)
        mean_geoInferTime = np.mean(geoInferTimeList)
        mean_geoTransmationTime = np.mean(geoTransmationTimeList)

        mean_renderPosNMLTime2 = np.mean(renderPosNMLTimeList2)
        mean_projectionPointTime2 = np.mean(projectionPointTimeList2)
        mean_getUnprojectionTime2 = np.mean(getUnprojectionTimeList2)
        mean_getGauInitMapTime2 = np.mean(getGauInitMapTimeList2)
        mean_gauInferTime = np.mean(gauInferTimeList)
        mean_gauRenderTime = np.mean(gauRenderTimeList)

        print()
        print("imgScale: {}".format(self.cfg.imgScale))
        print('texResGeo: {} | texResGau: {}'.format(cfg.dataset.texResGeo, cfg.dataset.texResGau))
        print(len(forwardKinamaticTimeList))
        print("stage I")
        print("mean_forwardKinamaticTime: ", mean_forwardKinamaticTime)
        print("mean_renderPosNMLTime: ", mean_renderPosNMLTime)
        print("mean_projectionPointTime: ", mean_projectionPointTime)
        print("mean_getUnprojectionTime: ", mean_getUnprojectionTime)
        print("mean_geoInferTime: ", mean_geoInferTime)
        print("mean_geoTransmationTime: ", mean_geoTransmationTime)
        print()
        print("stage II")
        print("mean_renderPosNMLTime2: ", mean_renderPosNMLTime2)
        print("mean_projectionPointTime2: ", mean_projectionPointTime2)
        print("mean_getUnprojectionTime2: ", mean_getUnprojectionTime2)
        print("mean_getGauInitMapTime2: ", mean_getGauInitMapTime2)
        print("mean_gauInferTime: ", mean_gauInferTime)
        print("mean_gauRenderTime: ", mean_gauRenderTime)

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

        # for view in ['lmain', 'rmain']:
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

    global saveTexFirst, saveTexSecond, sendData, saveDebug, sendImage, handScale, vis
    saveTexFirst = args.saveTexFirst
    saveTexSecond = args.saveTexSecond
    sendData = args.sendData
    saveDebug = args.saveDebug
    sendImage = args.sendImage
    handScale = args.handScale
    vis = args.vis


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
