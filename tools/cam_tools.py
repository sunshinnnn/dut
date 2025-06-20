"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-03-14
"""

import os
import os.path as osp
import pdb

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pandas
import json
from tqdm import tqdm

def load_camera_param_sparse(cam_path, camIdxs, scale = 1000.0):
    """
    Load camera parameters from a file or a list of files.

    param:
          cam_path: The path to the camera parameter file or a list of paths to multiple camera parameter files.

    return:
          A tuple containing the intrinsic matrices (Ks), extrinsic matrices (Es), image height (H), and image width (W).
    """
    if isinstance(cam_path, str):
        Ks, Es = [], []
        cIdxListJson = []
        with open(cam_path, 'r') as f:
            cam_data = json.load(f)
            # if camIdxs is None:
            #     camIdxs = np.arange(len(cam_data['frames']))
            # for i in range(len(camIdxs)):
            #     K_temp = np.eye(4)
            #     K_temp[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
            #     Ks.append(K_temp)
            #     tempE = np.array(cam_data['frames'][i]['transform_matrix'])
            #     tempE[:3, 3] /= scale
            #     tempE = np.linalg.inv(tempE)
            #     Es.append(tempE)

            for i in range(len(cam_data['frames'])):
                # cIdx =
                K_temp = np.eye(4)
                K_temp[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
                Ks.append(K_temp)
                tempE = np.array(cam_data['frames'][i]['transform_matrix'])
                tempE[:3, 3] /= scale
                tempE = np.linalg.inv(tempE)
                Es.append(tempE)
                cIdxListJson.append(int(cam_data['frames'][i]["file_path"].split('_')[2]))
        partList = []
        for cIdx in camIdxs:
            partList.append( cIdxListJson.index(cIdx) )

        # camera = np.load(f"{root}/cameras.npz")
        H, W = cam_data['h'], cam_data['w']
        Ks, Es = np.stack(Ks, 0).astype(np.float32), np.stack(Es, 0).astype(np.float32)
        Ks = Ks[np.array(partList)]
        Es = Es[np.array(partList)]

        return Ks, Es, H, W
    elif isinstance(cam_path, list):
        KsAll, EsAll = [], []
        for cam_path_temp in tqdm(cam_path):
            Ks, Es = [], []
            with open(cam_path_temp, 'r') as f:
                cam_data = json.load(f)
                # if camIdxs is None:
                #     camIdxs = np.arange(len(cam_data['frames']))
                # for i in camIdxs:
                for i in range(len(camIdxs)):
                    K_temp = np.eye(4)
                    K_temp[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
                    Ks.append(K_temp)
                    tempE = np.array(cam_data['frames'][i]['transform_matrix'])
                    tempE[:3, 3] /= scale
                    tempE = np.linalg.inv(tempE)
                    Es.append(tempE)
            # camera = np.load(f"{root}/cameras.npz")
            H, W = cam_data['h'], cam_data['w']
            Ks, Es = np.stack(Ks, 0), np.stack(Es, 0)
            KsAll.append(Ks), EsAll.append(Es)
        KsAll, EsAll = np.stack(KsAll, 0).astype(np.float32), np.stack(EsAll, 0).astype(np.float32)
        return KsAll, EsAll, H, W
    else:
        raise TypeError("Invalid input type. Expected a string or a list.")


def load_camera_param(cam_path, camIdxs=None, scale = 1000.0):
    """
    Load camera parameters from a file or a list of files.

    param:
          cam_path: The path to the camera parameter file or a list of paths to multiple camera parameter files.

    return:
          A tuple containing the intrinsic matrices (Ks), extrinsic matrices (Es), image height (H), and image width (W).
    """
    if isinstance(cam_path, str):
        Ks, Es = [], []
        with open(cam_path, 'r') as f:
            cam_data = json.load(f)
            if camIdxs is None:
                camIdxs = np.arange(len(cam_data['frames']))
            for i in camIdxs:
                K_temp = np.eye(4)
                K_temp[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
                Ks.append(K_temp)
                tempE = np.array(cam_data['frames'][i]['transform_matrix'])
                tempE[:3, 3] /= scale
                tempE = np.linalg.inv(tempE)
                Es.append(tempE)
        # camera = np.load(f"{root}/cameras.npz")
        H, W = cam_data['h'], cam_data['w']
        Ks, Es = np.stack(Ks, 0).astype(np.float32), np.stack(Es, 0).astype(np.float32)
        return Ks, Es, H, W
    elif isinstance(cam_path, list):
        KsAll, EsAll = [], []
        for cam_path_temp in cam_path:
            Ks, Es = [], []
            with open(cam_path_temp, 'r') as f:
                cam_data = json.load(f)
                if camIdxs is None:
                    camIdxs = np.arange(len(cam_data['frames']))
                for i in camIdxs:
                    K_temp = np.eye(4)
                    K_temp[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
                    Ks.append(K_temp)
                    tempE = np.array(cam_data['frames'][i]['transform_matrix'])
                    tempE[:3, 3] /= scale
                    tempE = np.linalg.inv(tempE)
                    Es.append(tempE)
            # camera = np.load(f"{root}/cameras.npz")
            H, W = cam_data['h'], cam_data['w']
            Ks, Es = np.stack(Ks, 0), np.stack(Es, 0)
            KsAll.append(Ks), EsAll.append(Es)
        KsAll, EsAll = np.stack(KsAll, 0).astype(np.float32), np.stack(EsAll, 0).astype(np.float32)
        return KsAll, EsAll, H, W
    else:
        raise TypeError("Invalid input type. Expected a string or a list.")



def points2mask(v3d,K,E,ref=None):
    if ref is None:
        h = 1280; w = 720
    else:
        h,w = ref.shape[:2]
    R = E[:3,:3].T
    t = -R.dot(E[:3,3:])
    mask = np.zeros((h,w),'uint8')
#     K = intrinc_list[0]
    v2d = K.dot(np.dot(R,v3d.T) + t.reshape(3,1))
    v2d = v2d/v2d[2:]
    v2d = np.round(v2d).astype('int')
    uv1_masked = v2d.T[np.array(list((v2d[0]>0)) and list(v2d[0]<w) and list(v2d[1]>0) and list(v2d[1]<h))].T
    uv1_masked = v2d.T[  (v2d[0]<w) & (v2d[0]>0) & (v2d[1]<h) & (v2d[1]>0)].T
    (v2d[0]<w) & (v2d[0]>0)
    mask[uv1_masked[1],uv1_masked[0]] = 255
    return mask

def depth2pointcloud_real(dep, mask, K, E):
    # h = 1280
    # w = 720
    h, w = dep.shape[:2]
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    xx, yy = np.meshgrid(x, y)
    xx = xx.astype('float32')
    yy = yy.astype('float32')
    # mask = dep>0
    # mask =

    dirx = (xx - K[0][2]) / K[0][0]
    diry = (yy - K[1][2]) / K[1][1]
    dirz = np.ones((h, w))
    dnm = np.sqrt(dirx ** 2 + diry ** 2 + dirz ** 2);
    #     dep_z = dep/dnm
    dep_z = dep
    #     plt.imshow(dnm)

    pcx = (xx - K[0][2]) / K[0][0] * dep_z
    pcy = (yy - K[1][2]) / K[1][1] * dep_z
    pcz = dep_z
    pc = np.stack([pcx, pcy, pcz], -1).reshape(-1,3)
    pc_m = pc[mask.reshape(-1) > 0]
    # pc_m_new = pc_m
    # pc_m_new = E[:3, :3].dot(pc_m.T) + E[:3, 3:]
    # pc_m_new = pc_m_new.T
    pc_m_new = pc_m.dot(E[:3, :3].T) + + E[:3, 3:].T

    return pc_m_new

def loadCameraJson(camPath, scale = 1000.0):
    Ks, Es, Esc2w = [], [], []
    with open(camPath, 'r') as f:
        cam_data = json.load(f)
        for i in range(len(cam_data['frames'])):
            Ks.append(np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3])
            tempE = np.array(cam_data['frames'][i]['transform_matrix'])
            tempE[:3, 3] /= scale
            tempEw2c = np.linalg.inv(tempE)
            Es.append(tempEw2c)
            Esc2w.append(tempE)
    # camera = np.load(f"{root}/cameras.npz")
    H, W = cam_data['h'], cam_data['w']

    Ks, Es, Esc2w = np.stack(Ks, 0), np.stack(Es, 0), np.stack(Esc2w, 0)
    # Ew2c = torch.Tensor(Es)
    # Ec2w = torch.inverse(Ew2c)
    # Ec2w = torch.Tensor(Esc2w)
    # Ks = torch.Tensor(Ks)
    Size = [H, W]
    return Ks, Es, Size


def loadCamera(camPath, H=-1, W=-1, returnTensor= False, returnNames=False, device=None, scale=1.0):
    """
        Input:
            camPath
        Output:
            Ks, Es, PsGL, Sizes[0]
        =====
        Es: world 2 camera
        Size: H,W
    """
    with open(os.path.join(camPath), 'r') as f:
        data = f.readlines()
    assert data[0] == 'Skeletool Camera Calibration File V1.0\n'
    Names = []
    Ks = []
    Es = []
    PsGL = []  #opengl style
    Sizes = []
    for line in data:
        splittedLine = line.split()
        if len(splittedLine) > 0:
            if (splittedLine[0] == 'name'):
                Names.append(splittedLine[1])
            if (splittedLine[0] == 'intrinsic'):
                tempK = np.zeros(16)
                for i in range(1, len(splittedLine)):
                    tempK[i-1] = float(splittedLine[i])
                Ks.append(tempK.reshape(4,4))
            if (splittedLine[0] == 'extrinsic'):
                tempE = np.zeros(16)
                for i in range(1, len(splittedLine)):
                    tempE[i-1] = float(splittedLine[i])
                Es.append(tempE.reshape(4,4))
            if (splittedLine[0] == 'size'):
                Sizes.append( [ float(splittedLine[2]), float(splittedLine[1]) ]) #H,W

    for i in range(len(Ks)):
        K = Ks[i]
        h, w = Sizes[0][0], Sizes[0][1]
        near, far = 0.01, 10000.0
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        tempP = np.array([
            [2 * fx / w, 0.0, (w - 2 * cx) / w, 0],
            [0, 2 * fy / h, (h - 2 * cy) / h, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0]
        ])
        PsGL.append(tempP)
    Ks = np.stack(Ks,0)
    Es = np.stack(Es,0)
    Es[:,:3,3] /= scale
    PsGL = np.stack(PsGL,0)
    Sizes = np.stack(Sizes, 0)
    if H>0 or W>0:
        assert H/Sizes[0, 0] == W/Sizes[0, 1]
        scale_ = H/Sizes[0, 0]
        Ks *= scale_
        Ks[:, 2, 2] = 1
        Ks[:, 3, 3] = 1
        Sizes *= scale_
        # Sizes = Es.astype('int')
    Sizes= Sizes.astype('int')

    if returnTensor:
        Ks = torch.from_numpy(Ks).float().to(device)
        Es = torch.from_numpy(Es).float().to(device)
        PsGL = torch.from_numpy(PsGL).float().to(device)
        # Sizes = torch.from_numpy(Sizes).to(device)
    if returnNames:
        return Ks, Es, PsGL, Sizes[0], Names
    else:
        return Ks, Es, PsGL, Sizes[0]


def saveCamera(camPath, Ks, Es, H=-1, W=-1):
    with open(os.path.join(camPath), 'w') as f:
        f.write("Skeletool Camera Calibration File V1.0\n")
        for i in range(len(Ks)):
            f.write("name          {}\n".format(i))
            f.write("  sensor      14.1864 10.3776\n")
            f.write("  size        {} {}\n".format(W, H))
            f.write("  animated    0\n")
            # pdb.set_trace()
            f.write("  intrinsic   {}\n".format(' '.join([str(item) for item in Ks[i].reshape(-1).tolist()])))
            f.write("  extrinsic   {}\n".format(' '.join([str(item) for item in  Es[i].reshape(-1).tolist()])))
            f.write("  radial    0\n")


def uv2Mask(vertsNDC, H=-1, W=-1, isNDC = True):
    """
        mask: numpy
    """
    if isinstance(vertsNDC,torch.Tensor):
        vertsNDC = vertsNDC.data.cpu().numpy()
    mask = np.zeros([H,W],dtype=np.uint8)
    if isNDC:
        xs = np.round((vertsNDC[:,0]+1)/2 * W).astype(np.int32)
        ys = np.round((vertsNDC[:,1]+1)/2 * H).astype(np.int32)
    else:
        xs = np.round(vertsNDC[:, 0]).astype(np.int32)
        ys = np.round(vertsNDC[:, 1]).astype(np.int32)
    # print(ys)
    xs[xs < 0] = 0
    ys[ys < 0] = 0
    xs[xs > W-1] = W-1
    ys[ys > H-1] = H-1
    # print(xs.shape)
    # print(xs)
    # print(ys)
    mask[ys,xs] = 255

    return mask

def ndc2Mask(verts, faces, H=-1, W=-1, verts_filter=None, returnBoundary=False, convert2NDC = False):
    """
        Input:
            verts: N,3   #N,D,C
            faces: F,3
            H: -1 int
            W: -1 int
            verts_filter: N      None
        Output:
            mask: H,W torch
    """

    if verts_filter is None:
        N = verts.shape[0]
        verts_filter = torch.ones((N), dtype=torch.uint8, device=verts.device)
    if isinstance(faces, np.ndarray):
        faces = torch.Tensor(faces).to(torch.int32).to(verts.device)
    elif isinstance(faces, torch.Tensor):
        faces = faces.to(torch.int32)
    if convert2NDC:
        verts[:, 0] = 2 * verts[:, 0].clone() / W - 1
        verts[:, 1] = 2 * verts[:, 1].clone() / H - 1
    verts[:, 2] = verts[:, 2].clone()/  1000.0
    verts = verts.contiguous()
    depth = project_mesh_cuda(
            verts, faces, verts[:,[2]], verts_filter,
            H, W)[:,:,0].data.cpu().numpy()

    mask = np.where(depth>0, 255, 0).astype('uint8')
    # return mask
    if returnBoundary:
        kernel = np.ones((3, 3), np.uint8)
        # Using cv2.erode() method
        mask0 = cv2.dilate(mask, kernel)
        mask2 = cv2.erode(mask, kernel)
        maskB = mask0 - mask2

        xs = np.round((verts[:, 0].data.cpu().numpy() + 1) / 2 * W).astype(np.int32)
        ys = np.round((verts[:, 1].data.cpu().numpy()  + 1) / 2 * H).astype(np.int32)
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs > W - 1] = W - 1
        ys[ys > H - 1] = H - 1

        boundaryFlags = np.array(maskB[ys,xs]>0)
        boundaryFlags = torch.Tensor(boundaryFlags).to(verts.device)
        return boundaryFlags
    else:
        return mask

def ndc2Mask(verts, faces, H=-1, W=-1, verts_filter=None, returnBoundary=False, convert2NDC = False):
    """
        Input:
            verts: N,3   #N,D,C
            faces: F,3
            H: -1 int
            W: -1 int
            verts_filter: N      None
        Output:
            mask: H,W torch
    """

    if verts_filter is None:
        N = verts.shape[0]
        verts_filter = torch.ones((N), dtype=torch.uint8, device=verts.device)
    if isinstance(faces, np.ndarray):
        faces = torch.Tensor(faces).to(torch.int32).to(verts.device)
    elif isinstance(faces, torch.Tensor):
        faces = faces.to(torch.int32)
    if convert2NDC:
        verts[:, 0] = 2 * verts[:, 0].clone() / W - 1
        verts[:, 1] = 2 * verts[:, 1].clone() / H - 1
    verts[:, 2] = verts[:, 2].clone()/  1000.0
    verts = verts.contiguous()
    depth = project_mesh_cuda(
            verts, faces, verts[:,[2]], verts_filter,
            H, W)[:,:,0].data.cpu().numpy()

    mask = np.where(depth>0, 255, 0).astype('uint8')
    # return mask
    if returnBoundary:
        kernel = np.ones((3, 3), np.uint8)
        # Using cv2.erode() method
        mask0 = cv2.dilate(mask, kernel)
        mask2 = cv2.erode(mask, kernel)
        maskB = mask0 - mask2

        xs = np.round((verts[:, 0].data.cpu().numpy() + 1) / 2 * W).astype(np.int32)
        ys = np.round((verts[:, 1].data.cpu().numpy()  + 1) / 2 * H).astype(np.int32)
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs > W - 1] = W - 1
        ys[ys > H - 1] = H - 1

        boundaryFlags = np.array(maskB[ys,xs]>0)
        boundaryFlags = torch.Tensor(boundaryFlags).to(verts.device)
        return boundaryFlags
    else:
        return mask

def projectPoints(verts, Ks, Es = None, H=-1, W=-1):
    """
        verts: B,N,3
        Ks: B,C,4,4
        Es: B,C,4,4   world2camera
        if H,W >0:
            return NDC  ==> u,v,Z
    """
    B, N, _ = verts.shape
    verts = torch.cat([verts, torch.ones(B,N,1).to(verts.device)], dim = -1) #B,N,4
    if not Es is None:
        verts = torch.einsum('bvm,bcmn->bcvn', verts, Es.transpose(2,3))
    verts = torch.einsum('bcvm,bcmn ->bcvn', verts, Ks.transpose(2, 3))
    verts[:,:,:, [0,1]] /= verts[:,:,:, [2]]
    if H>0 and W>0:
        verts[:, :, :, 0] = 2 * verts[:, :, :, 0] / W - 1
        verts[:, :, :, 1] = 2 * verts[:, :, :, 1] / H - 1
    return verts


def projectPoints2(verts, Ps, H=-1, W=-1):
    """
        verts: B,N,3
        Ps: B,C,4,4
        if H,W >0:
            return NDC  ==> u,v,Z
    """
    B, N, _ = verts.shape
    verts = torch.cat([verts, torch.ones(B,N,1).to(verts.device)], dim = -1) #B,N,4
    verts = torch.einsum('bvm,bcmn ->bcvn', verts, Ps.transpose(2, 3))
    verts[:,:,:, [0,1]] /= verts[:,:,:, [2]]
    if H>0 and W>0:
        verts[:, :, :, 0] = 2 * verts[:, :, :, 0] / W - 1
        verts[:, :, :, 1] = 2 * verts[:, :, :, 1] / H - 1
    return verts


def unprojectPoints(verts, PsGL, Es = None):
    """
        verts: B,N,3
        Ks: B,C,4,4
        Es: B,C,4,4   world2camera
        if H,W >0:
            return NDC  ==> u,v,Z
    """
    B, N, _ = verts.shape
    vertsCam = torch.cat([verts, torch.ones(B,N,1).to(verts.device)], dim = -1) #B,N,4
    if not Es is None:
        vertsCam = torch.einsum('bvm, bcmn->bcvn', vertsCam, Es.transpose(2,3))
    vertsCam[:,:,:,[2]] *= -1  # invert z-axis
    # vertsCam[:, :, :, [0, 1]] /= vertsCam[:, :, :, [2]]
    # print(vertsCam.shape)
    # print(PsGL.shape)
    vertsNDC = torch.einsum('bcvm,bcmn ->bcvn', vertsCam, PsGL.transpose(2, 3))

    return vertsNDC

def projectPointsGL(verts, PsGL, Es = None):
    """
        verts: B,N,3
        Ks: B,C,4,4
        Es: B,C,4,4   world2camera
        if H,W >0:
            return NDC  ==> u,v,Z
    """
    B, N, _ = verts.shape
    vertsCam = torch.cat([verts, torch.ones(B,N,1).to(verts.device)], dim = -1) #B,N,4
    if not Es is None:
        vertsCam = torch.einsum('bvm, bcmn->bcvn', vertsCam, Es.transpose(2,3))
    vertsCam[:,:,:,[2]] *= -1  # invert z-axis
    # vertsCam[:, :, :, [0, 1]] /= vertsCam[:, :, :, [2]]
    # print(vertsCam.shape)
    # print(PsGL.shape)
    vertsNDC = torch.einsum('bcvm,bcmn ->bcvn', vertsCam, PsGL.transpose(2, 3))

    return vertsNDC

def getBoundary(verts, faces, PsGL, Ks, Es = None, H=-1, W=-1):
    import nvdiffrast.torch as dr

    glctx = dr.RasterizeGLContext()
    B,C = PsGL.shape[:2]
    N = verts.shape[1]
    vertsNDC = projectPointsGL(verts, PsGL, Es)
    vertsNDC = vertsNDC.reshape(B*C, N, 4)
    rast, rast_db = dr.rasterize(glctx, vertsNDC[:], faces, resolution=[H, W])
    render, render_da = dr.interpolate(torch.ones((N, 1), device=verts.device), rast, faces, rast_db=rast_db,
                                       diff_attrs='all')
    a = dilate(render.permute(0, 3, 1, 2))
    b = erode(render.permute(0, 3, 1, 2))
    c = a - b
    c = torch.where(c>0.999,1,0) #B*C, 1, H, W

    vertsUV = projectPoints(verts, Ks, Es, H, W).reshape(B*C, N, 4)[:,:,:2]
    boundary = index(c, vertsUV)

    return boundary


def projectNormals(nmls, verts, Ks, Es = None, returnVertsNew = False, returnVerts = False):
    """
        verts: B,N,3
        Ks: B,C,4,4
        Es: B,C,4,4   world2camera
    """
    B, N, _ = nmls.shape
    verts = torch.cat([verts, torch.ones(B, N, 1).to(verts.device)], dim = -1)  # B,N,4
    nmls  = torch.cat([nmls, torch.zeros(B, N, 1).to(verts.device)], dim = -1)  # B,N,4
    if not Es is None:
        verts = torch.einsum('bvm,bcmn->bcvn', verts, Es.transpose(2,3))
        nmls = torch.einsum('bvm,bcmn->bcvn', nmls, Es.transpose(2,3))
        verts_new = verts + nmls

    verts = torch.einsum('bcvm,bcmn ->bcvn', verts, Ks.transpose(2, 3))
    verts_new = torch.einsum('bcvm,bcmn ->bcvn', verts_new, Ks.transpose(2, 3))
    verts[:, :, :, [0, 1]] /= verts[:, :, :, [2]]
    verts_new[:, :, :, [0, 1]] /= verts_new[:, :, :, [2]]

    nmls_new = verts_new[:,:,:,:2] - verts[:,:,:,:2]  # B,C,N,2
    nmls_new  = torch.nn.functional.normalize(nmls_new, dim = -1)

    if returnVertsNew:
        return nmls_new, verts_new
    elif returnVerts:
        return nmls_new, verts
    else:
        return nmls_new


def loadCrop(cropPath):
    basedir = cropPath
    fileNames = os.listdir(basedir)
    crops = []
    for name in fileNames:
        data = pandas.read_csv(osp.join(basedir, name), index_col=None, header=None)
        data = data.to_numpy()
        outls = []
        for idx in range(len(data)):
            temp_list = list(data[idx])
            temp_list = [str(item) for item in temp_list]
            outtxt = (' ').join(temp_list)
            outls.append(outtxt)
        outls = np.array(outls)
        crops.append(outls)
    crops = np.stack(crops,0)
    return crops



def index(feat, uv):        # [B, 2, N]
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # feat [B,C,H,W]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

def dilate(bin_img, ksize=3):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=3):
    out = 1 - dilate(1 - bin_img, ksize)
    return out