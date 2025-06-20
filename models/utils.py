import numpy as np
import torch
import torch.nn as nn
from sys import platform
import os
import logging
import time
import datetime

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def setup_logger(save_dir='', log_filename=None, log=True, save = True, format_type='m'):
    FORMAT_DICTS = {
        'tm': "%(asctime)s: %(message)s",
        'tlm': "%(asctime)s: [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        'lm': "[%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        'm': "%(message)s",
    }
    # if not os.path.exists(save_dir):
    #     print('[ERROR] You must give a dir path to save logger.')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s: %(message)s")
    # formatter = logging.Formatter("%(message)s")
    formatter = logging.Formatter(FORMAT_DICTS[format_type])
    if log_filename is None:
        log_filename = os.path.join(save_dir, time.strftime("%Y-%m-%d_%H:%M:%S@", time.localtime()).replace(':','~') +'logging.txt')
    else:
        log_filename = os.path.join(save_dir, log_filename + 'logging.txt')
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(log_filename, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if log:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger, log_filename

class AdamUniform(torch.optim.Optimizer):
    """
    Variant of Adam with uniform scaling by the second moment.

    Instead of dividing each component by the square root of its second moment,
    we divide all of them by the max.
    """
    def __init__(self, params, lr=0.1, betas=(0.9,0.999), weight_decay=0.0, eps=1e-8):
        # defaults = dict(lr=lr, betas=betas)
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super(AdamUniform, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamUniform, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state)==0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1-b1)
                g2.mul_(b2).add_(grad.square(), alpha=1-b2)
                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                # This is the only modification we make to the original Adam algorithm
                gr = m1 / (1e-8 + m2.sqrt().max())
                p.data.sub_(gr, alpha=lr)




class PatchSampler():
    def __init__(self, num_patch=1, patch_size=16, ratio_mask=1.0):
        self.n = num_patch
        self.patch_size = patch_size
        self.p = ratio_mask
        assert self.patch_size % 2 == 0, "patch size has to be even"

    def sample(self, mask, *args):
        patch = (self.patch_size, self.patch_size)
        shape = mask.shape[:2]
        if np.random.rand() < self.p:
            o = patch[0] // 2
            valid = mask[o:-o, o:-o] > 0
            (xs, ys) = torch.where(valid)
            idx = torch.randperm(len(xs))[:self.n]
            x, y = xs[idx], ys[idx]
        else:
            x = torch.random.randint(0, shape[0] - patch[0], size=self.n)
            y = torch.random.randint(0, shape[1] - patch[1], size=self.n)
        # return x, y
        return torch.cat([x,y],0)

def getProjectionMatrix_refine(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P


def getProjectionMatrix_refine_batch_batch(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    T = K.shape[0]
    B = K.shape[1]
    fx = K[:, :, 0, 0]
    fy = K[:, :, 1, 1]
    cx = K[:, :, 0, 2]
    cy = K[:, :, 1, 2]
    s =  K[:, :, 0, 1]
    P = torch.zeros(T, B, 4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[:, :, 0, 0] = 2 * fx / W
    P[:, :, 0, 1] = 2 * s / W
    P[:, :, 0, 2] = -1 + 2 * (cx / W)

    P[:, :, 1, 1] = 2 * fy / H
    P[:, :, 1, 2] = -1 + 2 * (cy / H)

    P[:, :, 2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[:, :, 2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[:, :, 3, 2] = z_sign

    return P
def getProjectionMatrix_refine_batch(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    B = K.shape[0]
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    s =  K[:, 0, 1]
    P = torch.zeros(B, 4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[:, 0, 0] = 2 * fx / W
    P[:, 0, 1] = 2 * s / W
    P[:, 0, 2] = -1 + 2 * (cx / W)

    P[:, 1, 1] = 2 * fy / H
    P[:, 1, 2] = -1 + 2 * (cy / H)

    P[:, 2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[:, 2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[:, 3, 2] = z_sign

    return P


def load_pointcoud(datapath):
    with open(datapath, 'r') as f:
        data = f.readlines()
        out = []
        for i in range(len(data)):
            out.append(data[i].split()[1:])
    return np.array(out).astype(np.float32)

def list_difference(list1, list2):
    """
    Returns the elements that are unique to each list.

    Args:
    list1 (list): The first list.
    list2 (list): The second list.

    Returns:
    tuple: A tuple containing two lists:
           - The elements in list1 but not in list2.
           - The elements in list2 but not in list1.
    """
    # Elements in list1 but not in list2
    diff1 = [x for x in list1 if x not in list2]

    return diff1

def save_ply(fname, points, faces=None, colors=None):
    if faces is None and colors is None:
        points = points.reshape(-1,3)
        to_save = points
        return np.savetxt(fname,
                  to_save,
                  fmt='%.6f %.6f %.6f',
                  comments='',
                  header=(
                      'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nend_header'.format(points.shape[0]))
                        )
    elif faces is None and not colors is None:
        points = points.reshape(-1,3)
        colors = colors.reshape(-1,3)
        to_save = np.concatenate([points, colors], axis=-1)
        return np.savetxt(fname,
                          to_save,
                          fmt='%.6f %.6f %.6f %d %d %d',
                          comments='',
                          header=(
                      'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header'.format(
                      points.shape[0])))

    elif not faces is None and colors is None:
        points = points.reshape(-1,3)
        faces = faces.reshape(-1,3)
        with open(fname,'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nelement face {:d}\nproperty list uchar int vertex_indices\nend_header\n'.format(
                      points.shape[0],faces.shape[0]))
            for i in range(points.shape[0]):
                f.write('%.6f %.6f %.6f\n'%(points[i,0],points[i,1],points[i,2]))
            for i in range(faces.shape[0]):
                f.write('3 %d %d %d\n'%(faces[i,0],faces[i,1],faces[i,2]))
    elif not faces is None and not colors is None:
        points = points.reshape(-1,3)
        colors = colors.reshape(-1,3)
        faces = faces.reshape(-1,3)
        with open(fname,'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {:d}\nproperty list uchar int vertex_indices\nend_header\n'.format(
                      points.shape[0],faces.shape[0]))
            for i in range(points.shape[0]):
                f.write('%.6f %.6f %.6f %d %d %d\n'%(points[i,0],points[i,1],points[i,2],colors[i,0],colors[i,1],colors[i,2]))
            for i in range(faces.shape[0]):
                f.write('3 %d %d %d\n'%(faces[i,0],faces[i,1],faces[i,2]))

def checkPlatformDir(path):
    if path == '' or path is None:
        return None
    if platform == "win32" or platform=="win64":
        win = True
    else:
        win = False
    if not win:
        if platform == "linux" or platform == "linux2":
            if path[:2]=='Z:':
                path = '/HPS'+ path[2:]
            elif path[:2]=='Y:':
                path = '/CT'+ path[2:]
            else:
                pass
    else:
        if platform == "win32":
            if path[:3] == '/HP':
                path = 'Z:' + path[4:]
            elif path[:3] == '/CT':
                path = 'Y:' + path[3:]
            else:
                pass
    return path

def makePath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path, exist_ok=True))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path,exist_ok=True)
    return desired_path

def compute_normal_torch(vertices, faces):
    # import torch
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    # norm = torch.zeros(vertices.shape, dtype=vertices.dtype)
    normals = torch.zeros_like(vertices)  # B,N,3
    faces = faces.to(torch.long)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[:, faces]  # B,F,N,3
    # n = torch.cross(tris[:, :, :, 1] - tris[:, :, :, 0], tris[:, :, :, 2] - tris[:, :, :, 0])
    n = torch.cross(tris[:, :, 1, :] - tris[:, :, 0, :], tris[:, :, 2, :] - tris[:, :, 0, :])
    # n = torch.cross(tris[:, :, :, 1] - tris[:, :, :, 0], tris[:, :, :, 2] - tris[:, :, :, 0])
    # n = torch.cross(tris[..., 1] - tris[..., 0], tris[..., 2] - tris[..., 0])
    # n = torch.cross(tris[..., 1] - tris[..., 0], tris[..., 2] - tris[..., 0])
    # n = torch.cross(tris[..., 1][0] - tris[..., 0][0], tris[..., 2][0] - tris[..., 0][0])
    # print()
    n = torch.nn.functional.normalize(n, dim=-1)
    # n = n / torch.linalg.norm(n, ord = 2,  dim=-1, keepdim = True)

    normals[:, faces[:, 0]] += n
    normals[:, faces[:, 1]] += n
    normals[:, faces[:, 2]] += n
    normals = torch.nn.functional.normalize(normals, dim=-1)
    # print(normals.shape)
    # normals = normals / torch.linalg.norm(normals, ord = 2,  dim=-1, keepdim = True)
    # print(torch.linalg.norm(normals, dim=-1, keepdim = True).shape)

    return normals

# def index(feat, uv):        # [B, 2, N]     # feat [B,C,H,W]
#     uv = uv.transpose(1, 2)  # [B, N, 2]
#     uv = uv.unsqueeze(2)  # [B, N, 1, 2]
#     samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
#     return samples[:, :, :, 0]  # [B, C, N]

def index_feat(feat, uv):        # [B, 2, N]     # feat [B,C,H,W]
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

def load_camera_param(cam_path):
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
            for i in range(len(cam_data['frames'])):
                K_temp = np.eye(4)
                K_temp[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
                Ks.append(K_temp)
                tempE = np.array(cam_data['frames'][i]['transform_matrix'])
                tempE[:3, 3] /= 1000
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
                for i in range(len(cam_data['frames'])):
                    K_temp = np.eye(4)
                    K_temp[:3, :3] = np.array(cam_data['frames'][i]['intrinsic_matrix'])[:3, :3]
                    Ks.append(K_temp)
                    tempE = np.array(cam_data['frames'][i]['transform_matrix'])
                    tempE[:3, 3] /= 1000
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

def load_ddc_param(path, returnTensor=False, frames = None):
    assert isinstance(frames, list)
    params = dict(np.load(str(path)))
    idxList = []
    for frame in frames:
        idx = list(params['frameList']).index(frame)
        idxList.append(idx)

    motion = params["motionList"].astype(np.float32)[idxList]
    deltaR = params["deltaRList"].astype(np.float32)[idxList]
    deltaT = params["deltaTList"].astype(np.float32)[idxList]
    displacement = params["displacementList"].astype(np.float32)[idxList]

    if isinstance(motion, tuple):
        motion = motion[0]
    if isinstance(deltaR, tuple):
        deltaR = deltaR[0]
    if isinstance(deltaT, tuple):
        deltaT = deltaT[0]
    if isinstance(displacement, tuple):
        displacement = displacement[0]
    if returnTensor:
        motion = torch.Tensor(motion)
        deltaR = torch.Tensor(deltaR)
        deltaT = torch.Tensor(deltaT)
        displacement = torch.Tensor(displacement)
    return {
        "frame": frames,
        "motion": motion,
        "deltaR": deltaR,
        "deltaT": deltaT,
        "displacement": displacement,
    }