"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2024-07-20
"""

import nvdiffrast.torch as dr
import torch
from .geometry_utils import transform_pos_batch_to_render_depth


def render_depth_batch(glctx, mtx, pos, pos_idx, resolutions):
    pos_clip = transform_pos_batch_to_render_depth(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolutions)
    out, _ = dr.interpolate(pos_clip[:,:,[2]], rast_out, pos_idx)
    return out

def render_uv(glctx, uvs, facesuv, resolutions, feats, faces):
    # pdb.set_trace()
    pos_clip_uv = torch.cat([(uvs - 0.5 )*2, 1 * torch.ones([uvs.shape[0], 2]).to(feats.device)], axis=1)
    rast_out_uv, rast_out_db_uv = dr.rasterize(glctx, pos_clip_uv[None], facesuv, resolution = resolutions )
    color, _ = dr.interpolate(feats[None, ...], rast_out_uv, faces)
    color = color * torch.clamp(rast_out_uv[..., -1:], 0, 1) # Mask out background.
    return color

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