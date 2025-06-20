#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math

from diff_gaussian_rasterization_mip import GaussianRasterizationSettings as GaussianRasterizationSettingsMip
from diff_gaussian_rasterization_mip import GaussianRasterizer as GaussianRasterizerMip

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import depth_to_normal
from utils.graphics_utils import focal2fov, getProjectionMatrix_refine
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply


def render(K, E, H, W, T_fw, pc : GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    znear = 0.01
    zfar = 10.0
    focalX, focalY = K[0, 0], K[1, 1]
    FovX, FovY = focal2fov(focalX, W), focal2fov(focalY, H)
    projection_matrix = getProjectionMatrix_refine(K, H, W, znear, zfar).transpose(0, 1)
    world_view_transform = E.transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    tanfovx, tanfovy = math.tan(FovX * 0.5), math.tan(FovY * 0.5)


    # if subpixel_offset is None:
    subpixel_offset = torch.zeros((int(H), int(W), 2), dtype=torch.float32, device="cuda")

    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(H),
    #     image_width=int(W),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=world_view_transform,
    #     projmatrix=full_proj_transform,
    #     sh_degree=pc.active_sh_degree,
    #     campos=camera_center,
    #     prefiltered=False,
    #     debug=False
    # )
    kernel_size = 0.1
    raster_settings = GaussianRasterizationSettingsMip(
        image_height=int(H),
        image_width=int(W),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )
    
    rasterizer = GaussianRasterizerMip(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    means3D = (T_fw[..., :3, :3] @ pc.get_xyz[..., None]).squeeze(-1) + T_fw[..., :3, 3]  # V,3
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
        pass
    else:
        scales = pc.get_scaling #* 2.2
        rotations = pc.get_rotation
        rotations = quaternion_multiply(matrix_to_quaternion(T_fw[..., :3, :3]), rotations)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            shs = pc.get_features
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    # rendered_image, radii, depth, alpha = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

