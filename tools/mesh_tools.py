"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-01-04
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def load_pointcoud(datapath):
    with open(datapath, 'r') as f:
        data = f.readlines()
        out = []
        for i in range(len(data)):
            out.append(data[i].split()[1:])
    return np.array(out).astype(np.float32)

def save_pointcloud(outputDir, fIdx, verts, scale = 1.0):
    with open(os.path.join(outputDir, 'depthMap_'+str(fIdx)+'.obj'), 'w') as f:
        for l in range(0, verts.shape[0]):
            f.write('v {:.4f} {:.4f} {:.4f}\n'.format(verts[l,0] * scale, verts[l,1] * scale, verts[l,2] * scale))

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


def load_obj_mesh(mesh_file, with_color=False, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []
    color_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            if with_color:
                c = list(map(float, values[4:7]))
                color_data.append(c)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1
    colors = np.array(color_data)

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_normal and with_color:
        norms = np.array(norm_data)
        # print(norms.shape)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1

        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, colors, norms , face_normals
    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        # norms = np.array(norm_data)
        # norms = normalize_v3(norms)
        # face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    if with_color:
        return vertices, faces, colors

    return vertices, faces

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    normalize_v3(n)
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)
    return norm


def compute_normal_torch(vertices, faces):
    normals = torch.zeros_like(vertices)  # B,N,3
    tris = vertices[:, faces]  # B,F,N,3
    n = torch.cross(tris[:, :, 1, :] - tris[:, :, 0, :], tris[:, :, 2, :] - tris[:, :, 0, :])
    n = torch.nn.functional.normalize(n, dim=-1)
    normals[:, faces[:, 0]] += n
    normals[:, faces[:, 1]] += n
    normals[:, faces[:, 2]] += n
    normals = torch.nn.functional.normalize(normals, dim=-1)
    return normals


# compute tangent and bitangent
def compute_tangent(vertices, faces, normals, uvs=None, faceuvs=None):
    # NOTE: this could be numerically unstable around [0,0,1]
    # but other current solutions are pretty freaky somehow
    c1 = np.cross(normals, np.array([0, 1, 0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)
    return tan, btan
