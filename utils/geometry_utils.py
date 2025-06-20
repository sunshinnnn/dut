"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2024-07-20
"""

import torch
import numpy as np
import nvdiffrast.torch as dr
import pymeshlab
def compute_self_intersection(verts, faces):
    m = pymeshlab.Mesh(verts,faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "baseMesh")
    ms.load_filter_script('datas/select_self_intersection.mlx')
    ms.apply_filter_script()

    return ms.current_mesh().selected_face_number() / m.face_number() * 100.0

def assign_unique_uvs(vertices, faces, uvs, faces_uv):
    uvs_out = torch.zeros(vertices.shape[0], 2).to(vertices.device)
    uvs_flag = torch.zeros(vertices.shape[0]).to(vertices.device)
    # Iterate over each face
    for face_index, uv_indices in enumerate(faces_uv):
        face_vertices = faces[face_index]
        for i, vertex_index in enumerate(face_vertices):
            uv_index = uv_indices[i]
            uv = uvs[uv_index]
            if not uvs_flag[vertex_index]:
                uvs_out[vertex_index] = uv
                uvs_flag[vertex_index] = 1.0
    return uvs_out

def find_edges_with_two_faces(faces, handMask=None):
    # faces: Tensor of shape (F, 3), where F is the number of faces, and each face contains 3 vertices (a triangle)

    edge_to_faces = {}  # Dictionary to map edges to the list of face indices they belong to

    # Iterate over each face and extract its edges
    for face_idx, face in enumerate(faces):
        v0, v1, v2 = face[0].item(), face[1].item(), face[2].item()
        if handMask is not None and ( v0 in handMask or v1 in handMask or v2 in handMask):
            continue

        # Define the three edges of the face, sort to ensure consistency (undirected edge)
        edge1 = tuple(sorted([v0, v1]))
        edge2 = tuple(sorted([v1, v2]))
        edge3 = tuple(sorted([v2, v0]))

        # For each edge, map it to the current face index
        for edge in [edge1, edge2, edge3]:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # Prepare a list to store only the edges that belong to exactly two faces
    edges_with_two_faces = []
    face_pairs = []

    # Filter edges that belong to exactly two faces
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:  # Only consider edges shared by exactly two faces
            edges_with_two_faces.append(edge)
            face_pairs.append(faces)

    # Convert to tensors
    # edges_tensor = torch.tensor(edges_with_two_faces, dtype=torch.long)
    # face_pairs_tensor = torch.tensor(face_pairs, dtype=torch.long)
    edge2vert = torch.tensor(edges_with_two_faces, dtype=torch.long)
    edge2face = torch.tensor(face_pairs, dtype=torch.long)


    # return edges_tensor, face_pairs_tensor
    return edge2vert, edge2face


def transform_pos_batch(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    if pos.shape[-1] == 3:
        posw = torch.cat([pos, torch.ones([pos.shape[0], pos.shape[1], 1]).cuda()], axis=-1)
    else:
        posw = pos
    return torch.einsum('bmd,bmdn->bmn', posw, t_mtx.transpose(2,3)).contiguous()[:,:,:3]

def transform_pos_batch_to_render_depth(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    if pos.shape[-1] == 3:
        posw = torch.cat([pos, torch.ones([pos.shape[0], pos.shape[1], 1]).cuda()], axis=-1)
    else:
        posw = pos
    # return torch.matmul(posw, t_mtx.t())# [None, ...]
    return torch.einsum('bmd,bdn->bmn',posw, t_mtx.transpose(1,2)).contiguous()

# def render_depth_batch(glctx, mtx, pos, pos_idx, resolutions):
#     pos_clip = transform_pos_batch(mtx, pos)
#     rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolutions)
#     out, _ = dr.interpolate(pos_clip[:,:,[2]], rast_out, pos_idx)
#     return out


def find_vertex_to_face_index(N, faces):
    V = N
    F = faces.shape[0]
    vertex_to_face_index = [[] for _ in range(V)]

    for face_idx in range(F):
        for vertex_idx in faces[face_idx]:
            vertex_to_face_index[vertex_idx.item()].append(face_idx)
    max_faces_per_vertex = max(len(indices) for indices in vertex_to_face_index)
    # vertex_to_face_tensor = torch.full((V, max_faces_per_vertex), -1, dtype=torch.long)
    vertex_to_face_tensor = torch.full((V, max_faces_per_vertex), -1, dtype=torch.long)


    for vertex_idx, indices in enumerate(vertex_to_face_index):
        vertex_to_face_tensor[vertex_idx, :len(indices)] = torch.tensor(indices, dtype=torch.long)
        vertex_to_face_tensor[vertex_idx, len(indices):] = torch.tensor(indices[-1], dtype=torch.long)

    return vertex_to_face_tensor

def compute_edge_length(vertices, edges):
    # verts: B,N,3
    # Extract the vertex coordinates for each edge
    edge_vertices = vertices[:, edges]

    # Compute the squared length of each edge
    squared_edge_lengths = torch.sum((edge_vertices[:,:, 0] - edge_vertices[:,:, 1]) ** 2, dim=2)

    # Compute the length of each edge
    edge_lengths = torch.sqrt(squared_edge_lengths)

    return edge_lengths

def remove_duplicates(v, f):
    """
    Generate a mesh representation with no duplicates and
    return it along with the mapping to the original mesh layout.
    """

    unique_verts, inverse = torch.unique(v, dim=0, return_inverse=True)
    new_faces = inverse[f.long()]
    return unique_verts, new_faces, inverse

def average_edge_length(verts, faces):
    """
    Compute the average length of all edges in a given mesh.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    return (A + B + C).sum() / faces.shape[0] / 3

def massmatrix_voronoi(verts, faces):
    """
    Compute the area of the Voronoi cell around each vertex in the mesh.
    http://www.alecjacobson.com/weblog/?p=863

    params
    ------

    v: vertex positions
    f: triangle indices
    """
    # Compute edge lengths
    l0 = (verts[faces[:,1]] - verts[faces[:,2]]).norm(dim=1)
    l1 = (verts[faces[:,2]] - verts[faces[:,0]]).norm(dim=1)
    l2 = (verts[faces[:,0]] - verts[faces[:,1]]).norm(dim=1)
    l = torch.stack((l0, l1, l2), dim=1)

    # Compute cosines of the corners of the triangles
    cos0 = (l[:,1].square() + l[:,2].square() - l[:,0].square())/(2*l[:,1]*l[:,2])
    cos1 = (l[:,2].square() + l[:,0].square() - l[:,1].square())/(2*l[:,2]*l[:,0])
    cos2 = (l[:,0].square() + l[:,1].square() - l[:,2].square())/(2*l[:,0]*l[:,1])
    cosines = torch.stack((cos0, cos1, cos2), dim=1)

    # Convert to barycentric coordinates
    barycentric = cosines * l
    barycentric = barycentric / torch.sum(barycentric, dim=1)[..., None]

    # Compute areas of the faces using Heron's formula
    areas = 0.25 * ((l0+l1+l2)*(l0+l1-l2)*(l0-l1+l2)*(-l0+l1+l2)).sqrt()

    # Compute the areas of the sub triangles
    tri_areas = areas[..., None] * barycentric

    # Compute the area of the quad
    cell0 = 0.5 * (tri_areas[:,1] + tri_areas[:, 2])
    cell1 = 0.5 * (tri_areas[:,2] + tri_areas[:, 0])
    cell2 = 0.5 * (tri_areas[:,0] + tri_areas[:, 1])
    cells = torch.stack((cell0, cell1, cell2), dim=1)

    # Different formulation for obtuse triangles
    # See http://www.alecjacobson.com/weblog/?p=874
    cells[:,0] = torch.where(cosines[:,0]<0, 0.5*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,0]<0, 0.25*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,0]<0, 0.25*areas, cells[:,2])

    cells[:,0] = torch.where(cosines[:,1]<0, 0.25*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,1]<0, 0.5*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,1]<0, 0.25*areas, cells[:,2])

    cells[:,0] = torch.where(cosines[:,2]<0, 0.25*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,2]<0, 0.25*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,2]<0, 0.5*areas, cells[:,2])

    # Sum the quad areas to get the voronoi cell
    return torch.zeros_like(verts).scatter_add_(0, faces, cells).sum(dim=1)

def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [verts.index_select(1, fi[0]),
                 verts.index_select(1, fi[1]),
                 verts.index_select(1, fi[2])]

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    n = c / torch.norm(c, dim=0)
    return n

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

def compute_vertex_normals(verts, faces, face_normals):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    face_normals : torch.Tensor
        Per-face normals
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [verts.index_select(1, fi[0]),
             verts.index_select(1, fi[1]),
             verts.index_select(1, fi[2])]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0*d1, 0)
        face_angle = safe_acos(torch.sum(d0*d1, 0))
        nn =  face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])
    return (normals / torch.norm(normals, dim=0)).transpose(0, 1)