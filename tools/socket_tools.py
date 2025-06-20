'''
  @ Date: 2021-05-25 11:14:48
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-05 19:32:56
'''

import socket
import time
from threading import Thread
from queue import Queue
from datetime import datetime
import cv2
import numpy as np
import os.path as osp
import os
import copy
import open3d as o3d

from .omni_tools import makePath
from .timer_tools import Timer

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh

load_mesh_o3d = o3d.io.read_triangle_mesh
load_pcd_o3d = o3d.io.read_point_cloud
vis_o3d = o3d.visualization.draw_geometries
write_mesh_o3d = o3d.io.write_triangle_mesh

def create_mesh_o3d(vertices, faces, colors=None, normal=True, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    if colors is not None and isinstance(colors, np.ndarray):
        mesh.vertex_colors = Vector3dVector(colors)
    elif colors is not None and isinstance(colors, list):
        mesh.paint_uniform_color(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    if normal:
        mesh.compute_vertex_normals()
    return mesh

import importlib
def load_object(module_name, module_args, **extra_args):
    module_path = '.'.join(module_name.split('.')[:-1])
    module = importlib.import_module(module_path)
    name = module_name.split('.')[-1]
    obj = getattr(module, name)(**extra_args, **module_args)
    return obj

def log(x):
    time_now = datetime.now().strftime("%m-%d-%H:%M:%S.%f ")
    print(time_now + x)

def mkout(x):
    if x is not None:
        makePath(osp.dirname(x))

colors_bar_rgb = [
    (94, 124, 226), # 青色
    (255, 200, 87), # yellow
    (74,  189,  172), # green
    (8, 76, 97), # blue
    (219, 58, 52), # red
    (77, 40, 49), # brown
]

colors_table = {
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    'r': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    'y': [ 250/255.,  230/255.,  154/255.],
    'g':[0,255/255,0],
    'k':[0,0,0],
    '_r':[255/255,0,0],
    '_g':[0,255/255,0],
    '_b':[0,0,255/255],
    '_k':[0,0,0],
    '_y':[255/255,255/255,0],
    'purple':[128/255,0,128/255],
    'smap_b':[51/255,153/255,255/255],
    'smap_r':[255/255,51/255,153/255],
    'person': [255/255,255/255,255/255],
    'handl': [255/255,51/255,153/255],
    'handr': [51/255,255/255,153/255],
}


def get_rgb(index):
    if isinstance(index, int):
        if index == -1:
            return (255, 255, 255)
        if index < -1:
            return (0, 0, 0)
        # elif index == 0:
        #     return (245, 150, 150)
        col = list(colors_bar_rgb[index%len(colors_bar_rgb)])[::-1]
    elif isinstance(index, str):
        col = colors_table.get(index, (1, 0, 0))
        col = tuple([int(c*255) for c in col[::-1]])
    else:
        raise TypeError('index should be int or str')
    return col

def get_rgb_01(index):
    col = get_rgb(index)
    return [i*1./255 for i in col[:3]]

class BaseCrit:
    def __init__(self, min_conf, min_joints=3) -> None:
        self.min_conf = min_conf
        self.min_joints = min_joints
        self.name = self.__class__.__name__

    def __call__(self, keypoints3d, **kwargs):
        # keypoints3d: (N, 4)
        conf = keypoints3d[..., -1]
        conf[conf<self.min_conf] = 0
        idx = keypoints3d[..., -1] > self.min_conf
        return len(idx) > self.min_joints

class CritRange(BaseCrit):
    def __init__(self, minr, maxr, rate_inlier, min_conf) -> None:
        super().__init__(min_conf)
        self.min = minr
        self.max = maxr
        self.rate = rate_inlier

    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        crit = (k3d[:, 0] > self.min[0]) & (k3d[:, 0] < self.max[0]) & \
               (k3d[:, 1] > self.min[1]) & (k3d[:, 1] < self.max[1]) & \
               (k3d[:, 2] > self.min[2]) & (k3d[:, 2] < self.max[2])
        self.log = '{}: {}'.format(self.name, k3d)
        return crit.sum() / crit.shape[0] > self.rate

def myarray2string(array, separator=', ', fmt='%7.7f', indent=8):
    assert len(array.shape) == 2, 'Only support MxN matrix, {}'.format(array.shape)
    blank = ' ' * indent
    res = ['[']
    for i in range(array.shape[0]):
        res.append(blank + '  ' + '[{}]'.format(separator.join([fmt%(d) for d in array[i]])))
        if i != array.shape[0] -1:
            res[-1] += ', '
    res.append(blank + ']')
    return '\r\n'.join(res)

def write_common_results(dumpname=None, results=[], keys=[], fmt='%2.3f'):
    format_out = {'float_kind':lambda x: fmt % x}
    out_text = []
    out_text.append('[\n')
    for idata, data in enumerate(results):
        out_text.append('    {\n')
        output = {}
        output['id'] = data['id']
        for k in ['type']:
            if k in data.keys():output[k] = '\"{}\"'.format(data[k])
        keys_current = [k for k in keys if k in data.keys()]
        for key in keys_current:
            # BUG: This function will failed if the rows of the data[key] is too large
            # output[key] = np.array2string(data[key], max_line_width=1000, separator=', ', formatter=format_out)
            output[key] = myarray2string(data[key], separator=', ', fmt=fmt)
        for key in output.keys():
            out_text.append('        \"{}\": {}'.format(key, output[key]))
            if key != keys_current[-1]:
                out_text.append(',\n')
            else:
                out_text.append('\n')
        out_text.append('    }')
        if idata != len(results) - 1:
            out_text.append(',\n')
        else:
            out_text.append('\n')
    out_text.append(']\n')
    if dumpname is not None:
        mkout(dumpname)
        with open(dumpname, 'w') as f:
            f.writelines(out_text)
    else:
        return ''.join(out_text)

def encode_detect(data):
    res = write_common_results(None, data, ['keypoints3d'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_motion(data):
    res = write_common_results(None, data, ['motion'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_vertices(data):
    res = write_common_results(None, data, ['vertices'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_smpl(data):
    res = write_common_results(None, data, ['poses', 'shapes', 'expression', 'Rh', 'Th'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_image_opencv(image):
    fourcc = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    #frame을 binary 형태로 변환 jpg로 decoding
    result, img_encode = cv2.imencode('.jpg', image, fourcc)
    data = np.array(img_encode) # numpy array로 안바꿔주면 ERROR
    data = data.reshape(-1,1)
    # stringData = data.tostring()
    # return stringData
    return data

def encode_image(data):
    res = write_common_results(None, data, ['images'], fmt='%d')
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

class BaseSocketClient:
    def __init__(self, host, port) -> None:
        if host == 'auto':
            host = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        self.s = s

    def send(self, data):
        val = encode_detect(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)

    def send_smpl(self, data):
        val = encode_smpl(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)
        
    def send_motion(self, data):
        val = encode_motion(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)

    def send_vertices(self, data):
        val = encode_vertices(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)

    def send_images(self, data):
        val = encode_image(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)

    def close(self):
        self.s.close()

class BaseSocket:
    def __init__(self, host, port, debug=False) -> None:
        # 创建 socket 对象
        print('[Info] server start')
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((host, port))
        serversocket.listen(1)
        self.serversocket = serversocket
        self.queue = Queue()
        self.t = Thread(target=self.run)
        self.t.start()
        self.debug = debug
        self.disconnect = False

    @staticmethod
    def recvLine(sock):
        flag = True
        result = b''
        while not result.endswith(b'\n'):
            res = sock.recv(1)
            if not res:
                flag = False
                break
            result += res
        return flag, result.strip().decode('ascii')

    @staticmethod
    def recvAll(sock, l):
        l = int(l)
        result = b''
        while (len(result) < l):
            t = sock.recv(l - len(result))
            result += t
        return result.decode('ascii')

    def run(self):
        while True:
            clientsocket, addr = self.serversocket.accept()
            print("[Info] Connect: %s" % str(addr))
            self.disconnect = False
            while True:
                flag, l = self.recvLine(clientsocket)
                if not flag:
                    print("[Info] Disonnect: %s" % str(addr))
                    self.disconnect = True
                    break
                data = self.recvAll(clientsocket, l)
                if self.debug: log('[Info] Recv data')
                self.queue.put(data)
            clientsocket.close()

    def update(self):
        time.sleep(1)
        while not self.queue.empty():
            log('update')
            data = self.queue.get()
            self.main(data)

    def main(self, datas):
        print(datas)

    def __del__(self):
        self.serversocket.close()
        self.t.join()


rotate = False


def o3d_callback_rotate(vis=None):
    global rotate
    rotate = not rotate
    return False


class VisOpen3DSocket(BaseSocket):
    def __init__(self, host, port, cfg) -> None:
        # output
        self.write = cfg.write
        self.out = cfg.out
        self.cfg = cfg
        if self.write:
            print('[Info] capture the screen to {}'.format(self.out))
            os.makedirs(self.out, exist_ok=True)
        # scene
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(ord('A'), o3d_callback_rotate)
        if cfg.rotate:
            o3d_callback_rotate()
        vis.create_window(window_name='Visualizer', width=cfg.width, height=cfg.height)
        self.vis = vis
        # load the scene
        for key, mesh_args in cfg.scene.items():
            mesh = load_object(key, mesh_args)
            self.vis.add_geometry(mesh)
        for key, val in cfg.extra.items():
            mesh = load_mesh_o3d(val["path"])
            trans = np.array(val['transform']).reshape(4, 4)
            mesh.transform(trans)
            self.vis.add_geometry(mesh)
        # create vis => update => super() init
        super().__init__(host, port, debug=cfg.debug)
        self.block = cfg.block
        if os.path.exists(cfg.body_model_template):
            body_template = o3d.io.read_triangle_mesh(cfg.body_model_template)
            self.body_template = body_template
        else:
            self.body_template = None
        self.body_model = load_object(cfg.body_model.module, cfg.body_model.args)
        zero_params = self.body_model.init_params(1)
        self.max_human = cfg.max_human
        self.track = cfg.track
        self.filter = cfg.filter
        self.camera_pose = cfg.camera.camera_pose
        self.init_camera(cfg.camera.camera_pose)
        self.zero_vertices = Vector3dVector(np.zeros((self.body_model.nVertices, 3)))

        self.vertices, self.meshes = [], []
        for i in range(self.max_human):
            self.add_human(zero_params)

        self.count = 0
        self.previous = {}
        self.critrange = CritRange(**cfg.range)
        self.new_frames = cfg.new_frames

    def add_human(self, zero_params):
        vertices = self.body_model(return_verts=True, return_tensor=False, **zero_params)[0]
        self.vertices.append(vertices)
        if self.body_template is None:  # create template
            mesh = create_mesh_o3d(vertices=vertices, faces=self.body_model.faces, colors=self.body_model.color)
        else:
            mesh = copy.deepcopy(self.body_template)
        self.meshes.append(mesh)
        self.vis.add_geometry(mesh)
        self.init_camera(self.camera_pose)

    @staticmethod
    def set_camera(cfg, camera_pose):
        theta, phi = np.deg2rad(-(cfg.camera.theta + 90)), np.deg2rad(cfg.camera.phi)
        theta = theta + np.pi
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        rot_x = np.array([
            [1., 0., 0.],
            [0., ct, -st],
            [0, st, ct]
        ])
        rot_z = np.array([
            [cp, -sp, 0],
            [sp, cp, 0.],
            [0., 0., 1.]
        ])
        camera_pose[:3, :3] = rot_x @ rot_z
        return camera_pose

    def init_camera(self, camera_pose):
        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        # init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
        init_param.extrinsic = np.array(camera_pose)
        ctr.convert_from_pinhole_camera_parameters(init_param)

    def get_camera(self):
        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        return np.array(init_param.extrinsic)

    def filter_human(self, datas):
        datas_new = []
        for data in datas:
            kpts3d = np.array(data['keypoints3d'])
            data['keypoints3d'] = kpts3d
            pid = data['id']
            if pid not in self.previous.keys():
                if not self.critrange(kpts3d):
                    continue
                self.previous[pid] = 0
            self.previous[pid] += 1
            if self.previous[pid] > self.new_frames:
                datas_new.append(data)
        return datas_new

    def main(self, datas):
        if self.debug: log('[Info] Load data {}'.format(self.count))
        if isinstance(datas, str):
            datas = osp.join.loads(datas)
        for data in datas:
            for key in data.keys():
                if key == 'id':
                    continue
                data[key] = np.array(data[key])
            if 'keypoints3d' not in data.keys() and self.filter:
                data['keypoints3d'] = self.body_model(return_verts=False, return_tensor=False, **data)[0]
        if self.filter:
            datas = self.filter_human(datas)
        with Timer('forward'):
            params = []
            for i, data in enumerate(datas):
                if i >= len(self.meshes):
                    print('[Error] the number of human exceeds!')
                    self.add_human(data)
                if 'vertices' in data.keys():
                    vertices = data['vertices']
                    self.vertices[i] = Vector3dVector(vertices)
                else:
                    params.append(data)
            if len(params) > 0:
                params = self.body_model.merge_params(params, share_shape=False)
                vertices = self.body_model(return_verts=True, return_tensor=False, **params)
                for i in range(vertices.shape[0]):
                    self.vertices[i] = Vector3dVector(vertices[i])
            for i in range(len(datas), len(self.meshes)):
                self.vertices[i] = self.zero_vertices
        # Open3D will lock the thread here
        with Timer('set vertices'):
            for i in range(len(self.vertices)):
                self.meshes[i].vertices = self.vertices[i]
                if i < len(datas) and self.track:
                    col = get_rgb_01(datas[i]['id'])
                    self.meshes[i].paint_uniform_color(col)

    def o3dcallback(self):
        if rotate:
            self.cfg.camera.phi += np.pi / 10
            camera_pose = self.set_camera(self.cfg, self.get_camera())
            self.init_camera(camera_pose)

    def update(self):
        if self.disconnect and not self.block:
            self.previous.clear()
        if not self.queue.empty():
            if self.debug: log('Update' + str(self.queue.qsize()))
            datas = self.queue.get()
            if not self.block:
                while self.queue.qsize() > 0:
                    datas = self.queue.get()
            self.main(datas)
            with Timer('update geometry'):
                for mesh in self.meshes:
                    mesh.compute_triangle_normals()
                    self.vis.update_geometry(mesh)
                self.o3dcallback()
                self.vis.poll_events()
                self.vis.update_renderer()
            if self.write:
                outname = osp.join(self.out, '{:06d}.jpg'.format(self.count))
                with Timer('capture'):
                    self.vis.capture_screen_image(outname)
            self.count += 1
        else:
            with Timer('update renderer', True):
                self.o3dcallback()
                self.vis.poll_events()
                self.vis.update_renderer()


from yacs.config import CfgNode as CN
import socket
import numpy as np
import json
def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        try:
            data = json.load(f)
        except:
            print('Reading error {}'.format(path))
            data = []
    return data

def read_smpl(filename):
    datas = read_json(filename)
    if isinstance(datas, dict):
        datas = datas['annots']
    outputs = []
    for data in datas:
        for key in ['Rh', 'Th', 'poses', 'handl', 'handr', 'shapes', 'expression', 'keypoints3d']:
            if key in data.keys():
                data[key] = np.array(data[key], dtype=np.float32)
        # for smplx results
        outputs.append(data)
    return outputs

def read_keypoints3d(filename):
    data = read_json(filename)
    res_ = []
    for d in data:
        pid = d['id'] if 'id' in d.keys() else d['personID']
        ret = {'id': pid, 'type': 'body25'}
        for key in ['keypoints3d', 'handl3d', 'handr3d', 'face3d']:
            if key not in d.keys():continue
            pose3d = np.array(d[key], dtype=np.float32)
            if pose3d.shape[1] == 3:
                pose3d = np.hstack([pose3d, np.ones((pose3d.shape[0], 1))])
            ret[key] = pose3d
        res_.append(ret)
    return res_
class BaseConfig:
    @classmethod
    def load_from_args(cls, default_cfg='config/vis/base.yml'):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default=default_cfg)
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument("--opts", default=[], nargs='+')
        args = parser.parse_args()
        return cls.load(filename=args.cfg, opts=args.opts, debug=args.debug)

    @classmethod
    def load_args(cls, usage=None):
        import argparse
        parser = argparse.ArgumentParser(usage=usage)
        parser.add_argument('--cfg', type=str, default='config/vis/base.yml')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--slurm', action='store_true')
        parser.add_argument("opts", default=None, nargs='+')
        args = parser.parse_args()
        return args, cls.load(filename=args.cfg, opts=args.opts, debug=args.debug)

    @classmethod
    def load(cls, filename=None, opts=[], debug=False) -> CN:
        cfg = CN()
        cfg = cls.init(cfg)
        if filename is not None:
            cfg.merge_from_file(filename)
        if len(opts) > 0:
            cfg.merge_from_list(opts)
        cls.parse(cfg)
        if debug:
            cls.print(cfg)
        return cfg

    @staticmethod
    def init(cfg):
        return cfg

    @staticmethod
    def parse(cfg):
        pass

    @staticmethod
    def print(cfg):
        print('[Info] --------------')
        print('[Info] Configuration:')
        print('[Info] --------------')
        print(cfg)

class Config(BaseConfig):
    @staticmethod
    def init(cfg):
        # input and output
        cfg.host = 'auto'
        cfg.port = 9999
        cfg.width = 1920
        cfg.height = 1080

        cfg.max_human = 5
        cfg.track = True
        cfg.block = True  # block visualization or not, True for visualize each frame, False in realtime applications
        cfg.rotate = False
        cfg.debug = False
        cfg.write = False
        cfg.out = '/'
        # scene:
        cfg.scene_module = "easymocap.visualize.o3dwrapper"
        cfg.scene = CN()
        cfg.extra = CN()
        cfg.range = CN()
        cfg.new_frames = 0

        # skel
        cfg.skel = CN()
        cfg.skel.joint_radius = 0.02
        cfg.body_model_template = "none"
        # camera
        cfg.camera = CN()
        cfg.camera.phi = 0
        cfg.camera.theta = -90 + 60
        cfg.camera.cx = 0.
        cfg.camera.cy = 0.
        cfg.camera.cz = 6.
        cfg.camera.set_camera = False
        cfg.camera.camera_pose = []
        # range
        cfg.range = CN()
        cfg.range.minr = [-100, -100, -10]
        cfg.range.maxr = [100, 100, 10]
        cfg.range.rate_inlier = 0.8
        cfg.range.min_conf = 0.1
        return cfg

    @staticmethod
    def parse(cfg):
        if cfg.host == 'auto':
            cfg.host = socket.gethostname()
        if cfg.camera.set_camera:
            pass
        else:  # use default camera
            # theta, phi = cfg.camera.theta, cfg.camera.phi
            theta, phi = np.deg2rad(cfg.camera.theta), np.deg2rad(cfg.camera.phi)
            cx, cy, cz = cfg.camera.cx, cfg.camera.cy, cfg.camera.cz
            st, ct = np.sin(theta), np.cos(theta)
            sp, cp = np.sin(phi), np.cos(phi)
            dist = 6
            camera_pose = np.array([
                [cp, -st * sp, ct * sp, cx],
                [sp, st * cp, -ct * cp, cy],
                [0., ct, st, cz],
                [0.0, 0.0, 0.0, 1.0]])
            cfg.camera.camera_pose = camera_pose.tolist()