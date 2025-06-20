"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2024-10-01
"""
import os
import os.path as osp
import sys
sys.path.append('..')
import random
import time
import pandas
import json
from multiprocessing import Pool
import cv2
import av
from av import time_base as AV_TIME_BASE
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
from tools.omni_tools import makePath
from tools.skel_tools import loadMotion
from tools.cam_tools import loadCamera


encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

def init_params(pos, H, W):
    output = {}

    output["w"] = W
    output["h"] = H
    output["aabb_scale"] = 1
    output["scale"] = 0.46 / 1000.0  # scale to fit person into unit box
    output["offset"] = [-0.6, 0.0, 0.7]  # studio

    x, y, z = pos
    scale_offset = 0.01
    scale = 1.0 / 2.0 * float(y) - scale_offset
    scale = max(0.4, scale)
    offset_x = 0.5 - float(x) * scale
    offset_z = 0.5 - float(z) * scale

    print('scale ' + str(scale))
    print('offset_x ' + str(offset_x))
    print('offset_z ' + str(offset_z))
    output["scale"] = scale / 1000.0
    output["offset"] = [offset_x, 0, offset_z]
    output["from_na"] = True

    return output

def prepare_json_file(iframe):

    global IMAGE_H, IMAGE_W, CROP_IMAGE, CENTER_OFFSETS, CROP_W, CROP_H, INTR, EXTR, OUTPUT_DIR, padding_global

    FRAME_DIR = os.path.join(OUTPUT_DIR, padding_global % iframe)

    print('preparing jsonfile ' + str(iframe))
    output = {}
    if CROP_IMAGE:
        init_params(output, CROP_W, CROP_H, iframe)
    else:
        init_params(output, IMAGE_W, IMAGE_H, iframe)

    cameras = []
    for camera_id in range(len(INTR)):
        camera = {}
        camera['pose'] = EXTR[camera_id]
        camera['intrinsic'] = INTR[camera_id].copy()

        if CROP_IMAGE:
            camera['intrinsic'][0][2] -= CENTER_OFFSETS[camera_id][0]
            camera['intrinsic'][1][2] -= CENTER_OFFSETS[camera_id][1]

        cameras.append(camera)

    output['frames'] = []

    camera_num = len(cameras)

    for cam_idx in range(camera_num):
        camera_data = f'{cam_idx}'
        # add one_frame
        one_frame = {}
        one_frame["file_path"] = 'image_c_' + str(cam_idx) + '_f_' + str(iframe) + '.png'
        one_frame["transform_matrix"] = cameras[cam_idx]['pose'].tolist()
        ixt = cameras[cam_idx]['intrinsic']
        # intrinsic_dir = join(base_ixt_dir,f'{cam_idx}',all_ixt_dir[frame_idx])
        # ixt = load_intrinsics(intrinsic_dir)
        one_frame["intrinsic_matrix"] = ixt.tolist()
        output['frames'].append(one_frame)

    file_dir = os.path.join(FRAME_DIR + '/temp/', f'transform_{iframe:06}.json')
    # file_dir = join(output_dir,f'transform_{frame_i:06}.json')
    # file_dir = join(output_frame_dir,file_name)
    with open(file_dir, 'w') as f:
        json.dump(output, f, indent=4)
    return file_dir

def cropImg(img, mask, BGR = True):
    global cropImgSize, imgSize
    CROP_H, CROP_W = cropImgSize
    IMAGE_H, IMAGE_W = imgSize

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)

    # img = np.asarray(frame.to_image().resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)).copy()
    # result, encimg = cv2.imencode('.jpg', img, encode_param)
    # img = cv2.imdecode(encimg, 1)
    # print('img: ', img.shape)
    # print('mask: ', mask.shape)
    img = np.where(np.stack([mask,mask,mask], -1), img, 0)
    concat_img = np.concatenate([img, mask[:,:,None]], axis=-1)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

    delta_x = (CROP_W - w) / 2.0
    if delta_x > 0:
        x -= int(delta_x)

    delta_y = (CROP_H - h) / 2.0
    if delta_y > 0:
        y -= int(delta_y)

    w = CROP_W
    h = CROP_H

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > IMAGE_W:
        x = IMAGE_W - w
    if y + h > IMAGE_H:
        y = IMAGE_H - h

    CENTER_OFFSET = [y, x]
    cropped_image = concat_img[y:y + h, x:x + w]
    return cropped_image, CENTER_OFFSET

def get_keyframe_interval(cap):
    frame_number = 0

    fps = cap.streams.video[0].average_rate

    video_stream = cap.streams.video[0]

    assert int(1 / video_stream.time_base) % fps == 0

    offset_timestamp = int(1 / video_stream.time_base / fps)
    video_stream.codec_context.skip_frame = "NONKEY"

    target_timestamp = int((frame_number * AV_TIME_BASE) / video_stream.average_rate)

    cap.seek(target_timestamp)
    result = []
    iter = 0

    for frame in cap.decode(video_stream):
    # print(frame)
        if (iter > 1):
            video_stream.codec_context.skip_frame = "DEFAULT"
            return result[1] - result[0]
            break
        result.append(int(frame.pts / offset_timestamp))
        iter += 1

    video_stream.codec_context.skip_frame = "DEFAULT"
    return -1

def get_timestamp_offset(cap):
    frame_number = 0

    fps = cap.streams.video[0].average_rate

    video_stream = cap.streams.video[0]

    assert int(1 / video_stream.time_base) % fps == 0

    offset_timestamp = int(1 / video_stream.time_base / fps)

    target_timestamp = int((frame_number * AV_TIME_BASE) / video_stream.average_rate)

    cap.seek(target_timestamp)

    for packet in cap.demux():

        for frame in packet.decode():
            return int(frame.dts / offset_timestamp)

    return -1

def get_frame_av(cap, frame_number, index_offset, keyframe_interval):
    # print('frame_number ' + str(frame_number) )
    fps = cap.streams.video[0].average_rate
    # print('fps  ' + str(fps))
    video_stream = cap.streams.video[0]
    assert int(1 / video_stream.time_base) % fps == 0
    # print('video_stream.time_base  ' + str(video_stream.time_base))
    offset_timestamp = int(1 / video_stream.time_base / fps)
    # print('offset  ' + str(offset_timestamp))
    # print('duration ' + str(cap.duration  /  AV_TIME_BASE  ))
    video_stream = cap.streams.video[0]

    target_frame = int(frame_number / keyframe_interval) * keyframe_interval
    target_timestamp = int(cap.duration * float(target_frame / int(cap.streams.video[0].frames)))
    # print('target_timestamp ' + str(target_timestamp/  AV_TIME_BASE))
    cap.seek(target_timestamp)
    framex_index = -1
    for packet in cap.demux():
        # print(packet)
        for frame in packet.decode():
            # print(frame)
            if frame.dts:
                framex_index = int(frame.dts / offset_timestamp) - index_offset
            else:
                framex_index += 1
            if (framex_index == frame_number):
                # print('found frame ' + str(frame.pts / offset_timestamp) + ' ' + str(frame_number))
                return None
            # [True,cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_YUV2BGR_I420)]	#np.asarray(frame.to_ndarray(format="bgr24"))	[True,cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_YUV2BGR_I420)]

            # if(packet.dts / offset_timestamp  - time_stamp_offset== frame_number):
        #	break

    # print('camera released' + str(camera))
    return None

def cropImgMask(camIdx):
    print("start camIdx: ", camIdx)
    centerOffsetAll = []
    st = time.time()

    for fIdx in indices:
        print("{}/{}".format(fIdx, len(indices)))
        source_img_path = osp.join(in_img_dir, "cam" + str(camIdx).zfill(2), str(fIdx).zfill(8) + '.jpg')
        source_mask_path = osp.join(in_mask_dir, "cam" + str(camIdx).zfill(2), str(fIdx).zfill(8) + '.jpg')
        if osp.exists(source_img_path):
            img = cv2.imread(source_img_path, -1)[:,:,::-1]
            mask = cv2.imread(source_mask_path, -1)
            # print(mask.shape)
            if len(mask.shape)==2:
                mask = np.stack([mask,mask,mask], -1)
            else:
                if mask.shape[-1] == 1:
                    mask = np.concatenate([mask, mask, mask] ,- 1)

            cropped_image, CENTER_OFFSET = cropImg(img[:,:,::-1], mask[:,:,::-1])

        else:
            cropped_image = np.zeros([cropImgSize[0], cropImgSize[1],4]).astype('uint8')
            CENTER_OFFSET = [0, 0]

        centerOffsetAll.append([fIdx] + CENTER_OFFSET)
        makePath(osp.join(out_dir, 'imgs', str(fIdx).zfill(6)))
        makePath(osp.join(out_dir, 'centerOffsets', str(fIdx).zfill(6)))
        cv2.imwrite(osp.join(out_dir, 'imgs', str(fIdx).zfill(6), f"image_c_{str(camIdx).zfill(3)}_f_{str(fIdx).zfill(6)}.png"), cropped_image)
        np.savetxt(osp.join(out_dir, 'centerOffsets', str(fIdx).zfill(6), f"image_c_{str(camIdx).zfill(3)}_f_{str(fIdx).zfill(6)}.txt"), np.array([CENTER_OFFSET]), fmt='%d')

    print('Time: ', time.time() - st)


if __name__ == '__main__':
    numCam =  24 
    subjectType = 'Subject2618'
    clothType =  'tight'
    sequenceType = 'training'

    global cropImgSize, imgSize, INTR, EXTR
    cropImgSize = [800, 600] # [1600, 1600]  #[1600, 1200]  # H,W
    imgSize = [1150, 1330]  # H,W

    parser = argparse.ArgumentParser(description="DUT")
    parser.add_argument("-si", '--slurm_id', default=-1, help="max epoch", type=int)
    parser.add_argument("-sn", "--slurm_num", default=13, help="max epoch", type=int)
    parser.add_argument("-cn", "--cpu_num", default=9, help="max epoch", type=int)
    parser.add_argument("-cs", "--create_scene", default=False, action='store_true')
    parser.add_argument("-sc", "--select_camera", default=False, action='store_true')
    parser.add_argument("-in", "--index_name", default= 'training_indices_all.txt', type=str)
    args = parser.parse_args()

    slurm_id = args.slurm_id
    slurm_num = args.slurm_num
    create_scene = args.create_scene
    select_camera = args.select_camera
    index_name  = args.index_name
    numCpu = args.cpu_num

    camList = list(np.arange(numCam))
    camListInput = camList[slurm_id * numCam // slurm_num: (slurm_id + 1) * numCam // slurm_num]

    base_dir = "/DataPathToDUT_official/{}/{}/{}".format(subjectType, clothType, sequenceType)
    makePath(base_dir)
    motion_dir = "/DataPathToDUT_official/{}/{}/{}".format(subjectType, clothType, sequenceType)


    oriDataDir = '/DataPathToTHUman4.0/subject01'
    in_img_dir = osp.join(oriDataDir, "images")
    in_mask_dir = osp.join(oriDataDir, "masks")

    out_dir = makePath(osp.join(base_dir, 'recon_neus2'))

    activeCamera = []

    Ks, Es, _, _ = loadCamera(osp.join(osp.dirname(base_dir), 'cameras.calibration') )
    INTR = Ks
    EXTR = np.linalg.inv(Es)

    indices = np.arange(0, 5058, 1).tolist()
    print('Indices:\n')
    print(indices)

    print('select_camera: ', select_camera)
    print('create_scene: ', create_scene)
    if create_scene:
        motionPath = osp.join(motion_dir, 'motions', 'poseAngles.motion')

        if not os.path.isfile(motionPath):
            motionPath = osp.join(motion_dir, 'motions', '159dof.motion')

        humanPosAll = loadMotion(motionPath)[indices,:3]
        indices_csv = indices
        IMAGE_H, IMAGE_W = imgSize

        for idx, fIdx in enumerate(indices_csv):
            print('{}/{}'.format(fIdx, max(indices_csv)), flush=True)
            output = init_params(humanPosAll[idx] + [0,-0.2,0], H=cropImgSize[0], W= cropImgSize[1])

            cameras = []
            for camIdx in range(0, numCam):
                shiftData = np.loadtxt(osp.join(out_dir, 'centerOffsets', str(fIdx).zfill(6), f"image_c_{str(camIdx).zfill(3)}_f_{str(fIdx).zfill(6)}.txt"))
                camera = {}
                camera['pose'] = EXTR[camIdx]
                camera['intrinsic'] = INTR[camIdx].copy()
                camera['intrinsic'][0][2] -= shiftData[1]
                camera['intrinsic'][1][2] -= shiftData[0]
                cameras.append(camera)

            if select_camera:

                output['frames'] = []
                for camIdx in activeCamera:
                    # add one_frame
                    one_frame = {}
                    one_frame["file_path"] = 'image_c_' + str(camIdx) + '_f_' + str(fIdx) + '.png'
                    one_frame["transform_matrix"] = cameras[camIdx]['pose'].tolist()
                    ixt = cameras[camIdx]['intrinsic']
                    one_frame["intrinsic_matrix"] = ixt.tolist()
                    output['frames'].append(one_frame)
                out_json_path = osp.join(out_dir, 'imgs', str(fIdx).zfill(6), f'transform_{fIdx:06}_selectCamera.json')
                with open(out_json_path, 'w') as f:
                    json.dump(output, f, indent=4)
            else:

                output['frames'] = []
                for camIdx in range(numCam):
                    one_frame = {}
                    one_frame["file_path"] = f"image_c_{str(camIdx).zfill(3)}_f_{str(fIdx).zfill(6)}.png"
                    one_frame["transform_matrix"] = cameras[camIdx]['pose'].tolist()
                    ixt = cameras[camIdx]['intrinsic']
                    one_frame["intrinsic_matrix"] = ixt.tolist()
                    output['frames'].append(one_frame)

                out_json_path = osp.join(out_dir, 'imgs', str(fIdx).zfill(6), f'transform_{fIdx:06}.json')
                with open(out_json_path, 'w') as f:
                    json.dump(output, f, indent=4)
    else:
        for i in range(0, numCam // slurm_num // numCpu + 1):
            camListInputFinal = camListInput[i * numCpu: (i + 1) * numCpu]
            print(camListInputFinal)
            p = Pool(numCpu)
            p.map(cropImgMask, camListInputFinal)
            p.close()
