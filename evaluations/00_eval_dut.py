import sys
sys.path.append("..")

import time
import numpy as np
import cv2
from tools.metric_tools import PSNR, get_psnr, ssim, get_ssim, my_lpips
from tools.omni_tools import checkPlatformDir, resizeImg
from tqdm import tqdm
import os.path as osp
import torch
import time
import torchvision.transforms as T



subjectType = 'Subject0003'
# subjectType = 'Subject0022'
# subjectType = 'Subject2618'
# resoType = '1k'
# resoType = '2k'
# resoType = '4k'

# scale = 1.0
scale = 0.25
tagName = "tagName"


if subjectType == 'Subject0003':
    fIdxs = np.arange(100, 7000, 10).tolist()
    testCameraIdxs = [23, 36, 45, 62]
    split = 'testing'
    baseDir = '/CT/HOIMOCAP4/work/data'
    if scale == 1.0:
        resoType = '4k'
    elif scale == 0.25:
        resoType = '1k'
elif subjectType == 'Subject0022':
    fIdxs = np.arange(110, 7200, 10).tolist()
    testCameraIdxs = [1, 14, 25, 39]
    split = 'testing'
    baseDir = '/CT/HOIMOCAP4/work/data'
    if scale == 1.0:
        resoType = '4k'
    elif scale == 0.25:
        resoType = '1k'
elif subjectType == 'Subject2618':
    fIdxs = np.arange(3500, 5000, 3).tolist()
    testCameraIdxs = [14, 9, 2, 21]
    split = 'training'
    baseDir = '/CT/HOIMOCAP5/work/data'
    if scale == 1.0:
        resoType = '2k'
    elif scale == 0.5:
        resoType = '1k'

get_lpips = my_lpips()
gtDir = checkPlatformDir('{}/{}/tight/{}/recon_neus2/imgs'.format(baseDir, subjectType, split))

dataDir = checkPlatformDir("/CT/HOIMOCAP5/static00/results/DUG/{}/debug/render_test".format(tagName))


psnrList = []
ssimList = []
lpipsList = []
print("Tag: {}".format(tagName))
print("DUG | {} | {}".format(subjectType, resoType))


element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
for idx, fIdx in enumerate(fIdxs):
    for cIdx in testCameraIdxs:
        st = time.time()
        imgGTTemp = cv2.imread(osp.join(gtDir, str(fIdx).zfill(6) ,"image_c_{}_f_{}.png".format(str(cIdx).zfill(3), str(fIdx).zfill(6))), -1)
        H, W= imgGTTemp.shape[:2]
        if scale!= 1.0:
            imgGTTemp = resizeImg(imgGTTemp, scale= scale)
        imgGT = imgGTTemp[:,:,[2,1,0]].astype(np.float32) / 255.0
        maskGT = (imgGTTemp[:,:,[3]] / 255.0).astype(np.uint8)
        maskGT = cv2.erode(maskGT, element)
        maskGT_float = maskGT.astype(np.float32)[:,:,None]
        maskGT_float_torch = torch.Tensor(maskGT_float)

        imgPred = cv2.imread(osp.join(dataDir, "{}_{}.png".format(str(fIdx).zfill(6), str(cIdx).zfill(3))), -1)
        imgPred = imgPred[:,:,[2,1,0]].astype(np.float32) / 255.0
        imgPred *= maskGT_float
        imgGT *= maskGT_float

        psnr_ = get_psnr(imgGT, imgPred)
        ssim_ = get_ssim(imgGT, imgPred, maskGT)
        lpips_ = get_lpips.forward(imgGT, imgPred, maskGT)

        psnrList.append(psnr_)
        ssimList.append(ssim_)
        lpipsList.append(lpips_)
        ed = time.time()
        print("fIdx-cIdx: {}-{} | PSNR: {:3.4f}  | SSIM: {:3.4f} | LPIPS: {:3.4f} | FPS: {}".format(fIdx, cIdx, \
                                                                                                    psnr_, ssim_, lpips_, 1/(ed-st)))

print("Tag: {}".format(tagName))
print("DUT | {} | {}".format(subjectType, resoType))
print("Mean PSNR: {:3.4f}".format(np.mean(psnrList)))
print("Mean SSIM: {:3.4f}".format(np.mean(ssimList)))
print("Mean LPIPS: {:3.4f}".format( np.mean(lpipsList) ))
