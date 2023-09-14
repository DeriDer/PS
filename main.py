import cv2
import numpy as np
from PIL import Image
# import all necessary scripts
from surfaceNorm import compute_Norm,disp_Normmap
from depthMap import compute_depth
from read_Images import read_Images
from calib import calib_light


def process(path:str) -> bool:
    #step 1 calibration light source
    calib_light()

    #assume calibrated light source
    I, L, mask, mask2 = read_Images('./data/cat','cat')
    height, width, _  = mask.shape
    print('='*20)

    #step 2 compute surface norm 
    normal = compute_Norm(I, L, mask)
    N = np.reshape(normal.copy(),(height,width,3))
    # RGB-> BGR
    N[:,:,0],N[:,:,2] = N[:,:,2], N[:,:,0].copy()
    zeromat=np.zeros((height,width,3),np.uint8)
    result = N + mask
    disp_Normmap(result)
    print('='*20)

    #step 3 compute depth map
    Z = compute_depth(mask2,normal)
    print('='*20)
    

    print('success')
    return True


if __name__ == '__main__':
    process('/')

