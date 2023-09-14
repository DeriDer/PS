import numpy as np
from sklearn.preprocessing import normalize
import cv2
import os

def compute_Norm(I, L, mask):
    '''compute the surface normal vector'''
    # least square for Normal calculation
    N = np.linalg.lstsq(L, I, rcond=-1)[0].T
    N = normalize(N, axis=1)    
    return N

def show_surfNorm(img,steps=3):
    height,width,_ = img.shape
    dst=np.zeros((height,width,3),np.float64)
    for i in range(3):
        for x in range(0,height,steps):
            for y in range(0,width,steps):
                dst[x][y][i]=img[x][y][i]

    return dst

def disp_Normmap(norm):
    """display the normal map """
    N = norm
    cv2.imshow('normal map', N)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print('save normal map') 
    N = N * 255
    path = './results'
    if not os.path.exists(path):
        os.makedirs(path)
        print("created result folder")
    cv2.imwrite("./results/"+"norm.jpg",N)
