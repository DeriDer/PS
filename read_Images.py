import cv2
from PIL import Image
import numpy as np

def read_Images(path:str,Image_name:str):

    # =================read  MASK=================
    print('reading mask')
    image_path = path + '/' + Image_name + '.mask.png'
    mask = cv2.imread(image_path)
    mask2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height,width,_=mask.shape
    dst=np.zeros((height,width,3),np.uint8)
    for k in range(3):
        for i in range(height):
            for j in range(width):
                dst[i,j][k]=255-mask[i,j][k]

    # ================read light vector=================
    print('reading light direction')
    file_path = path + "/lights.txt"
    file = open(file_path,'r')
    L=[]
    i=0
    while 1:
        line = file.readline()
        if not line:
            break
        if(i!=0):
            line = line.split(" ")
            line[2] = line[2].replace("\n",'')
            for l in range(3):
                line[l] = float(line[l])
            L.append(tuple(line))
        i+=1
    file.close()
    L = np.array(L)

    # =================read picture info=================
    print('reading images')
    I = []
    for i in range(12):
        picture = np.array(Image.open(path + "/"+Image_name+'.'+str(i)+'.png'),'f')
        picture = cv2.cvtColor(picture,cv2.COLOR_RGB2GRAY)
        height, width = picture.shape #(340, 512)
        picture = picture.reshape((-1,1)).squeeze(1)
        I.append(picture)
    I = np.array(I)
    print('reading done')
    return I, L, dst, mask2

