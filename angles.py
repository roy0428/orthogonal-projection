import numpy as np
import math
import cv2
import os
import sys
from ImageTransformer import ImageTransformer

result=[]
with open('D:/research/drive-download-20220512T115256Z-001/images2.txt','r') as f:
    for line in f.readlines():
        temp = line.split()
        result.append(temp)

result = np.array(result, dtype=object)
#%%
data = []
for i in range(78):
    data.append(result[2*i][1:8])
    
data = np.array(data, dtype=float)
#%%
R = np.zeros((78,3,3))

for i in range(len(R)):
    qw = data[i][0]
    qx = data[i][1]
    qy = data[i][2]
    qz = data[i][3]
    R[(i,0,0)] = 1 - 2*qz*qz- 2*qy*qy
    R[(i,0,1)] = -2*qz*qw + 2*qy*qx
    R[(i,0,2)] = 2*qy*qw + 2*qz*qx  
    
    R[(i,1,0)] = 2*qx*qy + 2*qw*qz  
    R[(i,1,1)] = 1 - 2*qz*qz - 2*qx*qx
    R[(i,1,2)] = 2*qz*qy - 2*qx*qw
    
    R[(i,2,0)] = 2*qx*qz - 2*qw*qy
    R[(i,2,1)] = 2*qy*qz + 2*qw*qx
    R[(i,2,2)] = 1- 2*qy*qy - 2*qx*qx
#%%
angles = np.zeros((78,3))
for i in range(len(angles)):
    angles[(i,0)]  = math.atan2(R[(i,2,1)], R[(i,2,2)])
    angles[(i,1)]  = math.atan2(-R[(i,2,0)], np.sqrt(R[(i,2,1)]**2 + R[(i,2,2)]**2))
    angles[(i,2)]  = math.atan2(R[(i,1,0)], R[(i,0,0)])

#%%
imgPath = 'D:/research/img'
imgList = os.listdir(imgPath)
imgs = []
for imgName in imgList:
    pathImg = os.path.join(imgPath, imgName)
    img = cv2.imread(pathImg)
    
    img = cv2.resize(img,(int(img.shape[1]/6), int(img.shape[0]/6)))
    if img is None:
       print("圖片不能讀取：" + imgName)
       sys.exit(-1)
    imgs.append(img)
#%%
n = 5
img = ImageTransformer(imgs[n])

result = img.rotate_along_axis(angles[(n,0)], angles[(n,1)], angles[(n,2)])

cv2.imshow("image", imgs[n])
cv2.imshow("result", result)
output = 'result' + '.jpg'
cv2.imwrite(output, result)
cv2.waitKey(0)
cv2.destroyAllWindows()














