from tkinter import Image
import numpy
import cv2
from pupil_apriltags import Detector
from ImageTransformer import *
import os, glob
import os.path as osp

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

image = cv2.imread('./raw/G0300633.JPG') #based pic.(apriltags)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
tags = at_detector.detect(gray_image, estimate_tag_pose=True, camera_params=[2466.3496376258695, 2592, 1944, 0.00054674077536223328], tag_size=0.16)
#print(tags)

corners = [o.corners for o in tags]
corners = corners[0]

def find_contours(output_img):
    output_img_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(output_img_gray, 5, 255, cv2.THRESH_BINARY)
    kernel = numpy.ones((5,5), numpy.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        output_img = output_img[y:y+h,x:x+w]

    return output_img

def transform_ortho(img0, corners):
    
    (tl, tr, br, bl) = corners
    #0, 1, 2, 3
    rotated_x = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    rotated_y = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

    points = numpy.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]],
        dtype=numpy.float32).reshape((-1, 1, 2))
    [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(numpy.int32)
    [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(numpy.int32)
    #print(x_max, y_max)

    src = numpy.float32(corners)
    dst = numpy.array([[y_max, x_max], 
        [y_max + rotated_x, x_max],
        [y_max + rotated_x, x_max - rotated_y],
        [y_max, x_max - rotated_y]], dtype = "float32")

    #print(dst)
    M = cv2.getPerspectiveTransform(src, dst)
    output_img = cv2.warpPerspective(img0, M, (img0.shape[1]*2, img0.shape[0]*2))
    output_img = find_contours(output_img)
   
    return output_img, M

result_image, mat = transform_ortho(image, corners)
#print(mat)

''' 
# camera rotational persepective transform
def rotate_axis(image, rot_a, rot_t):
    output_img = ImageTransformer(image)

    output_img_r = output_img.rotate_along_axis(rot_a[0], rot_a[1], 0, rot_t[0], rot_t[1], rot_t[2])
    cv2.imwrite('./xx.jpg', output_img_r)

    return output_img_r
'''

def transform_perspec(mat, image):
    output_img = cv2.warpPerspective(image, mat, (image.shape[1]*2, image.shape[0]*2))
    output_img = find_contours(output_img)
   
    return output_img
    
imgPath = './raw'
outPath = './result'
for filename in os.listdir(imgPath):
    if filename.endswith(".JPG"):
        #print(filename)
        image = cv2.imread('./raw/'+filename)
        ortho_image = transform_perspec(mat, image)
        outfile = '%s/%s.JPG' % (outPath, filename)
        print(outfile)
        cv2.imwrite(outfile,  ortho_image)
