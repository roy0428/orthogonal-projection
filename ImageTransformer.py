import numpy as np
import cv2

class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, img):
        #self.image_path = image_path
        self.image = img
 
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.num_channels = 3

    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        
        # Get radius of rotation along 3 axes
        #rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)
        dz = self.focal
        
        # Get projection matrix
        mat = self.get_M(theta, phi, gamma, dx, dy, dz)
        #print(mat)

        return cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))


    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):
        
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])
        
        RY = np.array([ [np.cos(phi), 0, np.sin(phi)],
                        [0, 1, 0],
                        [-np.sin(phi), 0, np.cos(phi)]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [dx],
                        [dy],
                        [dz]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2],
                        [0, f, h/2],
                        [0, 0, 1]])

        # Final transformation matrix
        A3 = np.dot(A2, np.concatenate((R, T), axis=1))
        
        return np.dot(A3, A1)