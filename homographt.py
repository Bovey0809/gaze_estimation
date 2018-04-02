import cv2
from matplotlib import pyplot as plt
import skimage
from scipy.io import loadmat
import numpy as np
import pandas as pd
import glob

def normalize(poinrts):
    for row in points:
        row /= points[-1]
    return points

def make_homog(points):
    return np.vstack((points, ones((1, points.shape[1]))))

def annotationDr(person):
    'return annotation file for specific person.'
    original_dr = '/nfshome/xueqin/deeplearning/gaze_estimation/data/all_writte/MPIIFaceGaze/'
    return original_dr + 'p{:02}/p{:02}.txt'.format(person, person)

def annotation_file_reader(file_address):
    '''
    Read annotation file and return right format as indicted in the paper.
    '''
    return pd.read_csv(file_address, delim_whitespace=True, names=[i for i in range(1, 29)], index_col=False)

def get_Calibration(person):
    '''
    Return camera.mat, monitorPose.mat, screenSize.mat files address.
    
    Example
    -------
    calibration_files[0] is the camera.mat file address.
    ''' 
    MPIIFaceGaze_address = '/nfshome/xueqin/deeplearning/gaze_estimation/data/all_writte/MPIIFaceGaze/'
    calibration_files = glob.glob(MPIIFaceGaze_address+'p{:02}/Calibration/*.mat'.format(person))
    # print(calibration_files)
    return calibration_files

# Read Camera matrix mat file.
def read_camara_mx(file_path):
    '''
    return mtx, dist from camera mx
    
    Parameter
    ---------
    get_calibration(person)[0]
    '''
    mx = loadmat(file_path)
    cameraMatrix = mx['cameraMatrix']
    distCoeffs = mx['distCoeffs']
    retval = mx['retval']
    rvecs = mx['rvecs'] # rotation vector
    tvecs = mx['tvecs'] # translation vector
    return cameraMatrix, distCoeffs, rvecs, tvecs

# gaze calculator
def gaze_cal(annotation_file):
    '''
    return gaze direction of three dimension vector
    Parameter
    ---------
    annotation_file: DataFrame annotation file.
    Return
    ------
    Numpy array of shape (m, x, y, z)
    '''
    return annotation_file.iloc[:, 24:27].values - annotation_file.iloc[:, 22:25].values

def read_image(file_address):
    '''
    Return RGB array format.
    '''
    return cv2.cvtColor(cv2.imread(file_address), cv2.COLOR_BGR2RGB)


def parameter_extracter(annotation_df, person):
    root_directory = '/nfshome/xueqin/deeplearning/gaze_estimation/data/all_writte/MPIIFaceGaze/p{:02}/'.format(person)
    file_address = root_directory + annotation_df[1]
    gaze_location = annotation_df.iloc[:, 1:3].values
    landmarks = annotation_df.iloc[:, 3:15].values
    estimated_head_pose = annotation_df.iloc[:, 15:21].values
    face_center = annotation_df.iloc[:, 21:24].values
    gaze_target = annotation_df.iloc[:, 24:27].values
    return file_address, gaze_location, landmarks, estimated_head_pose, face_center, gaze_target

# draw projection function
def draw_gaze(image, start, face_center, gaze_target, camera_matrix, diss, rotation, translation):
    gaze_direction = gaze_target - face_center
    gaze_direction = gaze_direction.reshape(1, 3)
    end_point, _ = cv2.projectPoints(gaze_direction, rotation, translation, camera_matrix, diss)
    cv2.line(image, tuple(start), tuple(map(int, (end_point[0][0]))), (255, 0, 0), 2)
    return image

def draw_image(person, photo_number, vector_number):
    '''
    Return image with gaze line drawed on it.
    
    Parameters
    ----------
    person: int, person that you want to analyse.
    photo_number: int, which photo you wanna use to draw pictures.
    vector_number: index of (rotation, translation) vectors to use as camera matrix
    
    Returns
    -------
    numpy array
    
    Example
    -------
    draw(1, 1, 8) --> numpy array with shape (1280, 720)
    '''
    # camera matrix, distortion vector, rotation vector, translation vector.
    t_cameraMatrix, t_dist, t_rotation, t_translation = read_camara_mx(get_Calibration(person)[0])
    annotation_dataframe = annotation_file_reader(annotationDr(person))
    parameters = parameter_extracter(annotation_dataframe,person)
    im = plt.imread(parameters[0][photo_number])
    image_points = parameters[2][photo_number].reshape((6,2))
    landmarks = parameters[2][photo_number]
    face_center = parameters[4][photo_number]
    gaze_target = parameters[5][photo_number]
    test_image = draw_gaze(im, landmarks.reshape(-1, 2)[0], 
                           face_center, gaze_target, 
                           t_cameraMatrix, t_dist, t_rotation[vector_number], t_translation[vector_number])
    return test_image

