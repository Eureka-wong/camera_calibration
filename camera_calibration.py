import cv2
import numpy as np
import glob
from math import *
import pandas as pd
import os
from PIL import Image

'''
This code is used to calculate the end to camera matrix. It takes input as a folder with images and a .xlsx file with 
position and orientation data of the robot. It performs camera calibration within these data and outputs the camera to
end matrix.
There are 4 optional parameters:
K: Camera intrinsic parameters
chess_board_x_num: Number of chessboard squares in the x direction
chess_board_y_num: Number of chessboard squares in the y direction
chess_board_len:Length of a single chessboard square, mm
'''

K = np.array([[4283.95148301687, -0.687179973528103, 2070.79900757240],
              [0, 4283.71915784510, 1514.87274457919],
              [0, 0, 1]], dtype=np.float64)  # Camera intrinsic parameters
chess_board_x_num = 11  # Number of chessboard squares in the x direction
chess_board_y_num = 8  # Number of chessboard squares in the y direction
chess_board_len = 35  # Length of a single chessboard square, mm

# Function to calculate the rotation matrix from Euler angles
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R

# Function to calculate the transformation matrix from pose
def pose_robot(x, y, z, Tx, Ty, Tz):
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # Column merge
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    # RT1 = np.linalg.inv(RT1)
    return RT1

# Function to get camera extrinsic parameters from chessboard images
def get_RT_from_chessboard(img_path, chess_board_x_num, chess_board_y_num, K, chess_board_len):
    '''
    :param img_path: Path to the image
    :param chess_board_x_num: Number of chessboard squares in the x direction
    :param chess_board_y_num: Number of chessboard squares in the y direction
    :param K: Camera intrinsic parameters
    :param chess_board_len: Length of a single chessboard square, mm
    :return: Camera extrinsic parameters
    '''
    # Read the image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None) # Find chessboard corners
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (chess_board_x_num, chess_board_y_num), corners, ret)
    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Demo', 640, 480)
    cv2.imshow('Demo', img)
    cv2.waitKey(0)
    # Prepare corner points in 2D
    corner_points = np.zeros((2, corners.shape[0]), dtype=np.float64)  # Create a zero matrix with 2 rows and the number of corners columns
    for i in range(corners.shape[0]):
        corner_points[:, i] = corners[i, 0, :]  # Get a matrix where each column represents the coordinates of a corner
    # Prepare object points in 3D
    object_points = np.zeros((3, chess_board_y_num * chess_board_x_num), dtype=np.float64)
    flag = 0
    for i in range(chess_board_y_num):
        for j in range(chess_board_x_num):
            object_points[:2, flag] = np.array([(11 - j - 1) * chess_board_len, (8 - i - 1) * chess_board_len])
            flag += 1
    # Solve the PnP problem
    retval, rvec, tvec = cv2.solvePnP(object_points.T, corner_points.T, K, distCoeffs=None)   
    # Convert rotation vector to rotation matrix and create the RT matrix
    RT = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    return RT

# Folder where chessboard images are stored
folder = r"C:\Users\Aliya\Desktop\github\calib"
# List all the files in the folder and count the number of files:
files = os.listdir(folder)
file_num = len(files)
RT_all = np.zeros((4, 4, file_num))

'''
This part has a strange characteristic; some chessboard points can be detected, some cannot.
You can judge by the runtime of the function get_RT_from_chessboard.
'''
# Store the images where chessboard corners can be detected
good_picture = [1, 2, 3] 
# file_num = len(good_picture)

# Calculate the transformation matrix from the board to the camera
R_all_chess_to_cam_1 = []
T_all_chess_to_cam_1 = []
for i in good_picture:
    # print(i)
    # image_path = os.path.join(folder, f"picture{i}.bmp")
    dir = os.path.split(os.path.realpath('picture' + str(i)))[0]
    file_name = os.path.split(os.path.realpath('picture' + str(i)))[1]
    target = dir + os.path.sep + file_name.split('.')[0] + '.bmp'
    im = Image.open(folder + '\\' + 'picture' + str(i) + '.jpg')
    im.save(target, "bmp")
    # image_path = folder + '/' + 'picture' + str(i) + '.bmp'
    # print(image_path)
    RT = get_RT_from_chessboard(target, chess_board_x_num, chess_board_y_num, K, chess_board_len)

    # RT = np.linalg.inv(RT)

    R_all_chess_to_cam_1.append(RT[:3, :3])
    T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3, 1)))

# Calculate the transformation matrix from the end-effector to the base
file_address = r"C:\Users\Aliya\Desktop\github\calib\position.xlsx" # Read the six poses of the robot from the record file
sheet_1 = pd.read_excel(file_address)
R_all_end_to_base_1 = []
T_all_end_to_base_1 = []
for i in good_picture:
    # print(sheet_1.iloc[i-1]['ax'], sheet_1.iloc[i-1]['ay'], sheet_1.iloc[i-1]['az'], sheet_1.iloc[i-1]['dx'],
    #       sheet_1.iloc[i-1]['dy'], sheet_1.iloc[i-1]['dz'])
    RT = pose_robot(sheet_1.iloc[i - 1]['ax'], sheet_1.iloc[i - 1]['ay'], sheet_1.iloc[i - 1]['az'], sheet_1.iloc[i - 1]['dx'],
                    sheet_1.iloc[i - 1]['dy'], sheet_1.iloc[i - 1]['dz'])
    # RT = np.column_stack(((cv2.Rodrigues(np.array([[sheet_1.iloc[i-1]['ax']], [sheet_1.iloc[i-1]['ay'], [sheet_1.iloc[i-1]['az']]])))[0],
    #                       np.array([[sheet_1.iloc[i-1]['dx']],
    #                                 [sheet_1.iloc[i-1]['dy']], [sheet_1.iloc[i-1]['dz']]])))
    # RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    # RT = np.linalg.inv(RT)

    R_all_end_to_base_1.append(RT[:3, :3])
    T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))

# Calculate the transformation matrix from the camera to the end:
R, T = cv2.calibrateHandEye(R_all_end_to_base_1, T_all_end_to_base_1, R_all_chess_to_cam_1, T_all_chess_to_cam_1)  # Hand-eye calibration
RT = np.column_stack((R, T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))  # Transformation matrix from camera to end-effector
print('The transformation matrix from the camera to the end-effector is:')
print(RT)

# Result verification, in principle, the difference is small each time
