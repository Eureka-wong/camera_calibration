# Camera calibration
This program is used to find the transformation matrix from camera to robot base.

## Background and purpose
We use stationary calibration board and moving robot to take at least two pictures. Input the pictures and robot pose, and ransformation matrix from camera to robot basec can be calculated by our program.

## Installation
 set up the development environment to match the required dependencies and versions for the project. Below are the necessary packages and their version:

 ### packgage version:

       Name                    Version                   Build  Channel
    ca-certificates           2024.3.11            haa95532_0
    certifi                   2024.6.2                 pypi_0    pypi  
    charset-normalizer        3.3.2                    pypi_0    pypi
    et-xmlfile                1.1.0                    pypi_0    pypi
    idna                      3.7                      pypi_0    pypi
    numpy                     1.26.4                   pypi_0    pypi
    opencv-python             4.10.0.82                pypi_0    pypi
    openpyxl                  3.1.4                    pypi_0    pypi
    openssl                   3.0.13               h2bbff1b_2
    panda                     0.3.1                    pypi_0    pypi
    pandas                    2.2.2                    pypi_0    pypi
    pillow                    10.3.0                   pypi_0    pypi
    pip                       24.0             py39haa95532_0
    pymssql                   2.3.0                    pypi_0    pypi
    python                    3.9.19               h1aa4202_1
    python-dateutil           2.9.0.post0              pypi_0    pypi
    pytz                      2024.1                   pypi_0    pypi
    requests                  2.32.3                   pypi_0    pypi
    setuptools                69.5.1           py39haa95532_0
    six                       1.16.0                   pypi_0    pypi
    sqlite                    3.45.3               h2bbff1b_0
    tzdata                    2024.1                   pypi_0    pypi
    urllib3                   2.2.1                    pypi_0    pypi
    vc                        14.2                 h2eaa2aa_1
    vs2015_runtime            14.29.30133          h43f2093_3
    wheel                     0.43.0           py39haa95532_0


### Setup Instructions:
1.Python Installation: 
Install Python version 3.9.19 using your preferred method.
2.Package Installation:
Open a terminal or command prompt and run the following commands to install each required package with its specified version:

    pip install ca-certificates==2024.3.11
    pip install certifi==2024.6.2
    pip install charset-normalizer==3.3.2
    pip install et-xmlfile==1.1.0
    pip install idna==3.7
    pip install numpy==1.26.4
    pip install opencv-python==4.10.0.82
    pip install openpyxl==3.1.4
    pip install openssl==3.0.13
    pip install panda==0.3.1
    pip install pandas==2.2.2
    pip install pillow==10.3.0
    pip install pymssql==2.3.0
    pip install python-dateutil==2.9.0.post0
    pip install pytz==2024.1
    pip install requests==2.32.3
    pip install setuptools==69.5.1
    pip install six==1.16.0
    pip install sqlite==3.45.3
    pip install tzdata==2024.1
    pip install urllib3==2.2.1
    pip install vc==14.2
    pip install vs2015_runtime==14.29.30133
    pip install wheel==0.43.0


3.Verification:

After installation, verify the installation by running pip list to ensure all packages are installed correctly:
'''
pip list
'''
You should see a list of installed packages matching the versions specified above.

## Usage
### read files
#### folder = r"The path to store photos taken by the robot."
eg. folder = r"C:\Users\Aliya\Desktop\calib"
#### filez_address = r"The path to store the robot's pose."
eg.file_address = r"C:\Users\Aliya\Desktop\calib\position.xlsx"

### functions:
#### myRPY2R_robot(x, y, z):to calculate the rotation matrix based on Euler angle
#### pose_robot(x, y, z, Tx, Ty, Tz)：Used to calculate the transformation matrix based on pose.
#### get_RT_from_chessboard(img_path,chess_board_x_num,chess_board_y_num,K,chess_board_len):To obtain the camera extrinsics from a chessboard image.

 
