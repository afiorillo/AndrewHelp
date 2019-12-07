1. Install OpenCV.  It was tested with opencv-3.4.7-vc14_vc15.exe
2. If using Visual-Studio-2015 use the vc14 of OpenCV.  If using Visual-Studio-2017 use the vc15 of OpenCV.
3. Create a system variable OPENCV_INC_DIR = (OPENCV DIRECTORY)\build\include
4. Create a system variable OPENCV_LIB_DIR = (OPENCV DIRECTORY)\build\x64\(vc14 or vc15)\lib
5. Add (OPENCV DIRECTORY)\build\x64\(vc14 or vc15)\bin to the system path

Now you are ready to use opencv.

I am using CUDA 10.2.  If you have installed another version you have to change configuration of the dll project.

Now you are ready to build the application.

Usage: Fish2PerspTest [options] fisheye_image|fisheye_video|fisheye_camera_number
Options
   -w n        perspective image width
   -h n        perspective image height
   -t n        field of view of perspective (degrees)(*)
   -s n        field of view of fisheye (degrees)(*)
   -c x y      center of the fisheye image(*)
   -r n        fisheye radius (horizontal)(*)
   -ry n       fisheye radius (vertical) for anamophic lens, default is circular fisheye(*)
   -x n        tilt angle (degrees)(*)
   -y n        roll angle (degrees)(*)
   -z n        pan angle (degrees)(*)
   -a n        antialiasing level(*)
   -p n n n n  4th order lens correction(*)
   
   -n 			NUMBER OF OUTPUTS (NEW PARAMETER)
   -corr 		(USE/NOT USE) LENS CORRECTION (NEW PARAMETER)(*)
   -Video		SPECIFY IF THE INPUT IS A VIDEO FILE OR CAMERA(*)
   
   (*) Have to enter one value per output.
   
Example:
Fish2PerspTest -n 2 -w 960 -h 960 -t 100 120 -s 180 170 -c 960 960 960 960 -r 960 960 -ry 960 960 -x 0 0 -y 0 0 -z 0 0 -a 2 2 -corr 0 0 -p 0 0 0 0 0 0 0 0 C:\\MOrozco\\FisheyeProject\\bin\\x64\\Debug\\38.bmp

Two outputs.  Output size 960 x 960.  field of view of perspective of first output = 100, for second output = 120.  center of fisheye image (x y)(x y) for first and second output respectively.

-------------------------------------------------------

Fish2PerspTest -Video -n 2 -w 960 -h 960 -t 100 120 -s 180 170 -c 960 960 960 960 -r 960 960 -ry 960 960 -x 0 0 -y 0 0 -z 0 0 -a 2 2 -corr 0 0 -p 0 0 0 0 0 0 0 0 C:\\MOrozco\\output_cam38.mp4

Two outputs.  Video file.
-------------------------------------------------------

Fish2PerspTest -Video -n 2 -w 100 -h 80 -t 100 120 -s 180 170 -c 100 80 100 80 -r 100 80 -ry 100 80 -x 0 0 -y 0 0 -z 0 0 -a 2 2 -corr 0 0 -p 0 0 0 0 0 0 0 0 0

Two outputs.  Camera labeled with 0.