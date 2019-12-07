import cv2
import numpy as np
import FisheyeProc

im = cv2.imread("../Test/img/38.bmp")

imChannels = 3
perspWidth = 960
perspHeight = 960
perspFoV = 100
fishFoV = 180
fishCenterX = 960
fishCenterY = 960
fishRadius = 960
fishRadiusY = 960
tiltAngle = 0
rollAngle = 0
panAngle = 0
antiAliasing = 2
lensCorrectionEnabled = false
lenscorrection = np.array([0, 0, 0, 0])
perspImg = np.array(shape = (im.width(), im.height()), dtype = np.float64)

FisheyeProc.fish2persp_CPU(
im.data, 
im.width(), 
im.height(),
imgChannels,
perspWidth,
perspHeight,
perspFoV,
fishFoV,
fishCenterX,
fishCenterY,
fishRadius,
fishRadiusY,
tiltAngle,
rollAngle,
panAngle,
antiAliasing,
lensCorrectionEnabled,
lenscorrection,
perspImg)

cv2.waitKey(5000)
 
# Closes all the frames
cv2.destroyAllWindows()
