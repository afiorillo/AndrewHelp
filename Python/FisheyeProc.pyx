from libcpp cimport bool

cimport cFisheyeProc

import numpy as np  	# Import the Python Numpy
cimport numpy as np  	# Import the C Numpy
from numpy cimport ndarray
from cython cimport floating

from cpython.pycapsule cimport *

# Python wrapper functions.

def fish2persp_CPU(np.ndarray[np.uint8_t] img,
					int imgWidth,
					int imgHeight,
					int imgChannels,
					int perspWidth,
					int perspHeight,
					double perspFoV,
					double fishFoV,
					int fishCenterX,
					int fishCenterY,
					int fishRadius,
					int fishRadiusY,
					double tiltAngle,
					double rollAngle,
					double panAngle,
					int antiAliasing,
					bool lensCorrectionEnabled,
					np.ndarray[np.float64_t] lenscorrection,
					np.ndarray[np.uint8_t] perspImg):
			
    """Calculates the perspective view from a Fisheye image."""
		
    # Call the imported DLL functions on the parameters.
    # Notice that we are passing a pointer to the first element in each array
		
    cFisheyeProc.Fish2Persp_8U_64F_CPU(&img[0], imgWidth, imgHeight, imgChannels,
	perspWidth, perspHeight, perspFoV, fishFoV, fishCenterX, fishCenterY,
	fishRadius, fishRadiusY, tiltAngle, rollAngle, panAngle, antiAliasing,
	lensCorrectionEnabled, &lenscorrection[0], &perspImg[0])
    