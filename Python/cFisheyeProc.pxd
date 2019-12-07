from libcpp cimport bool

cdef extern from "../FisheyeProc/inc/FisheyeProcessing.h":
	void Fish2Persp_8U_64F_CPU(const unsigned char * img,
	const int imgWidth,	const int imgHeight,
	const int imgChannels, const int perspWidth,
	const int perspHeight, const double perspFoV,
	const double fishFoV, const int fishCenterX,
	const int fishCenterY, const int fishRadius,
	const int fishRadiusY, const double tiltAngle,
	const double rollAngle, const double panAngle,
	const int antiAliasing, const bool lensCorrectionEnabled,
	const double * LensCorrection, unsigned char * perspImg
);
