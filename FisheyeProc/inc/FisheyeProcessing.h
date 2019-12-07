#pragma once

#define DLL_EXPORT extern "C" __declspec(dllexport) 

DLL_EXPORT void Fish2Persp_8U_64F_CPU
(
	const unsigned char * img,
	const int imgWidth,
	const int imgHeight,
	const int imgChannels,
	const int perspWidth,
	const int perspHeight,
	const double perspFoV,
	const double fishFoV,
	const int fishCenterX,
	const int fishCenterY,
	const int fishRadius,
	const int fishRadiusY,
	const double tiltAngle,
	const double rollAngle,
	const double panAngle,
	const int antiAliasing,
	const bool lensCorrectionEnabled,
	const double * LensCorrection,
	/* Output */
	unsigned char * perspImg
);

DLL_EXPORT int Fish2Persp_8U_64F_GPU
(
	const int nOutputs,
	const unsigned char * img,
	const int imgWidth,
	const int imgHeight,
	const int imgChannels,
	const int perspWidth,
	const int perspHeight,
	const double * perspFoV,
	const double * fishFoV,
	const int * fishCenterX,
	const int * fishCenterY,
	const int * fishRadius,
	const int * fishRadiusY,
	const double * tiltAngle,
	const double * rollAngle,
	const double * panAngle,
	const int * antiAliasing,
	const bool * lensCorrectionEnabled,
	const double * lensCorrection,
	/* Output */
	unsigned char * perspImg
);

DLL_EXPORT int Fish2Persp_8U_32F_GPU
(
	const int nOutputs,
	const unsigned char * img,
	const int imgWidth,
	const int imgHeight,
	const int imgChannels,
	const int perspWidth,
	const int perspHeight,
	const float * perspFoV,
	const float * fishFoV,
	const int * fishCenterX,
	const int * fishCenterY,
	const int * fishRadius,
	const int * fishRadiusY,
	const float * tiltAngle,
	const float * rollAngle,
	const float * panAngle,
	const int * antiAliasing,
	const bool * lensCorrectionEnabled,
	const float * lensCorrection,
	/* Output */
	unsigned char * perspImg
);