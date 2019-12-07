

#include "FisheyeProcessing.h"
#include "Fish2Persp.h"
#include "Fish2PerspGPU.cuh"

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
	const double * lensCorrection,
	/* Output */
	unsigned char * perspImg
)
{
	FisheyeProcessingCPU::LensCorrection lensCorr;
	lensCorr.isEnabled = lensCorrectionEnabled;
	if (lensCorrectionEnabled)
	{
		lensCorr.a1 = lensCorrection[0];
		lensCorr.a2 = lensCorrection[1];
		lensCorr.a3 = lensCorrection[2];
		lensCorr.a4 = lensCorrection[3];
	}

	FisheyeProcessingCPU::Fish2Persp(img, imgWidth, imgHeight, imgChannels, perspWidth, perspHeight,
		perspFoV, fishFoV, fishCenterX,	fishCenterY, fishRadius, fishRadiusY, 
		tiltAngle, rollAngle, panAngle,	antiAliasing, lensCorr, /* Output */ perspImg);
}

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
)
{
	return FisheyeProcessingGPU::Fish2Persp(nOutputs, img, imgWidth, imgHeight, 
		imgChannels, perspWidth, perspHeight, perspFoV, fishFoV, fishCenterX, 
		fishCenterY, fishRadius, fishRadiusY, tiltAngle, rollAngle, panAngle, 
		antiAliasing, lensCorrectionEnabled, lensCorrection, /* Output */ perspImg);
}

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
)
{
	return FisheyeProcessingGPU::Fish2Persp(nOutputs, img, imgWidth, imgHeight,
		imgChannels, perspWidth, perspHeight, perspFoV, fishFoV, fishCenterX,
		fishCenterY, fishRadius, fishRadiusY, tiltAngle, rollAngle, panAngle,
		antiAliasing, lensCorrectionEnabled, lensCorrection, /* Output */ perspImg);
}