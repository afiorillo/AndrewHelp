#pragma once

// #include "CImg.h"
#include <vector>
#include "MathUtils.h"

namespace FisheyeProcessingCPU
{
	const double DTOR = MathUtils::PI / 180.0;
	const double MAX_FISH_FOV = 360.0;
	const double MAX_PERSP_FOV = 170.0;
	const int MIN_ANTIALIASING = 1;

	typedef struct 
	{
		double x, y, z;
	} XYZ;

	enum TransformAxis { XTILT, YROLL, ZPAN };

	struct Pixel4
	{
		unsigned char r;
		unsigned char g;
		unsigned char b;
		unsigned char a;

		Pixel4();
	};

	struct Transform
	{
		TransformAxis axis;
		double value;
		double cValue;
		double sValue;
	};

	struct LensCorrection
	{
		bool isEnabled;
		double a1;
		double a2;
		double a3;
		double a4;

		LensCorrection();
	};

	struct Fish2PerspParameters
	{
		int perspWidth;				// Perspective width and height
		int perspHeight;
		double fishFoV;				// Field of view
		double perspFoV;			// Horizontal fov of perspective camera
		int fishRadius;				// Radius (horizontal) of the fisheye circle
		int fishRadiusY;			// Vertical radius, deals with anamorphic lenses
		double fishAspect;			// fishradiusy / fishradius
		int antiAliasing;			// Supersampling antialiasing		
		std::vector<Transform> trans;
	};

	XYZ VectorSum
	(
		const double d1,
		const XYZ& p1,
		const double d2,
		const XYZ& p2,
		const double d3,
		const XYZ& p3,
		const double d4,
		const XYZ& p4
	);

	struct CameraRayParams
	{
		XYZ p1;
		XYZ p2;
		XYZ p3;
		XYZ p4;
		XYZ deltaH;
		XYZ deltaV;
		double inverseW;
		double inverseH;
	};

	CameraRayParams PrecomputeCameraRay(Fish2PerspParameters& params);

	Fish2PerspParameters CreateInputParameters
	(
		const int perspWidth,
		const int perspHeight,
		const double perspFoV,
		const double fishFoV,
		const int fishRadius,
		const int fishRadiusY,
		const double tiltAngle,
		const double rollAngle,
		const double panAngle,
		const int antiAliasing
	);

	void Fish2Persp
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
		const LensCorrection p,
		/* Output */
		unsigned char * perspImg
	);
}