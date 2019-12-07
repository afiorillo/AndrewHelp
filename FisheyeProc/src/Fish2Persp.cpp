// #include "stdafx.h"
#include "Fish2Persp.h"

using namespace MathUtils;

namespace FisheyeProcessingCPU
{
	Pixel4::Pixel4()
	{
		r = g = b = a = 0;
	} // Pixel4

	// -------------------------------------------------------------------------------

	LensCorrection::LensCorrection()
	{
		isEnabled = false;
		a1 = a2 = a3 = a4 = 0.0;
	} // LensCorrection

	// -------------------------------------------------------------------------------

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
	)
	{
		Fish2PerspParameters out;

		out.perspWidth = perspWidth / 2 * 2;	// Truncate
		out.perspHeight = perspHeight / 2 * 2;	// Truncate

		out.fishFoV = fishFoV < MAX_FISH_FOV ? fishFoV : MAX_FISH_FOV;
		out.fishFoV /= 2.0;
		out.fishFoV *= DTOR;

		out.perspFoV = perspFoV < MAX_PERSP_FOV ? perspFoV : MAX_PERSP_FOV;
		out.perspFoV *= DTOR;

		out.antiAliasing = antiAliasing > MIN_ANTIALIASING ? antiAliasing : MIN_ANTIALIASING;

		out.fishRadius = fishRadius;
		out.fishRadiusY = fishRadiusY;

		Transform t;
		if (tiltAngle != 0)
		{
			t.axis = XTILT;
			t.value = DTOR * tiltAngle;
			out.trans.push_back(t);
		}
		if (rollAngle != 0)
		{
			t.axis = YROLL;
			t.value = DTOR * rollAngle;
			out.trans.push_back(t);
		}
		if (panAngle != 0)
		{
			t.axis = ZPAN;
			t.value = DTOR * panAngle;
			out.trans.push_back(t);
		}

		for (int i = 0; i < out.trans.size(); ++i)
		{
			out.trans[i].sValue = sin(out.trans[i].value);
			out.trans[i].cValue = cos(out.trans[i].value);
		}

		out.fishAspect = out.fishRadiusY / static_cast<double>(out.fishRadius);	

		return out;
	} // CreateInputParameters

	// -------------------------------------------------------------------------------
	// Sum 4 vectors each with a scaling factor
	// Only used 4 times for the first pixel
	// -------------------------------------------------------------------------------

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
	)
	{
		XYZ sum;

		sum.x = d1 * p1.x + d2 * p2.x + d3 * p3.x + d4 * p4.x;
		sum.y = d1 * p1.y + d2 * p2.y + d3 * p3.y + d4 * p4.y;
		sum.z = d1 * p1.z + d2 * p2.z + d3 * p3.z + d4 * p4.z;

		return sum;
	}

	// -------------------------------------------------------------------------------

	CameraRayParams PrecomputeCameraRay(Fish2PerspParameters& params)
	{
		CameraRayParams paramsCR;

		const XYZ vp = { 0, 0, 0 };
		const XYZ vd = { 0, 1, 0 };
		const XYZ vu = { 0, 0, 1 };		// Camera view position, direction, and up 
		const XYZ right = { 1, 0, 0 };

		double dh = tan(params.perspFoV / 2);
		double dv = params.perspHeight * dh / params.perspWidth;
		paramsCR.p1 = VectorSum(1.0, vp, 1.0, vd, -dh, right, dv, vu);
		paramsCR.p2 = VectorSum(1.0, vp, 1.0, vd, -dh, right, -dv, vu);
		paramsCR.p3 = VectorSum(1.0, vp, 1.0, vd, dh, right, -dv, vu);
		paramsCR.p4 = VectorSum(1.0, vp, 1.0, vd, dh, right, dv, vu);
		paramsCR.deltaH.x = paramsCR.p4.x - paramsCR.p1.x;
		paramsCR.deltaH.y = paramsCR.p4.y - paramsCR.p1.y;
		paramsCR.deltaH.z = paramsCR.p4.z - paramsCR.p1.z;
		paramsCR.deltaV.x = paramsCR.p2.x - paramsCR.p1.x;
		paramsCR.deltaV.y = paramsCR.p2.y - paramsCR.p1.y;
		paramsCR.deltaV.z = paramsCR.p2.z - paramsCR.p1.z;

		paramsCR.inverseW = 1.0 / params.perspWidth;
		paramsCR.inverseH = 1.0 / params.perspHeight;

		return paramsCR;
	} // PrecomputeCameraRay

	// -------------------------------------------------------------------------------

	void CameraRay
	(
		const double x, 
		const double y,
		const Fish2PerspParameters& params,
		CameraRayParams& paramsCR,
		/* Output */
		XYZ& p
	)
	{
		double h = x * paramsCR.inverseW;
		double v = (params.perspHeight - 1 - y) * paramsCR.inverseH;
		p.x = paramsCR.p1.x + h * paramsCR.deltaH.x + v * paramsCR.deltaV.x;
		p.y = paramsCR.p1.y + h * paramsCR.deltaH.y + v * paramsCR.deltaV.y;
		p.z = paramsCR.p1.z + h * paramsCR.deltaH.z + v * paramsCR.deltaV.z;
	}

	// -------------------------------------------------------------------------------

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
		const LensCorrection lensCorr,
		/* Output */
		unsigned char * perspImg
	)
	{
		Fish2PerspParameters params = CreateInputParameters(perspWidth, perspHeight,
			perspFoV, fishFoV, fishRadius, fishRadiusY,
			tiltAngle, rollAngle, panAngle, antiAliasing);

		// Step through each pixel in the output perspective image 		
		CameraRayParams paramsCR = PrecomputeCameraRay(params);
		for (int j = 0; j < params.perspHeight; ++j)
		{
			for (int i = 0; i < params.perspWidth; ++i)
			{
				std::vector<int> rgbsum(3, 0);

				// Antialiasing loops, sub-pixel sampling
				for (int ai = 0; ai < params.antiAliasing; ++ai)
				{
					double x = i + ai / static_cast<double>(params.antiAliasing);
					for (int aj = 0; aj < params.antiAliasing; ++aj)
					{
						double y = j + aj / static_cast<double>(params.antiAliasing);

						// Calculate vector to each pixel in the perspective image 
						XYZ p;
						CameraRay(x, y, params, paramsCR, /* Output */ p);

						// Apply rotations in order
						for (int k = 0; k < params.trans.size(); ++k)
						{
							XYZ q;
							switch (params.trans[k].axis)
							{
							case XTILT:
								q.x = p.x;
								q.y = p.y * params.trans[k].cValue + p.z * params.trans[k].sValue;
								q.z = -p.y * params.trans[k].sValue + p.z * params.trans[k].cValue;
								break;
							case YROLL:
								q.x = p.x * params.trans[k].cValue + p.z * params.trans[k].sValue;
								q.y = p.y;
								q.z = -p.x * params.trans[k].sValue + p.z * params.trans[k].cValue;
								break;
							case ZPAN:
								q.x = p.x * params.trans[k].cValue + p.y * params.trans[k].sValue;
								q.y = -p.x * params.trans[k].sValue + p.y * params.trans[k].cValue;
								q.z = p.z;
								break;
							}
							p = q;
						} // for k

						// Convert to fisheye image coordinates 
						double theta = std::atan2(p.z, p.x);
						double phi = std::atan2(std::sqrt(p.x * p.x + p.z * p.z), p.y);
						double r;
						if (!lensCorr.isEnabled)
						{
							r = phi / params.fishFoV;
						}
						else
						{
							r = phi * (lensCorr.a1 + phi * (lensCorr.a2 + 
										phi * (lensCorr.a3 + phi * lensCorr.a4)));
							if (phi > params.fishFoV)
							{
								r *= 10;
							}
						}

						// Convert to fisheye texture coordinates 
						int u = static_cast<int>(
							std::round(fishCenterX + r * params.fishRadius * cos(theta)));
						if (u < 0 || u >= imgWidth) 
						{ 
							continue; 
						};

						int v = static_cast<int>(std::round(
							fishCenterY + r * params.fishRadius * params.fishAspect * sin(theta)));
						if (v < 0 || v >= imgHeight) 
						{ 
							continue; 
						};

						// Add up antialias contribution
						for (int ch = 0; ch < 3; ++ch)
						{
							rgbsum[ch] += img[ch * imgWidth * imgHeight + v * imgWidth + u];
						}
					} // for aj
				} // for ai

				// Set the pixel 
				for (int ch = 0; ch < 3; ++ch)
				{
					perspImg[ch * perspWidth * perspHeight + j * perspWidth + i] =
						rgbsum[ch] / (params.antiAliasing * params.antiAliasing);
				} // for ch
			} // for i
		} // for j
	} // Fish2Persp
} // FisheyeProcessingCPU