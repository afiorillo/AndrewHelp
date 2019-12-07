
#include <cuda_runtime.h>
#include "Fish2PerspGPU.cuh"
#include "CheckErrors.cuh"

#define CAMERARAY_PARAMS 20
#define XTILT 1
#define YROLL 2
#define ZPAN 3

#define OPEN_CV_ACCESS

namespace FisheyeProcessingGPU
{
	template <class T>
	__device__ void CameraRayDevice
	(
		const T x,
		const T y,
		const int perspHeight,
		const T p1x,
		const T p1y,
		const T p1z,
		const T deltaHx,
		const T deltaHy,
		const T deltaHz,
		const T deltaVx,
		const T deltaVy,
		const T deltaVz,
		const T inverseW,
		const T inverseH,
		/* Output */
		T& px,
		T& py,
		T& pz
	)
	{
		T h = x * inverseW;
		T v = (perspHeight - 1 - y) * inverseH;
		px = p1x + h * deltaHx + v * deltaVx;
		py = p1y + h * deltaHy + v * deltaVy;
		pz = p1z + h * deltaHz + v * deltaVz;
	}

	template<class T>
	__global__ void ComputePerspective
	(
		const unsigned char * img,
		const int imgWidth,
		const int imgHeight,
		const int imgChannels,
		const int perspWidth,
		const int perspHeight,
		const T DTOR,
		const T * fishFoV,
		const int * fishCenterX,
		const int * fishCenterY,
		const int * fishRadius,
		const int * fishRadiusY,
		const T * tiltAngle,
		const T * rollAngle,
		const T * panAngle,
		const int * antiAliasing,
		const bool * corrEnabled,
		const T * lensCorrection,
		const T * cameraRayParams,
		/* Output */
		unsigned char * perspImg
	)
	{
		__shared__ T sharedCameraRayParams[CAMERARAY_PARAMS];
		__shared__ T sharedTransformValues[3];
		// __shared__ T sharedPerspFoV;
		__shared__ T sharedFishFoV;
		__shared__ T sharedA1;
		__shared__ T sharedA2;
		__shared__ T sharedA3;
		__shared__ T sharedA4;
		__shared__ int sharedTransformType[3];
		__shared__ int sharedNTransforms;
		__shared__ int sharedAntialiasing;
		__shared__ int sharedCorrEnabled;
		__shared__ int sharedFishCenterX;
		__shared__ int sharedFishCenterY;
		__shared__ int sharedFishRadius;
		__shared__ int sharedFishRadiusY;

		T& p1x = sharedCameraRayParams[0];
		T& p1y = sharedCameraRayParams[1];
		T& p1z = sharedCameraRayParams[2];
		T& deltaHx = sharedCameraRayParams[12];
		T& deltaHy = sharedCameraRayParams[13];
		T& deltaHz = sharedCameraRayParams[14];
		T& deltaVx = sharedCameraRayParams[15];
		T& deltaVy = sharedCameraRayParams[16];
		T& deltaVz = sharedCameraRayParams[17];
		T& inverseW = sharedCameraRayParams[18];
		T& inverseH = sharedCameraRayParams[19];

		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int ti = ty * blockDim.x + tx;	

		int z = blockIdx.z;

		int i = blockIdx.x * blockDim.x + tx;
		int j = blockIdx.y * blockDim.y + ty;

		for (int k = ti; k < CAMERARAY_PARAMS; k += blockDim.x * blockDim.y)
		{
			sharedCameraRayParams[k] = cameraRayParams[z * CAMERARAY_PARAMS + k];
		}

		__syncthreads();

		if (ti == 0)
		{
			sharedNTransforms = 0;
			T dtor = DTOR;
			if (tiltAngle[z] != 0)
			{
				sharedTransformType[sharedNTransforms] = XTILT;
				sharedTransformValues[sharedNTransforms++] = dtor * tiltAngle[z];
			}
			if (rollAngle[z] != 0)
			{
				sharedTransformType[sharedNTransforms] = YROLL;
				sharedTransformValues[sharedNTransforms++] = dtor * rollAngle[z];
			}
			if (panAngle[z] != 0)
			{
				sharedTransformType[sharedNTransforms] = ZPAN;
				sharedTransformValues[sharedNTransforms++] = dtor * panAngle[z];
			}
			sharedA1 = lensCorrection[z * 4 + 0];
			sharedA2 = lensCorrection[z * 4 + 1];
			sharedA3 = lensCorrection[z * 4 + 2];
			sharedA4 = lensCorrection[z * 4 + 3];

			sharedFishFoV = fishFoV[z] / 2 * dtor;
			// sharedPerspFoV = perspFoV[z];

			sharedAntialiasing = antiAliasing[z];

			sharedCorrEnabled = corrEnabled[z] ? 1 : 0;

			sharedFishCenterX = fishCenterX[z];
			sharedFishCenterY = fishCenterY[z];
			sharedFishRadius = fishRadius[z];
			sharedFishRadiusY = fishRadiusY[z];
		}

		__syncthreads();

		int rgbsum[3] = { 0, 0, 0 };
		if (i < perspWidth && j < perspHeight)
		{			
			// Antialiasing loops, sub-pixel sampling
			for (int ai = 0; ai < sharedAntialiasing; ++ai)
			{
				T x = i + ai / static_cast<T>(sharedAntialiasing);
				for (int aj = 0; aj < sharedAntialiasing; ++aj)
				{
					T y = j + aj / static_cast<T>(sharedAntialiasing);

					// Calculate vector to each pixel in the perspective image 
					T px, py, pz;
					CameraRayDevice<T>(x, y, perspHeight,p1x, p1y, p1z, deltaHx, deltaHy, deltaHz,
						deltaVx, deltaVy, deltaVz, inverseW, inverseH, /* Output */ px, py, pz);

					// Apply rotations in order
					for (int k = 0; k < sharedNTransforms; ++k)
					{
						T qx, qy, qz;
						switch (sharedTransformType[k])
						{
						case XTILT:
							qx = px;
							qy = py * cos(sharedTransformValues[k]) + pz * sin(sharedTransformValues[k]);
							qz = -py * sin(sharedTransformValues[k]) + pz * cos(sharedTransformValues[k]);
							break;
						case YROLL:
							qx = px * cos(sharedTransformValues[k]) + pz * sin(sharedTransformValues[k]);
							qy = py;
							qz = -px * sin(sharedTransformValues[k]) + pz * cos(sharedTransformValues[k]);
							break;
						case ZPAN:
							qx = px * cos(sharedTransformValues[k]) + py * sin(sharedTransformValues[k]);
							qy = -px * sin(sharedTransformValues[k]) + py * cos(sharedTransformValues[k]);
							qz = pz;
							break;
						}
						px = qx;
						py = qy;
						pz = qz;
					} // for k

					// Convert to fisheye image coordinates 
					T theta = atan2(pz, px);
					T phi = atan2(sqrt(px * px + pz * pz), py);
					T r;
					if (sharedCorrEnabled == 0)
					{
						r = phi / sharedFishFoV;
					}
					else
					{
						r = phi * (sharedA1 + phi * (sharedA2 + phi * (sharedA3 + phi * sharedA4)));
						if (phi > sharedFishFoV)
						{
							r *= 10;
						}
					}

					// Convert to fisheye texture coordinates 
					int u = static_cast<int>(
						round(sharedFishCenterX + r * sharedFishRadius * cos(theta)));
					if (u < 0 || u >= imgWidth)
					{
						continue;
					};

					int v = static_cast<int>(round(sharedFishCenterY + 
						r * sharedFishRadius * (sharedFishRadiusY / static_cast<T>(sharedFishRadius)) * sin(theta)));
					if (v < 0 || v >= imgHeight)
					{
						continue;
					};

					// Add up antialias contribution
					for (int ch = 0; ch < 3; ++ch)
					{
#ifdef OPEN_CV_ACCESS
						rgbsum[ch] += img[v * imgWidth * imgChannels + u * imgChannels + ch];
#else
						rgbsum[ch] += img[ch * imgWidth * imgHeight + v * imgWidth + u];
#endif
					}
				} // for aj
			} // for ai
		}

		__syncthreads();

		// Set the pixel 
		if (i < perspWidth && j < perspHeight)
		{
			int zOffset = z * perspWidth * perspHeight * imgChannels;
			for (int ch = 0; ch < 3; ++ch)
			{
#ifdef OPEN_CV_ACCESS
				perspImg[zOffset + j * perspWidth * imgChannels + i * imgChannels + ch] =
					rgbsum[ch] / (sharedAntialiasing * sharedAntialiasing);
#else				
				perspImg[zOffset + ch * perspWidth * perspHeight + j * perspWidth + i] =
					rgbsum[ch] / (sharedAntialiasing * sharedAntialiasing);
#endif
			} // for ch
		}
	} // ComputeMeanKernel

	template<class T>
	std::vector<T> PrecomputeCameraRay
	(
		const int nOutputs,
		const int perspWidth,
		const int perspHeight,
		const T * perspFoV
	)
	{
		std::vector<T> paramsCR;

		const FisheyeProcessingCPU::XYZ vp = { 0, 0, 0 };
		const FisheyeProcessingCPU::XYZ vd = { 0, 1, 0 };
		const FisheyeProcessingCPU::XYZ vu = { 0, 0, 1 };		// Camera view position, direction, and up 
		const FisheyeProcessingCPU::XYZ right = { 1, 0, 0 };

		for (int i = 0; i < nOutputs; ++i)
		{
			T dh = std::tan(perspFoV[i] * FisheyeProcessingCPU::DTOR / 2);
			T dv = perspHeight * dh / perspWidth;
			FisheyeProcessingCPU::XYZ p1 = VectorSum(1.0, vp, 1.0, vd, -dh, right, dv, vu);
			FisheyeProcessingCPU::XYZ p2 = VectorSum(1.0, vp, 1.0, vd, -dh, right, -dv, vu);
			FisheyeProcessingCPU::XYZ p3 = VectorSum(1.0, vp, 1.0, vd, dh, right, -dv, vu);
			FisheyeProcessingCPU::XYZ p4 = VectorSum(1.0, vp, 1.0, vd, dh, right, dv, vu);

			paramsCR.push_back(p1.x);
			paramsCR.push_back(p1.y);
			paramsCR.push_back(p1.z);

			paramsCR.push_back(p2.x);
			paramsCR.push_back(p2.y);
			paramsCR.push_back(p2.z);

			paramsCR.push_back(p3.x);
			paramsCR.push_back(p3.y);
			paramsCR.push_back(p3.z);

			paramsCR.push_back(p4.x);
			paramsCR.push_back(p4.y);
			paramsCR.push_back(p4.z);

			FisheyeProcessingCPU::XYZ deltaH;
			FisheyeProcessingCPU::XYZ deltaV;

			deltaH.x = p4.x - p1.x;
			deltaH.y = p4.y - p1.y;
			deltaH.z = p4.z - p1.z;
			deltaV.x = p2.x - p1.x;
			deltaV.y = p2.y - p1.y;
			deltaV.z = p2.z - p1.z;

			paramsCR.push_back(deltaH.x);
			paramsCR.push_back(deltaH.y);
			paramsCR.push_back(deltaH.z);

			paramsCR.push_back(deltaV.x);
			paramsCR.push_back(deltaV.y);
			paramsCR.push_back(deltaV.z);

			paramsCR.push_back(1.0 / perspWidth);	// InverseW
			paramsCR.push_back(1.0 / perspHeight);	// InverseH
		}

		return paramsCR;
	} // PrecomputeCameraRay

	template<class T>
	int Fish2Persp
	(
		const int nOutputs,
		const unsigned char * img,
		const int imgWidth,
		const int imgHeight,
		const int imgChannels,
		const int perspWidth,
		const int perspHeight,
		const T * perspFoV,
		const T * fishFoV,
		const int * fishCenterX,
		const int * fishCenterY,
		const int * fishRadius,
		const int * fishRadiusY,
		const T * tiltAngle,
		const T * rollAngle,
		const T * panAngle,
		const int * antiAliasing,
		const bool * lensCorrectionEnabled,
		const T * lensCorrection,
		/* Output */
		unsigned char * perspImg
	)
	{
		int imSize = imgWidth * imgHeight * imgChannels * sizeof(unsigned char);
		unsigned char *deviceImg;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceImg), imSize));
		GPU_ERROR_CHECK(cudaMemcpy(deviceImg, img, imSize, cudaMemcpyHostToDevice));

		int perspSize = nOutputs * perspWidth * perspHeight * imgChannels * sizeof(unsigned char);
		unsigned char *devicePersp;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePersp), perspSize));

		// ----------------------------------------------------------------------------

		std::vector<T> paramsCR = PrecomputeCameraRay(nOutputs, perspWidth, perspHeight, perspFoV);

		T * deviceCameraRayParams;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCameraRayParams),
			paramsCR.size() * sizeof(T)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceCameraRayParams, paramsCR.data(),
			paramsCR.size() * sizeof(T), cudaMemcpyHostToDevice));

		// ----------------------------------------------------------------------------

		//T * devicePerspFoV;
		//GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePerspFoV), nOutputs * sizeof(T)));
		//GPU_ERROR_CHECK(cudaMemcpy(devicePerspFoV, perspFoV, nOutputs * sizeof(T), cudaMemcpyHostToDevice));

		T * deviceFishFoV;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceFishFoV), nOutputs * sizeof(T)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceFishFoV, fishFoV, nOutputs * sizeof(T), cudaMemcpyHostToDevice));

		int * deviceFishCenterX;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceFishCenterX), nOutputs * sizeof(int)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceFishCenterX, fishCenterX, nOutputs * sizeof(int), cudaMemcpyHostToDevice));

		int * deviceFishCenterY;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceFishCenterY), nOutputs * sizeof(int)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceFishCenterY, fishCenterY, nOutputs * sizeof(int), cudaMemcpyHostToDevice));

		int * deviceFishRadius;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceFishRadius), nOutputs * sizeof(int)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceFishRadius, fishRadius, nOutputs * sizeof(int), cudaMemcpyHostToDevice));

		int * deviceFishRadiusY;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceFishRadiusY), nOutputs * sizeof(int)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceFishRadiusY, fishRadiusY, nOutputs * sizeof(int), cudaMemcpyHostToDevice));

		T * deviceTiltAngle;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTiltAngle), nOutputs * sizeof(T)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceTiltAngle, tiltAngle, nOutputs * sizeof(T), cudaMemcpyHostToDevice));

		T * deviceRollAngle;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceRollAngle), nOutputs * sizeof(T)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceRollAngle, rollAngle, nOutputs * sizeof(T), cudaMemcpyHostToDevice));

		T * devicePanAngle;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePanAngle), nOutputs * sizeof(T)));
		GPU_ERROR_CHECK(cudaMemcpy(devicePanAngle, panAngle, nOutputs * sizeof(T), cudaMemcpyHostToDevice));

		int * deviceAntialiasing;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceAntialiasing), nOutputs * sizeof(int)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceAntialiasing, antiAliasing, nOutputs * sizeof(int), cudaMemcpyHostToDevice));

		bool * deviceLensCorrectionEnabled;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceLensCorrectionEnabled), nOutputs * sizeof(bool)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceLensCorrectionEnabled, lensCorrectionEnabled, nOutputs * sizeof(bool), cudaMemcpyHostToDevice));

		T * deviceLensCorrection;
		GPU_ERROR_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceLensCorrection), 4 * nOutputs * sizeof(T)));
		GPU_ERROR_CHECK(cudaMemcpy(deviceLensCorrection, lensCorrection, 4 * nOutputs * sizeof(T), cudaMemcpyHostToDevice));

		// ----------------------------------------------------------------------------

		const dim3 threadsPerBlock(16, 16);
		const dim3 blockSize(1 + (perspWidth - 1) / threadsPerBlock.x, 1 + (perspHeight - 1) / threadsPerBlock.y, nOutputs);

		// Kernel Function
		ComputePerspective<T> <<< blockSize, threadsPerBlock >>> (
			deviceImg, imgWidth, imgHeight, imgChannels, perspWidth, perspHeight, 
			static_cast<T>(FisheyeProcessingCPU::DTOR),	deviceFishFoV, 
			deviceFishCenterX, deviceFishCenterY, deviceFishRadius, deviceFishRadiusY, 
			deviceTiltAngle, deviceRollAngle, devicePanAngle, deviceAntialiasing, 
			deviceLensCorrectionEnabled, deviceLensCorrection, deviceCameraRayParams,
			/* Output */ devicePersp);
		CHECK_KERNEL_FUNCTION_ERROR

		// ----------------------------------------------------------------------------

		// Copying results out
		GPU_ERROR_CHECK(cudaMemcpy(perspImg, devicePersp, perspSize, cudaMemcpyDeviceToHost));

		// ----------------------------------------------------------------------------

		// Free GPU resources
		GPU_ERROR_CHECK(cudaFree(deviceImg));
		GPU_ERROR_CHECK(cudaFree(deviceCameraRayParams));
		GPU_ERROR_CHECK(cudaFree(devicePersp));

		GPU_ERROR_CHECK(cudaFree(deviceFishFoV));
		GPU_ERROR_CHECK(cudaFree(deviceFishCenterX));
		GPU_ERROR_CHECK(cudaFree(deviceFishCenterY));
		GPU_ERROR_CHECK(cudaFree(deviceFishRadius));
		GPU_ERROR_CHECK(cudaFree(deviceFishRadiusY));
		GPU_ERROR_CHECK(cudaFree(deviceTiltAngle));
		GPU_ERROR_CHECK(cudaFree(deviceRollAngle));
		GPU_ERROR_CHECK(cudaFree(devicePanAngle));
		GPU_ERROR_CHECK(cudaFree(deviceAntialiasing));
		GPU_ERROR_CHECK(cudaFree(deviceLensCorrectionEnabled));
		GPU_ERROR_CHECK(cudaFree(deviceLensCorrection));

		return 0;
	}

	int Fish2Persp
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
		return Fish2Persp<double>(nOutputs, img, imgWidth, imgHeight, imgChannels,
			perspWidth, perspHeight, perspFoV, fishFoV, fishCenterX, fishCenterY,
			fishRadius, fishRadiusY, tiltAngle, rollAngle, panAngle, antiAliasing,
			lensCorrectionEnabled, lensCorrection, /* Output */ perspImg);
	}

	int Fish2Persp
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
		return Fish2Persp<float>(nOutputs, img, imgWidth, imgHeight, imgChannels,
			perspWidth, perspHeight, perspFoV, fishFoV, fishCenterX, fishCenterY,
			fishRadius, fishRadiusY, tiltAngle, rollAngle, panAngle, antiAliasing,
			lensCorrectionEnabled, lensCorrection, /* Output */ perspImg);
	}
}