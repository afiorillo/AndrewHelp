
#define GPU_ERROR_CHECK(e) { \
	if (e != cudaSuccess) { cudaDeviceReset(); return static_cast<int>(e); } }

#define GPU_ERROR_CHECK_NO_RETURN(ans) { CommonAlgorithmGPULibrary::GPUAssert((ans), __FILE__, __LINE__); }

#ifdef _DEBUG
#define CHECK_KERNEL_FUNCTION_ERROR { GPU_ERROR_CHECK(cudaDeviceSynchronize()); GPU_ERROR_CHECK(cudaGetLastError()); GPU_ERROR_CHECK(cudaDeviceSynchronize());}
#define CHECK_KERNEL_FUNCTION_ERROR_NO_RETURN { GPU_ERROR_CHECK_NO_RETURN(cudaGetLastError()); \
		GPU_ERROR_CHECK_NO_RETURN(cudaDeviceSynchronize());}
#else
#define CHECK_KERNEL_FUNCTION_ERROR { GPU_ERROR_CHECK(cudaGetLastError()); }
#define CHECK_KERNEL_FUNCTION_ERROR_NO_RETURN { GPU_ERROR_CHECK_NO_RETURN(cudaGetLastError()); }
#endif