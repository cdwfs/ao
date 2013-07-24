#ifndef UTILS_H__
#define UTILS_H__

#include <windows.h>

#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

class CpuTimer
{
public:
	CpuTimer(void);

	void Reset(void);
	void Start(void);
	double Update(void) { return UpdateWithScale(1.0); }
	double UpdateWithScale(double scale);
	void Stop(void);
	double GetElapsedSeconds(void) const;

private:
	static double GetRawSeconds(void);

	double m_elapsedSeconds;
	double m_lastRawSeconds;
	bool m_isRunning; ///< False if the timer is paused/stopped
	// TODO: 4 bytes of padding here...
};


// Use CUDA_CHECK() in host code to validate the result of CUDA functions that return an error.
#define ENABLE_CUDA_CHECK 1 // disable if you want CUDA_CHECK() to compile away
#if ENABLE_CUDA_CHECK
	#define CUDA_CHECK(val) \
		do{ \
			cudaError err = (val); \
			if (err != cudaSuccess) { \
				printf("CUDA error in %s:\n%d:\t%s\n%s\n", __FILE__, __LINE__, #val, cudaGetErrorString(err)); \
				if (IsDebuggerPresent()) \
					__debugbreak(); \
				else \
					assert(err == cudaSuccess); \
			} \
		__pragma(warning(push)) \
		__pragma(warning(disable:4127)) /* constant conditional */ \
		} while(0) \
		__pragma(warning(pop))
#else
	#define CUDA_CHECK(val) (val)
#endif

#endif
