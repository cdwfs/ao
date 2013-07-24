#include <cuda.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <time.h>

#include "utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int kThreadsPerBlock = 256;


__global__ void init_curand_states(curandState_t * const outStates)
{
	int seed = threadIdx.x;
	int id = threadIdx.x;
	curand_init(seed, id, 0, &outStates[id]);
}

__device__ bool rayHitsCell(const float3 ray, const float3 cellMin)
{
	// Test X=cellMin.x
	if (ray.x > 0)
	{
		float tx = cellMin.x / ray.x;
		float y4x = ray.y * tx;
		float z4x = ray.z * tx;
		if (y4x >= cellMin.y && y4x < cellMin.y + 1.0f &&
			z4x >= cellMin.z && z4x < cellMin.z + 1.0f)
		{
			return true;
		}
	}
	// Test Y=cellMin.y
	if (ray.y > 0)
	{
		float ty = cellMin.y / ray.y;
		float x4y = ray.x * ty;
		float z4y = ray.z * ty;
		if (x4y >= cellMin.x && x4y < cellMin.x + 1.0f &&
			z4y >= cellMin.z && z4y < cellMin.z + 1.0f)
		{
			return true;
		}
	}
	// Test Z=cellMin.z
	if (ray.z > 0)
	{
		float tz = cellMin.z / ray.z;
		float x4z = ray.x * tz;
		float y4z = ray.y * tz;
		if (x4z >= cellMin.x && x4z < cellMin.x + 1.0f &&
			y4z >= cellMin.y && y4z < cellMin.y + 1.0f)
		{
			return true;
		}
	}
	return false;
}

__global__ void computeOctantOcclusion(float * const outOcclusionFactors, const float3 * const cellMins, const int cellCount, const int raysPerThread,
	const curandState_t * const globalStates, const int blockIndexOffset)
{
	// This is a huge kernel with a ton of blocks. We launch them in smaller blocks to avoid GPU timeouts.
	// globalBlockIndex is the actual index of this block in the global workload.
	const int globalBlockIndex = blockIdx.x + blockIndexOffset;
	// optimization hack: if the lowest bit of the global block index is set, that means the cell at [0,0,0] is opaque,
	// and the vertex is fully occluded. This happens half the time, so let's kill those blocks early.
	if (globalBlockIndex & 0x1)
	{
		if (threadIdx.x == 0)
			outOcclusionFactors[globalBlockIndex] = 1.0f;
		return;
	}
	// Load curand states
	curandState_t localState = globalStates[threadIdx.x];
	// Load cell minimums into shared memory
	__shared__ float3 sCellMins[64];
	if (threadIdx.x < cellCount)
	{
		sCellMins[threadIdx.x] = cellMins[threadIdx.x];
	}
	__syncthreads();
	
	int hitCount = 0; // Count the rays that hit an opaque cell
	for(int iRay=0; iRay<raysPerThread; ++iRay)
	{
		// Compute a random point on the surface of the unit sphere.
		// Courtesy of http://mathproofs.blogspot.com/2005/04/uniform-random-distribution-on-sphere.html
		const float radius = 1.0f; // hard-coded, but included anyway just for completeness
		const float rand0 = curand_uniform(&localState), rand1 = curand_uniform(&localState);
		const float theta0 = 2.0f * (float)M_PI * rand0;
		const float cosT1 = 1.0f - 2.0f*rand1;
		const float rsinT1 = radius * sqrtf(1.0f - cosT1*cosT1);
		float sinT0, cosT0;
		sincosf(theta0, &sinT0, &cosT0); // TODO: sincospif(), and remove the multiplication by M_PI?
		// Take absolute value of all three components, to restrict to a single octant.
		float3 ray = make_float3(
			fabsf(sinT0 * rsinT1),
			fabsf(cosT0 * rsinT1),
			fabsf(radius * cosT1));

		for(int iCell=0; iCell<cellCount; ++iCell)
		{
			if ( (globalBlockIndex & (1<<iCell)) == 0 )
				continue; // Skip this cell if it's not opaque in this configuration
			if (rayHitsCell(ray, sCellMins[iCell]))
			{
				hitCount += 1;
				break; // early-out once after the first hit for each ray
			}
		}
	}

	// Reduce the hit ratio for all threads to compute the overall occlusion factor for this cell configuration.
	float hitRatio = (float)hitCount / (float)raysPerThread;
#if __CUDA_ARCH__ >= 300
	hitRatio += __shfl_down(hitRatio, 1);
	hitRatio += __shfl_down(hitRatio, 2);
	hitRatio += __shfl_down(hitRatio, 4);
	hitRatio += __shfl_down(hitRatio, 8);
	hitRatio += __shfl_down(hitRatio, 16);
	__shared__ float warpHitRatios[32];
	const int warpIdx = threadIdx.x / warpSize;
	const int warpTid = threadIdx.x % warpSize;
	const int warpCount = blockDim.x / warpSize;
	if (warpTid == 0)
		warpHitRatios[warpIdx] = hitRatio;
	__syncthreads();
	if (warpIdx == 0)
	{
		hitRatio = (warpIdx < warpCount) ? warpHitRatios[warpTid] : 0;
		hitRatio += __shfl_down(hitRatio, 1);
		hitRatio += __shfl_down(hitRatio, 2);
		hitRatio += __shfl_down(hitRatio, 4);
		hitRatio += __shfl_down(hitRatio, 8);
		hitRatio += __shfl_down(hitRatio, 16);
		if (warpTid == 0)
		{
			outOcclusionFactors[globalBlockIndex] = hitRatio / (float)blockDim.x;
		}
	}
#else
	// TODO: lame CC2.0 reduce
#endif

	// Store CURAND states back -- I don't think we want to do this?
	// globalStates[threadIdx.x] = localState;
}

///////////////////////////// Host code

typedef struct Vec3
{
	float m_x, m_y, m_z;
} Vec3;

static float randf01(void)
{
	static_assert(RAND_MAX == 0x7FFF, "Unexpected value of RAND_MAX");
	uint32_t n = (rand() & RAND_MAX) + ( (rand() & RAND_MAX)<<15 ) + ( (rand() & 0x3) << 30 );
	return (float)n / (float)0xFFFFFFFF;
}
static void randSphereSurface(Vec3 *outPoint)
{
	// Courtesy of http://mathproofs.blogspot.com/2005/04/uniform-random-distribution-on-sphere.html
	const float radius = 1.0f; // hard-coded, but included anyway just for completeness
	const float rand0 = randf01(), rand1 = randf01();
	const float theta0 = 2.0f * (float)M_PI * rand0;
#if 0// original, inefficient code
	const float theta1 = acosf(1.0f - 2.0f*rand1);
	outPoint->m_x = radius * sinf(theta0) * sinf(theta1);
	outPoint->m_y = radius * cosf(theta0) * sinf(theta1);
	outPoint->m_z = radius * cosf(theta1);
#else // new hotness
	const float cosT1 = 1.0f - 2.0f*rand1;
	const float rsinT1 = radius * sqrtf(1.0f - cosT1*cosT1);
	outPoint->m_x = sinf(theta0) * rsinT1;
	outPoint->m_y = cosf(theta0) * rsinT1;
	outPoint->m_z = radius * cosT1;
#endif
}

bool hitsCube(const Vec3& ray, const Vec3& cubeMin)
{
	// intersect the ray with the near X/Y/Z-oriented planes of the cube, and test against the face boundaries.

	// Test X=cubeMin.m_x
	if (ray.m_x > 0)
	{
		float tx = cubeMin.m_x / ray.m_x;
		float y4x = ray.m_y * tx;
		float z4x = ray.m_z * tx;
		if (y4x >= cubeMin.m_y && y4x < cubeMin.m_y + 1.0f &&
			z4x >= cubeMin.m_z && z4x < cubeMin.m_z + 1.0f)
		{
			return true;
		}
	}
	// Test Y=cubeMin.m_y
	if (ray.m_y > 0)
	{
		float ty = cubeMin.m_y / ray.m_y;
		float x4y = ray.m_x * ty;
		float z4y = ray.m_z * ty;
		if (x4y >= cubeMin.m_x && x4y < cubeMin.m_x + 1.0f &&
			z4y >= cubeMin.m_z && z4y < cubeMin.m_z + 1.0f)
		{
			return true;
		}
	}
	// Test Z=cubeMin.m_z
	if (ray.m_z > 0)
	{
		float tz = cubeMin.m_z / ray.m_z;
		float x4z = ray.m_y * tz;
		float y4z = ray.m_z * tz;
		if (x4z >= cubeMin.m_x && x4z < cubeMin.m_x + 1.0f &&
			y4z >= cubeMin.m_y && y4z < cubeMin.m_y + 1.0f)
		{
			return true;
		}
	}
	return false;
}

int computeOctantOcclusionLUT(void)
{
	int deviceCount = 0;
	CUDA_CHECK( cudaGetDeviceCount(&deviceCount) );
	if (deviceCount < 1)
	{
		printf("ERROR: expected at least one CUDA device; found %d.\n", 0);
		return -1;
	}
	const int deviceId = 0;
	CUDA_CHECK( cudaSetDevice(deviceId) );
	cudaDeviceProp deviceProp = {};
	CUDA_CHECK( cudaGetDeviceProperties(&deviceProp, deviceId) );
	printf("Using CUDA device %d: \"%s\"\n", deviceId, deviceProp.name);

	curandState_t *d_globalStates = nullptr;
	CUDA_CHECK( cudaMalloc(&d_globalStates, kThreadsPerBlock*sizeof(curandState_t)) );

	init_curand_states<<<1,kThreadsPerBlock>>>(d_globalStates);
	CUDA_CHECK( cudaGetLastError() );

	const int kCellCount = 20;
	const int kCellPermutationCount = 1<<kCellCount;
	const int kRaysPerThread = 128;
	float3 h_cellMins[kCellCount] = {
		make_float3(0,0,0),

		make_float3(1,0,0),
		make_float3(0,1,0),
		make_float3(0,0,1),

		make_float3(2,0,0),
		make_float3(1,1,0),
		make_float3(0,2,0),
		make_float3(1,0,1),
		make_float3(0,1,1),
		make_float3(0,0,2),

		make_float3(3,0,0),
		make_float3(2,1,0),
		make_float3(1,2,0),
		make_float3(0,3,0),
		make_float3(2,0,1),
		make_float3(1,1,1),
		make_float3(0,2,1),
		make_float3(1,0,2),
		make_float3(0,1,2),
		make_float3(0,0,3),
	};
	float *h_occlusionFactors = (float*)malloc(kCellPermutationCount*sizeof(float));

	float3 *d_cellMins = nullptr;
	float *d_occlusionFactors = nullptr;
	CUDA_CHECK( cudaMalloc(&d_cellMins, sizeof(h_cellMins)) );
	CUDA_CHECK( cudaMalloc(&d_occlusionFactors, kCellPermutationCount*sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(d_cellMins, h_cellMins, sizeof(h_cellMins), cudaMemcpyHostToDevice) );

	{
		GpuTimer timer;
		timer.Start();
		const int kBlocksPerGrid = 1024;
		for(int iOffset=0; iOffset<kCellPermutationCount; iOffset += kBlocksPerGrid)
		{
			computeOctantOcclusion<<<kBlocksPerGrid,kThreadsPerBlock>>>(d_occlusionFactors, d_cellMins, kCellCount, kRaysPerThread, d_globalStates, iOffset);
			CUDA_CHECK( cudaGetLastError() );
			//cudaDeviceSynchronize();
			printf("*");
		}
		timer.Stop();
		printf("%d configurations of %d cells, %d rays/configuration: %.3f seconds\n",
			kCellPermutationCount, kCellCount, kRaysPerThread*kThreadsPerBlock, timer.Elapsed() / 1000.0f);
	}

	CUDA_CHECK( cudaMemcpy(h_occlusionFactors, d_occlusionFactors, kCellPermutationCount*sizeof(float), cudaMemcpyDeviceToHost) );

	CUDA_CHECK( cudaFree(d_globalStates) );
	CUDA_CHECK( cudaFree(d_cellMins) );
	CUDA_CHECK( cudaFree(d_occlusionFactors) );

#if 0
	srand((unsigned int)time(NULL));
	
	const Vec3 cubeMin = {1,0,0};
	uint32_t numTests = 100000000, numHits = 0;
	for(int iRay=0; iRay<numTests; ++iRay)
	{
		const Vec3 ray = {randf01(), randf01(), randf01()};
		if (hitsCube(ray, cubeMin))
		{
			++numHits;
		}
		
		Vec3 spherePoint;
		randSphereSurface(&spherePoint);
		(void)spherePoint;
	}
	printf("%11d out of %11d (%6.2f%%) of rats hit cube [%d,%d,%d]\n",
		numHits, numTests, 100.0 * (double)numHits / (double)numTests, (int)cubeMin.m_x, (int)cubeMin.m_y, (int)cubeMin.m_z);
#endif

	for(int i=0; i<16; ++i)
	{
		printf("factor %4d: %.3f\n", i, h_occlusionFactors[i]);
	}
	CUDA_CHECK( cudaDeviceReset() );

	free(h_occlusionFactors);
	return 0;
}
