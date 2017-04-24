#include "chirplet_cuda.h"

#include <helper_cuda.h>
#include <helper_timer.h>

#define cutilSafeCall(x) checkCudaErrors(x)
#define cutilCheckMsg(x) getLastCudaError(x)
#define cutStartTimer(x) sdkStartTimer(x)
#define cutCreateTimer(x) sdkCreateTimer(x)
#define cutStopTimer(x) sdkStopTimer(x)
#define cutGetTimerValue(x) sdkGetTimerValue(x)
#define cutResetTimer(x) sdkResetTimer(x)
#define cutDeleteTimer(x) sdkDeleteTimer(x)
__constant__ float sqrtOfPI = 1.77245385;

__constant__ float muB = 11.823222f;
__constant__ float muG = 16.465048f;
__constant__ float muR = 18.527411f;

__constant__ float histRes = (float)histogramSize;
__constant__ int cdf_samples_per_thread = CDFSAMPLES;
__constant__ int cdf_threads = CDFSMEM;

//unsigned int htimer;
StopWatchInterface *htimer = NULL;
long imgsize;

unsigned char* d_in;
unsigned char* d_output;


float* d_res;
float* d_weight;
float* d_light;
float* d_log_light;
float* d_log_light_lowpassed;
float* d_log_normalized;
float* d_log_normalized_transposed;
float* d_H;
float* d_V;
float* d_blur;
float* d_blur_transposed;
float* d_detail;
float* d_detail_domains;

float* h_log_light;

cudaError_t cudaStat ;
cublasStatus_t stat ;
cublasHandle_t handle;

int _tpb;
int _tpbh;
int _bpg;
int _bpgc;
int _bpgh;

int _wid;
int _hi;

float _q_scalar;
int _num_img;

void hdrInit(int wid, int hi, int colrDep, unsigned int num_img){
	cudaDeviceReset();
	imgsize = wid*hi*colrDep;
	_tpb = TPB;
	_tpbh = TPB_HIST;//TPB;//histogramSize;
	_bpg = (wid*hi + _tpb - 1)/_tpb;
	_bpgc = (imgsize + _tpb - 1)/_tpb;
	_bpgh = (imgsize + _tpbh*PARTIAL_HIST - 1)/(_tpbh*PARTIAL_HIST);
	_wid = wid;
	_hi = hi;
	_num_img = num_img;
	cutCreateTimer (&htimer);
	int i;
	
	cutilSafeCall(cudaMalloc((void**)&d_blur, _wid*_hi*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_blur,0,_wid*_hi*sizeof(float)));

	cutilSafeCall(cudaMalloc((void**)&d_detail, _wid*_hi*NUM_DOMAINS*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_detail,0,_wid*_hi*NUM_DOMAINS*sizeof(float)));

	cutilSafeCall(cudaMalloc((void**)&d_detail_domains, _wid*_hi*NUM_DOMAINS*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_detail_domains,0,_wid*_hi*NUM_DOMAINS*sizeof(float)));

	cutilSafeCall(cudaMalloc((void**)&d_blur_transposed, _wid*_hi*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_blur_transposed,0,_wid*_hi*sizeof(float)));
	
	cutilSafeCall(cudaMalloc((void**)&d_in, imgsize*_num_img*sizeof(unsigned char)));
	cutilSafeCall(cudaMemset((void*)d_in,0,imgsize*_num_img*sizeof(unsigned char)));
	
	cutilSafeCall(cudaMalloc((void**)&d_light, imgsize*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_light,0,imgsize*sizeof(float)));

	cutilSafeCall(cudaMalloc((void**)&d_log_light, _wid*_hi*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_log_light,0,_wid*_hi*sizeof(float)));

	cutilSafeCall(cudaMalloc((void**)&d_log_light_lowpassed, _wid*_hi*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_log_light_lowpassed,0,_wid*_hi*sizeof(float)));

	cutilSafeCall(cudaMalloc((void**)&d_log_normalized, _wid*_hi*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_log_normalized,0,_wid*_hi*sizeof(float)));

	cutilSafeCall(cudaMalloc((void**)&d_log_normalized_transposed, _wid*_hi*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_log_normalized_transposed,0,_wid*_hi*sizeof(float)));
	
	cutilSafeCall(cudaMalloc((void**)&d_H, _wid*_hi*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_H,0,_wid*_hi*sizeof(float)));
	
	cutilSafeCall(cudaMalloc((void**)&d_V, _wid*_hi*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_V,0,_wid*_hi*sizeof(float)));
	
	cutilSafeCall(cudaMalloc((void**)&d_output, imgsize*sizeof(unsigned char)));
	cutilSafeCall(cudaMemset((void*)d_output,0,imgsize*sizeof(unsigned char)));
	
	cutilSafeCall(cudaMalloc((void**)&d_res, PIX_DEP*colrDep*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_res,0,PIX_DEP*colrDep*sizeof(float)));
	
	cutilSafeCall(cudaMalloc((void**)&d_weight, PIX_DEP*num_img*sizeof(float)));
	cutilSafeCall(cudaMemset((void*)d_weight,0,PIX_DEP*num_img*sizeof(float)));
	
	h_log_light = (float*)malloc(_wid*_hi*sizeof(float));

	stat = cublasCreate(&handle);
}

void RIW_CopyToDevice(float* input_res, float* input_weight, float q_scalar){
	int tmpsize = PIX_DEP;
	//printf("copying %d\n",tmpsize);
	cutStartTimer(&htimer);
	_q_scalar = q_scalar;
	cutilSafeCall(cudaMemcpy(d_res, input_res, PIX_DEP*3*sizeof(float), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_weight, input_weight, PIX_DEP*_num_img*sizeof(float), cudaMemcpyHostToDevice));
	cutStopTimer(&htimer);
	printf("RIW_CopyToDevice: transfer to device ms time: %f ms\n", cutGetTimerValue(&htimer));
	cutResetTimer(&htimer);
}

void frameCopyToDevice(unsigned char* input, int index){
	cutStartTimer(&htimer);
	cutilSafeCall(cudaMemcpy(d_in+index*imgsize, input, imgsize*sizeof(unsigned char), cudaMemcpyHostToDevice));
	cutStopTimer(&htimer);
	printf ("frameCopyToDevice: transfer to device ms time: %f ms\n", cutGetTimerValue(&htimer));
	cutResetTimer(&htimer);
}

void frameCopyToHost(unsigned char* output){
	cutStartTimer(&htimer);
	cutilSafeCall(cudaMemcpy(output, d_output, imgsize*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	cutStopTimer(&htimer);
	printf ("frameCopyToHost: transfer to host ms time: %f ms\n", cutGetTimerValue(&htimer));
	cutResetTimer(&htimer);
}

__global__ void light_composition(unsigned char* input, float* dRes, float* C, const float pref, const float q_scalar,
		const unsigned int num_img, const float pref_exposure, float* output, unsigned int imgsize){
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j, k; //j == rgb, k == bmd
	register unsigned char p[NUM_IMG];
	register float q[NUM_IMG];
	register float c[NUM_IMG];
	register float c_hat[NUM_IMG];
	float total;
	float light;
	long offset = 0;
	//float k_hat;
	const float pref_added=pref_exposure-pref;

	if (i < imgsize){
		//Optimized for staged operations.
		//https://mc.stanford.edu/cgi-bin/images/0/0a/M02_4.pdf
		__shared__ float hdrPref[NUM_IMG];
		#pragma unroll
		for (k=0; k<num_img; k++){
			hdrPref[k] = __powf(q_scalar, (float)k-pref_added);
		}
		__syncthreads();
		light = 0.0f;
		total = 0.0f;
		j = i % 3;

		#pragma unroll
		for(k=0;k<num_img;k++){
			p[k] = input[i+offset];
			offset += imgsize;
		}
		#pragma unroll
		for(k=0;k<num_img;k++){
    		q[k] = dRes[p[k]*3+j];
    		c[k] = C[p[k]*num_img+k];
		}
		#pragma unroll
    	for(k=0;k<num_img;k++){
		    total += c[k];
    		light += q[k]*c[k]*hdrPref[k];
    	}

		output[i] = light/(total+0.001f);
	}
}

__global__
void ConvertToLogLuminance_cuda(float* img, float* log_lum, int N){
    int b_i = blockIdx.x * blockDim.x + threadIdx.x;
    int b_ii = b_i * 3;
    int g_i = b_ii + 1;
    int r_i = b_ii + 2;
    float log_brightness=10000.0f; //3 frames

    //float log_brightness=10*10*10*10; 
    if(b_i < N){
        float value = 0.3f*img[r_i]+0.6f*img[g_i]+0.1f*img[b_ii];
        float value_2 = 1.0f/value;
        img[b_ii] = img[b_ii]*value_2;
        img[g_i] = img[g_i]*value_2;
        img[r_i] = img[r_i]*value_2;
        //log_lum[b_i] = __powf(value*log_brightness, 1/8.0);
        log_lum[b_i] = __logf(1.0+value*log_brightness)/__logf(2)-1.0;
    }
}

__global__
void NormalizeRange(float* img, float range, float* output, float upper_bound, float lower_bound, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        output[i] = (img[i] - lower_bound)/(upper_bound - lower_bound)*range;
    }
}

__global__
void NormalizeRange2(float* img, float* img_lowpassed, float range, float* output,
		const unsigned int maxIndex, const unsigned int minIndex, const int N){
    float upper_bound = img_lowpassed[maxIndex];
    const float lower_bound = img_lowpassed[minIndex];
    if(upper_bound < 8.5f){
    	upper_bound = 8.5f;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float norm = 1.0f/(upper_bound - lower_bound)*range;

    if(i < N){
        output[i] = (img[i] - lower_bound)*norm;
    }
}

__global__ void transpose(float *odata, float *idata, const int width, const int height, int dummy)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

__global__ void gradient_vertical(float* input, float* output, unsigned int width, unsigned int height, unsigned int imgsize){
	unsigned int i = (blockIdx.x*blockDim.x+threadIdx.x);
	unsigned int row = i / width;
	if (i < imgsize){
    		if(row == 0)
        		output[i] = 0.0f;
    		else
        		output[i] = fabs(input[i]+input[i-width]);
	}
}

__global__ void domain_filter_vertical_RGB_noreg(float* img, float* dVdy, float a,
		const float s_div_r, const int wid, const int hi, const int n){

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_ij = j;

	const int rowsize = wid*n;

	float tmpH;
	float q_e; 
	float tmpD;
		
	if(j< rowsize){
		for(int i = 0; i < hi; i++){
			tmpD = 1.0f+s_div_r*dVdy[idx_ij];
			if(i>0){
				q_e = img[idx_ij];
				tmpH = q_e + __powf(a, tmpD) * ( tmpH - q_e );
				img[idx_ij] = tmpH;
			}
			else
				tmpH = img[idx_ij];
			idx_ij += rowsize;
		}
		for(int i = hi-1; i >= 0; i--){
			idx_ij -= rowsize;
			if(i<hi-1){
			    q_e = img[idx_ij];
				tmpH = q_e + __powf(a, tmpD) * ( tmpH - q_e );
			    img[idx_ij] = tmpH;
			}
			tmpD = 1.0f+s_div_r*dVdy[idx_ij];
		}

	}
}

__global__
void extract_detail(float* img, float* blur, float* output, const unsigned int num_domains, const int N){
	unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j=0;
	if(i < N){
		for(j=0;j<num_domains;j++){
			if(j==0)
				output[i] = img[i]-blur[i];
			else
				output[i] = blur[i-N]-blur[i];
			i += N;
		}
	}
}

__global__
void CompressionAndSaturation(float* img, float* detail, float* J3, float meanB, unsigned char* out_img, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int ii = i*3;
	unsigned int j;
	unsigned long detail_offset = i;
	if(i < N){
		const float brightness = -0.05f;
		float layer[NUM_DOMAINS] = {0.4f,0.3f,0.3f}; //detail _ fine texture
		const float sFRGB[3] = {0.55f,0.55f,0.55f};
		const float base_value = 1.0; //mid-tone value (may saturate the high lights)
		const float local_min = 0.2f;
		const float local_max = 7.0f;
		const float range = 1.0f;
		float J3_local = (J3[i]-local_min)/(local_max-local_min)*range;
		float LC = 0.0f;
		float output = 0.0f;
		//saturation factor (0.5 = low) & (1 = high)

		for(j=0;j<NUM_DOMAINS;j++){
			if(J3_local<0.5f)
				layer[j] = J3_local*layer[j]/0.5f;
			else if(J3_local>6.0f && J3_local <= 7.0f)
				layer[j] = (7.0f-J3_local)*layer[j]/6.0f;
			else if(J3_local>7.0f)
				layer[j] = 0.0f;
			LC += layer[j]*detail[detail_offset];
			detail_offset += N;
		}
		LC += brightness + meanB + base_value * (J3_local - meanB);

		for(j=0;j<3;j++){
			output = __powf(img[ii+j], sFRGB[j]) * LC*255.0f;
			if(output > 255.0f)
				output = 255.0f;
			else if(output < 0.0f)
				output = 0.0f;
			out_img[ii+j] = (unsigned char)output;
		}
	}
}

__global__
void lowpass(float* img, float* img_filter, unsigned int filter, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N){
		if(filter)
			img_filter[i] = img_filter[i]*0.9f + 0.1f*img[i];
		else
			img_filter[i] = img[i];
	}
}

float avgHDR_HE = 0.0f;
float cnt = 0.0f;
int indexMax;
int indexMin;
float img_mean;

float max_light = 0.0f;
float min_light = 0.0f;


//unsigned int useLIGHT(float muLight, float sigS, float sigR, float logEffect, float exposure_pref, float rescalar,
//		float* h_light,
//		unsigned int* h_hist,
//		float* h_CDF,
//		float* h_normalized,
//		float* h_equalize,
//		float* h_detail){
//	cnt += 1.0f;
//
//	cutStartTimer(htimer);
//	light_composition <<< _bpgc, _tpb >>> (d_in, d_res, d_weight, muLight, _q_scalar, _num_img, exposure_pref, d_light, imgsize);
//	cudaThreadSynchronize();
//
//	stat = cublasIsamax(handle, imgsize, d_light, 1, &indexMax);
//	stat = cublasIsamin(handle, imgsize, d_light, 1, &indexMin);
//	normalizeImage <<< _bpgc, _tpb >>> (d_light, indexMax-1, indexMin-1, d_normalized, d_index, imgsize);
//	cudaThreadSynchronize();
//
//	dim3 transpose_HtoV_grid3((_wid*3+BLOCK_DIM-1)/BLOCK_DIM, (_hi+BLOCK_DIM-1)/BLOCK_DIM), transpose_HtoV_threads3(BLOCK_DIM,BLOCK_DIM);
//	dim3 transpose_VtoH_grid3((_hi+BLOCK_DIM-1)/BLOCK_DIM, (_wid*3+BLOCK_DIM-1)/BLOCK_DIM), transpose_VtoH_threads3(BLOCK_DIM,BLOCK_DIM);
//	transpose <<<transpose_HtoV_grid3, transpose_HtoV_threads3 >>> (d_normalized_transposed, d_normalized, _wid*3, _hi, 1);
//	gradient_vertical <<< _bpgc, _tpb >>> (d_normalized_transposed, d_H3, _hi*3, _wid, imgsize);
//	gradient_vertical <<< _bpgc, _tpb >>> (d_normalized, d_V3, _wid*3, _hi, imgsize);
//	cudaThreadSynchronize();
//
//	cutilSafeCall(cudaMemcpy(d_blur_transposed, d_normalized_transposed, imgsize*sizeof(float), cudaMemcpyDeviceToDevice));
//	float _a0;
//	unsigned int num_iterations = 1;
//	for(int i = 0; i<num_iterations;i++){
//		_a0 = exp( (-1 * sqrt(2)) / (sigS * sqrt(3.0) * (powf(2, ( num_iterations-i-1 )) / sqrt( powf(4,num_iterations) - 1 ))) );
//		domain_filter_vertical_RGB_noreg <<< (_hi*3+_tpb-1)/_tpb, _tpb >>> (d_blur_transposed, d_H3, _a0, sigS/sigR, _hi, _wid, 3);
//		transpose <<<transpose_VtoH_grid3, transpose_VtoH_threads3>>> (d_blur, d_blur_transposed, _hi, _wid*3, 1);
//		domain_filter_vertical_RGB_noreg <<< (_wid*3+_tpb-1)/_tpb, _tpb >>> (d_blur, d_V3, _a0, sigS/sigR, _wid, _hi, 3);
//		cudaThreadSynchronize();
//	}
//
//	compose_onelayer<<< _bpgc, _tpb >>>(d_normalized, d_blur, imgsize);
//	cudaThreadSynchronize();
//
//	stat = cublasIsamax(handle, imgsize, d_normalized, 1, &indexMax);
//	stat = cublasIsamin(handle, imgsize, d_normalized, 1, &indexMin);
//	normalizeImage <<< _bpgc, _tpb >>> (d_normalized, indexMax-1, indexMin-1, d_light, d_index, imgsize);
//	cudaThreadSynchronize();
//
//	cudaMemset((void*)d_histogramInteger,0,histogramSize*sizeof(unsigned int));
//	cudaMemset((void*)d_histogramCombined,0,histogramSize*sizeof(float));
//
//	hist_multiple <<< _bpgh, _tpbh >>> (d_index, d_histogramInteger, _tpbh*_bpgh, imgsize);
//	cudaThreadSynchronize();
//
//	convertToFloat <<< (histogramSize+_tpb-1)/_tpb, _tpb >>> (d_histogramInteger, d_histogramCombined, histogramSize);
//	cudaThreadSynchronize();
//
//	stat = cublasIsamax(handle, histogramSize, d_histogramCombined, 1, &indexMax);
//	cudaThreadSynchronize();
//
//	produceCDF <<< 1, _tpb >>> (d_histogramInteger, indexMax-1, histogramSize, d_CDF, logEffect);
//	cudaThreadSynchronize();
//
//	//equalizeLIGHT <<< _bpgc, _tpb >>> (d_index, d_CDF, d_output, imgsize);
//	equalize <<< _bpgc, _tpb >>> (d_index, d_CDF, d_normalized, imgsize);
//	cudaThreadSynchronize();
//
//	convertCharToFloat <<< (imgsize+_tpb-1)/_tpb, _tpb >>> (d_in, d_in32, imgsize);
//	cudaThreadSynchronize();
//
//    stat = cublasIsamax(handle, imgsize, d_normalized, 1, &indexMax);
//    stat = cublasIsamin(handle, imgsize, d_normalized, 1, &indexMin);
//    rescale_cuda <<< _bpgc, _tpb >>> (rescalar, d_normalized, indexMax-1, d_output, imgsize);
//    cudaThreadSynchronize();
//
//	if(h_normalized != NULL)
//	    cutilSafeCall(cudaMemcpy(h_normalized, d_normalized, imgsize*sizeof(float), cudaMemcpyDeviceToHost));
//
//    stat = cublasIsamax(handle, imgsize, d_in32, 1, &indexMax32);
//	stat = cublasIsamin(handle, imgsize, d_in32, 1, &indexMin32);
//
//	cutStopTimer(htimer);
//	avgHDR_HE += cutGetTimerValue(htimer);
//	cutResetTimer(htimer);
//
//	return (indexMax32-1);
//}


unsigned int run = 1;
unsigned int useLIGHT(float muLight, float exposure_pref){
	cnt += 1.0f;
	printf ("*******************************************************************\n");
	cutStartTimer(&htimer);
	light_composition <<< _bpgc, _tpb >>> (d_in, d_res, d_weight, muLight, _q_scalar, _num_img, exposure_pref, d_light, imgsize);
	cudaThreadSynchronize();
	printf ("HDR After Composition: %f ms\n", cutGetTimerValue(&htimer));
	
	ConvertToLogLuminance_cuda<<<_bpg, _tpb>>>(d_light, d_log_light, _wid*_hi);
	cudaThreadSynchronize();
	printf ("HDR After Log: %f ms\n", cutGetTimerValue(&htimer));

	stat = cublasIsamax(handle, _wid*_hi, d_log_light, 1, &indexMax);
	stat = cublasIsamin(handle, _wid*_hi, d_log_light, 1, &indexMin);

	lowpass <<< _bpg, _tpb >>> (d_log_light, d_log_light_lowpassed, run, _wid*_hi);
	cudaThreadSynchronize();
	printf ("HDR After Low Pass: %f ms\n", cutGetTimerValue(&htimer));

	if(run == 1)
		run = 0;

	float range = 8.0f;
	NormalizeRange2 <<< _bpg, _tpb >>> (d_log_light, d_log_light_lowpassed, range, d_log_normalized, indexMax-1, indexMin-1, _wid*_hi);
	cudaThreadSynchronize();

	printf ("HDR After Normalized: %f ms\n", cutGetTimerValue(&htimer));

	dim3 transpose_HtoV_grid3((_wid+BLOCK_DIM-1)/BLOCK_DIM, (_hi+BLOCK_DIM-1)/BLOCK_DIM), transpose_HtoV_threads3(BLOCK_DIM,BLOCK_DIM);
	dim3 transpose_VtoH_grid3((_hi+BLOCK_DIM-1)/BLOCK_DIM, (_wid+BLOCK_DIM-1)/BLOCK_DIM), transpose_VtoH_threads3(BLOCK_DIM,BLOCK_DIM);

	transpose <<<transpose_HtoV_grid3, transpose_HtoV_threads3 >>> (d_log_normalized_transposed, d_log_normalized, _wid, _hi, 1);

	gradient_vertical <<< _bpg, _tpb >>> (d_log_normalized_transposed, d_H, _hi, _wid, _wid*_hi);
	gradient_vertical <<< _bpg, _tpb >>> (d_log_normalized, d_V, _wid, _hi, _wid*_hi);
	cudaThreadSynchronize();   

	printf ("HDR After Gradient: %f ms\n", cutGetTimerValue(&htimer));

	unsigned long detail_offset = 0;
	float s_scale = 0.25;
	float r_scale = 1.0;
	float local_sigS[NUM_DOMAINS] = {s_scale*20.0f, s_scale*50.0f, s_scale*100.0f};
	float local_sigR[NUM_DOMAINS] = {r_scale*0.33f, r_scale*0.67f, r_scale*1.34f};

	for(int j=0;j<NUM_DOMAINS;j++){
		cutilSafeCall(cudaMemcpy(d_blur_transposed, d_log_normalized_transposed, _wid*_hi*sizeof(float), cudaMemcpyDeviceToDevice));
		float _a0;
		unsigned int num_iterations = 2;
		for(int i = 0; i<num_iterations;i++){
			_a0 = exp( (-1 * sqrt(2)) / (local_sigS[j] * sqrt(3.0) * (powf(2, ( num_iterations-i-1 )) / sqrt( powf(4,num_iterations) - 1 ))) );
			domain_filter_vertical_RGB_noreg <<< (_hi+_tpb-1)/_tpb, _tpb >>> (d_blur_transposed, d_H, _a0, local_sigS[j]/local_sigR[j], _hi, _wid, 1);
			transpose <<<transpose_VtoH_grid3, transpose_VtoH_threads3>>> (d_blur, d_blur_transposed, _hi, _wid, 1);
			cudaThreadSynchronize();
			domain_filter_vertical_RGB_noreg <<< (_wid+_tpb-1)/_tpb, _tpb >>> (d_blur, d_V, _a0, local_sigS[j]/local_sigR[j], _wid, _hi, 1);
		}
		cutilSafeCall(cudaMemcpy(d_detail+detail_offset, d_blur, _wid*_hi*sizeof(float), cudaMemcpyDeviceToDevice));
		detail_offset += _wid*_hi;
		cudaThreadSynchronize();
		printf ("HDR After Detail: %d %f ms\n", j, cutGetTimerValue(&htimer));
	}

	extract_detail <<< _bpg, _tpb >>> (d_log_normalized, d_detail, d_detail_domains, NUM_DOMAINS, _wid*_hi);
	cudaThreadSynchronize();
	printf ("HDR After Detail: %f ms\n", cutGetTimerValue(&htimer));

	cublasSasum(handle, _wid*_hi, d_blur, 1, &img_mean);

	CompressionAndSaturation <<< _bpg, _tpb >>> (d_light, d_detail_domains, d_blur, img_mean/(float)(_wid*_hi), d_output, _wid*_hi);
	cudaThreadSynchronize();
	printf ("HDR After Compress & Sat Tuning: %f ms\n", cutGetTimerValue(&htimer));

	cutStopTimer(&htimer);
	avgHDR_HE += cutGetTimerValue(&htimer);
	//printf ("Total HDR Processing time: %f ms\n", cutGetTimerValue(&htimer));
	cutResetTimer(&htimer);

	printf ("*******************************************************************\n\n");

	return 0;
}

void freeMem(){
#if BENCHMARK
	printf ("AVG ms time: %f ms\n", avgHDR_HE/cnt);
	printf ("AVG FPS: %f\n", 1000.0f/(avgHDR_HE/cnt));
#endif
	cutDeleteTimer(&htimer);
	
	cudaFree(d_blur);
	cudaFree(d_blur_transposed);
	
	cudaFree(d_in);
	cudaFree(d_light);
	cudaFree(d_log_light);
	cudaFree(d_log_light_lowpassed);
	cudaFree(d_log_normalized);
	cudaFree(d_log_normalized_transposed);
	cudaFree(d_H);
	cudaFree(d_V);

	cudaFree(d_detail);
	cudaFree(d_detail_domains);

	cudaFree(d_output);
	cudaFree(d_res);
	cudaFree(d_weight);

	free(h_log_light);
	cublasDestroy(handle);
	//cudaThreadExit();
}
