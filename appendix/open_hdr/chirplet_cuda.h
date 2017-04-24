#ifndef CHIRPLET_CUDA_H
#define CHIRPLET_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include <cublas_v2.h>

#define PI 3.14159265
#define natural_e 2.71828183
#define NUM_IMG 9
#define PIX_DEP 256
#define INV_DEP PIX_DEP
#define splResSize INV_DEP*16
#define histogramSize 256
#define QCOMP_SIZE PIX_DEP

#define TPB 128

#define CDFSMEM TPB
#define CDFSAMPLES histogramSize/CDFSMEM

#define BLOCK_DIM 32

#define BENCHMARK 1

#define CALIBRATION 0

#define PLOT_RESPONSE 0
#define PLOT_WEIGHT 0
#define DISPLAY_RESULTS 0

#define SMOOTH_IMG 1
#define PARTIAL_HIST 512
#define TPB_HIST 128

#define n_control 3
#define NUM_DOMAINS 3


extern "C"{
  void freeMem();
  void hdrInit(int wid, int hi, int colrDep, unsigned int num_img);
  void RIW_CopyToDevice(float* input_res, float* input_weight, float q_scalar);
  void frameCopyToDevice(unsigned char* input, int index);
  void frameCopyToHost(unsigned char* output);
  unsigned int useLIGHT(float muLight, float pref_exposure);
};
#endif

