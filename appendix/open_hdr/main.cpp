#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <math.h>
#include <stdio.h>
#include "chirplet_cuda.h"

float basis[16] = { -1.0f, 3.0f, -3.0f, 1.0f, 3.0f, -6.0f, 3.0f, 0.0f, -3.0f,
		0.0f, 3.0f, 0.0f, 1.0f, 4.0f, 1.0f, 0.0f };

IplImage* frame;
CvCapture* capture;

float mu[3] = { 11.823222f, 16.465048f, 18.527411f };
float var[3] = { 46.427956f, 64.919624f, 83.983139f };

char* windowName[NUM_IMG];
CvScalar rgb[3] = { cvScalar(255, 0, 0, 0), cvScalar(0, 255, 0, 0), cvScalar(0,
		0, 255, 0) };

bool responseLoad(FILE* file, float* I, int M);

int compare(const void * a, const void * b) {
	if ((*(float*) a - *(float*) b) > 0.0f)
		return 1;
	else
		return -1;
}

void spline(float* b, float* x, int xPts, float* y, int yPts, int n, int xInc) {
	int i, j, k, l, m;
	float t;
	float tmp;
	int ptsInterpolate = yPts / xPts * xInc;
	int exponent;
	float tmpX;
	int xIndex;
	//bspline requires x[l-1], x[l], x[l+1], and x[l+2]
	//so we need paddings for x[-1] for x[0] and x[xPts] x[xPts+1] for x[xPts-1]
	//printf("spline size %d\n", yPts);
	for (m = 0; m < n; m++) {
		for (l = 0; l < xPts / xInc; l++) {
			for (k = 0; k < ptsInterpolate; k++) {
				t = (float) k / (float) ptsInterpolate;
				tmp = 0.0f;
				for (i = 0; i < 4; i++) {
					exponent = 3 - i;
					for (j = 0; j < 4; j++) {
						xIndex = (l - 1 + j) * xInc;
						if (xIndex < 0)
							tmpX = x[m];
						else if (xIndex >= xPts)
							tmpX = x[(xPts - 1) * n + m] * 1.1f;
						else
							tmpX = x[xIndex * n + m];
						tmp += powf(t, exponent) * b[i * 4 + j] * tmpX;
					}
				}
				y[(l * ptsInterpolate + k) * n + m] = tmp / 6.0f;
			}
		}
	}
}

void vecUpdatedBySpline(float* x, int xPts, float* y, int yPts, int n) {
	int i, j;
	int inc = xPts / yPts;
	for (j = 0; j < n; j++) {
		for (i = 0; i < yPts; i++) {
			y[i * n + j] = x[i * n * inc + j];
		}
	}
}

void plot(float* y, int yPts, int n, IplImage* dst) {
	float tmp = 0.0f;
	float _tmp = 65535.0f;
	unsigned char* dstpt = (unsigned char*) dst->imageData;
	int wid = dst->width;
	int hi = dst->height;
	int skip = yPts / wid;
	int i, j;
	float scalar = 1.0f;
	for (j = 0; j < n; j++) {
		for (i = 0; i < yPts; i++) {
			if (_tmp > y[i * n + j])
				_tmp = y[i * n + j];
		}
	}
	for (j = 0; j < n; j++) {
		for (i = 0; i < yPts; i++) {
			if (tmp < y[i * n + j] - _tmp)
				tmp = y[i * n + j] - _tmp;
		}
	}
	if (tmp > 0.0f) {
		scalar = (float) (hi - 1) / tmp;
	}
	int yPos;

	cvZero(dst);
	for (j = 0; j < 3; j++) {
		for (i = 0; i < wid; i++) {
			//yPos = i;
			if ((y[i * n * skip + j] - _tmp) != NAN) {
				yPos = (int) ((y[i * n * skip + j] - _tmp) * scalar);
				dstpt[(yPos * wid + i) * 3 + j] = 255;
			}
		}
	}
}

void plotInteger(unsigned int* y, int yPts, int n, IplImage* dst) {
	float tmp = 0.0f;
	float _tmp = 65535.0f;
	unsigned char* dstpt = (unsigned char*) dst->imageData;
	int wid = dst->width;
	int hi = dst->height;
	int skip = yPts / wid;
	int i, j;
	float scalar = 1.0f;
	for (j = 0; j < n; j++) {
		for (i = 0; i < yPts; i++) {
			if (_tmp > (float) y[i * n + j])
				_tmp = (float) y[i * n + j];
		}
	}
	for (j = 0; j < n; j++) {
		for (i = 0; i < yPts; i++) {
			if (tmp < (float) y[i * n + j] - _tmp)
				tmp = (float) y[i * n + j] - _tmp;
		}
	}
	if (tmp > 0.0f) {
		scalar = (float) (hi - 1) / tmp;
	}
	int yPos;

	cvZero(dst);
	for (j = 0; j < 3; j++) {
		for (i = 0; i < wid; i++) {
			//yPos = i;
			if (((float) y[i * n * skip + j] - _tmp) != NAN) {
				yPos = (int) (((float) y[i * n * skip + j] - _tmp) * scalar);
				dstpt[(yPos * wid + i) * 3 + j] = 255;
			}
		}
	}
}

void display_vector(float* x, int xPts, int n) {
	int i, j;
	for (j = 0; j < n; j++) {
		printf("\n[");
		for (i = 0; i < xPts; i++) {
			printf("%f", x[i * n + j]);
			if (i < xPts - 1)
				printf(",");
		}
		printf("]\n");
	}
}

void beta_dist(float* x, int xPts, int __a, int __b, int __c, int sum, int n) {
	int i, j;
	float a = (float) __a / (float) sum;
	float b = (float) __b / (float) sum;
	float c = (float) __c / (float) sum;
	float t;
	float B = 0.0f;
	float inst;
	float tmp[xPts];
	float offset = 0.0001f;
	for (i = 0; i < xPts; i++) {
		t = (float) i / (float) (xPts - 1);
		B += powf(t, a - 1.0f) * powf(1.0f - t, b - 1.0f);
	}
	for (j = 0; j < n; j++) {
		for (i = 0; i < xPts; i++) {
			t = (float) i / (float) (xPts - 1);
			inst = powf(t, a - 1.0f) * powf(1.0f - t, b - 1.0f);
			tmp[i] = inst / B;
		}
		for (i = 0; i < xPts; i++) {
			x[i * n + j] = tmp[i] * c + offset;    	//tmp[xPts-1-i];
		}
	}
}

float sigmoid(float x, float a, float c) {
	return a / (1.0f + powf(natural_e, -1.0f * x)) + c;
}

// y = a/(1+e^-x)+c
// e^-x = a/(y-c) - 1
// x = -log(a/(y-c)-1)
// c = 0
// a = 1
// x = -127/31 .. 128/31

void sigmoid_dist(float* x, int xPts, int n) {
	int i, j;
	for (j = 0; j < n; j++) {
		for (i = 0; i < xPts; i++) {
			x[i * n + j] = sigmoid(
					(float) (i - (xPts / 2 - 1)) / (float) (xPts / 8 - 1), 1.0f,
					0.0f);
		}
	}
}

float normalize_vector(float* x, int xPts, int n, float c) {
	int i;
	float tmp = 0.0f;
	float _tmp = 65535.0f;
	float value;
	for (i = 0; i < xPts * n; i++) {
		value = x[i];
		if (tmp < value)
			tmp = value;
		if (_tmp > value)
			_tmp = value;
	}
	if (tmp > 0.0f) {
		for (i = 0; i < xPts * n; i++) {
			x[i] = (x[i] - _tmp) / (tmp - _tmp) + c;
		}
	}
	return tmp;
}

float sum_frame(IplImage* src) {
	float tmp = 0.0f;
	unsigned char* pt = (unsigned char*) src->imageData;
	int wid = src->width;
	int hi = src->height;
	int n = src->nChannels;
	int i;
	for (i = 0; i < wid * hi * n; i++) {
		tmp += (float) pt[i] / 255.0f / 1000.0f;
	}
	return tmp;
}

unsigned int resize_param(unsigned int* width, unsigned int* height) {
	float tmpWid = (float) (*width);
	float tmpHi = (float) (*height);
	float proper_ratio = 1280.0f / 720.0f;
	float proper_resolution = 1280.0f * 720.0f;
	float input_ratio = tmpWid / tmpHi;
	float input_resolution = tmpWid * tmpHi;
	unsigned int resize_required = 0;
	if (input_resolution > proper_resolution) {
		if (input_ratio > proper_ratio) {
			*width = 1280;
			*height = (unsigned int) (tmpHi * 1280.0f / tmpWid);
		} else if (input_ratio < proper_ratio) {
			*width = (unsigned int) (tmpWid * 720.0f / tmpHi);
			*height = 720;
		} else {
			*width = 1280;
			*height = 720;
		}
		resize_required = 1;
	}
	return resize_required;
}

unsigned int resize_strict(unsigned int* width, unsigned int* height) {
	float tmpWid = (float) (*width);
	float tmpHi = (float) (*height);
	unsigned int resize_required = 0;
	if (tmpWid > 1920.0f || tmpHi > 1080.0f) {
		*width = 1920;
		*height = 1080;
		resize_required = 1;
	}
	return resize_required;
}

float q_comparagram(unsigned char* src0, unsigned char* src1,
		unsigned int img_size, float* inverse_response, float* certainty) {
	float k_hat_cumulative = 0.0f;
	float c = 0.0f;
	float c0;
	float c1;
	float q0;
	float q1;
	float weights = 0.0f;
	unsigned int j;
	for (unsigned int i = 0; i < img_size; i++) {
		j = i % 3;
		q0 = inverse_response[src0[i] * 3 + j];
		q1 = inverse_response[src1[i] * 3 + j];
		c0 = certainty[src0[i] * 3 + j];
		c1 = certainty[src1[i] * 3 + j];
		if (c0 > c1)
			c = c0;
		else
			c = c1;
		//c = (c0+c1)/2.0f;
		c = 1.0f;
		weights += c;

		if (q1 > 0.0f)
			k_hat_cumulative += c * q0 / q1;
	}
	k_hat_cumulative = logf(k_hat_cumulative / weights) / logf(2.0f);
	return fabs(k_hat_cumulative);
}

void compute_weight(float* x, unsigned int xPts, float* tmp_x, unsigned int n,
		unsigned int __a, unsigned int __b, unsigned int __c,
		unsigned int __total) {
	beta_dist(tmp_x, PIX_DEP, __a, __b, __c, __total, 1);
	normalize_vector(tmp_x, PIX_DEP, 1, 0.0f);
	for (unsigned int j = 1; j < n - 1; j++) {
		for (unsigned int i = 0; i < PIX_DEP; i++) {
			x[i * n + j] = tmp_x[i];
		}
	}
	printf("\n");
	sigmoid_dist(tmp_x, PIX_DEP, 1);
	normalize_vector(tmp_x, PIX_DEP, 1, 0.0f);
	for (unsigned int i = 0; i < PIX_DEP; i++) {
		x[i * n] = 1.0f - tmp_x[i];
		x[i * n + (n - 1)] = tmp_x[i];
	}
}

int main(int argc, char *argv[]) {
	unsigned char c = 0;
	int cnt;

	int num_img = 2;
	float q_scalar = 16.0f;
	float fps = 60.0f;
	unsigned int downSize = 0;
	float known_exposure = -1.0f;

	//$hdr_exec $path/$file $path/$hdr_video_only $stops $skip $frames $fps

	if (argc >= 7) {
		capture = cvCaptureFromFile(argv[1]);
		known_exposure = atof(argv[3]);
		num_img = atoi(argv[5]);
		fps = atof(argv[6]);
	}else{
		printf("Usage: renderer $source_hdr_video $output_video $stops_part $skip $frames $output_fps\n");
	}

	if (capture == NULL) {
		printf("capture did not initialize\n");
		exit(0);
	}

	float tmpPixSum = 0.0f;
	float pixSum;
	int tmpIndexSum = 0;
	for (int i = 0; i < num_img; i++) {
		frame = cvQueryFrame(capture);
		if (frame == NULL)
			break;
	}

	if (frame == NULL) {
		printf("frame did not initialize\n");
		exit(0);
	}

	int _a = 50;
	int _b = 50;
	int _c = 50;
	int _total = 14;
	int _toneP = 225;
	int sigS = 20;
	int sigR = 100;
	int m = 150;
	int histogram_thresh = 1000;
	int alpha = 10;
	int gamma = 35;

	int i;
	int j;
	unsigned int wid = frame->width;
	unsigned int hi = frame->height;
	int n = frame->nChannels;

	downSize = resize_strict(&wid, &hi);

	float *spline_response = (float *) malloc(sizeof(float) * 3 * splResSize);
	float *sample_weight = (float *) malloc(sizeof(float) * num_img * PIX_DEP);
	float *tmp_weight = (float *) malloc(sizeof(float) * PIX_DEP);
	float *cumulative_sum = (float *) malloc(sizeof(float) * 3 * histogramSize);
	unsigned int *histogram = (unsigned int*) malloc(
			sizeof(unsigned int) * histogramSize);

	IplImage* res = cvCreateImage(cvSize(PIX_DEP, PIX_DEP), 8, 3);
	IplImage* comb = cvCreateImage(cvSize(wid, hi), 8, 3);
	IplImage* qFrame = cvCreateImage(cvSize(wid, hi), 8, 3);
	IplImage* prev_frame = cvCreateImage(cvSize(wid, hi), 8, 3);
	IplImage* lightProfile = cvCreateImage(cvSize(wid, hi), 8, 3);
	IplImage* light = cvCreateImage(cvSize(wid, hi), 32, 3);
	IplImage* light_transposed = cvCreateImage(cvSize(hi, wid), 32, 3);
	IplImage* normalized = cvCreateImage(cvSize(wid, hi), 32, 3);
	IplImage* equalized = cvCreateImage(cvSize(wid, hi), 32, 3);
	IplImage** img = (IplImage**) malloc(sizeof(IplImage*) * num_img);
	for (i = 0; i < num_img; i++)
		img[i] = cvCreateImage(cvSize(wid, hi), 8, 3);


	float *response_CPU = (float *) malloc(sizeof(float) * 3 * PIX_DEP);
	float *tmp_response = (float *) malloc(sizeof(float) * 3 * PIX_DEP);

	FILE *response_file;
	if (argc >= 8)
		response_file = fopen(argv[7], "r");
	else
		response_file = fopen("response.m", "r");

	if (!response_file) {
		printf("Response Curve is missing ... please put it in response.m\n");
	}

	bool load_curve = false;
	load_curve = responseLoad(response_file, tmp_response, 256);
	load_curve = responseLoad(response_file, tmp_response + 256, 256);
	load_curve = responseLoad(response_file, tmp_response + 512, 256);

	qsort(tmp_response, PIX_DEP, sizeof(float), compare);
	qsort(tmp_response + PIX_DEP, PIX_DEP, sizeof(float), compare);
	qsort(tmp_response + PIX_DEP * 2, PIX_DEP, sizeof(float), compare);

	for (i = 0, j = 0; i < 256 * 3; i += 3, j++) {
		response_CPU[i] = tmp_response[j];
		response_CPU[i + 1] = tmp_response[j + 256];
		response_CPU[i + 2] = tmp_response[j + 512];
	}

	spline(basis, response_CPU, PIX_DEP, spline_response, splResSize, 3, 2);
	vecUpdatedBySpline(spline_response, splResSize, response_CPU, PIX_DEP, 3);
	compute_weight(sample_weight, PIX_DEP, tmp_weight, num_img, _a, _b, _c,
			_total);

	hdrInit(wid, hi, n, num_img);

	int waitTime = 2;

	CvVideoWriter* video = cvCreateVideoWriter(argv[2],
			CV_FOURCC('M', 'J', 'P', 'G'), (double) fps, cvSize(wid, hi), 1);

	float pair_stop = 0.0f;
	float estimate_stop = 0.0f;
	float error_stop = 0.0f;
	float prev_error = 0.0f;
	float kp = 0.5f;
	float ki = 0.2f;
	float kd = 0.1f;
	float ep = 0.0f;
	float ei = 0.0f;
	float ed = 0.0f;
	unsigned int stop_detected = 0;
	if (known_exposure != -1.0f) {
		stop_detected = 1;
		estimate_stop = known_exposure;
	}
	for (int i = 0; (stop_detected == 0) || (i < 50 * num_img); i++) {
		frame = cvQueryFrame(capture);
		if (frame == NULL)
			break;
		if (downSize == 0) {
			if (i > 0 && stop_detected == 0)
				pair_stop += q_comparagram((unsigned char*) frame->imageData,
						(unsigned char*) prev_frame->imageData, wid * hi * 3,
						response_CPU, sample_weight);
			pixSum = sum_frame(frame);
			cvCopy(frame, prev_frame, NULL);
		} else {
			cvResize(frame, qFrame, CV_INTER_LINEAR);
			if (i > 0 && stop_detected == 0)
				pair_stop += q_comparagram((unsigned char*) qFrame->imageData,
						(unsigned char*) prev_frame->imageData, wid * hi * 3,
						response_CPU, sample_weight);
			pixSum = sum_frame(qFrame);
			cvCopy(qFrame, prev_frame, NULL);
		}
		if (tmpPixSum < pixSum) {
			tmpPixSum = pixSum;
			tmpIndexSum = i % num_img;
		}
		if (i > 0 && stop_detected == 0) {
			error_stop = pair_stop / (float) i - estimate_stop;
			if (error_stop > 0.01f * pair_stop / (float) i) {
				ep = error_stop;
				ei += error_stop;
				ed = error_stop - prev_error;
				prev_error = error_stop;
				estimate_stop = kp * ep + ki * ei + kd * ed;
				//printf("average stops %f on frame %d estimate %f\n", pair_stop/(float)i, i, estimate_stop);
			} else
				stop_detected = 1;
		}
	}

	cnt = (num_img - tmpIndexSum) % num_img;
	RIW_CopyToDevice(response_CPU, sample_weight, powf(2.0f, estimate_stop));

	capture = cvCaptureFromFile(argv[1]);
	cvZero(frame);
	cvZero(lightProfile);
	unsigned int zero_cnt = 0;

	unsigned int max_index = 0;
	float rescale = 255.0f;
	unsigned char* pt;

	printf(";%d;%f\n", cnt, estimate_stop);




	unsigned int bench_cnt = 0;
	while ((frame = cvQueryFrame(capture)) != NULL) {
		if (downSize == 0) {
			frameCopyToDevice((unsigned char*) frame->imageData, cnt);
			cvShowImage("frame", frame);
		} else {
			cvResize(frame, qFrame, CV_INTER_LINEAR);
			frameCopyToDevice((unsigned char*) qFrame->imageData, cnt);
			cvShowImage("frame", qFrame);
		}

		max_index = useLIGHT((float) (_toneP - 200) / 10.0f,
				(float) num_img / 2.0f);

		if (cnt == 0) {
			if (downSize == 0)
				pt = (unsigned char*) frame->imageData;
			else
				pt = (unsigned char*) qFrame->imageData;

			rescale = rescale * 0.95f + 0.05f * (float) pt[max_index];
		}

		frameCopyToHost((unsigned char*) lightProfile->imageData);

		cvShowImage("stage3", lightProfile);
		c = cvWaitKey(waitTime);
		if (c == 'q')
			break;
		else if (c == 'p')
			waitTime = 0;
		else if (c == 'r')
			waitTime = 2;

		if (zero_cnt < 10) {
			cvZero(lightProfile);
			zero_cnt++;
		}
		cvWriteFrame(video, lightProfile);
		cnt++;
		cnt %= num_img;

		bench_cnt++;
	}
	//printf("video stopped\n");
	cvDestroyAllWindows();
	cvReleaseCapture(&capture);
	cvReleaseVideoWriter(&video);
	freeMem();
	free(response_CPU);
	free(cumulative_sum);
	free(histogram);
	free(sample_weight);
	free(tmp_weight);
	free(spline_response);
	cvReleaseImage(&comb);
	cvReleaseImage(&res);
	cvReleaseImage(&light);
	cvReleaseImage(&normalized);
	cvReleaseImage(&equalized);
	cvReleaseImage(&lightProfile);
	cvReleaseImage(&qFrame);

	for (i = 0; i < num_img; i++) {
		cvReleaseImage(&img[i]);
	}
}

bool responseLoad(FILE* file, float* I, int M) {
	char line[1024];
	int m = 0, c = 0;

	// parse response curve matrix header
	while (fgets(line, 1024, file))
		if (sscanf(line, "# rows: %d\n", &m) == 1)
			break;
	if (m != M) {
		printf("responseLoad: number of input levels is different\n");
		return false;
	}
	while (fgets(line, 1024, file))
		if (sscanf(line, "# columns: %d\n", &c) == 1)
			break;
	if (c != 3)
		return false;

	// read response
	float ignore;
	for (int i = 0; i < M; i++) {
		float val;
		if (fscanf(file, " %f %d %f\n", &ignore, &m, &val) != 3)
			return false;
		if (m < 0 || m > M)
			printf("response: camera value out of range\n");
		else
			I[m] = val;
	}

	return true;
}
