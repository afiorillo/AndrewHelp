// Fish2PerspTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FisheyeProcessing.h"
#include "CImg.h"
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

#ifdef WIN32
#include <windows.h>
#else
#include <X11/Xlib.h>
#endif

//...

void GetScreenResolution(int &width, int &height) 
{
#ifdef WIN32
	width = static_cast<int>(GetSystemMetrics(SM_CXSCREEN));
	height = static_cast<int>(GetSystemMetrics(SM_CYSCREEN));
#else
	Display* disp = XOpenDisplay(NULL);
	Screen*  scrn = DefaultScreenOfDisplay(disp);
	width = scrn->width;
	height = scrn->height;
#endif
}

int main(int argc, char** argv)
{
	bool isVideo = false;

	int perspwidth;
	int perspheight;
	int nOutputs = 1;

	std::vector<double> perspFoV;
	std::vector<double> fishFoV;
	std::vector<int> fishCenterX;
	std::vector<int> fishCenterY;
	std::vector<int> fishRadius;
	std::vector<int> fishRadiusY;
	std::vector<double> tiltAngle;
	std::vector<double> rollAngle;
	std::vector<double> panAngle;
	std::vector<int> antiAliasing;
	std::vector<unsigned char> lensCorrectionEnabled;
	std::vector<double> lensCorrection;

	// Reading data
	int i = 1;
	while (i < argc - 1) 
	{
		if (strcmp(argv[i], "-w") == 0) 
		{
			i++;
			if ((perspwidth = atoi(argv[i++])) < 8) 
			{
				fprintf(stderr, "Bad output perspective image width, must be > 8\n");
				exit(-1);
			}
			perspwidth /= 2;
			perspwidth *= 2; // Even
		}
		else if (strcmp(argv[i], "-h") == 0) 
		{
			i++;
			if ((perspheight = atoi(argv[i++])) < 8) 
			{
				fprintf(stderr, "Bad output perspective image height, must be > 8\n");
				exit(-1);
			}
			perspheight /= 2;
			perspheight *= 2; // Even
		}
		else if (strcmp(argv[i], "-n") == 0)
		{
			i++;
			nOutputs = atoi(argv[i++]);
		}
		else if (strcmp(argv[i], "-s") == 0) 
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				double data;
				if ((data = atof(argv[i++])) > 360)
				{
					fprintf(stderr, "Maximum fisheye FOV is 360 degrees\n");
					data = 360;
				}
				fishFoV.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-t") == 0) {
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				double data;
				if ((data = atof(argv[i++])) > 170)
				{
					fprintf(stderr, "Maximum fisheye FOV is 360 degrees\n");
					data = 170;
				}
				perspFoV.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-x") == 0) 
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				double data = atof(argv[i++]);
				tiltAngle.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-y") == 0) 
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				double data = atof(argv[i++]);
				rollAngle.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-z") == 0)
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				double data = atof(argv[i++]);
				panAngle.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-a") == 0) 
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				int data = atoi(argv[i++]);
				antiAliasing.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-c") == 0) 
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				int data = atoi(argv[i++]);
				fishCenterX.push_back(data);
				data = atoi(argv[i++]);
				fishCenterY.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-r") == 0) 
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				int data = atoi(argv[i++]);
				fishRadius.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-ry") == 0) 
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				int data = atoi(argv[i++]);
				fishRadiusY.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-corr") == 0)
		{
			i++;
			for (int j = 0; j < nOutputs; ++j)
			{
				int data = atoi(argv[i++]);
				lensCorrectionEnabled.push_back(static_cast<bool>(data != 0));
			}
		}
		else if (strcmp(argv[i], "-p") == 0) 
		{
			i++;
			for (int j = 0; j < 4 * nOutputs; ++j)
			{
				double data = atof(argv[i++]);
				lensCorrection.push_back(data);
			}
		}
		else if (strcmp(argv[i], "-Video") == 0)
		{
			i++;
			isVideo = true;
		}
	}

	std::string fname(argv[argc - 1]);
	bool isCamera = isVideo;
	if (isCamera)
	{
		for (int i = 0; i < fname.size(); ++i)
		{
			if (!std::isdigit(fname[i]))
			{
				isCamera = false;
				break;
			}
		}
	}

	unsigned char * perspImgPtr = reinterpret_cast<unsigned char *>(
		malloc(nOutputs * perspwidth * perspheight * 3));

	if (!isVideo)
	{
		cv::Mat img = cv::imread(fname);
		unsigned char * imgPtr = img.data;
		int rc = img.rows;
		int cc = img.cols;
		int ch = img.channels();

		clock_t starttime = clock();

		int err = Fish2Persp_8U_64F_GPU
		(
			nOutputs,
			imgPtr, rc, cc, ch,
			perspwidth,
			perspheight,
			perspFoV.data(),
			fishFoV.data(),
			fishCenterX.data(),
			fishCenterY.data(),
			fishRadius.data(),
			fishRadiusY.data(),
			tiltAngle.data(),
			rollAngle.data(),
			panAngle.data(),
			antiAliasing.data(),
			reinterpret_cast<bool*>(lensCorrectionEnabled.data()),
			lensCorrection.data(),
			/* Output */
			perspImgPtr
		);

		clock_t endtime = clock();

		if (err != 0)
		{
			printf("Error Code: %d\n", err);
		}

		double t = (endtime - starttime);
		printf("Compute time: %f ms\n", t);

		int dim[3] = { perspwidth, perspheight, ch };

		for (int i = 0; i < nOutputs; ++i)
		{
			// Mat cv_image(3, dim, CV_8U, &perspImgPtr[i * perspwidth * perspheight * ch]);
			Mat cv_image(Size(perspwidth, perspheight), CV_8UC3, 
				&perspImgPtr[i * perspwidth * perspheight * ch], cv::Mat::AUTO_STEP);
			std::string label = "Output " + std::to_string(i);
			namedWindow("", WINDOW_NORMAL);
			imshow(label, cv_image);
		}

		cvWaitKey(0);
	}
	else
	{
		VideoCapture cap;
		if (isCamera)
		{
			cap.open(std::stoi(fname));
		}
		else
		{
			cap.open(fname);
		}

		if (!cap.isOpened()) 
		{
			printf("ERROR! Unable to open video\n");
			return -1;
		}

		cv::Mat img;
		cv::Mat imgr;

		int scWidth;
		int scHeight;
		GetScreenResolution(scWidth, scHeight);
		int fact = (scWidth - perspwidth) / nOutputs;

		int flag = CV_WINDOW_AUTOSIZE;
		namedWindow("ORIGINAL VIDEO", flag);
		moveWindow("ORIGINAL VIDEO", 0, 0);
		for (int i = 0; i < nOutputs; ++i)
		{
			std::string label = "Output " + std::to_string(i);
			namedWindow(label, flag);
			moveWindow(label, perspwidth + i * fact, 0);
		}

		int nframes = 0;

		clock_t starttimeall = clock();
		for (;;)
		{
			// wait for a new frame from camera and store it into 'frame'
			cap.read(img);

			if (img.empty()) 
			{
				break;
			}

			nframes++;

			unsigned char * imgPtr = img.data;
			int rc = img.rows;
			int cc = img.cols;
			int ch = img.channels();

			clock_t starttime = clock();

			int err = Fish2Persp_8U_64F_GPU
			(
				nOutputs,
				imgPtr, rc, cc, ch,
				perspwidth,
				perspheight,
				perspFoV.data(),
				fishFoV.data(),
				fishCenterX.data(),
				fishCenterY.data(),
				fishRadius.data(),
				fishRadiusY.data(),
				tiltAngle.data(),
				rollAngle.data(),
				panAngle.data(),
				antiAliasing.data(),
				reinterpret_cast<bool*>(lensCorrectionEnabled.data()),
				lensCorrection.data(),
				/* Output */
				perspImgPtr
			);

			clock_t endtime = clock();

			if (err != 0)
			{
				printf("Error Code: %d\n", err);
			}

			double t = (endtime - starttime);
			printf("Compute time: %f ms\n", t);

			int dim[3] = { perspwidth, perspheight, ch };

			resize(img, imgr, Size(perspwidth, perspheight));			
			imshow("ORIGINAL VIDEO", imgr);

			for (int i = 0; i < nOutputs; ++i)
			{
				Mat cv_image(Size(perspwidth, perspheight), CV_8UC3,
					&perspImgPtr[i * perspwidth * perspheight * ch], cv::Mat::AUTO_STEP);
				std::string label = "Output " + std::to_string(i);
				namedWindow(label, flag);
				imshow(label, cv_image);
			}

			cvWaitKey(1);
		}

		clock_t endtimeall = clock();
		double ttime = endtimeall - starttimeall;
		printf("Compute time all: %f ms\n", ttime);
		printf("FPS: %f\n", nframes / (ttime / 1000));
	}

	cvWaitKey(0);

	cvDestroyAllWindows();

	return 0;
}

