#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;

Mat performConvolution(Mat& src, Mat kernel) {
    Mat convImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    int kernel_r = 1;
    for (int i = 0; i < src.rows - (2 * kernel_r); i++)
    {
        for (int j = 0; j < src.cols - (2 * kernel_r); j++)
        {
            double value = 0;
            for (int k = 0; k < 2 * kernel_r + 1; k++) {
                for (int l = 0; l < 2 * kernel_r + 1; l++) {
                    value += src.at<unsigned char>(i + k, j + l) * kernel.at<double>(k,l);

                }
            }
            convImg.at<unsigned char>(i + kernel_r, j + kernel_r) = 3 * abs(value);
        }
    }
    return convImg;
}

Mat calculateEnergy(Mat& src, int w = 15) {
    Mat energImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    double weight = 1 / pow(2 * w + 1, 2);
    for (int i = w; i < src.rows - w; i++) {
        for (int j = w; j < src.cols - w; j++) {
            double value = 0;
            for (int m = i - w; m < i + w; m++) {
                for (int n = j - w; n < j + w; n++) {
                    value += abs((double)src.at<uchar>(m, n));
                }
            }
            energImg.at<uchar>(i, j) = weight * value;
        }
    }
    return energImg;
}


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, "{@input | laws_texture.bmp | input image}" 
        "{@input2 | laws_input.bmp | input image}");
    Mat src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_GRAYSCALE);
    Mat srcInput = imread(samples::findFile(parser.get<String>("@input2")), IMREAD_GRAYSCALE);
    Mat path(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    if (src.empty())
    {
        return EXIT_FAILURE;
    }

    Mat l1 = (Mat_<double>(3, 3) << 1.0 / 36.0, 2.0/36.0, 1.0 / 36.0, 
        2.0/36.0, 4.0/ 36.0, 2.0 / 36.0, 
        1.0 / 36.0, 2.0 / 36.0, 1.0 / 36.0);

    Mat l2 = (Mat_<double>(3, 3) << 1.0 / 12.0, 0, -1.0 / 12.0,
        2.0 / 12.0, 0, -2.0 / 12.0,
        1.0 / 12.0, 0, -1.0 / 12.0);

    Mat l3 = (Mat_<double>(3, 3) << -1.0 / 12.0, 2.0 / 12.0, -1.0 / 12.0,
        -2.0 / 12.0, 4.0 / 12.0, -2.0 / 12.0,
        -1.0 / 12.0, 2.0 / 12.0, -1.0 / 12.0);

    Mat l4 = (Mat_<double>(3, 3) << -1.0 / 12.0, -2.0 / 12.0, -1.0 / 12.0,
        0.0, 0.0, 0.0,
        1.0 / 12.0, 2.0 / 12.0, 1.0 / 12.0);

    Mat l5 = (Mat_<double>(3, 3) << 1.0 / 4.0, 0, -1.0 / 4.0,
        0, 0, 0,
        -1.0 / 4.0, 0, 1.0 / 4.0);

    Mat l6 = (Mat_<double>(3, 3) << -1.0 / 4.0, 2.0 / 4.0, -1.0 / 4.0,
        0, 0, 0,
        1.0 / 4.0, -2.0 / 4.0, 1.0 / 4.0);

    Mat l7 = (Mat_<double>(3, 3) << -1.0 / 12.0, -2.0 / 12.0, -1.0 / 12.0,
        2.0 / 12.0, 4.0 / 12.0, 2.0 / 12.0,
        -1.0 / 12.0, -2.0 / 12.0, -1.0 / 12.0);

    Mat l8 = (Mat_<double>(3, 3) << -1.0 / 4.0, 0, 1.0 / 4.0,
        2.0 / 4.0, 0, -2.0 / 4.0,
        -1.0 / 4.0, 0, 1.0 / 4.0);

    Mat l9 = (Mat_<double>(3, 3) << 1.0 / 4.0, -2.0 / 4.0, 1.0 / 4.0,
        -2.0 / 4.0, 1.0, -2.0 / 4.0,
        1.0 / 4.0, -2.0 / 4.0, 1.0 / 4.0);

    Mat sample1 = src(Rect(63 - 15, 49 - 15, 30, 30));
    //Mat sample2 = src(Rect(63 - 15, 149 - 15, 30, 30));
    //Mat sample3 = src(Rect(191 - 15, 49 - 15, 30, 30));
    //Mat sample4 = src(Rect(191 - 15, 149 - 15, 30, 30));

    Mat sample1Conv = performConvolution(src, l1);
    Mat sample2Conv = performConvolution(src, l2);
    Mat sample3Conv = performConvolution(src, l3);
    Mat sample4Conv = performConvolution(src, l4);
    Mat sample5Conv = performConvolution(src, l5);
    Mat sample6Conv = performConvolution(src, l6);
    Mat sample7Conv = performConvolution(src, l7);
    Mat sample8Conv = performConvolution(src, l8);
    Mat sample9Conv = performConvolution(src, l9);


    Mat sample1Energy = calculateEnergy(sample1Conv);
    Mat sample2Energy = calculateEnergy(sample2Conv);
    Mat sample3Energy = calculateEnergy(sample3Conv);
    Mat sample4Energy = calculateEnergy(sample4Conv);
    Mat sample5Energy = calculateEnergy(sample5Conv);
    Mat sample6Energy = calculateEnergy(sample6Conv);
    Mat sample7Energy = calculateEnergy(sample7Conv);
    Mat sample8Energy = calculateEnergy(sample8Conv);
    Mat sample9Energy = calculateEnergy(sample9Conv);

    Mat inputConv1 = performConvolution(srcInput, l1);
    Mat inputConv2 = performConvolution(srcInput, l2);
    Mat inputConv3 = performConvolution(srcInput, l3);
    Mat inputConv4 = performConvolution(srcInput, l4);
    Mat inputConv5 = performConvolution(srcInput, l5);
    Mat inputConv6 = performConvolution(srcInput, l6);
    Mat inputConv7 = performConvolution(srcInput, l7);
    Mat inputConv8 = performConvolution(srcInput, l8);
    Mat inputConv9 = performConvolution(srcInput, l9);

    Mat inputEnergy1 = calculateEnergy(inputConv1);
    Mat inputEnergy2 = calculateEnergy(inputConv2);
    Mat inputEnergy3 = calculateEnergy(inputConv3);
    Mat inputEnergy4 = calculateEnergy(inputConv4);
    Mat inputEnergy5 = calculateEnergy(inputConv5);
    Mat inputEnergy6 = calculateEnergy(inputConv6);
    Mat inputEnergy7 = calculateEnergy(inputConv7);
    Mat inputEnergy8 = calculateEnergy(inputConv8);
    Mat inputEnergy9 = calculateEnergy(inputConv9);

    imshow("Tanito", src);
    imshow("Input", srcInput);
    waitKey();
    return EXIT_SUCCESS;
}