#include <iostream>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


template <int N>
Mat dilation(Mat src, int kernel[N][N])
{
    Mat tmp(src.rows, src.cols, CV_8UC1);

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {

            bool dilat = false;
            int offset = N / 2;
            for (int u = -offset; u < offset + 1 && !dilat; u++)
            {
                for (int v = -offset; v < offset + 1 && !dilat; v++)
                {
                    if (!((y + v < 0) || ((x + u) < 0) || ((y + v) > src.rows - 1) || ((x + u) > src.cols - 1)))
                    {
                        if (kernel[v + offset][u + offset] == 1)
                        {
                            int im = src.at<uchar>(y + v, x + u);
                            if (im == 0)
                            {
                                dilat = true;
                            }
                        }
                    }
                }
            }
            if (dilat)
            {
                tmp.at<uchar>(y, x) = 0;
            }
            else
            {
                tmp.at<uchar>(y, x) = 255;
            }
        }
    }

    return tmp;
}

template <int N>
Mat erosion(Mat src, int kernel[N][N])
{
    Mat tmp(src.rows, src.cols, CV_8UC1);

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            bool eros = false;
            int offset = N / 2;
            for (int u = -offset; u < offset + 1 && !eros; u++)
            {
                for (int v = -offset; v < offset + 1 && !eros; v++)
                {
                    if (!((y + v < 0) || ((x + u) < 0) || ((y + v) > src.rows - 1) || ((x + u) > src.cols - 1)))
                    {
                        if (kernel[v + offset][u + offset] == 1)
                        {
                            int im = src.at<uchar>(y + v, x + u);
                            if (im == 255)
                            {
                                eros = true;
                            }
                        }
                    }
                }
            }
            if (eros)
            {
                tmp.at<uchar>(y, x) = 255;
            }
            else
            {
                tmp.at<uchar>(y, x) = 0;
            }
        }
    }

    return tmp;
}

template <int N>
Mat nyitas(Mat src, int kernel[N][N])
{
    imshow("Nyitas", dilation<N>(erosion<N>(src, kernel), kernel));

    return dilation<N>(erosion<N>(src, kernel), kernel);
}

template <int N>
Mat zaras(Mat src, int kernel[N][N])
{
    imshow("Zaras", erosion<N>(dilation<N>(src, kernel), kernel));

    return erosion<N>(dilation<N>(src, kernel), kernel);
}

void findTulkozeli(Mat& src, Mat& zart)
{
    Mat newImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double pixelFirstImage = src.at<uchar>(i, j);
            double pixelSecondImage = zart.at<uchar>(i, j);
            if (pixelFirstImage != pixelSecondImage) newImg.at<unsigned char>(i, j) = 255;
        }
    }
    imshow("Vastag reszek", newImg);
}

void findVekony(Mat& src, Mat& nyitott)
{
    Mat newImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double pixelFirstImage = src.at<uchar>(i, j);
            double pixelSecondImage = nyitott.at<uchar>(i, j);
            if (pixelFirstImage != pixelSecondImage) newImg.at<unsigned char>(i, j) = 255;
        }
    }
    imshow("Vekony reszek", newImg);
}

int main(int argc, char** argv) {
    Mat src = imread("./pcb-hibas-8bpp3.bmp", IMREAD_GRAYSCALE);
    //Mat src = imread("./pcb-hibas-8bpp.bmp", IMREAD_GRAYSCALE);

    int kernel[3][3] =
    {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1 };

    imshow("Dilate", dilation<3>(src, kernel));

    imshow("Erose image", erosion<3>(src, kernel));

    //Nyitás és zárás morfológiai művelet
    Mat nyitott = nyitas<3>(src, kernel);

    Mat zart = zaras<3>(src, kernel);

    findVekony(src, nyitott);
    findTulkozeli(src, zart);
    waitKey();
    return EXIT_SUCCESS;
}