#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace samples;

void saveAndShowImage(Mat src, string name);

void morphology(Mat src, string name);
Mat findThin(Mat src, Mat open);
Mat findTooClose(Mat src, Mat closed);
template <int N> Mat close(Mat src, int kernel[N][N]);
template <int N> Mat open(Mat src, int kernel[N][N]);
template <int N> Mat dilation(Mat src, int kernel[N][N]);
template <int N> Mat erosion(Mat src, int kernel[N][N]);

const bool UNIFORM = true;
const bool ACCUMULATE = false;
const int RANGE_MIN = 0;
const int RANGE_MAX = 256;
const int RADIUS = 3;
const int HIST_W = 512;
const int HIST_H = 400;
const string INPUT_PATH = "images/original/";
const string OUTPUT_PATH = "images/output/";

int main() {
    Mat bug = imread(findFile(INPUT_PATH + "bug.bmp"), IMREAD_GRAYSCALE);
    Mat bug7 = imread(findFile(INPUT_PATH + "bug7.bmp"), IMREAD_GRAYSCALE);
    Mat laws_input5 = imread(findFile(INPUT_PATH + "laws_input5.bmp"), IMREAD_GRAYSCALE);
    Mat laws_texture = imread(findFile(INPUT_PATH + "laws_texture.bmp"), IMREAD_GRAYSCALE);
    Mat laws_texture5 = imread(findFile(INPUT_PATH + "laws_texture5.bmp"), IMREAD_GRAYSCALE);
    Mat pcb_hibas_8bpp = imread(findFile(INPUT_PATH + "pcb-hibas-8bpp.bmp"), IMREAD_GRAYSCALE);
    Mat pcb_hibas_8bpp7 = imread(findFile(INPUT_PATH + "pcb-hibas-8bpp7.bmp"), IMREAD_GRAYSCALE);

    morphology(pcb_hibas_8bpp, "pcb-hibas-8bpp");
    morphology(pcb_hibas_8bpp, "pcb-hibas-8bpp7");

    waitKey();

    return EXIT_SUCCESS;
}

void saveAndShowImage(Mat src, string name) {
    vector<int> compressionParams;
    compressionParams.push_back(IMWRITE_JPEG_QUALITY);
    compressionParams.push_back(90);
    bool result = false;

    try {
        result = imwrite(OUTPUT_PATH + name + ".jpg", src, compressionParams);
        imshow(name, src);
    }
    catch (const cv::Exception& ex) {
        fprintf(stderr, "Exception saving file: %s\n", ex.what());
    }

    if (result) { printf("File saved.\n"); }
    else { printf("ERROR: Can't save file.\n"); }

    compressionParams.pop_back();
    compressionParams.pop_back();
}

template <int N>
Mat dilation(Mat src, int kernel[N][N]) {
    Mat result(src.rows, src.cols, CV_8UC1);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            bool dilated = false;
            int offset = N / 2;

            for (int u = -offset; u < offset + 1 && !dilated; u++) {
                for (int v = -offset; v < offset + 1 && !dilated; v++) {
                    if (!((y + v < 0) || ((x + u) < 0) || ((y + v) > src.rows - 1) || ((x + u) > src.cols - 1))) {
                        if (kernel[v + offset][u + offset] == 1) {
                            int im = src.at<uchar>(y + v, x + u);
                            if (im == 0)
                                dilated = true;
                        }
                    }
                }
            }
            if (dilated) {
                result.at<uchar>(y, x) = 0;
            } else {
                result.at<uchar>(y, x) = 255;
            }
        }
    }
    return result;
}

template <int N>
Mat erosion(Mat src, int kernel[N][N]) {
    Mat result(src.rows, src.cols, CV_8UC1);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            bool eroded = false;
            int offset = N / 2;
            for (int u = -offset; u < offset + 1 && !eroded; u++) {
                for (int v = -offset; v < offset + 1 && !eroded; v++) {
                    if (!((y + v < 0) || ((x + u) < 0) || ((y + v) > src.rows - 1) || ((x + u) > src.cols - 1))) {
                        if (kernel[v + offset][u + offset] == 1) {
                            int im = src.at<uchar>(y + v, x + u);
                            if (im == 255)
                                eroded = true;
                        }
                    }
                }
            }
            if (eroded) {
                result.at<uchar>(y, x) = 255;
            } else {
                result.at<uchar>(y, x) = 0;
            }
        }
    }
    return result;
}

template <int N>
Mat open(Mat src, int kernel[N][N]) {
    return dilation<N>(erosion<N>(src, kernel), kernel);
}

template <int N>
Mat close(Mat src, int kernel[N][N]) {
    return erosion<N>(dilation<N>(src, kernel), kernel);
}

Mat findTooClose(Mat src, Mat closed) {
    Mat result(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double originalPixel = src.at<uchar>(i, j);
            double closedPixel = closed.at<uchar>(i, j);
            if (originalPixel != closedPixel)
                result.at<unsigned char>(i, j) = 255;
        }
    }
    return result;
}

Mat findThin(Mat src, Mat opened) {
    Mat result(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double originalPixel = src.at<uchar>(i, j);
            double openedPixel = opened.at<uchar>(i, j);
            if (originalPixel != openedPixel)
                result.at<unsigned char>(i, j) = 255;
        }
    }
    return result;
}

void morphology(Mat src, string name) {
    int kernel[3][3] = { 1, 1, 1,
                         1, 1, 1,
                         1, 1, 1 };

    Mat opened = open<3>(src, kernel);
    Mat closed = close<3>(src, kernel);
    Mat thin = findThin(src, opened);
    Mat tooClose = findTooClose(src, closed);
}