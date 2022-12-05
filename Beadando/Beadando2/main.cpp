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

enum direction {
    N,
    E,
    S,
    W
};

struct Bug {
    direction dir;
    int pointI;
    int pointJ;
};

void saveAndShowImage(Mat src, string name);

void morphology(Mat src, string name);
Mat findThin(Mat src, Mat open);
Mat findTooClose(Mat src, Mat closed);
template <int N> Mat close(Mat src, int kernel[N][N]);
template <int N> Mat open(Mat src, int kernel[N][N]);
template <int N> Mat dilation(Mat src, int kernel[N][N]);
template <int N> Mat erosion(Mat src, int kernel[N][N]);

void bugAlgo(Mat src, string name);
direction turnBack(direction dir);
direction turnRight(direction dir);
direction turnLeft(direction dir);
std::pair<int, int> findFirstPixel(Mat src);
bool isInObject(Bug bug, Mat src);
void moveBug(Bug& bug, Mat src);
Mat bugFollow(Mat src);
Mat backtrackBugFollow(Mat src);

void laws(Mat inputSrc, Mat textureSrc, string name);
Mat performConvolution(Mat src, Mat kernel);
Mat calculateEnergy(Mat src, int w = 15);

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
    Mat laws_input = imread(findFile(INPUT_PATH + "laws_input.bmp"), IMREAD_GRAYSCALE);
    Mat laws_input5 = imread(findFile(INPUT_PATH + "laws_input5.bmp"), IMREAD_GRAYSCALE);
    Mat laws_texture = imread(findFile(INPUT_PATH + "laws_texture.bmp"), IMREAD_GRAYSCALE);
    Mat laws_texture5 = imread(findFile(INPUT_PATH + "laws_texture5.bmp"), IMREAD_GRAYSCALE);
    Mat pcb_hibas_8bpp = imread(findFile(INPUT_PATH + "pcb-hibas-8bpp.bmp"), IMREAD_GRAYSCALE);
    Mat pcb_hibas_8bpp7 = imread(findFile(INPUT_PATH + "pcb-hibas-8bpp7.bmp"), IMREAD_GRAYSCALE);

    morphology(pcb_hibas_8bpp, "pcb-hibas-8bpp");
    morphology(pcb_hibas_8bpp, "pcb-hibas-8bpp7");

    bugAlgo(bug, "bug");
    bugAlgo(bug7, "bug7");
    waitKey();

    return EXIT_SUCCESS;
}


//Segéd függvény
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

//6. feladat
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
    Mat tooThin = findThin(src, opened);
    Mat tooClose = findTooClose(src, closed);

    saveAndShowImage(opened, "morph/" + name + "_opened");
    saveAndShowImage(closed, "morph/" + name + "_closed");
    saveAndShowImage(tooThin, "morph/" + name + "_too_thin");
    saveAndShowImage(tooClose, "morph/" + name + "_too_close");
}

//7. feladat
direction turnBack(direction d) {
    switch (d) {
        case N:
            return S;
            break;
        case W:
            return E;
            break;
        case S:
            return N;
            break;
        case E:
            return W;
            break;
        default:
            return N;
            break;
    }
}

direction turnRight(direction d) {
    switch (d) {
        case N:
            return E;
            break;
        case E:
            return S;
            break;
        case S:
            return W;
            break;
        case W:
            return N;
            break;
        default:
            return N;
            break;
    }
}

direction turnLeft(direction d) {
    switch (d) {
        case N:
            return W;
            break;
        case W:
            return S;
            break;
        case S:
            return E;
            break;
        case E:
            return N;
            break;
        default:
            return N;
            break;
    }
}

std::pair<int, int> findFirstPixel(Mat src) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int pixel = (int)src.at<unsigned char>(i, j);
            if (pixel == 255) {
                return std::pair<int, int>(i, j);
            }
        }
    }
    return std::pair<int, int>(-1, -1);
}

bool isInObject(Bug bug, Mat src) {
    return src.at<unsigned char>(bug.pointI, bug.pointJ) != 0;
}

void moveBug(Bug& bug, Mat src) {
    int height = src.rows;
    int width = src.cols;

    switch (bug.dir) {
        case direction::W:
            if (0 < bug.pointJ)
                bug.pointJ--;
            else
                bug.dir = turnLeft(bug.dir);
            break;
        case direction::E:
            if (width > bug.pointJ)
                bug.pointJ++;
            else
                bug.dir = turnRight(bug.dir);
            break;
        case direction::N:
            if (0 < bug.pointI)
                bug.pointI--;
            else
                bug.dir = turnRight(bug.dir);
            break;
        case direction::S:
            if (height > bug.pointI)
                bug.pointI++;
            else
                bug.dir = turnRight(bug.dir);
            break;
        default:
            break;
    }
}

Mat bugFollow(Mat src){
    Mat trackImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));

    std::pair<int, int> firstLocation = findFirstPixel(src);
    int firstPointI = firstLocation.first;
    int firstPointJ = firstLocation.second - 1;

    Bug bug;
    bug.pointI = firstPointI;
    bug.pointJ = firstPointJ;
    bug.dir = direction::N;

    do {
        int pixel = src.at<uchar>(bug.pointI, bug.pointJ);
        if (pixel == 255) {
            bug.dir = turnLeft(bug.dir);
            trackImg.at<uchar>(bug.pointI, bug.pointJ) = 255;
        }
        else
            bug.dir = turnRight(bug.dir);
        moveBug(bug, src);
    } while (!(bug.pointI == firstPointI && bug.pointJ == firstPointJ && bug.dir == direction::N));
    return trackImg;
}

Mat backtrackBugFollow(Mat src) {
    Mat trackImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));

    std::pair<int, int> firstLocation = findFirstPixel(src);
    int firstPointI = firstLocation.first;
    int firstPointJ = firstLocation.second - 1;

    Bug bug;
    bug.pointI = firstPointI;
    bug.pointJ = firstPointJ;
    bug.dir = direction::N;

    do {
        int pixel = src.at<uchar>(bug.pointI, bug.pointJ);
        if (pixel == 255) {
            bug.dir = turnBack(bug.dir);
            trackImg.at<uchar>(bug.pointI, bug.pointJ) = 255;
        }
        else
            bug.dir = turnRight(bug.dir);
        moveBug(bug, src);
    } while (!(bug.pointI == firstPointI && bug.pointJ == firstPointJ && bug.dir == direction::N));
    return trackImg;
}

void bugAlgo(Mat src, string name) {
    Mat bugFollowImage = bugFollow(src);
    Mat bugBacktrackImage = backtrackBugFollow(src);

    saveAndShowImage(bugFollowImage, "bug/" + name + "_bug_follow");
    saveAndShowImage(bugBacktrackImage, "bug/" + name + "_bug_backtrack");
}

//8. feladat

Mat performConvolution(Mat src, Mat kernel) {
    Mat convImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    int kernel_r = 1;
    for (int i = 0; i < src.rows - (2 * kernel_r); i++) {
        for (int j = 0; j < src.cols - (2 * kernel_r); j++) {
            double value = 0;
            for (int k = 0; k < 2 * kernel_r + 1; k++) {
                for (int l = 0; l < 2 * kernel_r + 1; l++) {
                    value += src.at<unsigned char>(i + k, j + l) * kernel.at<double>(k, l);
                }
            }
            convImg.at<unsigned char>(i + kernel_r, j + kernel_r) = 3 * abs(value);
        }
    }
    return convImg;
}

Mat calculateEnergy(Mat src, int w = 15) {
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

void laws(Mat inputSrc, Mat textureSrc, string name) {
    Mat path(textureSrc.rows, textureSrc.cols, CV_8UC1, Scalar(0, 0, 0));

    Mat l1 = (Mat_<double>(3, 3) << 1.0 / 36.0, 2.0 / 36.0, 1.0 / 36.0,
        2.0 / 36.0, 4.0 / 36.0, 2.0 / 36.0,
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

    Mat sample1 = textureSrc(Rect(63 - 15, 49 - 15, 30, 30));
    //Mat sample2 = src(Rect(63 - 15, 149 - 15, 30, 30));
    //Mat sample3 = src(Rect(191 - 15, 49 - 15, 30, 30));
    //Mat sample4 = src(Rect(191 - 15, 149 - 15, 30, 30));

    Mat sample1Conv = performConvolution(textureSrc, l1);
    Mat sample2Conv = performConvolution(textureSrc, l2);
    Mat sample3Conv = performConvolution(textureSrc, l3);
    Mat sample4Conv = performConvolution(textureSrc, l4);
    Mat sample5Conv = performConvolution(textureSrc, l5);
    Mat sample6Conv = performConvolution(textureSrc, l6);
    Mat sample7Conv = performConvolution(textureSrc, l7);
    Mat sample8Conv = performConvolution(textureSrc, l8);
    Mat sample9Conv = performConvolution(textureSrc, l9);


    Mat sample1Energy = calculateEnergy(sample1Conv);
    Mat sample2Energy = calculateEnergy(sample2Conv);
    Mat sample3Energy = calculateEnergy(sample3Conv);
    Mat sample4Energy = calculateEnergy(sample4Conv);
    Mat sample5Energy = calculateEnergy(sample5Conv);
    Mat sample6Energy = calculateEnergy(sample6Conv);
    Mat sample7Energy = calculateEnergy(sample7Conv);
    Mat sample8Energy = calculateEnergy(sample8Conv);
    Mat sample9Energy = calculateEnergy(sample9Conv);

    Mat inputConv1 = performConvolution(inputSrc, l1);
    Mat inputConv2 = performConvolution(inputSrc, l2);
    Mat inputConv3 = performConvolution(inputSrc, l3);
    Mat inputConv4 = performConvolution(inputSrc, l4);
    Mat inputConv5 = performConvolution(inputSrc, l5);
    Mat inputConv6 = performConvolution(inputSrc, l6);
    Mat inputConv7 = performConvolution(inputSrc, l7);
    Mat inputConv8 = performConvolution(inputSrc, l8);
    Mat inputConv9 = performConvolution(inputSrc, l9);

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
}