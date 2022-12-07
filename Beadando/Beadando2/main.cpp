#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
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

void bugFollowAlgorithms(Mat src, string name);
direction turnBack(direction dir, string* code);
direction turnRight(direction dir, string* code);
direction turnLeft(direction dir, string* code);
std::pair<int, int> findFirstPixel(Mat src);
bool isInObject(Bug bug, Mat src);
void moveBug(Bug& bug, Mat src, string* code);
Mat bugFollow(Mat src, string* code);
Mat backtrackBugFollow(Mat src, string* code);

void laws(Mat src, Mat texture, string name);
float sampleTexture(Mat src, Mat kernel, int centerY, int centerX);
vector<float> calculateEnergy(Mat texture, int centerY, int centerX);
Mat performConvolution(Mat src, Mat kernel);
Mat getEnergy(Mat src, Mat kernel);

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

    bugFollowAlgorithms(bug, "bug");
    bugFollowAlgorithms(bug7, "bug7");

    laws(laws_input, laws_texture, "laws_input");
    laws(laws_input5, laws_texture5, "laws_input5");
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
direction turnBack(direction d, string* code) {
    switch (d) {
        case N:
            *code = *code + "S";
            return S;
        case W:
            *code = *code + "E";
            return E;
        case S:
            *code = *code + "N";
            return N;
        case E:
            *code = *code + "W";
            return W;
        default:
            *code = *code + "N";
            return N;
    }
}

direction turnRight(direction d, string* code) {
    switch (d) {
        case N:
            *code = *code + "E";
            return E;
        case E:
            *code = *code + "S";
            return S;
        case S:
            *code = *code + "W";
            return W;
        case W:
            *code = *code + "N";
            return N;
        default:
            *code = *code + "N";
            return N;
    }
}

direction turnLeft(direction d, string* code) {
    switch (d) {
        case N:
            *code = *code + "W";
            return W;
        case W:
            *code = *code + "S";
            return S;
        case S:
            *code = *code + "E";
            return E;
        case E:
            *code = *code + "N";
            return N;
        default:
            *code = *code + "N";
            return N;
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

void moveBug(Bug& bug, Mat src, string* code) {
    int height = src.rows;
    int width = src.cols;

    switch (bug.dir) {
        case direction::W:
            if (0 < bug.pointJ)
                bug.pointJ--;
            else
                bug.dir = turnLeft(bug.dir, code);
            break;
        case direction::E:
            if (width > bug.pointJ)
                bug.pointJ++;
            else
                bug.dir = turnRight(bug.dir, code);
            break;
        case direction::N:
            if (0 < bug.pointI)
                bug.pointI--;
            else
                bug.dir = turnRight(bug.dir, code);
            break;
        case direction::S:
            if (height > bug.pointI)
                bug.pointI++;
            else
                bug.dir = turnRight(bug.dir, code);
            break;
        default:
            break;
    }
}

Mat bugFollow(Mat src, string* code) {
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
            bug.dir = turnLeft(bug.dir, code);
            trackImg.at<uchar>(bug.pointI, bug.pointJ) = 255;
        }
        else
            bug.dir = turnRight(bug.dir, code);
        moveBug(bug, src, code);
    } while (!(bug.pointI == firstPointI && bug.pointJ == firstPointJ && bug.dir == direction::N));
    return trackImg;
}

Mat backtrackBugFollow(Mat src, string* code) {
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
            bug.dir = turnBack(bug.dir, code);
            trackImg.at<uchar>(bug.pointI, bug.pointJ) = 255;
        }
        else
            bug.dir = turnRight(bug.dir, code);
        moveBug(bug, src, code);
    } while (!(bug.pointI == firstPointI && bug.pointJ == firstPointJ && bug.dir == direction::N));
    return trackImg;
}

void bugFollowAlgorithms(Mat src, string name) {
    string bugFollowCode = "";
    string bugBacktrckCode = "";
    Mat bugFollowImage = bugFollow(src, &bugFollowCode);
    Mat bugBacktrackImage = backtrackBugFollow(src, &bugBacktrckCode);

    saveAndShowImage(bugFollowImage, "bug/" + name + "_bug_follow");
    saveAndShowImage(bugBacktrackImage, "bug/" + name + "_bug_backtrack");

    ofstream bugFollowFile("images/output/bug/bugFollowCode.txt");
    bugFollowFile << bugFollowCode;
    cout << bugFollowCode << endl;
    ofstream bugBacktrackFile("images/output/bug/bugBacktrackCode.txt");
    bugBacktrackFile << bugBacktrckCode;
    bugFollowFile.close();
    bugBacktrackFile.close();
}

//8. feladat
Mat performConvolution(Mat src, Mat kernel){
    Mat result = src.clone();
    float sum = 0;
    int radiusRows = (kernel.rows - 1) / 2;
    int radiusCols = (kernel.cols - 1) / 2;

    Mat dst(Size(src.cols * 3, src.rows * 3), src.type());
    Mat roi;

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            roi = dst(Rect(src.cols * x, src.rows * y, src.cols, src.rows));
            src.copyTo(roi);
        }
    }

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int k = 0; k < kernel.rows; k++) {
                for (int l = 0; l < kernel.cols; l++) {
                    sum += (dst.at<uchar>(y - radiusRows + k + src.rows, x - radiusCols + l + src.cols)) * kernel.at<float>(k, l);
                }
            }
            result.at<uchar>(y, x) = saturate_cast<uchar>(sum);
            sum = 0;
        }

    }
    return result;
}

float sampleTexture(Mat src, Mat kernel, int centerY, int centerX) {
    float result = 0;
    int radiusRows = (kernel.rows - 1) / 2;
    int radiusCols = (kernel.cols - 1) / 2;

    Mat dst(Size(src.cols * 3, src.rows * 3), src.type());
    Mat roi;

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            roi = dst(Rect(src.cols * x, src.rows * y, src.cols, src.rows));
            src.copyTo(roi);
        }
    }

    for (int k = 0; k < kernel.rows; k++) {
        for (int l = 0; l < kernel.cols; l++) {
            result += (float)abs((dst.at<uchar>(centerY - radiusRows + k + src.rows, centerX - radiusCols + l + src.cols)));
        }
    }
    result = result / (kernel.rows * kernel.rows);
    return result;
}

vector<float> calculateEnergy(Mat texture, int centerY, int centerX) {
    vector<float> result;
    Mat sample = Mat::ones(31, 31, CV_32F) / (31 * 31);

    vector<Mat> kernels{
        (Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 36,
        (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1) / 12,
        (Mat_<float>(3, 3) << -1, 2, -1, -2, 4, -2, -1, 2, -1) / 12,
        (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1) / 12,
        (Mat_<float>(3, 3) << 1, 0, -1, 0, 0, 0, -1, 0, 1) / 4,
        (Mat_<float>(3, 3) << -1, 2, -1, 0, 0, 0, 1, -2, 1) / 4,
        (Mat_<float>(3, 3) << -1, -2, -1, 2, 4, 2, -1, -2, -1) / 12,
        (Mat_<float>(3, 3) << -1, 0, 1, 2, 0, -2, -1, 0, 1) / 4,
        (Mat_<float>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1) / 4 };

    for (Mat i : kernels) {
        result.push_back(sampleTexture(performConvolution(texture, i), sample, centerY, centerX));
    }

    return result;
}

Mat getEnergy(Mat src, Mat kernel) {
    Mat result(src.cols, src.rows, CV_32F);
    float sum = 0;
    int radiusRows = (kernel.rows - 1) / 2;
    int radiusCols = (kernel.cols - 1) / 2;

    Mat dst(Size(src.cols * 3, src.rows * 3), src.type());
    Mat roi;

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            roi = dst(Rect(src.cols * x, src.rows * y, src.cols, src.rows));
            src.copyTo(roi);
        }
    }

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int k = 0; k < kernel.rows; k++) {
                for (int l = 0; l < kernel.cols; l++) {
                    sum += abs((dst.at<uchar>(y - radiusRows + k + src.rows, x - radiusCols + l + src.cols))); //TODO
                }
            }
            result.at<float>(y, x) = sum / (kernel.cols * kernel.cols);
            sum = 0;
        }

    }
    return result;
}

void laws(Mat src, Mat texture, string name) {
    vector<Mat> src_conv;
    vector<Mat> src_energies;

    Mat laws_filtered = src.clone();

    vector<Mat> kernels {
        (Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 36,
        (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1) / 12,
        (Mat_<float>(3, 3) << -1, 2, -1, -2, 4, -2, -1, 2, -1) / 12,
        (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1) / 12,
        (Mat_<float>(3, 3) << 1, 0, -1, 0, 0, 0, -1, 0, 1) / 4,
        (Mat_<float>(3, 3) << -1, 2, -1, 0, 0, 0, 1, -2, 1) / 4,
        (Mat_<float>(3, 3) << -1, -2, -1, 2, 4, 2, -1, -2, -1) / 12,
        (Mat_<float>(3, 3) << -1, 0, 1, 2, 0, -2, -1, 0, 1) / 4,
        (Mat_<float>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1) / 4 };

    vector<vector<float>> textures{
        calculateEnergy(texture,49,63),
        calculateEnergy(texture,149,63),
        calculateEnergy(texture,49,191),
        calculateEnergy(texture,149,191)
    };

    Mat sample = Mat::ones(31, 31, CV_32F);

    for (Mat i : kernels) {
        src_conv.push_back(performConvolution(src, i));
    }

    for (Mat i : src_conv) {
        src_energies.push_back(getEnergy(i, sample));
    }

    float difference;
    int index = 0, sum = 0;
    vector<float> d;

    for (int y = 0; y < laws_filtered.rows; y++) {
        for (int x = 0; x < laws_filtered.cols; x++) {
            for (vector<float> i : textures) {
                for (int z = 0; z < kernels.size(); z++) {
                    sum += abs(i.at(z) - src_energies.at(z).at<float>(y, x));
                }
                d.push_back(sum);
                sum = 0;
            }
            index = std::min_element(d.begin(), d.end()) - d.begin();
            laws_filtered.at<uchar>(y, x) = 255 - index * 70;
            d.clear();
        }
    }

    saveAndShowImage(laws_filtered, "laws/" + name + "_laws_filtered");
}
