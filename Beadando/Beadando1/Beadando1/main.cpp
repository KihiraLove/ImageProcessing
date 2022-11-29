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

enum class stretch_type {
    linear,
    quadratic,
    squared
};

Mat linearStretch(Mat src, int max, int min);
Mat squaredStretch(Mat src, int max, int min);
Mat quadraticStretch(Mat src, int max, int min);
Mat equalize(Mat src, Mat hist, int N, int K);
Mat avgOrScatter(Mat src, int radius, Mat avg);
Mat wallis(Mat src, Mat avg, Mat variance, int sd, float r, int md, float amax);
Mat sobel(Mat src, bool rowWise);
Mat laplace4(Mat src);
Mat laplace8(Mat src);
Mat outlier(Mat src, int radius, int threshold);

void saveImage(Mat src, string name);
constexpr const char* enumToString(stretch_type e);
unsigned char pixel(Mat src, int c, int r);
int compare(const void* p1, const void* p2);

void histStretch(Mat src, stretch_type type, string name);
void histEq(Mat src, string name);
void convolution(Mat src, string name);
void wallisFilter(Mat src, string name, int sd, float r, int md, float amax);
void noiseFilters(Mat src, string name);

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
    Mat lena = imread(findFile(INPUT_PATH + "lena.bmp"), IMREAD_GRAYSCALE);
    Mat lena_vilagos = imread(findFile(INPUT_PATH + "lena_vilagos.bmp"), IMREAD_GRAYSCALE);
    Mat bridge = imread(findFile(INPUT_PATH + "bridge.bmp"), IMREAD_GRAYSCALE);
    Mat boat_sotet = imread(findFile(INPUT_PATH + "boat_sotet.bmp"), IMREAD_GRAYSCALE);
    Mat airplane = imread(findFile(INPUT_PATH + "airplane.bmp"), IMREAD_GRAYSCALE);
    Mat peppers_sotet = imread(findFile(INPUT_PATH + "peppers_sotet.bmp"), IMREAD_GRAYSCALE);
    Mat peppers_vilagos = imread(findFile(INPUT_PATH + "peppers_vilagos.bmp"), IMREAD_GRAYSCALE);
    Mat barbara_gauss = imread(findFile(INPUT_PATH + "0.025.bmp"), IMREAD_GRAYSCALE);
    Mat barbara_saltpepper = imread(findFile(INPUT_PATH + "0.1.bmp"), IMREAD_GRAYSCALE);
    Mat montage = imread(findFile(INPUT_PATH + "montage.jpg"), IMREAD_GRAYSCALE);
    Mat montage_zajos = imread(findFile(INPUT_PATH + "montage_zajos.jpg"), IMREAD_GRAYSCALE);

    saveImage(montage, "montage/montage");
    saveImage(montage_zajos, "montage_zajos/montage_zajos");

    //2.1
    histStretch(peppers_sotet, stretch_type::linear, "peppers_sotet/peppers_sotet");
    /*histStretch(boat_sotet, stretch_type::linear, "boat_sotet/boat_sotet");

    histStretch(peppers_sotet, stretch_type::quadratic, "peppers_sotet/peppers_sotet");
    histStretch(peppers_vilagos, stretch_type::quadratic, "peppers_vilagos/peppers_vilagos");
    histStretch(lena_vilagos, stretch_type::quadratic, "lena_vilagos/lena_vilagos");

    histStretch(peppers_sotet, stretch_type::squared, "peppers_sotet/peppers_sotet");
    histStretch(peppers_vilagos, stretch_type::squared, "peppers_vilagos/peppers_vilagos");
    histStretch(lena_vilagos, stretch_type::squared, "lena_vilagos/lena_vilagos");

    //2.2
    histEq(airplane, "airplane/airplane");
    histEq(lena_vilagos, "lena_vilagos/lena_vilagos");


    //2.3
    convolution(lena, "lena/lena");
    convolution(montage, "montage/montage");

    //2.4
    wallisFilter(montage, "montage/montage", 100, 2.5f, 128, 0.8f);
    wallisFilter(bridge, "bridge/bridge", 100, 2.5f, 128, 0.8f);

    wallisFilter(montage, "montage/montage", 100, 2.5f, 32, 0.8f);
    wallisFilter(bridge, "bridge/bridge", 100, 2.5f, 32, 0.8f);

    wallisFilter(montage, "montage/montage", 100, 2.5f, 256, 0.8f);
    wallisFilter(bridge, "bridge/bridge", 100, 2.5f, 256, 0.8f);

    wallisFilter(montage, "montage/montage", 50, 2.5f, 256, 0.8f);
    wallisFilter(bridge, "bridge/bridge", 50, 2.5f, 256, 0.8f);

    //2.5
    noiseFilters(montage_zajos, "montage_zajos/montage_zajos");
    noiseFilters(barbara_gauss, "0.025/0.025");
    noiseFilters(barbara_saltpepper, "0.1/0.1");*/

    waitKey();

    return EXIT_SUCCESS;
}

void saveImage(Mat src, string name) {
    vector<int> compressionParams;
    compressionParams.push_back(IMWRITE_JPEG_QUALITY);
    compressionParams.push_back(90);
    bool result = false;

    try {
        result = imwrite(OUTPUT_PATH + name + ".jpg", src, compressionParams);
    }
    catch (const cv::Exception& ex) {
        fprintf(stderr, "Exception saving file: %s\n", ex.what());
    }

    if (result) { printf("File saved.\n"); }
    else { printf("ERROR: Can't save file.\n"); }

    compressionParams.pop_back();
    compressionParams.pop_back();
}

constexpr const char* enumToString(stretch_type e) {
    switch (e) {
    case stretch_type::linear: return "linearis";
    case stretch_type::quadratic: return "negyzetes";
    case stretch_type::squared: return "gyokos";
    default: return "[ERROR]";
    }
}

unsigned char pixel(Mat src, int c, int r) {

    if (c < 0)
        c = 0;
    if (c >= src.cols)
        c = src.cols - 1;
    if (r < 0)
        r = 0;
    if (r >= src.rows)
        r = src.rows - 1;

    return src.at<unsigned char>(r, c);
}

int compare(const void* p1, const void* p2)
{
    return *(const unsigned char*)p1 - *(const unsigned char*)p2;
}

Mat linearStretch(Mat src, int max, int min) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            src.at<unsigned char>(i, j) = 255 * (src.at<unsigned char>(i, j) - min) / (max - min);
        }
    }
    return src;
}

Mat squaredStretch(Mat src, int max, int min) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            src.at<unsigned char>(i, j) = 255 * sqrtf((src.at<unsigned char>(i, j) - min) / (max - min * 1.0f));
        }
    }
    return src;
}

Mat quadraticStretch(Mat src, int max, int min) {
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            src.at<unsigned char>(i, j) = 255 * pow((src.at<unsigned char>(i, j) - min) / (max - min * 1.0f), 2);
        }
    }
    return src;
}

Mat equalize(Mat src, Mat hist, int N, int K) {
    int lookUpTable[RANGE_MAX];
    float sum = 0;
    int i = 0;
    for (int j = 0; j < RANGE_MAX; j++) {
        if (sum < N / K) {
            sum += hist.at<float>(j);
        }
        else {
            i++;
            sum = 0;
        }
        lookUpTable[j] = i * (float)RANGE_MAX / K;
    }
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            src.at<unsigned char>(i, j) = lookUpTable[src.at<unsigned char>(i, j)];
        }
    }
    return src;
}

Mat avgOrScatter(Mat src, int radius, Mat avg) {
    Mat result = Mat::zeros(src.rows, src.cols, src.type());

    for (int resultRow = 0; resultRow < result.rows; resultRow++) {
        for (int resultCol = 0; resultCol < result.cols; resultCol++) {
            float sum = 0;
            for (int srcRow = resultRow - radius; srcRow <= resultRow + radius; srcRow++) {
                for (int srcCol = resultCol - radius; srcCol <= resultCol + radius; srcCol++) {
                    if (!avg.empty()) {
                        sum += pow((pixel(src, srcCol, srcRow) - pixel(avg, resultCol, resultRow)), 2);
                    }
                    else {
                        sum += pixel(src, srcCol, srcRow);
                    }
                }
            }
            result.at<unsigned char>(resultRow, resultCol) = sum / (float)pow((2 * radius + 1), 2);
        }
    }

    return result;
}

Mat wallis(Mat src, Mat avg, Mat variance, int sd, float r, int md, float amax) {
    Mat wallis = Mat::zeros(src.rows, src.cols, src.type());

    for (int wallisRow = 0; wallisRow < wallis.rows; wallisRow++) {
        for (int wallisCol = 0; wallisCol < wallis.cols; wallisCol++) {
            unsigned char tmp = ((src.at<unsigned char>(wallisRow, wallisCol) - avg.at<unsigned char>(wallisRow, wallisCol)) * (((double)r * sd) / (sd + (r * sqrt(variance.at<unsigned char>(wallisRow, wallisCol)))))) + (((double)amax * md) + ((1.0f - amax) * avg.at<unsigned char>(wallisRow, wallisCol)));

            if (tmp > 255)
                tmp = 255;
            else if (tmp < 0)
                tmp = 0;

            wallis.at<unsigned char>(wallisRow, wallisCol) = tmp;
        }
    }

    return wallis;
}

Mat outlier(Mat src, int radius, int threshold) {
    Mat avg = Mat::zeros(src.rows, src.cols, src.type());
    Mat result = Mat::zeros(src.rows, src.cols, src.type());

    for (int avgRow = 0; avgRow < avg.rows; avgRow++) {
        for (int avgCol = 0; avgCol < avg.cols; avgCol++) {
            float sum = 0;

            for (int srcRow = avgRow - radius; srcRow <= avgRow + radius; srcRow++) {
                for (int srcCol = avgCol - radius; srcCol <= avgCol + radius; srcCol++) {
                    if (srcRow == avgRow && srcCol == avgCol) {
                        continue;
                    }

                    sum += pixel(src, srcCol, srcRow);
                }
            }

            avg.at<unsigned char>(avgRow, avgCol) = sum / (float)(pow((2 * radius + 1), 2) - 1);
        }
    }

    for (int resultRow = 0; resultRow < result.rows; resultRow++) {
        for (int resultCol = 0; resultCol < result.cols; resultCol++) {
            if (abs(avg.at<unsigned char>(resultRow, resultCol) - src.at<unsigned char>(resultRow, resultCol)) <= threshold)
                result.at<unsigned char>(resultRow, resultCol) = src.at<unsigned char>(resultRow, resultCol);
            else
                result.at<unsigned char>(resultRow, resultCol) = avg.at<unsigned char>(resultRow, resultCol);
        }
    }

    return result;
}

Mat sobel(Mat src, bool rowWise) {
    Mat result = src.clone();

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            unsigned char n = pixel(src, c, r - 1);
            unsigned char ne = pixel(src, c + 1, r - 1);
            unsigned char e = pixel(src, c + 1, r);
            unsigned char se = pixel(src, c + 1, r + 1);
            unsigned char s = pixel(src, c, r + 1);
            unsigned char sw = pixel(src, c - 1, r + 1);
            unsigned char w = pixel(src, c - 1, r);
            unsigned char nw = pixel(src, c - 1, r - 1);
            float tmp = cv::abs(0.25f * (nw + 2 * w + sw - ne - 2 * e - se));

            if (rowWise)
                tmp = cv::abs(0.25f * (-nw - 2 * n - ne + sw + 2 * s + se));

            result.at<unsigned char>(r, c) = tmp;
        }
    }

    return result;
}

Mat laplace4(Mat src) {
    Mat result = src.clone();

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            unsigned char center = pixel(src, c, r);
            unsigned char top = pixel(src, c, r - 1);
            unsigned char bottom = pixel(src, c, r + 1);
            unsigned char left = pixel(src, c - 1, r);
            unsigned char right = pixel(src, c + 1, r);

            result.at<unsigned char>(r, c) = cv::abs(left + right + top + bottom - 4 * center);
        }
    }

    return result;
}

Mat laplace8(Mat src) {
    Mat result = src.clone();

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            unsigned char center = pixel(src, c, r);
            unsigned char n = pixel(src, c, r - 1);
            unsigned char ne = pixel(src, c + 1, r - 1);
            unsigned char e = pixel(src, c + 1, r);
            unsigned char se = pixel(src, c + 1, r + 1);
            unsigned char s = pixel(src, c, r + 1);
            unsigned char sw = pixel(src, c - 1, r + 1);
            unsigned char w = pixel(src, c - 1, r);
            unsigned char nw = pixel(src, c - 1, r - 1);

            result.at<unsigned char>(r, c) = cv::abs(n + ne + e + se + s + sw + w + nw - 8 * center);
        }
    }

    return result;
}

void histStretch(Mat src, stretch_type type, string name) {
    int histSize = RANGE_MAX;
    float range[] = { RANGE_MIN, RANGE_MAX };
    const float* histRange = { range };
    int binW = cvRound((double)HIST_W / histSize);
    int max = 0;
    int min = 0;
    Mat stretchedImg;
    Mat hist;
    Mat histImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat stretchedHist;
    Mat stretchedHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<float> histArr;

    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    imshow(name, src);
    
    if (hist.isContinuous())
        histArr.assign((float*)hist.data, (float*)hist.data + hist.total() * hist.channels());
    else {
        for (int i = 0; i < hist.rows; ++i)
            histArr.insert(histArr.end(), hist.ptr<float>(i), hist.ptr<float>(i) + (long)hist.cols * hist.channels());
    }
    normalize(hist, hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(histImg, cv::Point(binW * (i - 1), HIST_H - cvRound(hist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(hist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " hisztogram", histImg);
    saveImage(histImg, name + "_alap_hisztogram");

    for (int i = 0; i < histArr.size(); i++) {
        if (histArr[i] != 0) {
            min = i;
            break;
        }
    }

    for (int i = histArr.size() - 1; i >= 0; i--) {
        if (histArr[i] != 0) {
            max = i;
            break;
        }
    }

    if (type == stretch_type::linear)
        stretchedImg = linearStretch(src.clone(), max, min);
    if (type == stretch_type::squared)
        stretchedImg = squaredStretch(src.clone(), max, min);
    if (type == stretch_type::quadratic)
        stretchedImg = quadraticStretch(src.clone(), max, min);

    imshow(name + enumToString(type) + " szethuzott kep", stretchedImg);
    saveImage(stretchedImg, name + "_" + enumToString(type) + "_szethuzott");

    calcHist(&stretchedImg, 1, 0, Mat(), stretchedHist, 1, &histSize, &histRange, true, false);
    normalize(stretchedHist, stretchedHist, 0, stretchedHistImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(stretchedHistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(stretchedHist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(hist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + enumToString(type) + " szethuzott hisztogram", stretchedHistImg);
    saveImage(stretchedHistImg, name + "_" + enumToString(type) + "_szethuzott_hisztogram");
}

void histEq(Mat src, string name) {
    int histSize = RANGE_MAX;
    float range[] = { RANGE_MIN, RANGE_MAX };
    const float* histRange = { range };
    int binW = cvRound((double)HIST_W / histSize);
    Mat hist;
    Mat histTable;
    Mat histImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat fourShadeEqImg;
    Mat fourShadeEqImgHist;
    Mat fourShadeEqImgHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat sixteenShadeEqImg;
    Mat sixteenShadeEqImgHist;
    Mat sixteenShadeEqImgHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
    histTable = hist.clone();
    normalize(hist, hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(histImg, cv::Point(binW * (i - 1), HIST_H - cvRound(hist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(hist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name, src);
    imshow(name + " hisztogram", histImg);
    saveImage(histImg, name + "_hisztogram");

    fourShadeEqImg = equalize(src.clone(), histTable, src.size().area(), 4);
    calcHist(&fourShadeEqImg, 1, 0, Mat(), fourShadeEqImgHist, 1, &histSize, &histRange, true, false);
    normalize(fourShadeEqImgHist, fourShadeEqImgHist, 0, fourShadeEqImgHistImg.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(fourShadeEqImgHistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(fourShadeEqImgHist.at<float>(i - 1))), cv::Point(binW * (i), HIST_H - cvRound(fourShadeEqImgHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " kiegyenlitett 4 arnyalat", fourShadeEqImg);
    saveImage(fourShadeEqImg, name + "_4_arnyalatos_kiegyenlitett");
    imshow(name + " kiegyenlitett hisztogram, 4 arnyalat", fourShadeEqImgHistImg);
    saveImage(fourShadeEqImgHistImg, name + "_4_arnyalatos_kiegyenlitett_hisztogram");

    sixteenShadeEqImg = equalize(src.clone(), histTable, src.size().area(), 16);
    calcHist(&sixteenShadeEqImg, 1, 0, Mat(), sixteenShadeEqImgHist, 1, &histSize, &histRange, true, false);
    normalize(sixteenShadeEqImgHist, sixteenShadeEqImgHist, 0, sixteenShadeEqImgHistImg.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(sixteenShadeEqImgHistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(sixteenShadeEqImgHist.at<float>(i - 1))), cv::Point(binW * (i), HIST_H - cvRound(sixteenShadeEqImgHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " kiegyenlitett 16 arnyalat", sixteenShadeEqImg);
    saveImage(sixteenShadeEqImg, name + "_16_arnyalatos_kiegyenlitett");
    imshow(name + " kiegyenlitett hisztogram, 16 arnyalat", sixteenShadeEqImgHistImg);
    saveImage(sixteenShadeEqImgHistImg, name + "_16_arnyalatos_kiegyenlitett_hisztogram");
}

void convolution(Mat src, string name) {
    int histSize = RANGE_MAX;
    float range[] = { RANGE_MIN, RANGE_MAX };
    const float* histRange = { range };
    int binW = cvRound((double)HIST_W / histSize);
    Mat hist;
    Mat histLUT;
    Mat histImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat sobelRowImg = sobel(src.clone(), true); //Sobel vizszintes
    Mat sobelRowHist;
    Mat sobelRowHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat sobelColImg = sobel(src.clone(), false); //Sobel fuggoleges
    Mat sobelColHist;
    Mat sobelColHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat sobelAddImg;
    Mat sobelAddHist;
    Mat sobelAddHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(hist, hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
    histLUT = hist.clone();

    for (int i = 1; i < histSize; i++)
        line(histImg, cv::Point(binW * (i - 1), HIST_H - cvRound(hist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(hist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name, src);
    imshow(name + " hisztogram", histImg);
    saveImage(histImg, name + "_hisztogram");

    calcHist(&sobelRowImg, 1, 0, Mat(), sobelRowHist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(sobelRowHist, sobelRowHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(sobelRowHistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(sobelRowHist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(sobelRowHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " sobel vizszintes", sobelRowImg);
    saveImage(sobelRowImg, name + "_sobel_vizszintes");
    imshow(name + " sobel vizszintes hisztogram", sobelRowHistImg);
    saveImage(sobelRowHistImg, name + "_sobel_vizszintes_hisztogram");

    calcHist(&sobelColImg, 1, 0, Mat(), sobelColHist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(sobelColHist, sobelColHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(sobelColHistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(sobelColHist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(sobelColHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " sobel fuggoleges", sobelColImg);
    saveImage(sobelColImg, name + "_sobel_fuggoleges");
    imshow(name + " sobel fuggoleges hisztogram", sobelColHistImg);
    saveImage(sobelColHistImg, name + "_sobel_fuggoleges_hisztogram");

    //Sobel osszeg
    sobelAddImg = sobelRowImg.clone();
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            sobelAddImg.at<unsigned char>(r, c) += sobelColImg.at<unsigned char>(r, c);
        }
    }

    calcHist(&sobelAddImg, 1, 0, Mat(), sobelAddHist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(sobelAddHist, sobelAddHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(sobelAddHistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(sobelAddHist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(sobelAddHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " sobel osszetett", sobelAddImg);
    saveImage(sobelAddImg, name + "_sobel_osszetett");
    imshow(name + " sobel osszetett hisztogram", sobelAddHistImg);
    saveImage(sobelAddHistImg, name + "_sobel_osszetett_hisztogram");

    //Laplace 4 szomszéd
    Mat laplace4Img = laplace4(src.clone());
    Mat laplace4Hist;
    Mat laplace4HistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&laplace4Img, 1, 0, Mat(), laplace4Hist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(laplace4Hist, laplace4Hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(laplace4HistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(laplace4Hist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(laplace4Hist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " laplace4", laplace4Img);
    saveImage(laplace4Img, name + "_laplace4");
    imshow(name + " laplace4 hisztogram", laplace4HistImg);
    saveImage(laplace4HistImg, name + "_laplace4_hisztogram");

    //Laplace 8 szomszéd

    Mat laplace8Img = laplace8(src.clone());
    Mat laplace8Hist;
    Mat laplace8HistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&laplace8Img, 1, 0, Mat(), laplace8Hist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(laplace8Hist, laplace8Hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(laplace8HistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(laplace8Hist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(laplace8Hist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " laplace8", laplace8Img);
    saveImage(laplace8Img, name + "_laplace8");
    imshow(name + " laplace8 hisztogram", laplace8HistImg);
    saveImage(laplace8HistImg, name + "_laplace8_hisztogram");

    //Laplace 4 szomszéd + Hist Eq

    Mat laplace4EqImg = laplace4(equalize(src.clone(), histLUT, src.size().area(), 16));
    Mat laplace4EqHist;
    Mat laplace4EqHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&laplace4EqImg, 1, 0, Mat(), laplace4EqHist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(laplace4EqHist, laplace4EqHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
        line(laplace4EqHistImg, cv::Point(binW * (i - 1), HIST_H - cvRound(laplace4EqHist.at<float>(i - 1))), Point(binW * (i), HIST_H - cvRound(laplace4EqHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow("Laplace4EQ", laplace4EqImg);
    saveImage(laplace4EqImg, name + "_laplace4_eq");
    imshow("Laplace4EQ hisztogram", laplace4EqHistImg);
    saveImage(laplace4EqHistImg, name + "_laplace4_eq_hisztogram");
}

void wallisFilter(Mat src, string name, int sd, float r, int md, float amax) {
    int histSize = RANGE_MAX;
    float range[] = { RANGE_MIN, RANGE_MAX };
    const float* histRange = { range };
    int colW = cvRound((double)HIST_W / histSize);
    Mat hist;
    Mat histImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat avgImg = avgOrScatter(src, 8, cv::Mat());
    Mat scatterImg = avgOrScatter(src, 8, avgImg);
    Mat wallisImg = wallis(src, avgImg, scatterImg, sd, r, md, amax);
    Mat wallisHist;
    Mat wallisHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(hist, hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(histImg, cv::Point(colW * (i - 1), HIST_H - cvRound(hist.at<float>(i - 1))), Point(colW * (i), HIST_H - cvRound(hist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name, src);
    imshow(name + " hisztogram", histImg);
    saveImage(histImg, name + "_hisztogram");

    calcHist(&wallisImg, 1, 0, Mat(), wallisHist, 1, &histSize, &histRange, true, false);
    normalize(wallisHist, wallisHist, 0, wallisHistImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(wallisHistImg, cv::Point(colW * (i - 1), HIST_H - cvRound(wallisHist.at<float>(i - 1))), Point(colW * (i), HIST_H - cvRound(wallisHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " wallis", wallisImg);
    saveImage(wallisImg, name + "_wallis" + "_md" + to_string(md) + "_sd" + to_string(sd));
    imshow(name + " wallis hisztogram", wallisHistImg);
    saveImage(wallisHistImg, name + "_wallis" + "_md" + to_string(md) + "_sd" + to_string(sd) + "_hisztogram");
}

void noiseFilters(Mat src, string name) {
    int histSize = RANGE_MAX;
    float range[] = { RANGE_MIN, RANGE_MAX };
    const float* histRange = { range };
    const int len = (2 * RADIUS + 1) * (2 * RADIUS + 1);
    const int lenFast = 2 * RADIUS + 1;
    int colW = cvRound((double)HIST_W / histSize);
    Mat hist;
    Mat histImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat outlierImg = outlier(src, RADIUS, 35);
    Mat outlierHist;
    Mat outlierHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat median = Mat::zeros(src.rows, src.cols, src.type());
    Mat medianHist;
    Mat medianHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));
    Mat fastMedian = Mat::zeros(src.rows, src.cols, src.type());
    Mat fastMedianHist;
    Mat fastMedianHistImg(HIST_H, HIST_W, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(hist, hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(histImg, cv::Point(colW * (i - 1), HIST_H - cvRound(hist.at<float>(i - 1))), Point(colW * (i), HIST_H - cvRound(hist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name, src);
    imshow(name + " hisztogram", histImg);
    saveImage(histImg, name + "_hisztogram");

    calcHist(&outlierImg, 1, 0, Mat(), outlierHist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(outlierHist, outlierHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(outlierHistImg, cv::Point(colW * (i - 1), HIST_H - cvRound(outlierHist.at<float>(i - 1))), Point(colW * (i), HIST_H - cvRound(outlierHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " outlier szurt", outlierImg);
    saveImage(outlierImg, name + "_outlier");
    imshow(name + " outlier szurt hisztogram", outlierHistImg);
    saveImage(outlierHistImg, name + "_outlier_hisztogram");

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            unsigned char pixels[len];
            int index = 0;

            for (int k = i - 3; k <= i + 3; k++) {
                for (int l = j - 3; l <= j + 3; l++) {
                    int k2 = k, l2 = l;

                    if (k < 0)
                        k2 = 0;
                    if (k >= src.rows)
                        k2 = src.rows - 1;
                    if (l < 0)
                        l2 = 0;
                    if (l >= src.cols)
                        l2 = src.cols - 1;
                    pixels[index] = src.at<unsigned char>(k2, l2);
                    index++;
                }
            }

            qsort(pixels, len, sizeof(unsigned char), compare);
            median.at<unsigned char>(i, j) = pixels[(len + 1) / 2];
        }
    }

    calcHist(&median, 1, 0, Mat(), medianHist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(medianHist, medianHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(medianHistImg, cv::Point(colW * (i - 1), HIST_H - cvRound(medianHist.at<float>(i - 1))), Point(colW * (i), HIST_H - cvRound(medianHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " median", median);
    saveImage(median, name + "_median");
    imshow(name + " median hisztogram", medianHistImg);
    saveImage(medianHistImg, name + "_median_hisztogram");

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            unsigned char pixels[lenFast];
            int index = 0;

            for (int k = i - RADIUS; k <= i + RADIUS; k++) {
                unsigned char pixels2[lenFast];
                int index2 = 0;

                for (int l = j - RADIUS; l <= j + RADIUS; l++) {
                    int k2 = k, l2 = l;

                    if (k < 0)
                        k2 = 0;
                    if (k >= src.rows)
                        k2 = src.rows - 1;
                    if (l < 0)
                        l2 = 0;
                    if (l >= src.cols)
                        l2 = src.cols - 1;

                    pixels2[index2] = src.at<unsigned char>(k2, l2);
                    index2++;
                }

                qsort(pixels2, lenFast, sizeof(unsigned char), compare);
                pixels[index] = pixels2[(lenFast + 1) / 2];
                index++;
            }

            qsort(pixels, lenFast, sizeof(unsigned char), compare);
            fastMedian.at<unsigned char>(i, j) = pixels[(lenFast + 1) / 2];
        }
    }

    calcHist(&fastMedian, 1, 0, Mat(), fastMedianHist, 1, &histSize, &histRange, UNIFORM, ACCUMULATE);
    normalize(fastMedianHist, fastMedianHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(fastMedianHistImg, cv::Point(colW * (i - 1), HIST_H - cvRound(fastMedianHist.at<float>(i - 1))), Point(colW * (i), HIST_H - cvRound(fastMedianHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);

    imshow(name + " gyors Median szurt", fastMedian);
    saveImage(fastMedian, name + "_gyors_median");
    imshow(name + " gyors Median szurt hisztogram", fastMedianHistImg);
    saveImage(fastMedianHistImg, name + "_gyors_median_hisztogram");
}