#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

const int RANGEMIN = 0;
const int RANGEMAX = 256;
const int RADIUS = 3;

Mat Szethuzas(Mat kep, int max, int min, short int type) {
    //type 1:sima, 2: gyokos, 3: negyzetes
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            if (type == 1) {
                kep.at<unsigned char>(i, j) = 255 * (kep.at<unsigned char>(i, j) - min) / (max - min);
            }
            else if (type == 2) {
                kep.at<unsigned char>(i, j) = 255 * sqrtf((kep.at<unsigned char>(i, j) - min) / (max - min * 1.0f));
            }
            else {
                kep.at<unsigned char>(i, j) = 255 * pow((kep.at<unsigned char>(i, j) - min) / (max - min * 1.0f), 2);
            }
        }
    }
    return kep;
}

void hisztogramSzethuzas(Mat kep) {
    int histogramSize = RANGEMAX;
    float range[] = { RANGEMIN, RANGEMAX };
    const float* histogramRange = { range };
    bool uniform = true;
    bool accumulate = false;

    Mat hist;
    calcHist(&kep, 1, 0, Mat(), hist, 1, &histogramSize, &histogramRange, uniform, accumulate);

    imshow("Alap Kep", kep);

    std::vector<float> histogramArray;
    if (hist.isContinuous()) {
        histogramArray.assign((float*)hist.data, (float*)hist.data + hist.total() * hist.channels());
    }
    else {
        for (int i = 0; i < hist.rows; ++i) {
            histogramArray.insert(histogramArray.end(), hist.ptr<float>(i), hist.ptr<float>(i) + hist.cols * hist.channels());
        }
    }

    int histogramWidth = 512;
    int histogramHeight = 400;
    int stretchRate = cvRound((double)histogramWidth / histogramSize);

    Mat histogramImage(histogramHeight, histogramWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(hist, hist, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(histogramImage, cv::Point(stretchRate * (i - 1), histogramHeight - cvRound(hist.at<float>(i - 1))),
            Point(stretchRate * (i), histogramHeight - cvRound(hist.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Alap Hisztogramm", histogramImage);

    int max, min;

    for (int i = 0; i < histogramArray.size(); i++) {
        if (histogramArray[i] != 0) {
            min = i;
        }
    }

    for (int i = histogramArray.size() - 1; i >= 0; i--) {
        if (histogramArray[i] != 0) {
            max = i;
        }
    }

    //Mat szurkeKepSzethuzas = Szethuzas(kep.clone(), max, min, 1);
    //Mat szurkeKepSzethuzas = Szethuzas(kep.clone(), max, min, 2);
    Mat szurkeKepSzethuzas = Szethuzas(kep.clone(), max, min, 3);

    imshow("Szethuzott kep", szurkeKepSzethuzas);

    Mat stretchedHistogram;
    calcHist(&szurkeKepSzethuzas, 1, 0, Mat(), stretchedHistogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    Mat stretchHistogramImage(histogramHeight, histogramWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(stretchedHistogram, stretchedHistogram, 0, stretchHistogramImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(stretchHistogramImage, cv::Point(stretchRate * (i - 1), histogramHeight - cvRound(stretchedHistogram.at<float>(i - 1))),
            Point(stretchRate * (i), histogramHeight - cvRound(hist.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Szethuzott hisztogram", stretchHistogramImage);
}

auto Kiegyenlites(Mat kep, Mat hist, int N, int K) {
    int LUT[RANGEMAX];
    float sum = 0;
    int i = 0;
    for (int j = 0; j < RANGEMAX; j++) {
        if (sum < N / K) {
            sum += hist.at<float>(j);
        }
        else {
            i++;
            sum = 0;
        }
        LUT[j] = i * (float)RANGEMAX / K;
    }
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = LUT[kep.at<unsigned char>(i, j)];
        }
    }
    return kep;
}

void hisztogramKiegyenlites(Mat kep) {
    int histogramSize = RANGEMAX;
    float range[] = { RANGEMIN, RANGEMAX };
    const float* histogramRange = { range };
    bool uniform = true;
    bool accumulate = false;

    Mat szurkeKep_histogram;
    calcHist(&kep, 1, 0, Mat(), szurkeKep_histogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    Mat LUThist = szurkeKep_histogram.clone();

    int hist_width = 512, hist_height = 400;
    int bin_w = cvRound((double)hist_width / histogramSize);

    Mat szurkeKep_histogramImage(hist_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(szurkeKep_histogram, szurkeKep_histogram, 0, szurkeKep_histogramImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(szurkeKep_histogramImage, cv::Point(bin_w * (i - 1), hist_height - cvRound(szurkeKep_histogram.at<float>(i - 1))),
            Point(bin_w * (i), hist_height - cvRound(szurkeKep_histogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kep", kep);
    imshow("Hisztogram", szurkeKep_histogramImage);

    Mat negyArnyalatosKiegyenlitettKep = Kiegyenlites(kep.clone(), LUThist, kep.size().area(), 4);
    Mat negyArnyalatosKiegyenlitettKepHistogram;
    Mat negyArnyalatosKiegyenlitettKepHistogramImage(hist_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&negyArnyalatosKiegyenlitettKep, 1, 0, Mat(), negyArnyalatosKiegyenlitettKepHistogram, 1, &histogramSize, &histogramRange, uniform, accumulate);
    normalize(negyArnyalatosKiegyenlitettKepHistogram, negyArnyalatosKiegyenlitettKepHistogram, 0, negyArnyalatosKiegyenlitettKepHistogramImage.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(negyArnyalatosKiegyenlitettKepHistogramImage, cv::Point(bin_w * (i - 1), hist_height - cvRound(negyArnyalatosKiegyenlitettKepHistogram.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_height - cvRound(negyArnyalatosKiegyenlitettKepHistogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kiegyenlitett kep, 4 arnyalat", negyArnyalatosKiegyenlitettKep);
    imshow("Kiegyenlitett hisztogram, 4 arnyalat", negyArnyalatosKiegyenlitettKepHistogramImage);

    Mat tizenhatArnyalatosKiegyenlitettKep = Kiegyenlites(kep.clone(), LUThist, kep.size().area(), 16);
    Mat tizenhatArnyalatosKiegyenlitettKepHistogram;
    Mat tizenhatArnyalatosKiegyenlitettKepHistogramImage(hist_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&tizenhatArnyalatosKiegyenlitettKep, 1, 0, Mat(), tizenhatArnyalatosKiegyenlitettKepHistogram, 1, &histogramSize, &histogramRange, uniform, accumulate);
    normalize(tizenhatArnyalatosKiegyenlitettKepHistogram, tizenhatArnyalatosKiegyenlitettKepHistogram, 0, tizenhatArnyalatosKiegyenlitettKepHistogramImage.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(tizenhatArnyalatosKiegyenlitettKepHistogramImage, cv::Point(bin_w * (i - 1), hist_height - cvRound(tizenhatArnyalatosKiegyenlitettKepHistogram.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_height - cvRound(tizenhatArnyalatosKiegyenlitettKepHistogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kiegyenlitett kep, 16 arnyalat", tizenhatArnyalatosKiegyenlitettKep);
    imshow("Kiegyenlitett hisztogram, 16 arnyalat", tizenhatArnyalatosKiegyenlitettKepHistogramImage);
}

void konvulucio(Mat src) {
    int histogramSize = RANGEMAX;
    float range[] = { RANGEMIN, RANGEMAX };
    const float* histogramRange = { range };
    bool uniform = true;
    bool accumulate = false;

    Mat histogram;
    calcHist(&src, 1, 0, Mat(), histogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    int histogram_width = 512, histogram_height = 400;
    int column_width = cvRound((double)histogram_width / histogramSize);

    Mat histogramImage(histogram_height, histogram_width, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(histogram, histogram, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(histogramImage, cv::Point(column_width * (i - 1), histogram_height - cvRound(histogram.at<float>(i - 1))),
            Point(column_width * (i), histogram_height - cvRound(histogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    Mat base = src.clone();
    imshow("Elotte", src);
    imshow("Konvolucio Elotte Histogram", histogramImage);

    Size s = base.size();
    int HEIGHT = s.height;
    int WIDTH = s.width;
    Mat utanaKep = base.clone();

    Mat kernel = (Mat_<double>(3, 3) << 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0
        , 0, 0, 0
        , -1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0);

    int size = 5;
    double gauss[5][5];
    int sideStep = (size - 1) / 2;
    int kernelSugar = ((kernel.size().height - 1) / 2);

    for (int z = 0; z < 1; z++)
    {
        for (int i = 0; i < HEIGHT - (2 * kernelSugar); i++) {
            for (int j = 0; j < WIDTH - (2 * kernelSugar); j++) {
                double sum = 0;
                for (int k = 0; k < 2 * kernelSugar + 1; k++) {
                    for (int l = 0; l < 2 * kernelSugar + 1; l++) {
                        if ((i + k < HEIGHT - 1 && i + k >= 0) || (j + l < WIDTH - 1 && j + l >= 0)) {
                            sum += (double)base.at<unsigned char>(i + k, j + l) * kernel.at<double>(k, l);
                        }
                    }
                }
                utanaKep.at<unsigned char>(i + kernelSugar, j + kernelSugar) = (unsigned char)abs(sum);
            }
        }
        base = utanaKep.clone();
    }
    Mat stretchHistogram;
    calcHist(&src, 1, 0, Mat(), stretchHistogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    Mat stretchHistogramImage(histogram_height, histogram_width, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(stretchHistogram, stretchHistogram, 0, stretchHistogramImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(stretchHistogramImage, cv::Point(column_width * (i - 1), histogram_height - cvRound(stretchHistogram.at<float>(i - 1))),
            Point(column_width * (i), histogram_height - cvRound(histogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Konvulucio Utan Histogram", stretchHistogramImage);
    imshow("Konvulucio Utan", utanaKep);
}

auto pixel(Mat src, int c, int r) {

    if (c < 0) {
        c = 0;
    }
    if (c >= src.cols) {
        c = src.cols - 1;
    }
    if (r < 0) {
        r = 0;
    }
    if (r >= src.rows) {
        r = src.rows - 1;
    }

    return src.at<unsigned char>(r, c);
}

auto kepAtlagVagySzoras(Mat kep, int radius, Mat avg) {
    Mat eredmeny = Mat::zeros(kep.rows, kep.cols, kep.type());
    for (int eredmeny_row = 0; eredmeny_row < eredmeny.rows; eredmeny_row++) {
        for (int eredmeny_column = 0; eredmeny_column < eredmeny.cols; eredmeny_column++) {
            float sum = 0;
            for (int kep_row = eredmeny_row - radius; kep_row <= eredmeny_row + radius; kep_row++) {
                for (int kep_column = eredmeny_column - radius; kep_column <= eredmeny_column + radius; kep_column++) {
                    if (!avg.empty()) {
                        sum += pow((pixel(kep, kep_column, kep_row) - pixel(avg, eredmeny_column, eredmeny_row)), 2);
                    }
                    else {
                        sum += pixel(kep, kep_column, kep_row);
                    }
                }
            }
            eredmeny.at<unsigned char>(eredmeny_row, eredmeny_column) = sum / (float)pow((2 * radius + 1), 2);
        }
    }
    return eredmeny;
}

auto Wallis(Mat kep, Mat avg, Mat variance, int contrast, float cont_mod, int brightness, float bright_mod) {
    Mat wallis = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int wallis_row = 0; wallis_row < wallis.rows; wallis_row++) {
        for (int wallis_column = 0; wallis_column < wallis.cols; wallis_column++) {
            auto tmp = ((kep.at<unsigned char>(wallis_row, wallis_column) - avg.at<unsigned char>(wallis_row, wallis_column))
                * ((cont_mod * contrast) / (contrast + (cont_mod * sqrt(variance.at<unsigned char>(wallis_row, wallis_column))))))
                + ((bright_mod * brightness) + ((1.0f - bright_mod) * avg.at<unsigned char>(wallis_row, wallis_column)));

            if (tmp > 255) {
                tmp = 255;
            }
            else if (tmp < 0) {
                tmp = 0;
            }
            wallis.at<unsigned char>(wallis_row, wallis_column) = tmp;
        }
    }

    return wallis;
}

void wallisSzuro(Mat kep) {
    int histogramSize = RANGEMAX;
    float range[] = { RANGEMIN, RANGEMAX };
    const float* histogramRange = { range };
    bool uniform = true;
    bool accumulate = false;

    Mat histogram;
    calcHist(&kep, 1, 0, Mat(), histogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    int histogram_width = 512, histogram_height = 400;
    int column_width = cvRound((double)histogram_width / histogramSize);

    Mat histogramImage(histogram_height, histogram_width, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(histogram, histogram, 0, histogramImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(histogramImage, cv::Point(column_width * (i - 1), histogram_height - cvRound(histogram.at<float>(i - 1))),
            Point(column_width * (i), histogram_height - cvRound(histogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kep", kep);
    imshow("Hisztogram", histogramImage);

    Mat atlagKep = kepAtlagVagySzoras(kep, 8, cv::Mat());
    Mat szorasKep = kepAtlagVagySzoras(kep, 8, atlagKep);
    Mat wallisKep = Wallis(kep, atlagKep, szorasKep, 100, 2.5f, 50, 0.8);

    Mat wallisHistogram;
    calcHist(&wallisKep, 1, 0, Mat(), wallisHistogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    Mat wallisHistImage(histogram_height, histogram_width, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(wallisHistogram, wallisHistogram, 0, wallisHistImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histogramSize; i++)
    {
        line(wallisHistImage, cv::Point(column_width * (i - 1), histogram_height - cvRound(wallisHistogram.at<float>(i - 1))),
            Point(column_width * (i), histogram_height - cvRound(wallisHistogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Wallis kep", wallisKep);
    imshow("Wallis hisztogram", wallisHistImage);
}

int compare(const void* p1, const void* p2)
{
    return *(const unsigned char*)p1 - *(const unsigned char*)p2;
}

auto Outlier(Mat kep, int radius, int threshold) {

    Mat atlag = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int atlag_row = 0; atlag_row < atlag.rows; atlag_row++) {
        for (int atlag_column = 0; atlag_column < atlag.cols; atlag_column++) {
            float sum = 0;

            for (int src_r = atlag_row - radius; src_r <= atlag_row + radius; src_r++) {
                for (int src_c = atlag_column - radius; src_c <= atlag_column + radius; src_c++) {

                    if (src_r == atlag_row && src_c == atlag_column) {
                        continue;
                    }

                    sum += pixel(kep, src_c, src_r);
                }
            }
            atlag.at<unsigned char>(atlag_row, atlag_column) = sum / (float)(pow((2 * radius + 1), 2) - 1);
        }
    }

    Mat kimenet = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int kimenet_row = 0; kimenet_row < kimenet.rows; kimenet_row++) {
        for (int kimenet_column = 0; kimenet_column < kimenet.cols; kimenet_column++) {
            if (abs(atlag.at<unsigned char>(kimenet_row, kimenet_column) - kep.at<unsigned char>(kimenet_row, kimenet_column)) <= threshold)
                kimenet.at<unsigned char>(kimenet_row, kimenet_column) = kep.at<unsigned char>(kimenet_row, kimenet_column);
            else
                kimenet.at<unsigned char>(kimenet_row, kimenet_column) = atlag.at<unsigned char>(kimenet_row, kimenet_column);
        }
    }

    return kimenet;
}

void nemlinearis(Mat kep) {
    int histogramSize = RANGEMAX;
    float range[] = { RANGEMIN, RANGEMAX };
    const float* histogramRange = { range };
    bool uniform = true;
    bool accumulate = false;

    Mat hist;
    calcHist(&kep, 1, 0, Mat(), hist, 1, &histogramSize, &histogramRange, uniform, accumulate);

    int hist_width = 512, histogram_height = 400;
    int column_width = cvRound((double)hist_width / histogramSize);

    Mat histImage(histogram_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histogramSize; i++)
    {
        line(histImage, cv::Point(column_width * (i - 1), histogram_height - cvRound(hist.at<float>(i - 1))),
            Point(column_width * (i), histogram_height - cvRound(hist.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Zajos kep", kep);
    imshow("Zajos hisztogram", histImage);

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);

    Mat outlierSrc = Outlier(kep, RADIUS, 35);
    Mat outlierHist;
    calcHist(&outlierSrc, 1, 0, Mat(), outlierHist, 1, &histogramSize, &histogramRange, uniform, accumulate);

    Mat outlierHistImage(histogram_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(outlierHist, outlierHist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histogramSize; i++)
    {
        line(outlierHistImage, cv::Point(column_width * (i - 1), histogram_height - cvRound(outlierHist.at<float>(i - 1))),
            Point(column_width * (i), histogram_height - cvRound(outlierHist.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Outlier szurt kep", outlierSrc);
    imshow("Outlier szurt hisztogram", outlierHistImage);

    Mat median = Mat::zeros(kep.rows, kep.cols, kep.type());
    const int len = (2 * RADIUS + 1) * (2 * RADIUS + 1);

    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {

            unsigned char pixels[len];
            int index = 0;
            for (int k = i - RADIUS; k <= i + RADIUS; k++) {
                for (int l = j - RADIUS; l <= j + RADIUS; l++) {

                    int k2 = k, l2 = l;
                    if (k < 0) {
                        k2 = 0;
                    }
                    if (k >= kep.rows) {
                        k2 = kep.rows - 1;
                    }
                    if (l < 0) {
                        l2 = 0;
                    }
                    if (l >= kep.cols) {
                        l2 = kep.cols - 1;
                    }
                    pixels[index] = kep.at<unsigned char>(k2, l2);
                    index++;
                }
            }
            qsort(pixels, len, sizeof(unsigned char), compare);
            median.at<unsigned char>(i, j) = pixels[(len + 1) / 2];
        }
    }

    Mat medianHistogram;
    calcHist(&median, 1, 0, Mat(), medianHistogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    Mat medianHistImage(histogram_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(medianHistogram, medianHistogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histogramSize; i++)
    {
        line(medianHistImage, cv::Point(column_width * (i - 1), histogram_height - cvRound(medianHistogram.at<float>(i - 1))),
            Point(column_width * (i), histogram_height - cvRound(medianHistogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Median szurt kep", median);
    imshow("Median szurt hisztogram", medianHistImage);

    const int r = RADIUS;
    Mat fast_median = Mat::zeros(kep.rows, kep.cols, kep.type());
    const int len2 = 2 * r + 1;

    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {

            unsigned char pixels[len2];
            int index = 0;
            for (int k = i - r; k <= i + r; k++) {

                unsigned char pixels2[len2];
                int index2 = 0;
                for (int l = j - r; l <= j + r; l++) {

                    int k2 = k, l2 = l;
                    if (k < 0) {
                        k2 = 0;
                    }
                    if (k >= kep.rows) {
                        k2 = kep.rows - 1;
                    }
                    if (l < 0) {
                        l2 = 0;
                    }
                    if (l >= kep.cols) {
                        l2 = kep.cols - 1;
                    }
                    pixels2[index2] = kep.at<unsigned char>(k2, l2);
                    index2++;
                }
                qsort(pixels2, len2, sizeof(unsigned char), compare);
                pixels[index] = pixels2[(len2 + 1) / 2];
                index++;
            }
            qsort(pixels, len2, sizeof(unsigned char), compare);
            fast_median.at<unsigned char>(i, j) = pixels[(len2 + 1) / 2];
        }
    }

    Mat fastMedianHistogram;
    calcHist(&fast_median, 1, 0, Mat(), fastMedianHistogram, 1, &histogramSize, &histogramRange, uniform, accumulate);

    Mat fastmedianHistImage(histogram_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(fastMedianHistogram, fastMedianHistogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histogramSize; i++)
    {
        line(fastmedianHistImage, cv::Point(column_width * (i - 1), histogram_height - cvRound(fastMedianHistogram.at<float>(i - 1))),
            Point(column_width * (i), histogram_height - cvRound(fastMedianHistogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Gyors Median szurt kep", fast_median);
    imshow("Gyors Median szurt hisztogram", fastmedianHistImage);
}

int main(int argc, char** argv) {

    CommandLineParser parser(argc, argv, "{@input | peppers_sotet.bmp | input image}");
    Mat szurkeKep = imread(samples::findFile(parser.get<String>("@input")), IMREAD_GRAYSCALE);

    hisztogramSzethuzas(szurkeKep);

    //hisztogramKiegyenlites(szurkeKep);

    //konvulucio(szurkeKep);

    //wallisSzuro(szurkeKep);

    //nemlinearis(szurkeKep);

    waitKey();

    return EXIT_SUCCESS;
}