#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

unsigned char pixel(Mat src, int c, int r) {

    if (c < 0) { c = 0; }
    if (c >= src.cols) { c = src.cols - 1; }
    if (r < 0) { r = 0; }
    if (r >= src.rows) { r = src.rows - 1; }

    return src.at<unsigned char>(r, c);
}

int osszeHasonlit(const void* p1, const void* p2)
{
    return *(const unsigned char*)p1 - *(const unsigned char*)p2;
}


Mat outlier(Mat kep, int radius, int threshold) {
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

int main(int argc, char** argv) {

    CommandLineParser parser(argc, argv, "{@input | input.jpg | input image}");
    Mat kep = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);

    Mat szurkeSkalas;
    cvtColor(kep, szurkeSkalas, COLOR_BGR2GRAY);

    int hisztogramMeret = 256;
    float range[] = { 0, 256 };
    const float* hisztogramRange = { range };

    Mat hisztogram;
    calcHist(&szurkeSkalas, 1, 0, Mat(), hisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    int hisztogramSzelesseg = 512;
    int hisztogramMagassag = 400;
    int oszlopSzelesseg = cvRound((double)hisztogramSzelesseg / hisztogramMeret);

    Mat hisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(hisztogram, hisztogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(hisztogramKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(hisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(hisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Zajos kep", szurkeSkalas);
    imshow("Zajos hisztogram", hisztogramKep);

    std::vector<int> tomoritesPrameterek;
    tomoritesPrameterek.push_back(cv::IMWRITE_JPEG_QUALITY);
    tomoritesPrameterek.push_back(100);

    Mat outlierKep = outlier(szurkeSkalas, 3, 35);
    Mat outlierHisztogram;
    calcHist(&outlierKep, 1, 0, Mat(), outlierHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat outlierHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(outlierHisztogram, outlierHisztogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(outlierHisztogramKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(outlierHisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(outlierHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Outlier szurt kep", outlierKep);
    imshow("Outlier szurt hisztogram", outlierHisztogramKep);

    Mat median = Mat::zeros(szurkeSkalas.rows, szurkeSkalas.cols, szurkeSkalas.type());
    const int hossz = (2 * 3 + 1) * (2 * 3 + 1);

    for (int i = 0; i < szurkeSkalas.rows; i++) {
        for (int j = 0; j < szurkeSkalas.cols; j++) {

            unsigned char pixels[hossz];
            int index = 0;
            for (int k = i - 3; k <= i + 3; k++) {
                for (int l = j - 3; l <= j + 3; l++) {

                    int k2 = k, l2 = l;
                    if (k < 0) {
                        k2 = 0;
                    }
                    if (k >= szurkeSkalas.rows) {
                        k2 = szurkeSkalas.rows - 1;
                    }
                    if (l < 0) {
                        l2 = 0;
                    }
                    if (l >= szurkeSkalas.cols) {
                        l2 = szurkeSkalas.cols - 1;
                    }
                    pixels[index] = szurkeSkalas.at<unsigned char>(k2, l2);
                    index++;
                }
            }
            qsort(pixels, hossz, sizeof(unsigned char), osszeHasonlit);
            median.at<unsigned char>(i, j) = pixels[(hossz + 1) / 2];
        }
    }

    Mat medianHistogram;
    calcHist(&median, 1, 0, Mat(), medianHistogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat medianHistImage(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(medianHistogram, medianHistogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(medianHistImage, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(medianHistogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(medianHistogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Median szurt kep", median);
    imshow("Median szurt hisztogram", medianHistImage);

    const int r = 3;
    Mat gyors_median = Mat::zeros(szurkeSkalas.rows, szurkeSkalas.cols, szurkeSkalas.type());
    const int hossz2 = 2 * r + 1;

    for (int i = 0; i < szurkeSkalas.rows; i++) {
        for (int j = 0; j < szurkeSkalas.cols; j++) {

            unsigned char pixels[hossz2];
            int index = 0;
            for (int k = i - r; k <= i + r; k++) {

                unsigned char pixels2[hossz2];
                int index2 = 0;
                for (int l = j - r; l <= j + r; l++) {

                    int k2 = k, l2 = l;
                    if (k < 0) {
                        k2 = 0;
                    }
                    if (k >= szurkeSkalas.rows) {
                        k2 = szurkeSkalas.rows - 1;
                    }
                    if (l < 0) {
                        l2 = 0;
                    }
                    if (l >= szurkeSkalas.cols) {
                        l2 = szurkeSkalas.cols - 1;
                    }
                    pixels2[index2] = szurkeSkalas.at<unsigned char>(k2, l2);
                    index2++;
                }
                qsort(pixels2, hossz2, sizeof(unsigned char), osszeHasonlit);
                pixels[index] = pixels2[(hossz2 + 1) / 2];
                index++;
            }
            qsort(pixels, hossz2, sizeof(unsigned char), osszeHasonlit);
            gyors_median.at<unsigned char>(i, j) = pixels[(hossz2 + 1) / 2];
        }
    }

    Mat gyorsMedianHisztogram;
    calcHist(&gyors_median, 1, 0, Mat(), gyorsMedianHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat gyorsMedianHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(gyorsMedianHisztogram, gyorsMedianHisztogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(gyorsMedianHisztogramKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(gyorsMedianHisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(gyorsMedianHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Gyors Median szurt kep", gyors_median);
    imshow("Gyors Median szurt hisztogram", gyorsMedianHisztogramKep);

    waitKey();

    return EXIT_SUCCESS;
}