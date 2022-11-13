#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

Mat kiegyenlites(Mat kep, Mat hist, int N, int K) {
    int table[256];
    float ossz = 0;
    int i = 0;
    for (int j = 0; j < 256; j++) {
        if (ossz < N / K) {
            ossz += hist.at<float>(j);
        }
        else {
            i++;
            ossz = 0;
        }
        table[j] = i * (float)256 / K;
    }
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = table[kep.at<unsigned char>(i, j)];
        }
    }
    return kep;
}

int main(int argc, char** argv) {

    CommandLineParser parser(argc, argv, "{@input | input.jpg | input image}");
    Mat kep = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);

    Mat szurkeskalas;
    cvtColor(kep, szurkeskalas, COLOR_BGR2GRAY);


    int hisztogramMeret = 256;
    float range[] = { 0, 256 };
    const float* hisztogramRange = { range };

    Mat szurkeSkalasHisztogram;
    calcHist(&kep, 1, 0, Mat(), szurkeSkalasHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat hisztogramTable = szurkeSkalasHisztogram.clone();

    int hisztogramSzelesseg = 512;
    int hisztogramMagassag = 400;
    int bin_w = cvRound((double)hisztogramSzelesseg / hisztogramMeret);

    Mat szurkeSkalasHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(szurkeSkalasHisztogram, szurkeSkalasHisztogram, 0, szurkeSkalasHisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(szurkeSkalasHisztogramKep, cv::Point(bin_w * (i - 1), hisztogramMagassag - cvRound(szurkeSkalasHisztogram.at<float>(i - 1))),
            Point(bin_w * (i), hisztogramMagassag - cvRound(szurkeSkalasHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kep", kep);
    imshow("Hisztogram", szurkeSkalasHisztogramKep);

    Mat negyArnyalatosKiegyenlitettKep = kiegyenlites(kep.clone(), hisztogramTable, kep.size().area(), 4);
    Mat negyArnyalatosKiegyenlitettKepHisztogram;
    Mat negyArnyalatosKiegyenlitettKepHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&negyArnyalatosKiegyenlitettKep, 1, 0, Mat(), negyArnyalatosKiegyenlitettKepHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);
    normalize(negyArnyalatosKiegyenlitettKepHisztogram, negyArnyalatosKiegyenlitettKepHisztogram, 0, negyArnyalatosKiegyenlitettKepHisztogramKep.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(negyArnyalatosKiegyenlitettKepHisztogramKep, cv::Point(bin_w * (i - 1), hisztogramMagassag - cvRound(negyArnyalatosKiegyenlitettKepHisztogram.at<float>(i - 1))),
            cv::Point(bin_w * (i), hisztogramMagassag - cvRound(negyArnyalatosKiegyenlitettKepHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kiegyenlitett kep, 4 arnyalat", negyArnyalatosKiegyenlitettKep);
    imshow("Kiegyenlitett hisztogram, 4 arnyalat", negyArnyalatosKiegyenlitettKepHisztogramKep);

    Mat tizenhatArnyalatosKiegyenlitettKep = kiegyenlites(kep.clone(), hisztogramTable, kep.size().area(), 16);
    Mat tizenhatArnyalatosKiegyenlitettKepHisztogram;
    Mat tizenhatArnyalatosKiegyenlitettKepHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&tizenhatArnyalatosKiegyenlitettKep, 1, 0, Mat(), tizenhatArnyalatosKiegyenlitettKepHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);
    normalize(tizenhatArnyalatosKiegyenlitettKepHisztogram, tizenhatArnyalatosKiegyenlitettKepHisztogram, 0, tizenhatArnyalatosKiegyenlitettKepHisztogramKep.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(tizenhatArnyalatosKiegyenlitettKepHisztogramKep, cv::Point(bin_w * (i - 1), hisztogramMagassag - cvRound(tizenhatArnyalatosKiegyenlitettKepHisztogram.at<float>(i - 1))),
            cv::Point(bin_w * (i), hisztogramMagassag - cvRound(tizenhatArnyalatosKiegyenlitettKepHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kiegyenlitett kep, 16 arnyalat", tizenhatArnyalatosKiegyenlitettKep);
    imshow("Kiegyenlitett hisztogram, 16 arnyalat", tizenhatArnyalatosKiegyenlitettKepHisztogramKep);

    waitKey();

    return EXIT_SUCCESS;
}