#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

Mat negyzetesSzethuzas(Mat kep, int max, int min) {
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = 255 * pow((kep.at<unsigned char>(i, j) - min) / (max - min * 1.0f), 2);
        }
    }
    return kep;
}

Mat gyokosSzethuzas(Mat kep, int max, int min) {
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = 255 * sqrtf((kep.at<unsigned char>(i, j) - min) / (max - min * 1.0f));
        }
    }
    return kep;
}

Mat negyzetesSzethuzas(Mat kep, int max, int min) {
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = 255 * pow((kep.at<unsigned char>(i, j) - min) / (max - min * 1.0f), 2);
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

    Mat hisztogram;
    calcHist(&szurkeskalas, 1, 0, Mat(), hisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    imshow("Alap Kep", szurkeskalas);

    std::vector<float> hisztogramTomb;
    if (hisztogram.isContinuous()) {
        hisztogramTomb.assign((float*)hisztogram.data, (float*)hisztogram.data + hisztogram.total() * hisztogram.channels());
    }
    else {
        for (int i = 0; i < hisztogram.rows; ++i) {
            hisztogramTomb.insert(hisztogramTomb.end(), hisztogram.ptr<float>(i), hisztogram.ptr<float>(i) + hisztogram.cols * hisztogram.channels());
        }
    }

    int hisztogramSzelesseg = 512;
    int hisztogramMagassag = 400;
    int szethuzasMertek = cvRound((double)hisztogramSzelesseg / hisztogramMeret);

    Mat hisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(hisztogram, hisztogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(hisztogramKep, cv::Point(szethuzasMertek * (i - 1), hisztogramMagassag - cvRound(hisztogram.at<float>(i - 1))),
            Point(szethuzasMertek * (i), hisztogramMagassag - cvRound(hisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Alap hisztogram", hisztogramKep);

    int max = 0;
    int min = 0;

    for (int i = 0; i < hisztogramTomb.size(); i++) {
        if (hisztogramTomb[i] != 0) {
            min = i;
        }
    }

    for (int i = hisztogramTomb.size() - 1; i >= 0; i--) {
        if (hisztogramTomb[i] != 0) {
            max = i;
        }
    }

    //Mat szurkeKepSzethuzas = szethuzas(kep.clone(), max, min);
    //Mat szurkeKepSzethuzas = gyokosSzethuzas(kep.clone(), max, min);
    Mat szurkeKepSzethuzas = negyzetesSzethuzas(kep.clone(), max, min);

    imshow("Szethuzott kep", szurkeKepSzethuzas);

    Mat szethuzottHisztogram;
    calcHist(&szurkeKepSzethuzas, 1, 0, Mat(), szethuzottHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat szethuzottHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(szethuzottHisztogram, szethuzottHisztogram, 0, szethuzottHisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(szethuzottHisztogramKep, cv::Point(szethuzasMertek * (i - 1), hisztogramMagassag - cvRound(szethuzottHisztogram.at<float>(i - 1))),
            Point(szethuzasMertek * (i), hisztogramMagassag - cvRound(hisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Szethuzott hisztogram", szethuzottHisztogramKep);

    waitKey();

    return EXIT_SUCCESS;
}