#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

const int RANGEMIN = 0;
const int RANGEMAX = 256;

auto GetPixel(Mat src, int c, int r) {

    if (c < 0) c = 0;
    if (c >= src.cols) c = src.cols - 1;
    if (r < 0) r = 0;
    if (r >= src.rows) r = src.rows - 1;

    return src.at<unsigned char>(r, c);
}

auto Laplace4(Mat src) {

    Mat dest = src.clone();
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {

            auto center = GetPixel(src, c, r);
            auto top = GetPixel(src, c, r-1);
            auto bottom = GetPixel(src, c, r+1);
            auto left = GetPixel(src, c-1, r);
            auto right = GetPixel(src, c+1, r);

            dest.at<unsigned char>(r, c) = cv::abs(left + right + top + bottom - 4 * center);
        }
    }

    return dest;
}

auto Laplace8(Mat src) {
    Mat dest = src.clone();
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {

            auto center = GetPixel(src, c, r);
            auto n = GetPixel(src, c, r-1);
            auto ne = GetPixel(src, c+1, r-1);
            auto e = GetPixel(src, c+1, r);
            auto se = GetPixel(src, c+1, r+1);
            auto s = GetPixel(src, c, r+1);
            auto sw = GetPixel(src, c-1, r+1);
            auto w = GetPixel(src, c-1, r);
            auto nw = GetPixel(src, c-1, r-1);

            dest.at<unsigned char>(r, c) = cv::abs(n + ne + e + se + s + sw + w + nw - 8 * center);
        }
    }

    return dest;
}

auto Equalize(Mat src, Mat hist, int N, int K) {
    //K féle árnyalat lesz a kiegyenlített képen; N=pixelek száma a képen

    //Lookup table
    int LUT[RANGEMAX];

    //Feltöltés
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

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            src.at<unsigned char>(i, j) = LUT[src.at<unsigned char>(i, j)];
        }
    }

    return src;
}



int main(int argc, char** argv) {

    //normalize(graySrc, graySrcStretch, 0, 255, CV_MINMAX);
        
    //Kép betöltése
    CommandLineParser parser(argc, argv, "{@input | lena.bmp | input image}");
    //CommandLineParser parser(argc, argv, "{@input | pisa.bmp | input image}");
    Mat src = imread(samples::findFile(parser.get<cv::String>( "@input" )), IMREAD_COLOR);
    if( src.empty() )
    {
        return EXIT_FAILURE;
    }
    
    Mat graySrc;
    cvtColor(src, graySrc, COLOR_BGR2GRAY);

    //Hisztogram készítése
    int histSize = RANGEMAX;
    float range[] = {RANGEMIN, RANGEMAX};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;

    Mat hist;
    calcHist(&graySrc, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    Mat LUThist = hist.clone();

    //Hisztogram rajzolás: a hisztogram képének dimenziói, majd egy oszlop szélessége
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w/histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kep", graySrc);

    imshow("Hisztogram", histImage);
    waitKey();

    //Laplace 4 szomszéd

    Mat laplace4Src = Laplace4(graySrc.clone());


    Mat laplace4Hist;
    calcHist(&laplace4Src, 1, 0, Mat(), laplace4Hist, 1, &histSize, &histRange, uniform, accumulate);


    Mat laplace4HistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cout << "teszt" << endl;
    normalize(laplace4Hist, laplace4Hist, 0, laplace4HistImage.rows, NORM_MINMAX, -1, Mat());


    for (int i = 1; i < histSize; i++)
    {
        line(laplace4HistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(laplace4Hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(laplace4Hist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Laplace4 kep", laplace4Src);
    imshow("Laplace4 hisztogram", laplace4HistImage);
    waitKey();


    //Laplace 8 szomszéd

    Mat laplace8Src = Laplace8(graySrc.clone());

    Mat laplace8Hist;
    calcHist(&laplace8Src, 1, 0, Mat(), laplace8Hist, 1, &histSize, &histRange, uniform, accumulate);

    Mat laplace8HistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(laplace8Hist, laplace8Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(laplace8HistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(laplace8Hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(laplace8Hist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Laplace8 kep", laplace8Src);

    imshow("Laplace8 hisztogram", laplace8HistImage);
    waitKey();

    //Laplace 4 szomszéd + Hist EQ

    Mat laplaceEQSrc = Laplace4(Equalize(graySrc.clone(), LUThist, graySrc.size().area(), 16));

    Mat laplaceEQHist;
    calcHist(&laplaceEQSrc, 1, 0, Mat(), laplaceEQHist, 1, &histSize, &histRange, uniform, accumulate);

    Mat laplaceEQHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(laplaceEQHist, laplaceEQHist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(laplaceEQHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(laplaceEQHist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(laplaceEQHist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }



    imshow("Laplace4EQ kep", laplaceEQSrc);

    imshow("Laplace4EQ hisztogram", laplaceEQHistImage);
    waitKey();



    return EXIT_SUCCESS;
}

