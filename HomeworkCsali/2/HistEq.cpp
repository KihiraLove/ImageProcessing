#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

const int RANGEMIN = 0;
const int RANGEMAX = 256;

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


int main(int argc, char** argv)
{

    //Kép beolvasása
   // CommandLineParser parser(argc, argv, "{@input | airplane.bmp | input image}");
    CommandLineParser parser(argc, argv, "{@input | lena_vilagos.bmp | input image}");
    Mat src = imread(samples::findFile(parser.get<cv::String>( "@input" )), IMREAD_COLOR);
    if( src.empty() )
    {
        return EXIT_FAILURE;
    }

    //Szürkeárnyalatosra konvertálás
    Mat graySrc;
    cvtColor(src, graySrc, COLOR_BGR2GRAY);

    //Normalizálás előkészítése
    int histSize = RANGEMAX;
    float range[] = {RANGEMIN, RANGEMAX};

    const float* histRange = {range};
    bool uniform = true, accumulate = false;

    //A hisztogram kiszámolása
    Mat graySrc_hist;
    calcHist(&graySrc, 1, 0, Mat(), graySrc_hist, 1, &histSize, &histRange, uniform, accumulate);

    //Hisztogram mentése mielőtt normalizáljuk
    Mat LUThist = graySrc_hist.clone();
    
    //Hisztogram rajzolás: a hisztogram képének dimenziói, majd egy oszlop szélessége
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w/histSize);

    Mat graySrc_histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    
    normalize(graySrc_hist, graySrc_hist, 0, graySrc_histImage.rows, NORM_MINMAX, -1, Mat());
    
    for (int i = 1; i < histSize; i++)
    {
        line(graySrc_histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(graySrc_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(graySrc_hist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kep", graySrc);
    imshow("Hisztogram", graySrc_histImage);
    waitKey();


    //Kiegyenlített kép
    Mat K4Src = Equalize(graySrc.clone(), LUThist, graySrc.size().area(), 4);

    //Hisztogram képe: ugyanaz pepitában
    Mat K4SrcHist;
    Mat K4SrcHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&K4Src, 1, 0, Mat(), K4SrcHist, 1, &histSize, &histRange, uniform, accumulate);
    normalize(K4SrcHist, K4SrcHist, 0, K4SrcHistImage.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(K4SrcHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(K4SrcHist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(K4SrcHist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Kiegyenlitett kep, 4 arnyalat", K4Src);
    imshow("Kiegyenlitett hisztogram, 4 arnyalat", K4SrcHistImage);
    waitKey();


    //Kiegyenlített kép
    Mat K16Src = Equalize(graySrc.clone(), LUThist, graySrc.size().area(), 16);

    //Hisztogram képe: ugyanaz pepitában
    Mat K16SrcHist;
    Mat K16SrcHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&K16Src, 1, 0, Mat(), K16SrcHist, 1, &histSize, &histRange, uniform, accumulate);
    normalize(K16SrcHist, K16SrcHist, 0, K16SrcHistImage.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(K16SrcHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(K16SrcHist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(K16SrcHist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }
    
    imshow("Kiegyenlitett kep, 16 arnyalat", K16Src);
    imshow("Kiegyenlitett hisztogram, 16 arnyalat", K16SrcHistImage);
    waitKey();
    
    return EXIT_SUCCESS;
}
