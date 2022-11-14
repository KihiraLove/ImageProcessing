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

auto Sobel(Mat src, bool rowwise) {

    Mat dest = src.clone();
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {

            auto n = GetPixel(src, c, r-1);
            auto ne = GetPixel(src, c+1, r-1);
            auto e = GetPixel(src, c+1, r);
            auto se = GetPixel(src, c+1, r+1);
            auto s = GetPixel(src, c, r+1);
            auto sw = GetPixel(src, c-1, r+1);
            auto w = GetPixel(src, c-1, r);
            auto nw = GetPixel(src, c-1, r-1);


            auto tmp = cv::abs(0.25f * (nw + 2*w + sw - ne - 2*e - se));
            if (rowwise) {
                 tmp = cv::abs(0.25f * (-nw - 2*n - ne + sw + 2*s + se));
            }

            dest.at<unsigned char>(r, c) = tmp;
        }
    }

    return dest;

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

    //Sobel vizszintes
    Mat sobelRowSrc = Sobel(graySrc.clone(), true);

    Mat sobelRowHist;
    calcHist(&sobelRowSrc, 1, 0, Mat(), sobelRowHist, 1, &histSize, &histRange, uniform, accumulate);

    Mat sobelRowHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(sobelRowHist, sobelRowHist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(sobelRowHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(sobelRowHist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(sobelRowHist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Sobel vizszintes kep", sobelRowSrc);

    imshow("Sobel vizszintes hisztogram", sobelRowHistImage);
    waitKey();


    //Sobel fuggoleges
    Mat sobelColSrc = Sobel(graySrc.clone(), false);

    Mat sobelColHist;
    calcHist(&sobelColSrc, 1, 0, Mat(), sobelColHist, 1, &histSize, &histRange, uniform, accumulate);

    Mat sobelColHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(sobelColHist, sobelColHist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(sobelColHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(sobelColHist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(sobelColHist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Sobel fuggoleges kep", sobelColSrc);
    imshow("Sobel fuggoleges hisztogram", sobelColHistImage);
    waitKey();

    //Sobel osszeg
    Mat sobelTotalSrc = sobelRowSrc.clone();
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
                sobelTotalSrc.at<unsigned char>(r, c) += sobelColSrc.at<unsigned char>(r, c);
            }
        }

        Mat sobelTotalHist;
    calcHist(&sobelTotalSrc, 1, 0, Mat(), sobelTotalHist, 1, &histSize, &histRange, uniform, accumulate);

    Mat sobelTotalHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(sobelTotalHist, sobelTotalHist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(sobelTotalHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(sobelTotalHist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(sobelTotalHist.at<float>(i))),
             cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Sobel osszetett kep", sobelTotalSrc);
    imshow("Sobel osszetett hisztogram", sobelTotalHistImage);
    waitKey();


    return EXIT_SUCCESS;
}

