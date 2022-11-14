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

auto AverageMat(Mat src, int radius) {

    cout << "Lokalis atlag..." << endl;

    //LOKÁLIS ÁTLAG
    //Legyen egy mátrix a forráskép méreteivel, tele 0 értékekkel ; M(i,j)
    Mat avg = Mat::zeros(src.rows, src.cols, src.type());

    //Lokális átlag mátrix ciklusai
    for (int avg_r = 0; avg_r < avg.rows; avg_r++) {
        for (int avg_c = 0; avg_c < avg.cols; avg_c++) {
            float sum = 0;

            //Kép mátrix ciklusai
            for (int src_r = avg_r - radius; src_r <= avg_r + radius; src_r++) {
                for (int src_c = avg_c - radius; src_c <= avg_c + radius; src_c++) {

                    sum += GetPixel(src, src_c, src_r);
                }
            }
            avg.at<unsigned char>(avg_r, avg_c) = sum / (float)pow((2 * radius + 1), 2);
        }
    }

    return avg;
}

auto VarianceMat(Mat src, Mat avg, int radius) {

    cout << "Szorasnegyzet..." << endl;

    //SZÓRÁSNÉGYZET
    Mat variance = Mat::zeros(src.rows, src.cols, src.type());
    for (int var_r = 0; var_r < variance.rows; var_r++) {
        for (int var_c = 0; var_c < variance.cols; var_c++) {
            float sum = 0;

            //Kép mátrix ciklusai
            for (int src_r = var_r - radius; src_r <= var_r + radius; src_r++) {
                for (int src_c = var_c - radius; src_c <= var_c + radius; src_c++) {
                    sum += pow((GetPixel(src, src_c, src_r) - GetPixel(avg, var_c, var_r)), 2);
                }
            }
            variance.at<unsigned char>(var_r, var_c) = sum / (float)pow((2 * radius + 1), 2);
        }
    }

    return variance;
}

auto Wallis(Mat src, Mat avg, Mat variance, int contrast, float cont_mod, int brightness, float bright_mod) {

    cout << "Kezdeti beallitasok..." << endl;
    //Kezdeti beállítások
    /*
    int contrast = 128;     //avagy Sd ; elvárt kontraszt
    int brightness = 128;   //avagy Md ; elvárt világosság
    float bright_mod = 0.5f;  //avagy r ; "brightness modifier"
    float cont_mod = 1.25f;  //avagy Amax ; "contrast modifier"
     */


    cout << "Vegeredmeny..." << endl;

    //VÉGEREDMÉNY
    Mat dest = Mat::zeros(src.rows, src.cols, src.type());
    for (int dest_r = 0; dest_r < dest.rows; dest_r++) {
        for (int dest_c = 0; dest_c < dest.cols; dest_c++) {

            auto tmp = (( src.at<unsigned char>(dest_r, dest_c) - avg.at<unsigned char>(dest_r, dest_c) )
                            * ( ( cont_mod * contrast ) / ( contrast + ( cont_mod * sqrt(variance.at<unsigned char>(dest_r, dest_c)) ) ) ))
                            + ( ( bright_mod * brightness ) + ( ( 1.0f - bright_mod ) * avg.at<unsigned char>(dest_r, dest_c) ) );

            if (tmp > 255) tmp = 255;
            else if (tmp < 0) tmp = 0;
            dest.at<unsigned char>(dest_r, dest_c) = tmp;
        }
    }

    return dest;
}

int main(int argc, char** argv) {

    //Kép betöltése
    CommandLineParser parser(argc, argv, "{@input | bridge.bmp | input image}");
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

    //Wallis
    Mat avgSrc = AverageMat(graySrc, 8);
    Mat varSrc = VarianceMat(graySrc, avgSrc, 8);
    Mat wallisSrc = Wallis(graySrc, avgSrc, varSrc, 50, 2.5f, 256, 0.5);

    Mat wallisHist;
    calcHist(&wallisSrc, 1, 0, Mat(), wallisHist, 1, &histSize, &histRange, uniform, accumulate);

    Mat wallisHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(wallisHist, wallisHist, 0, wallisHistImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(wallisHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(wallisHist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(wallisHist.at<float>(i))),
             cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    imshow("Wallis kep Sd 50, Md 256", wallisSrc);
    imshow("Wallis hisztogram Sd 50, Md 256", wallisHistImage);
    waitKey();

    return EXIT_SUCCESS;
}

