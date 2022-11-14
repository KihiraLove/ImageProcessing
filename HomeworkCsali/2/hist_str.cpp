#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

const int RANGEMIN = 0;
const int RANGEMAX = 256;

Mat LinearStretch(Mat src, int max, int min) {

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            src.at<unsigned char>(i, j) = 255 * (src.at<unsigned char>(i, j) - min) / (max - min);
        }
    }

    return src;
}

Mat SqrtStretch(Mat src, int max, int min) {

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            src.at<unsigned char>(i, j) = 255 * sqrtf((src.at<unsigned char>(i, j) - min) / (max - min * 1.0f));
            //src.at<unsigned char>(i, j) = 255 * sqrtf((src.at<unsigned char>(i, j) - min) / 255.0f);
        }
    }

    return src;
}

Mat QuadraticStretch(Mat src, int max, int min) {

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            src.at<unsigned char>(i, j) = 255 * pow((src.at<unsigned char>(i, j) - min) / (max - min * 1.0f), 2);
            //src.at<unsigned char>(i, j) = 255 * pow((src.at<unsigned char>(i, j) - min) / 255.0f, 2);
        }
    }

    return src;
}

int main(int argc, char** argv) {

    //normalize(graySrc, graySrcStretch, 0, 255, CV_MINMAX);

    //Kép betöltése
    CommandLineParser parser(argc, argv, "{@input |  peppers_sotet.bmp | input image}");
    Mat src = imread(samples::findFile(parser.get<cv::String>("@input")), IMREAD_COLOR);
    if (src.empty())
    {
        return EXIT_FAILURE;
    }

    Mat graySrc;
    cvtColor(src, graySrc, COLOR_BGR2GRAY);

    //Hisztogram készítése
    int histSize = RANGEMAX;
    float range[] = { RANGEMIN, RANGEMAX };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    cout << "teszt" << endl;


    Mat hist;
    calcHist(&graySrc, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    imshow("Kep", graySrc);
    waitKey();

    //Széthúzás
    std::vector<float> histarr;
    if (hist.isContinuous()) {
        histarr.assign((float*)hist.data, (float*)hist.data + hist.total() * hist.channels());
    }
    else {
        for (int i = 0; i < hist.rows; ++i) {
            histarr.insert(histarr.end(), hist.ptr<float>(i), hist.ptr<float>(i) + hist.cols * hist.channels());
        }
    }

    //Hisztogram rajzolás: a hisztogram képének dimenziói, majd egy oszlop szélessége
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }


    imshow("Hisztogram", histImage);
    waitKey();

    int max, min;

    for (int i = 0; i < histarr.size(); i++) {
        if (histarr[i] != 0) {
            min = i;
            break;
        }
    }

    for (int i = histarr.size() - 1; i >= 0; i--) {
        if (histarr[i] != 0) {
            max = i;
            break;
        }
    }

     Mat graySrcStretch = LinearStretch(graySrc.clone(), max, min);
    // Mat graySrcStretch = SqrtStretch(graySrc.clone(), max, min);
    //Mat graySrcStretch = QuadraticStretch(graySrc.clone(), max, min);

    imshow("Szethuzott kep", graySrcStretch);
    waitKey();

    Mat stretchHist;
    calcHist(&graySrcStretch, 1, 0, Mat(), stretchHist, 1, &histSize, &histRange, uniform, accumulate);

    Mat stretchHistImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(stretchHist, stretchHist, 0, stretchHistImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(stretchHistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(stretchHist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Szethuzott hisztogram", stretchHistImage);
    waitKey();

    return EXIT_SUCCESS;
}

