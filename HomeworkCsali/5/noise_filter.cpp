#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <algorithm>



using namespace cv;

//Lokális átlag meghatározása
Mat localAverage(Mat& src, int w = 1)
{

    //Mat copyImg = src.clone();
    Mat copyImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));

    double helperNum = ((2 * w + 1.0) * (2 * w + 1.0)) - 1.0;
    double weight = 1 / helperNum;
    std::cout << weight;
    double value = 0;
    for (int i = w; i < src.rows - w; i++) {
        for (int j = w; j < src.cols - w; j++) {
            value = 0;
            for (int m = i - w; m < i + w; m++) {
                for (int n = j - w; n < j + w; n++) {
                    value += src.at<unsigned char>(m, n);
                }
            }
            if (value > 255) value = 255;
            else if (value < 0) value = 0;
            double temp = weight * value;
            copyImg.at<unsigned char>(i, j) = temp;
        }
    }

    //konvolucioval 
     //Mat kernel = (Mat_<double>(3, 3) << 1.0/8.0, 1.0 / 8.0, 1.0 / 8.0, 
     //    1.0 / 8.0, 0, 1.0 / 8.0, 
     //    1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0);
     //int kernel_r = 1;
     //for (int i = 0; i < copyImg.rows - (2 * kernel_r); i++)
     //{
     //    for (int j = 0; j < copyImg.cols - (2 * kernel_r); j++)
     //    {
     //        double value = 0;
     //        for (int k = 0; k < 2 * kernel_r + 1; k++) {
     //            for (int l = 0; l < 2 * kernel_r + 1; l++) {
     //                value += src.at<unsigned char>(i + k, j + l) * kernel.at<double>(k, l);

     //            }
     //        }
     //        copyImg.at<unsigned char>(i + kernel_r, j + kernel_r) = 3 * abs(value);
     //    }
     //}
    return copyImg;
}

//Outlier szűrő
Mat outlierFilter(Mat& src, double Th)
{
    //Mat copyImg = src.clone();
    Mat M = localAverage(src);
    Mat copyImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.rows; j++) {
            double pixelM = M.at<unsigned char>(i, j);
            double pixelF = src.at<unsigned char>(i, j);
            double subtraction = std::abs(pixelM - pixelF);

            if (subtraction <= Th) {
                //std::cout << "I'm here" << "\n";
                copyImg.at<unsigned char>(i, j) = src.at<unsigned char>(i, j);
            }
            else {
                //std::cout << "I'm here2" << "\n";
                copyImg.at<unsigned char>(i, j) = M.at<unsigned char>(i, j);
            }
        }
    }
    return copyImg;
}

Mat medianFilter(Mat& src, int w = 1)
{
    //Mat copyImg = src.clone();
    Mat copyImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    for (int i = w; i < src.rows - w; i++) {
        for (int j = w; j < src.cols - w; j++) {
            std::vector<double> pixels;
            for (int m = i - w; m < i + w; m++) {
                for (int n = j - w; n < j + w; n++) {
                    pixels.push_back(src.at<unsigned char>(m, n));
                }
            }
            std::sort(pixels.begin(), pixels.end(), [](double a, double b) {
                return a < b;
                });

            const auto median_it = pixels.begin() + pixels.size() / 2;
            std::nth_element(pixels.begin(), median_it, pixels.end());
            auto median = *median_it;
            copyImg.at<unsigned char>(i, j) = median;
        }
    }
    return copyImg;
}

Mat fastMedianFilter(Mat& src, int w = 1)
{
    Mat copyImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    for (int i = w; i < src.rows - w; i++) {
        for (int j = w; j < src.cols - w; j++) {
            std::vector<double> pixelsCol1;
            std::vector<double> pixelsCol2;
            std::vector<double> pixelsCol3;
            std::vector<double> medians;
            pixelsCol1.push_back(src.at<unsigned char>(i - w, j - w));
            pixelsCol1.push_back(src.at<unsigned char>(i, j - w));
            pixelsCol1.push_back(src.at<unsigned char>(i + w, j - w));

            pixelsCol2.push_back(src.at<unsigned char>(i - w, j));
            pixelsCol2.push_back(src.at<unsigned char>(i, j));
            pixelsCol2.push_back(src.at<unsigned char>(i + w, j));

            pixelsCol3.push_back(src.at<unsigned char>(i - w, j + w));
            pixelsCol3.push_back(src.at<unsigned char>(i, j + w));
            pixelsCol3.push_back(src.at<unsigned char>(i + w, j + w));

            const auto median_it1 = pixelsCol1.begin() + pixelsCol1.size() / 2;
            std::nth_element(pixelsCol1.begin(), median_it1, pixelsCol1.end());
            auto median1 = *median_it1;

            const auto median_it2 = pixelsCol2.begin() + pixelsCol2.size() / 2;
            std::nth_element(pixelsCol2.begin(), median_it2, pixelsCol2.end());
            auto median2 = *median_it2;

            const auto median_it3 = pixelsCol3.begin() + pixelsCol3.size() / 2;
            std::nth_element(pixelsCol3.begin(), median_it3, pixelsCol3.end());
            auto median3 = *median_it3;

            medians.push_back(median1);
            medians.push_back(median2);
            medians.push_back(median3);

            const auto median_it4 = medians.begin() + medians.size() / 2;
            std::nth_element(medians.begin(), median_it4, medians.end());
            auto median4 = *median_it4;

            copyImg.at<unsigned char>(i, j) = median4;
        }
    }
    return copyImg;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, "{@input | barbara.bmp | input image}");
    Mat src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    if (src.empty())
    {
        return EXIT_FAILURE;
    }

    Mat M = localAverage(src_gray);
    Mat O = outlierFilter(src_gray, 200.0);
    Mat median = medianFilter(src_gray, 2);
    Mat fastMedian = fastMedianFilter(src_gray);
    imshow("Source", src);
    imshow("Grayscale", src_gray);
    imshow("Avg", M);
    imshow("Outlier", O);
    imshow("Median", median);
    imshow("FastMedian", fastMedian);
    //imwrite("grascale.jpg", src_gray);
   /* imwrite("barbara_outlier.jpg", O);
    imwrite("barbara_median.jpg", median);
    imwrite("barbara_fastMedian.jpg", fastMedian);*/
    waitKey();
    return EXIT_SUCCESS;
}

