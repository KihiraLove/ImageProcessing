#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

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

    Mat alap = szurkeSkalas.clone();
    imshow("Elotte", szurkeSkalas);
    imshow("Konvolucio Elotte Histogram", hisztogramKep);

    Size meret = alap.size();
    int magassag = meret.height;
    int szelesseg = meret.width;
    Mat utanaKep = alap.clone();

    Mat kernel = (Mat_<double>(3, 3) << 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0
        , 0, 0, 0
        , -1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0);

    int size = 5;
    double gauss[5][5];
    int sideStep = (size - 1) / 2;
    int kernelSugar = ((kernel.size().height - 1) / 2);

    for (int z = 0; z < 1; z++)
    {
        for (int i = 0; i < magassag - (2 * kernelSugar); i++) {
            for (int j = 0; j < szelesseg - (2 * kernelSugar); j++) {
                double sum = 0;
                for (int k = 0; k < 2 * kernelSugar + 1; k++) {
                    for (int l = 0; l < 2 * kernelSugar + 1; l++) {
                        if ((i + k < magassag - 1 && i + k >= 0) || (j + l < szelesseg - 1 && j + l >= 0)) {
                            sum += (double)alap.at<unsigned char>(i + k, j + l) * kernel.at<double>(k, l);
                        }
                    }
                }
                utanaKep.at<unsigned char>(i + kernelSugar, j + kernelSugar) = (unsigned char)abs(sum);
            }
        }
        alap = utanaKep.clone();
    }
    Mat szethuzottHisztogram;
    calcHist(&szurkeSkalas, 1, 0, Mat(), szethuzottHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat szethuztottHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(szethuzottHisztogram, szethuzottHisztogram, 0, szethuztottHisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(szethuztottHisztogramKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(szethuzottHisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(hisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Konvulucio Utan Histogram", szethuztottHisztogramKep);
    imshow("Konvulucio Utan", utanaKep);

    waitKey();

    return EXIT_SUCCESS;
}