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

Mat kepAtlagVagySzoras(Mat kep, int radius, Mat atlag) {
    Mat eredmeny = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int eredmeny_row = 0; eredmeny_row < eredmeny.rows; eredmeny_row++) {
        for (int eredmeny_column = 0; eredmeny_column < eredmeny.cols; eredmeny_column++) {
            float sum = 0;
            for (int kep_row = eredmeny_row - radius; kep_row <= eredmeny_row + radius; kep_row++) {
                for (int kep_column = eredmeny_column - radius; kep_column <= eredmeny_column + radius; kep_column++) {
                    if (!atlag.empty()) {
                        sum += pow((pixel(kep, kep_column, kep_row) - pixel(atlag, eredmeny_column, eredmeny_row)), 2);
                    }
                    else {
                        sum += pixel(kep, kep_column, kep_row);
                    }
                }
            }
            eredmeny.at<unsigned char>(eredmeny_row, eredmeny_column) = sum / (float)pow((2 * radius + 1), 2);
        }
    }

    return eredmeny;
}

Mat wallis(Mat kep, Mat atlag, Mat varians, int kontraszt, float kont_mod, int vilagossag, float vil_mod) {
    Mat wallis = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int wallis_row = 0; wallis_row < wallis.rows; wallis_row++) {
        for (int wallis_column = 0; wallis_column < wallis.cols; wallis_column++) {
            auto tmp = ((kep.at<unsigned char>(wallis_row, wallis_column) - atlag.at<unsigned char>(wallis_row, wallis_column))
                * ((kont_mod * kontraszt) / (kontraszt + (kont_mod * sqrt(varians.at<unsigned char>(wallis_row, wallis_column))))))
                + ((vil_mod * vilagossag) + ((1.0f - vil_mod) * atlag.at<unsigned char>(wallis_row, wallis_column)));

            if (tmp > 255) {
                tmp = 255;
            }
            else if (tmp < 0) {
                tmp = 0;
            }
            wallis.at<unsigned char>(wallis_row, wallis_column) = tmp;
        }
    }

    return wallis;
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

    imshow("Kep", szurkeSkalas);
    imshow("Hisztogram", hisztogramKep);

    Mat atlagKep = kepAtlagVagySzoras(szurkeSkalas, 8, cv::Mat());
    Mat szorasKep = kepAtlagVagySzoras(szurkeSkalas, 8, atlagKep);
    Mat wallisKep = wallis(szurkeSkalas, atlagKep, szorasKep, 100, 2.5f, 50, 0.8);

    Mat wallisHisztogram;
    calcHist(&wallisKep, 1, 0, Mat(), wallisHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);
    Mat wallisHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(wallisHisztogram, wallisHisztogram, 0, wallisHisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(wallisHisztogramKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(wallisHisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(wallisHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow("Wallis kep", wallisKep);
    imshow("Wallis hisztogram", wallisHisztogramKep);

    waitKey();

    return EXIT_SUCCESS;

}