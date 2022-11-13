#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, "{@input | lena.jpg | input image}");

    //Színes beolvasás
    Mat src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    imshow("Original", src);

    //Grayscale beolvasás IMREAD_GRAYSCALE flaggel
    Mat gsrc1 = imread(samples::findFile(parser.get<String>("@input")), IMREAD_GRAYSCALE);
    imshow("Gray scale 1", gsrc1);

    //Színesként kiolvassuk és szürke color_space-t húzunk rá majd elmentjük a gsrc obejctbe
    Mat gsrc2;
    cvtColor(src, gsrc2, cv::COLOR_BGR2GRAY);
    imshow("Gray scale 2", gsrc2);

    //Hisztogram kiszámolása
    int histSize = 256; // hány elemű lesz a hisztogram
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat hist; // itt fogjuk tárolni a hisztogramot
    calcHist(&gsrc1, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    //Hisztogram kirajzolása
    int hist_w = 512, hist_h = 400; // ezek a változók határozzák meg, hogy mekkora lesz a hisztogramot kirajzoló kép
    int bin_w = cvRound((double)hist_w / histSize); // oszlopok szélessége
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0)); // létrehozunk egy adott méretű, 3 csatornás, csatornánként 8 bites fekete képet
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat()); // normalizáljuk magát a hisztogramot, hogy az oszlopok magassága megfeleljen a kép magasságának
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow("Hisztogram", histImage);

    //Invertálás 1. Végig iterálunk a pixeleken és kivonjuk minden r/g/b értékét 255-ből
    Mat invertsrc = src.clone();
    for (int i = 0; i < invertsrc.rows; i++)
    {
        for (int j = 0; j < invertsrc.cols; j++)
        {
            Vec3b intensity;
            intensity = invertsrc.at<Vec3b>(i, j);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            invertsrc.at<Vec3b>(i, j)[0] = 255 - blue;
            invertsrc.at<Vec3b>(i, j)[1] = 255 - green;
            invertsrc.at<Vec3b>(i, j)[2] = 255 - red;
        }
    }

    imshow("Inverted 1", invertsrc);

    //Invertálás 2. substract() függvénnyel
    Mat invertsrc2 = src.clone();
    Mat full255 = src.clone();

    for (int i = 0; i < full255.rows; i++) //full255 feltöltése 255, 255, 255 értékü pixelekkel
    {
        for (int j = 0; j < full255.cols; j++)
        {
            full255.at<Vec3b>(i, j)[0] = 255;
            full255.at<Vec3b>(i, j)[1] = 255;
            full255.at<Vec3b>(i, j)[2] = 255;
        }
    }
    invertsrc2 = full255 - invertsrc2; // kivonjuk a két Mat objectet egymásból

    imshow("Inverted 2", invertsrc2);
    
    //Invertálás bitwise_not() függvénnyel
    Mat invertsrc3 = src.clone();
    bitwise_not(src, invertsrc3);
    
    imshow("Inverted 3", invertsrc3);

    //grayscale invertálás 255 - gsrc
    Mat negsrc = gsrc2.clone();
    Mat gr255;
    cvtColor(full255, gr255, cv::COLOR_BGR2GRAY);
    negsrc = gr255 - negsrc;

    imshow("Gray Inverted", negsrc);

    //Kép elmentése
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(90); // itt adjuk meg a minőséget
    bool result = false;
    try
    {
        result = imwrite("saved_image1.jpg", gsrc1, compression_params);
    }
    catch (const cv::Exception& ex)
    {
        fprintf(stderr, "Exception saving file: %s\n", ex.what());
    }
    if (result)
        printf("File saved.\n");
    else
        printf("ERROR: Can't save file.\n");

    compression_params.pop_back();
    compression_params.pop_back();
    waitKey();

    return EXIT_SUCCESS;
}