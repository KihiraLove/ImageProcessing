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

    //Grayscale beolvasás IMREAD_GRAYSCALE flaggel
    Mat gsrc1 = imread(samples::findFile(parser.get<String>("@input")), IMREAD_GRAYSCALE);

    //Színesként kiolvassuk és szürke color_space-t húzunk rá majd elmentjük a gsrc obejctbe
    Mat gsrc2;
    cvtColor(src, gsrc2, cv::COLOR_BGR2GRAY);

    //Másolások
    Mat invertsrc = src.clone();
    Mat invertsrc2 = src.clone();
    Mat invertsrc3 = src.clone();
    Mat negsrc = gsrc2.clone();
    
    //
    bitwise_not(src, invertsrc3);
    
    //full255 feltöltése 255,255,255 értékü pixelekkel
    Mat full255 = src.clone();
    for (int i = 0; i < full255.rows; i++)
    {
        for (int j = 0; j < full255.cols; j++)
        {
            full255.at<Vec3b>(i, j)[0] = 255;
            full255.at<Vec3b>(i, j)[1] = 255;
            full255.at<Vec3b>(i, j)[2] = 255;
        }
    }

    // kivonjuk a két Mat objectet egymásból
    invertsrc2 = full255 - invertsrc2;

    //255 - gsrc
    Mat gr255;
    cvtColor(full255, gr255, cv::COLOR_BGR2GRAY);
    negsrc = gr255 - negsrc;

    //Végig iterálunk a pixeleken és kivonjuk minden r/g/b értékét 255-ből
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

    imshow("Original", src);
    waitKey();
    imshow("Inverted 1", invertsrc);
    waitKey();
    imshow("Inverted 2", invertsrc2);
    waitKey();
    imshow("Inverted 3", invertsrc3);
    waitKey();
    imshow("Gray scale 1", gsrc1);
    waitKey();
    imshow("Gray scale 2", gsrc2);
    waitKey();
    imshow("Gray Inverted", negsrc);
    waitKey();
    return EXIT_SUCCESS;
}