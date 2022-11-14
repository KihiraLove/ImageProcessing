#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <utility>
#include <chrono>
#include <thread>


using namespace cv;


//North, East, West, South
enum direction {
    N,
    E,
    S,
    W
};

struct Bug {
    direction dir;
    int pointI;
    int pointJ;
};

direction rotate(direction dir) {
    switch (dir)
    {
    case N:
        return S;
        break;
    case W:
        return E;
        break;
    case S:
        return N;
        break;
    case E:
        return W;
        break;
    default:
        return N;
        break;
    }
}


direction changeRight(direction dir) {
    switch (dir)
    {
    case N:
        return E;
        break;
    case E:
        return S;
        break;
    case S:
        return W;
        break;
    case W:
        return N;
        break;
    default:
        return N;
        break;
    }
}

direction changeLeft(direction dir) {
    switch (dir)
    {
    case N:
        return W;
        break;
    case W:
        return S;
        break;
    case S:
        return E;
        break;
    case E:
        return N;
        break;
    default:
        return N;
        break;
    }
}

std::pair<int,int> findFirstPixel(Mat *src) {
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            int pixel = (int) src->at<unsigned char>(i, j);
            if (pixel == 255) {
                return std::pair<int, int>(i, j);
            }
        }
    }
    return std::pair<int, int>(-1, -1);
}

bool isInObject(Bug& bug, Mat& src) {
    return src.at<unsigned char>(bug.pointI, bug.pointJ) != 0;
}

void moveBug(Bug &bug, Mat &src)
{
    int height = src.rows;
    int width = src.cols;


    switch (bug.dir)
    {
    case direction::W:
        if (0 < bug.pointJ) {
            bug.pointJ--;
        }
        else {
            bug.dir = changeLeft(bug.dir);
        }
        break;
    case direction::E:
        if (width > bug.pointJ) {
            bug.pointJ++;
        }
        else {
            bug.dir = changeRight(bug.dir);
        }
        break;
    case direction::N:
        if (0 < bug.pointI) {
            bug.pointI--;
        }
        else {
            bug.dir = changeRight(bug.dir);
        }
        break;
    case direction::S:
        if (height > bug.pointI) {
            bug.pointI++;
        }
        else {
            bug.dir = changeRight(bug.dir);
        }
        break;
    default:
        break;
    }
}

Mat bugFollow(Mat& src)
{
    Mat trackImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    std::pair<int, int> firstLocation = findFirstPixel(&src);
    int firstPointI = findFirstPixel(&src).first;
    int firstPointJ = findFirstPixel(&src).second - 1;

    Bug bug;
    bug.pointI = firstPointI;
    bug.pointJ = firstPointJ;
    bug.dir = direction::N;

    int i = 0;
    do {
        int pixel = src.at<uchar>(bug.pointI, bug.pointJ);
        if (pixel == 255) {
            bug.dir = changeLeft(bug.dir);
            trackImg.at<uchar>(bug.pointI, bug.pointJ) = 255;
        }
        else bug.dir = changeRight(bug.dir);
        //std::cout << "i" << bug.pointI << "j" << bug.pointJ << std::endl;
        
        
        moveBug(bug, src);
    } while (!(bug.pointI == firstPointI && bug.pointJ == firstPointJ && bug.dir == direction::N));
    //imshow("Bug follow", trackImg);
    return trackImg;
}
Mat backtrackBugFollow(Mat &src) {
   
    Mat trackImg(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    std::pair<int, int> firstLocation = findFirstPixel(&src);
    int firstPointI = findFirstPixel(&src).first;
    int firstPointJ = findFirstPixel(&src).second - 1;

    Bug bug;
    bug.pointI = firstPointI;
    bug.pointJ = firstPointJ;
    bug.dir = direction::N;

    int i = 0;
    do {
        int pixel = src.at<uchar>(bug.pointI, bug.pointJ);
        if (pixel == 255) {
            bug.dir = rotate(bug.dir);
            trackImg.at<uchar>(bug.pointI, bug.pointJ) = 255;
        }
        else bug.dir = changeRight(bug.dir);
        //std::cout << "i" << bug.pointI << "j" << bug.pointJ << std::endl;


        moveBug(bug, src);
    } while (!(bug.pointI == firstPointI && bug.pointJ == firstPointJ && bug.dir == direction::N));
    //imshow("Bug follow", trackImg);
    return trackImg;
}

int main(int argc, char** argv)
{
    //CommandLineParser parser(argc, argv, "{@input | bug.bmp | input image}");
    CommandLineParser parser(argc, argv, "{@input | bug3.bmp | input image}");
    Mat src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_GRAYSCALE);
    Mat path(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
    if (src.empty())
    {
        return EXIT_FAILURE;
    }
  
    Mat bugFollowImage = bugFollow(src);
    Mat bugBacktrackImage = backtrackBugFollow(src);
    imshow("Source", src);
    imshow("Bug follow", bugFollowImage);
    imshow("Bug backtrack", bugBacktrackImage);
    waitKey();
    return EXIT_SUCCESS;
}