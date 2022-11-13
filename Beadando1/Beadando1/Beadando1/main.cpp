#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace samples;

enum class szethuzas_tipus {
    linearis,
    negyzetes,
    gyokos
};

Mat szethuzas(Mat kep, int max, int min);
Mat gyokosSzethuzas(Mat kep, int max, int min);
Mat negyzetesSzethuzas(Mat kep, int max, int min);
Mat kiegyenlites(Mat kep, Mat hiszt, int N, int K);
Mat kepAtlagVagySzoras(Mat kep, int radiusz, Mat atlag);
Mat wallis(Mat kep, Mat atlag, Mat variancia, int kontraszt, float kont_mod, int fenyero, float feny_mod);
Mat outlier(Mat kep, int radiusz, int hatar);

void kepMentes(Mat kep, string nev);
constexpr const char* enumToString(szethuzas_tipus e);
unsigned char pixel(Mat kep, int c, int r);
int osszeHasonlit(const void* p1, const void* p2);

void hisztogramSzethuzas(Mat kep, szethuzas_tipus tipus, string nev);
void hisztogramKiegyenlites(Mat kep, string nev);
void konvulucio(Mat kep, string nev);
void wallisSzuro(Mat kep, string nev);
void nemlinearis(Mat kep, string nev);

const bool _uniform = true;
const bool _accumulate = false;
const int _rangeMin = 0;
const int _rangeMax = 256;
const int _radius = 3;
const int _hisztogramSzelesseg = 512;
const int _hisztogramMagassag = 400;

int main() {
    Mat lena = imread(findFile("images/lena.bmp"), IMREAD_GRAYSCALE);
    Mat lena_vilagos = imread(findFile("images/lena_vilagos.bmp"), IMREAD_GRAYSCALE);
    Mat bridge = imread(findFile("images/bridge.bmp"), IMREAD_GRAYSCALE);
    Mat boat_sotet = imread(findFile("images/boat_sotet.bmp"), IMREAD_GRAYSCALE);
    Mat airplane = imread(findFile("images/airplane.bmp"), IMREAD_GRAYSCALE);
    Mat peppers_sotet = imread(findFile("images/peppers_sotet.bmp"), IMREAD_GRAYSCALE);
    Mat peppers_vilagos = imread(findFile("images/peppers_vilagos.bmp"), IMREAD_GRAYSCALE);
    Mat barbara_gauss = imread(findFile("images/0.025.bmp"), IMREAD_GRAYSCALE);
    Mat barbara_saltpepper = imread(findFile("images/0.1.bmp"), IMREAD_GRAYSCALE);
    Mat montage = imread(findFile("images/montage.jpg"), IMREAD_GRAYSCALE);
    Mat mopntage_zajos = imread(findFile("images/montage_zajos.jpg"), IMREAD_GRAYSCALE);

    //2.1
    hisztogramSzethuzas(peppers_sotet, szethuzas_tipus::linearis, "peppers_sotet");
    hisztogramSzethuzas(boat_sotet, szethuzas_tipus::linearis, "boat_sotet");

    hisztogramSzethuzas(peppers_sotet, szethuzas_tipus::negyzetes, "peppers_sotet");
    hisztogramSzethuzas(peppers_vilagos, szethuzas_tipus::negyzetes, "peppers_vilagos");
    hisztogramSzethuzas(lena_vilagos, szethuzas_tipus::negyzetes, "lena_vilagos");

    hisztogramSzethuzas(peppers_sotet, szethuzas_tipus::gyokos, "peppers_sotet");
    hisztogramSzethuzas(peppers_vilagos, szethuzas_tipus::gyokos, "peppers_vilagos");
    hisztogramSzethuzas(lena_vilagos, szethuzas_tipus::gyokos, "lena_vilagos");

    //2.2
    hisztogramKiegyenlites(airplane, "airplane");
    hisztogramKiegyenlites(lena_vilagos, "lena_vilagos");


    //2.3.1
    
    //2.4
    wallisSzuro(montage, "montage");
    wallisSzuro(bridge, "bridge");

    //2.5
    nemlinearis(mopntage_zajos, "montage_zajos");
    nemlinearis(barbara_gauss, "barbara_gauss");
    nemlinearis(barbara_saltpepper, "barbara_saltpepper");

    waitKey();

    return EXIT_SUCCESS;
}

void kepMentes(Mat kep, string nev) {
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(90);
    bool result = false;

    try {
        result = imwrite("images/" + nev + ".jpg", kep, compression_params);
    } catch (const cv::Exception& ex) {
        fprintf(stderr, "Exception saving file: %s\n", ex.what());
    }

    if (result) { printf("File saved.\n"); }
    else { printf("ERROR: Can't save file.\n"); }

    compression_params.pop_back();
    compression_params.pop_back();
}

constexpr const char* enumToString(szethuzas_tipus e) {
    switch (e)  {
        case szethuzas_tipus::linearis: return "linearis";
        case szethuzas_tipus::negyzetes: return "negyzetes";
        case szethuzas_tipus::gyokos: return "gyokos";
        default: return "[ERROR]";
    }
}

unsigned char pixel(Mat kep, int c, int r) {

    if (c < 0) { c = 0; }
    if (c >= kep.cols) { c = kep.cols - 1; }
    if (r < 0) { r = 0; }
    if (r >= kep.rows) { r = kep.rows - 1; }

    return kep.at<unsigned char>(r, c);
}

int osszeHasonlit(const void* p1, const void* p2)
{
    return *(const unsigned char*)p1 - *(const unsigned char*)p2;
}

Mat szethuzas(Mat kep, int max, int min) {
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = 255 * (kep.at<unsigned char>(i, j) - min) / (max - min);
        }
    }
    return kep;
}

Mat gyokosSzethuzas(Mat kep, int max, int min) {
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = 255 * sqrtf((kep.at<unsigned char>(i, j) - min) / (max - min * 1.0f));
        }
    }
    return kep;
}

Mat negyzetesSzethuzas(Mat kep, int max, int min) {
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = 255 * pow((kep.at<unsigned char>(i, j) - min) / (max - min * 1.0f), 2);
        }
    }
    return kep;
}

Mat kiegyenlites(Mat kep, Mat hiszt, int N, int K) {
    int table[256];
    float ossz = 0;
    int i = 0;
    for (int j = 0; j < 256; j++) {
        if (ossz < N / K) {
            ossz += hiszt.at<float>(j);
        }
        else {
            i++;
            ossz = 0;
        }
        table[j] = i * (float)256 / K;
    }
    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {
            kep.at<unsigned char>(i, j) = table[kep.at<unsigned char>(i, j)];
        }
    }
    return kep;
}

Mat kepAtlagVagySzoras(Mat kep, int radiusz, Mat atlag) {
    Mat eredmeny = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int eredmeny_row = 0; eredmeny_row < eredmeny.rows; eredmeny_row++) {
        for (int eredmeny_column = 0; eredmeny_column < eredmeny.cols; eredmeny_column++) {
            float sum = 0;
            for (int kep_row = eredmeny_row - radiusz; kep_row <= eredmeny_row + radiusz; kep_row++) {
                for (int kep_column = eredmeny_column - radiusz; kep_column <= eredmeny_column + radiusz; kep_column++) {
                    if (!atlag.empty()) {
                        sum += pow((pixel(kep, kep_column, kep_row) - pixel(atlag, eredmeny_column, eredmeny_row)), 2);
                    }
                    else {
                        sum += pixel(kep, kep_column, kep_row);
                    }
                }
            }
            eredmeny.at<unsigned char>(eredmeny_row, eredmeny_column) = sum / (float)pow((2 * radiusz + 1), 2);
        }
    }

    return eredmeny;
}

Mat wallis(Mat kep, Mat atlag, Mat variancia, int kontraszt, float kont_mod, int fenyero, float feny_mod) {
    Mat wallis = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int wallis_row = 0; wallis_row < wallis.rows; wallis_row++) {
        for (int wallis_column = 0; wallis_column < wallis.cols; wallis_column++) {
            auto tmp = ((kep.at<unsigned char>(wallis_row, wallis_column) - atlag.at<unsigned char>(wallis_row, wallis_column)) * (((double)kont_mod * kontraszt) / (kontraszt + (kont_mod * sqrt(variancia.at<unsigned char>(wallis_row, wallis_column)))))) + (((double)feny_mod * fenyero) + ((1.0f - feny_mod) * atlag.at<unsigned char>(wallis_row, wallis_column)));

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

Mat outlier(Mat kep, int radiusz, int hatar) {
    Mat atlag = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int atlag_row = 0; atlag_row < atlag.rows; atlag_row++) {
        for (int atlag_column = 0; atlag_column < atlag.cols; atlag_column++) {
            float sum = 0;

            for (int src_r = atlag_row - radiusz; src_r <= atlag_row + radiusz; src_r++) {
                for (int src_c = atlag_column - radiusz; src_c <= atlag_column + radiusz; src_c++) {

                    if (src_r == atlag_row && src_c == atlag_column) {
                        continue;
                    }

                    sum += pixel(kep, src_c, src_r);
                }
            }
            atlag.at<unsigned char>(atlag_row, atlag_column) = sum / (float)(pow((2 * radiusz + 1), 2) - 1);
        }
    }

    Mat kimenet = Mat::zeros(kep.rows, kep.cols, kep.type());

    for (int kimenet_row = 0; kimenet_row < kimenet.rows; kimenet_row++) {
        for (int kimenet_column = 0; kimenet_column < kimenet.cols; kimenet_column++) {
            if (abs(atlag.at<unsigned char>(kimenet_row, kimenet_column) - kep.at<unsigned char>(kimenet_row, kimenet_column)) <= hatar)
                kimenet.at<unsigned char>(kimenet_row, kimenet_column) = kep.at<unsigned char>(kimenet_row, kimenet_column);
            else
                kimenet.at<unsigned char>(kimenet_row, kimenet_column) = atlag.at<unsigned char>(kimenet_row, kimenet_column);
        }
    }

    return kimenet;
}

void hisztogramSzethuzas(Mat kep, szethuzas_tipus tipus, string nev) {
    int hisztogramMeret = _rangeMax;
    float range[] = { -_rangeMin, _rangeMax };
    const float* hisztogramRange = { range };

    Mat hisztogram;
    calcHist(&kep, 1, 0, Mat(), hisztogram, 1, &hisztogramMeret, &hisztogramRange, _uniform, _accumulate);

    imshow(nev, kep);

    std::vector<float> hisztogramTomb;
    if (hisztogram.isContinuous()) {
        hisztogramTomb.assign((float*)hisztogram.data, (float*)hisztogram.data + hisztogram.total() * hisztogram.channels());
    }
    else {
        for (int i = 0; i < hisztogram.rows; ++i) {
            hisztogramTomb.insert(hisztogramTomb.end(), hisztogram.ptr<float>(i), hisztogram.ptr<float>(i) + (long)hisztogram.cols * hisztogram.channels());
        }
    }

    int szethuzasMertek = cvRound((double)_hisztogramSzelesseg / hisztogramMeret);

    Mat hisztogramKep(_hisztogramMagassag, _hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(hisztogram, hisztogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(hisztogramKep, cv::Point(szethuzasMertek * (i - 1), _hisztogramMagassag - cvRound(hisztogram.at<float>(i - 1))),
            Point(szethuzasMertek * (i), _hisztogramMagassag - cvRound(hisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev + " alap hisztogram", hisztogramKep);
    kepMentes(hisztogramKep, nev + "_alap_hisztogram");

    int max;
    int min;

    for (int i = 0; i < hisztogramTomb.size(); i++) {
        if (hisztogramTomb[i] != 0) {
            min = i;
            break;
        }
    }

    for (int i = hisztogramTomb.size() - 1; i >= 0; i--) {
        if (hisztogramTomb[i] != 0) {
            max = i;
        }
    }

    Mat szurkeKepSzethuzas;

    if (tipus == szethuzas_tipus::linearis)
        szurkeKepSzethuzas = szethuzas(kep.clone(), max, min);
    if (tipus == szethuzas_tipus::gyokos)
        szurkeKepSzethuzas = gyokosSzethuzas(kep.clone(), max, min);
    if (tipus == szethuzas_tipus::negyzetes)
        szurkeKepSzethuzas = negyzetesSzethuzas(kep.clone(), max, min);

    imshow(nev + enumToString(tipus) + " szethuzott kep", szurkeKepSzethuzas);
    kepMentes(szurkeKepSzethuzas, nev + "_" + enumToString(tipus) + "_szethuzott");

    Mat szethuzottHisztogram;
    calcHist(&szurkeKepSzethuzas, 1, 0, Mat(), szethuzottHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat szethuzottHisztogramKep(_hisztogramMagassag, _hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(szethuzottHisztogram, szethuzottHisztogram, 0, szethuzottHisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(szethuzottHisztogramKep, cv::Point(szethuzasMertek * (i - 1), _hisztogramMagassag - cvRound(szethuzottHisztogram.at<float>(i - 1))),
            Point(szethuzasMertek * (i), _hisztogramMagassag - cvRound(hisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev + enumToString(tipus) + " szethuzott hisztogram", szethuzottHisztogramKep);
    kepMentes(szethuzottHisztogramKep, nev + "_" + enumToString(tipus) + "_szethuzott_hisztogram");
}

void hisztogramKiegyenlites(Mat kep, string nev) {
    int hisztogramMeret = 256;
    float range[] = { 0, 256 };
    const float* hisztogramRange = { range };

    Mat szurkeSkalasHisztogram;
    calcHist(&kep, 1, 0, Mat(), szurkeSkalasHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat hisztogramTable = szurkeSkalasHisztogram.clone();

    int hisztogramSzelesseg = 512;
    int hisztogramMagassag = 400;
    int bin_w = cvRound((double)hisztogramSzelesseg / hisztogramMeret);

    Mat szurkeSkalasHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(szurkeSkalasHisztogram, szurkeSkalasHisztogram, 0, szurkeSkalasHisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(szurkeSkalasHisztogramKep, cv::Point(bin_w * (i - 1), hisztogramMagassag - cvRound(szurkeSkalasHisztogram.at<float>(i - 1))),
            Point(bin_w * (i), hisztogramMagassag - cvRound(szurkeSkalasHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev, kep);
    imshow(nev + " hisztogram", szurkeSkalasHisztogramKep);
    kepMentes(szurkeSkalasHisztogramKep, nev + "_hisztogram");

    Mat negyArnyalatosKiegyenlitettKep = kiegyenlites(kep.clone(), hisztogramTable, kep.size().area(), 4);
    Mat negyArnyalatosKiegyenlitettKepHisztogram;
    Mat negyArnyalatosKiegyenlitettKepHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&negyArnyalatosKiegyenlitettKep, 1, 0, Mat(), negyArnyalatosKiegyenlitettKepHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);
    normalize(negyArnyalatosKiegyenlitettKepHisztogram, negyArnyalatosKiegyenlitettKepHisztogram, 0, negyArnyalatosKiegyenlitettKepHisztogramKep.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(negyArnyalatosKiegyenlitettKepHisztogramKep, cv::Point(bin_w * (i - 1), hisztogramMagassag - cvRound(negyArnyalatosKiegyenlitettKepHisztogram.at<float>(i - 1))),
            cv::Point(bin_w * (i), hisztogramMagassag - cvRound(negyArnyalatosKiegyenlitettKepHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev + " kiegyenlitett kep, 4 arnyalat", negyArnyalatosKiegyenlitettKep);
    kepMentes(negyArnyalatosKiegyenlitettKep, nev + "_4_arnyalatos_kiegyenlitett");
    imshow(nev + " kiegyenlitett hisztogram, 4 arnyalat", negyArnyalatosKiegyenlitettKepHisztogramKep);
    kepMentes(negyArnyalatosKiegyenlitettKepHisztogramKep, nev + "_4_arnyalatos_kiegyenlitett_hisztogram");

    Mat tizenhatArnyalatosKiegyenlitettKep = kiegyenlites(kep.clone(), hisztogramTable, kep.size().area(), 16);
    Mat tizenhatArnyalatosKiegyenlitettKepHisztogram;
    Mat tizenhatArnyalatosKiegyenlitettKepHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));

    calcHist(&tizenhatArnyalatosKiegyenlitettKep, 1, 0, Mat(), tizenhatArnyalatosKiegyenlitettKepHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);
    normalize(tizenhatArnyalatosKiegyenlitettKepHisztogram, tizenhatArnyalatosKiegyenlitettKepHisztogram, 0, tizenhatArnyalatosKiegyenlitettKepHisztogramKep.rows, cv::NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(tizenhatArnyalatosKiegyenlitettKepHisztogramKep, cv::Point(bin_w * (i - 1), hisztogramMagassag - cvRound(tizenhatArnyalatosKiegyenlitettKepHisztogram.at<float>(i - 1))),
            cv::Point(bin_w * (i), hisztogramMagassag - cvRound(tizenhatArnyalatosKiegyenlitettKepHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev + " kiegyenlitett kep, 16 arnyalat", tizenhatArnyalatosKiegyenlitettKep);
    kepMentes(tizenhatArnyalatosKiegyenlitettKep, nev + "_16_arnyalatos_kiegyenlitett");
    imshow(nev + " kiegyenlitett hisztogram, 16 arnyalat", tizenhatArnyalatosKiegyenlitettKepHisztogramKep);
    kepMentes(negyArnyalatosKiegyenlitettKepHisztogramKep, nev + "_16_arnyalatos_kiegyenlitett_hisztogram");
}

void konvulucio(Mat kep, string nev) {
    int hisztogramMeret = 256;
    float range[] = { 0, 256 };
    const float* hisztogramRange = { range };

    Mat hisztogram;
    calcHist(&kep, 1, 0, Mat(), hisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

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

    Mat alap = kep.clone();
    imshow(nev , kep);
    imshow(nev + " konvolucio Elotti Histogram", hisztogramKep);
    kepMentes(hisztogramKep, nev + "_hisztogram");

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
    calcHist(&kep, 1, 0, Mat(), szethuzottHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat szethuztottHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(szethuzottHisztogram, szethuzottHisztogram, 0, szethuztottHisztogramKep.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(szethuztottHisztogramKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(szethuzottHisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(hisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev + " konvulucio Utan", utanaKep);
    kepMentes(utanaKep, nev + "_konvolucio");
    imshow(nev + " konvulucio Utan Hisztogram", szethuztottHisztogramKep);
    kepMentes(szethuztottHisztogramKep, nev + "_konvolucio_hisztogram");
}

void wallisSzuro(Mat kep, string nev) {
    int hisztogramMeret = 256;
    float range[] = { 0, 256 };
    const float* hisztogramRange = { range };

    Mat hisztogram;
    calcHist(&kep, 1, 0, Mat(), hisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

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

    imshow(nev , kep);
    imshow(nev + " hisztogram", hisztogramKep);
    kepMentes(hisztogramKep, nev + "_hisztogram");

    Mat atlagKep = kepAtlagVagySzoras(kep, 8, cv::Mat());
    Mat szorasKep = kepAtlagVagySzoras(kep, 8, atlagKep);
    Mat wallisKep = wallis(kep, atlagKep, szorasKep, 100, 2.5f, 50, 0.8);

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

    imshow(nev + " wallis kep", wallisKep);
    kepMentes(wallisKep, nev + "_wallis");
    imshow(nev + " wallis hisztogram", wallisHisztogramKep);
    kepMentes(wallisHisztogramKep, nev + "_wallis_hisztogram");
}

void nemlinearis(Mat kep, string nev) {
    int hisztogramMeret = 256;
    float range[] = { 0, 256 };
    const float* hisztogramRange = { range };

    Mat hisztogram;
    calcHist(&kep, 1, 0, Mat(), hisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

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

    imshow(nev, kep);
    imshow(nev +" hisztogram", hisztogramKep);
    kepMentes(hisztogramKep, nev + "_hisztogram");

    Mat outlierKep = outlier(kep, 3, 35);
    Mat outlierHisztogram;
    calcHist(&outlierKep, 1, 0, Mat(), outlierHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat outlierHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(outlierHisztogram, outlierHisztogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(outlierHisztogramKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(outlierHisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(outlierHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev + " outlier szurt kep", outlierKep);
    kepMentes(outlierKep, nev + "_outlier");
    imshow(nev + " outlier szurt hisztogram", outlierHisztogramKep);
    kepMentes(outlierHisztogramKep, nev + "_outlier_hisztogram");

    Mat median = Mat::zeros(kep.rows, kep.cols, kep.type());
    const int hossz = (2 * 3 + 1) * (2 * 3 + 1);

    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {

            unsigned char pixels[hossz];
            int index = 0;
            for (int k = i - 3; k <= i + 3; k++) {
                for (int l = j - 3; l <= j + 3; l++) {

                    int k2 = k, l2 = l;
                    if (k < 0) {
                        k2 = 0;
                    }
                    if (k >= kep.rows) {
                        k2 = kep.rows - 1;
                    }
                    if (l < 0) {
                        l2 = 0;
                    }
                    if (l >= kep.cols) {
                        l2 = kep.cols - 1;
                    }
                    pixels[index] = kep.at<unsigned char>(k2, l2);
                    index++;
                }
            }
            qsort(pixels, hossz, sizeof(unsigned char), osszeHasonlit);
            median.at<unsigned char>(i, j) = pixels[(hossz + 1) / 2];
        }
    }

    Mat medianHisztogram;
    calcHist(&median, 1, 0, Mat(), medianHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat medianHisztorgamKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(medianHisztogram, medianHisztogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(medianHisztorgamKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(medianHisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(medianHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev + " median szurt kep", median);
    kepMentes(median, nev + "_median");
    imshow(nev + " median szurt hisztogram", medianHisztorgamKep);
    kepMentes(medianHisztorgamKep, nev + "_median_hisztogram");

    const int r = 3;
    Mat gyors_median = Mat::zeros(kep.rows, kep.cols, kep.type());
    const int hossz2 = 2 * r + 1;

    for (int i = 0; i < kep.rows; i++) {
        for (int j = 0; j < kep.cols; j++) {

            unsigned char pixels[hossz2];
            int index = 0;
            for (int k = i - r; k <= i + r; k++) {

                unsigned char pixels2[hossz2];
                int index2 = 0;
                for (int l = j - r; l <= j + r; l++) {

                    int k2 = k, l2 = l;
                    if (k < 0) {
                        k2 = 0;
                    }
                    if (k >= kep.rows) {
                        k2 = kep.rows - 1;
                    }
                    if (l < 0) {
                        l2 = 0;
                    }
                    if (l >= kep.cols) {
                        l2 = kep.cols - 1;
                    }
                    pixels2[index2] = kep.at<unsigned char>(k2, l2);
                    index2++;
                }
                qsort(pixels2, hossz2, sizeof(unsigned char), osszeHasonlit);
                pixels[index] = pixels2[(hossz2 + 1) / 2];
                index++;
            }
            qsort(pixels, hossz2, sizeof(unsigned char), osszeHasonlit);
            gyors_median.at<unsigned char>(i, j) = pixels[(hossz2 + 1) / 2];
        }
    }

    Mat gyorsMedianHisztogram;
    calcHist(&gyors_median, 1, 0, Mat(), gyorsMedianHisztogram, 1, &hisztogramMeret, &hisztogramRange, true, false);

    Mat gyorsMedianHisztogramKep(hisztogramMagassag, hisztogramSzelesseg, CV_8UC3, cv::Scalar(0, 0, 0));
    normalize(gyorsMedianHisztogram, gyorsMedianHisztogram, 0, hisztogramKep.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < hisztogramMeret; i++)
    {
        line(gyorsMedianHisztogramKep, cv::Point(oszlopSzelesseg * (i - 1), hisztogramMagassag - cvRound(gyorsMedianHisztogram.at<float>(i - 1))),
            Point(oszlopSzelesseg * (i), hisztogramMagassag - cvRound(gyorsMedianHisztogram.at<float>(i))),
            cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    imshow(nev + " gyors Median szurt kep", gyors_median);
    kepMentes(gyors_median, nev + "_gyors_median");
    imshow(nev + " gyors Median szurt hisztogram", gyorsMedianHisztogramKep);
    kepMentes(gyorsMedianHisztogramKep, nev + "_gyors_median_hisztogram");
}