#include <cstdlib>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace std;
using namespace cv;
using namespace cuda;

float photometric_error(Mat &m1, Mat &m2, float u, float v)
{
  return 0;
}

int main()
{
  Mat img1 = imread("../data/testimg/b001.jpg");
  Mat img2 = imread("../data/testimg/b005.jpg");

  resize(img1, img1, Size(480, 640), 0, 0, CV_INTER_CUBIC);
  resize(img2, img2, Size(480, 640), 0, 0, CV_INTER_CUBIC);

  Mat gray1, gray2;
  cvtColor(img1, gray1, COLOR_BGR2GRAY);
  cvtColor(img2, gray2, COLOR_BGR2GRAY);

  int PYR_LEVEL = 7;
  vector<Mat> pyr1, pyr2;
  pyr1.push_back(gray1);
  pyr2.push_back(gray2);
  for(int i=1; i<=PYR_LEVEL; i++)
  {
    Mat m1, m2;
    pyrDown(pyr1.back(), m1);
    pyrDown(pyr2.back(), m2);
    pyr1.push_back(m1);
    pyr2.push_back(m2);
  }


  

  // Mat disp;
  // imshow("X", disp);

  imshow("frame1", gray1);
  imshow("frame2", gray2);

  waitKey(0);

}
