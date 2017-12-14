#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#define drawCross( center, color, d ) \
line( img, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( img, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

using namespace cv;
using namespace std;

cv::RotatedRect getErrorEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat);

static void help()
{
	printf("\nOpenCV's Kalman filter example.\n"
		"   Tracking mouse position.\n"
		"   Original code from.\n"
		"   http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html\n"
		"\n"
		"   Pressing any key (except ESC) will reset the tracking with a different speed.\n"
		"   Pressing ESC will stop the program.\n"
	);
}

bool init = false;
Point init_mousePos;
Point mousePos;

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
	 if (event == EVENT_RBUTTONDOWN)
	 {
		 std::cout << "Initial position (" << x << ", " << y << ")" << std::endl;
		 if (!init) {
			 init = true;
			 init_mousePos.x = x;
			 init_mousePos.y = y;
		 }
	 }
	 else if (event == EVENT_MOUSEMOVE) {
		mousePos.x = x;
		mousePos.y = y;
	}
}

int main()
{
	help();

	std::cout << '\n' << "Right-click on a point to start...";
	
	namedWindow("mouse kalman", 1);
	setMouseCallback("mouse kalman", mouseCallback, NULL);
	// Image to show mouse tracking
	Mat img(600, 800, CV_8UC3);
	img = Scalar::all(0);

	while (!init) {
		imshow("mouse kalman", img);
		waitKey(30);
	}

	KalmanFilter KF(4, 2, 0);

	// intialization of KF...

	// Transition Matrix
	// [ 1 0 dT 0  ]
	// [ 0 1 0  dT ]
	// [ 0 0 1  0  ]
	// [ 0 0 0  1  ]
	KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

	Mat_<float> measurement(2, 1);
	measurement(0) = (float)init_mousePos.x;
	measurement(1) = (float)init_mousePos.y;

	KF.statePre.at<float>(0) = (float)init_mousePos.x;
	KF.statePre.at<float>(1) = (float)init_mousePos.y;
	KF.statePre.at<float>(2) = 0.f;
	KF.statePre.at<float>(3) = 0.f;

	KF.statePost.at<float>(0) = (float)init_mousePos.x;
	KF.statePost.at<float>(1) = (float)init_mousePos.y;
	KF.statePost.at<float>(2) = 0.f;
	KF.statePost.at<float>(3) = 0.f;

	// Measurement Matrix H
	// [ 1 0 0 0 ]
	// [ 0 1 0 0 ]
	// [ 0 0 1 0 ]
	// [ 0 0 0 1 ]
	setIdentity(KF.measurementMatrix);

	// Process Noise Covariance Matrix
	// [ Ex 0  0    0    ]
	// [ 0  Ey 0    0    ]
	// [ 0  0  Ev_x 0    ]
	// [ 0  0  0    Ev_y ]
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));

	std::cout << "Process Noise Covariance Matrix" << std::endl;
	std::cout << KF.processNoiseCov << std::endl;

	setIdentity(KF.measurementNoiseCov, Scalar::all(10));

	std::cout << "Measurement Noise Covariance Matrix" << std::endl;
	std::cout << KF.measurementNoiseCov << std::endl;

	setIdentity(KF.errorCovPost, Scalar::all(.1));
	
	std::vector<Point> mousev, kalmanv;
	mousev.clear();
	kalmanv.clear();

	namedWindow("mouse kalman", 1);
	setMouseCallback("mouse kalman", mouseCallback, NULL);

	char key = (char)-1;

	while (1)
	{
		// First predict, to update the internal statePre variable
		Mat prediction = KF.predict();
		Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

		std::cout << "predictPt: " << predictPt << std::endl;

		// Get mouse point
		measurement(0) = mousePos.x;
		measurement(1) = mousePos.y;

		// The update phase
		Mat estimated = KF.correct(measurement);

		Point statePt(estimated.at<float>(0), estimated.at<float>(1));
		Point measPt(measurement(0), measurement(1));

		std::cout << "statePt: " << statePt << std::endl;
		std::cout << "measPt: " << measPt << std::endl;

		cv::Point2f mean(statePt.x, statePt.y);
		cout << "KF.statePost " << KF.statePost << endl;
		//cv::RotatedRect ellipse = getErrorEllipse(2.4477, mean, KF.statePost);

		//std::cout << "ellipse: " << ellipse.size << std::endl;

		//cv::ellipse(img, ellipse, cv::Scalar::all(255), 2);

		// plot points
		imshow("mouse kalman", img);
		img = Scalar::all(0);

		mousev.push_back(measPt);
		kalmanv.push_back(statePt);
		drawCross(statePt, Scalar(255, 255, 255), 5);
		drawCross(measPt, Scalar(0, 0, 255), 5);

		for (int i = 0; i < mousev.size() - 1; i++)
			line(img, mousev[i], mousev[i + 1], Scalar(255, 255, 0), 1);

		for (int i = 0; i < kalmanv.size() - 1; i++)
			line(img, kalmanv[i], kalmanv[i + 1], Scalar(0, 155, 255), 1);

		key = (char)waitKey(30);

		if (key == 27 || key == 'q' || key == 'Q')
			break;
		else if (key == 'c' || key == 'C') {
			img = Scalar::all(0);
			mousev.clear();
			kalmanv.clear();
		}
	}

	return 0;
}

cv::RotatedRect getErrorEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat) {

	//Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(covmat, eigenvalues, eigenvectors);

	//Calculate the angle between the largest eigenvector and the x-axis
	double angle = atan2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(0, 0));

	//Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
	if (angle < 0)
		angle += 6.28318530718;

	//Conver to degrees instead of radians
	angle = 180 * angle / 3.14159265359;

	//Calculate the size of the minor and major axes
	double halfmajoraxissize = chisquare_val*sqrt(eigenvalues.at<double>(0));
	double halfminoraxissize = chisquare_val*sqrt(eigenvalues.at<double>(1));

	//Return the oriented ellipse
	//The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
	return cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize), -angle);

}
