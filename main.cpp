/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/
#include <math.h>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "filter.h"
#include "imconv.h"
#include "segment-image.h"
//#include "cv.h"
//#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

#define ITER 5       // number of BP iterations at each scale
#define LEVELS 1     // number of scales

#define DISC_K 1.7F         // truncation of discontinuity cost
#define DATA_K 15.0F        // truncation of data cost
#define LAMBDA 0.07F        // weighting of data cost

#define INF 1E20     // large cost
#define VALUES 255    // number of possible disparities
#define SCALE 16     // scaling from disparity to graylevel in output

#define SIGMA 0.7    // amount to smooth the input images
#define CLEARBAND 20

#define focus_x 228
#define focus_y 172

// dt of 1d function
static void dt(float f[VALUES]) {
	for (int q = 1; q < VALUES; q++) {
		float prev = f[q - 1] + 1.0F;
		if (prev < f[q])
			f[q] = prev;
	}
	for (int q = VALUES - 2; q >= 0; q--) {
		float prev = f[q + 1] + 1.0F;
		if (prev < f[q])
			f[q] = prev;
	}
}

// compute message
void msg(float s1[VALUES], float s2[VALUES], float s3[VALUES], float s4[VALUES], float dst[VALUES]) {
	float val;

	// aggregate and find min
	float minimum = INF;
	for (int value = 0; value < VALUES; value++) {
		dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
		if (dst[value] < minimum)
			minimum = dst[value];
	}

	int a = 0;
	// dt
	dt(dst);

	// truncate 

	minimum += DISC_K;
	for (int value = 0; value < VALUES; value++)
	if (minimum < dst[value])
		dst[value] = minimum;


	// normalize
	val = 0;
	for (int value = 0; value < VALUES; value++)
		val += dst[value];

	val /= VALUES;
	for (int value = 0; value < VALUES; value++)
		dst[value] -= val;
}

// computation of data costs
image<float[VALUES]> *comp_data(vec2di &represent_xy, vec2di &each_pix, int segCnt) {
	//int width = each_pix.size();
	int height = each_pix[0].size();

	image<float[VALUES]> *data = new image<float[VALUES]>(segCnt, 1);



	float val;
	//float center_cycle = 50;
	int focusSeg = each_pix[focus_x][focus_y]; // get the focus segment
	int focus_h=represent_xy[focusSeg][1];
	float def;
	int max_y;



	/* find the max y distance */
	if (focus_h > height - focus_h){
		max_y = focus_h;
	}
	else{
		max_y = height - focus_h;
	}
	max_y=max_y-CLEARBAND;
	/* compute the data cost */
	cout << "default label" << endl;
	for (int i = 0; i < segCnt; i++){
		//val = abs(abs(represent_xy[i][1] - focus_h) - CLEARBAND) / (float)max_y * 255;
		val = abs(represent_xy[i][1] - focus_h)- CLEARBAND ;
		if(val >=0) val=val/ (float)max_y * 255;
		else val=0;
		cout << i<< ":" << val << endl;
		for (int lbl = 0; lbl<VALUES; lbl++){
			imRef(data, i, 0)[lbl] = abs(lbl - val);
		}
	}

	// delete sm1;
	return data;
}

// generate output from current messages
image<uchar> *output(image<float[VALUES]> *message,
	image<float[VALUES]> *data, vec2di &each_pix, vec2di &represent_xy, int segCnt) {

	int width = each_pix.size();
	int height = each_pix[0].size();

	image<uchar> *out = new image<uchar>(width, height);
	image<uchar> *label_for_seg = new image<uchar>(segCnt, 1); // Record the label for every segment representative
	
	/* First, Get the label of every segment*/
	cout << "final label" << endl;
	for (int i = 0; i < segCnt; i++)
	{
		// keep track of best value for current pixel
		int best = 0;
		float best_val = INF;
		for (int value = 0; value < VALUES; value++){
			float val =
				imRef(message, i, 0)[value] +
				imRef(message, i, 1)[value] +
				imRef(message, i, 2)[value] +
				imRef(message, i, 3)[value] +
				imRef(data, i, 0)[value];
			if (val < best_val){
				best_val = val;
				best = value;
			}
		}
		imRef(label_for_seg, i, 0)=best;
		//imRef(label_for_seg, i, 0)=255-best;
		cout << i << ":" << best << endl;
	}

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			int ind = each_pix[i][j];
			imRef(out, i, j) = imRef(label_for_seg, ind, 0);
		}
	}
	return out;
}

// belief propagation using checkerboard update scheme
void bp_cb(image<float[VALUES]> **message,
	image<float[VALUES]> *data, vec2di &neighbor) {

	int segCnt = neighbor.size();
	image<float[VALUES]> *oldMes = *message;
	image<float[VALUES]> *newMes;


	int neiCnt;
	int ind;
	for (int t = 0; t < ITER; t++){
		newMes = new image<float[VALUES]>(4, segCnt);

		for (int i = 0; i < segCnt; i++){
			//std::cout << "iter " << t << "\n";
			neiCnt = neighbor[i].size();

			if (neiCnt >= 1){ //  1 neighbor
				ind = find(neighbor[neighbor[i][0]].begin(), neighbor[neighbor[i][0]].end(), i) - neighbor[neighbor[i][0]].begin();
				msg(imRef(oldMes, i, 1), imRef(oldMes, i, 2), imRef(oldMes, i, 3)
					, imRef(data, i, 0), imRef(newMes, neighbor[i][0], ind));
			}
			if (neiCnt >= 2){ //  2 neighbor
				ind = find(neighbor[neighbor[i][1]].begin(), neighbor[neighbor[i][1]].end(), i) - neighbor[neighbor[i][1]].begin();
				msg(imRef(oldMes, i, 0), imRef(oldMes, i, 2), imRef(oldMes, i, 3)
					, imRef(data, i, 0), imRef(newMes, neighbor[i][1], ind));
			}
			if (neiCnt >= 3){ //  3 neighbor
				ind = find(neighbor[neighbor[i][2]].begin(), neighbor[neighbor[i][2]].end(), i) - neighbor[neighbor[i][2]].begin();
				msg(imRef(oldMes, i, 1), imRef(oldMes, i, 0), imRef(oldMes, i, 3)
					, imRef(data, i, 0), imRef(newMes, neighbor[i][2], ind));
			}
			if (neiCnt >= 4){ // 4 neighbor
				ind = find(neighbor[neighbor[i][3]].begin(), neighbor[neighbor[i][3]].end(), i) - neighbor[neighbor[i][3]].begin();
				msg(imRef(oldMes, i, 1), imRef(oldMes, i, 2), imRef(oldMes, i, 0)
					, imRef(data, i, 0), imRef(newMes, neighbor[i][3], ind));
			}
		}
		delete oldMes;
		oldMes = newMes;
	}
	*message = oldMes;
}



// multiscale belief propagation for image restoration
image<uchar> *stereo_ms(vec2di &represent_xy, vec2di &each_pix, vec2di &neighbor) {

	image<float[VALUES]> *data[LEVELS];
	image<float[VALUES]> *message[LEVELS];



	int width = each_pix.size();
	int height = each_pix[0].size();
	int segCnt = neighbor.size();

	// data costs
	data[0] = comp_data(represent_xy, each_pix, segCnt);


	// run bp from coarse to fine
	for (int i = LEVELS - 1; i >= 0; i--) {
		//int width = data[i]->width();
		//int height = data[i]->height();

		// allocate & init memory for messages
		if (i == LEVELS - 1) {
			// in the coarsest level messages are initialized to zero
			/*u[i] = new image<float[VALUES]>(width, height);
			d[i] = new image<float[VALUES]>(width, height);
			l[i] = new image<float[VALUES]>(width, height);
			r[i] = new image<float[VALUES]>(width, height);*/
			message[i] = new image<float[VALUES]>(segCnt, 4);
		}
		/*
		else {
		// initialize messages from values of previous level
		u[i] = new image<float[VALUES]>(width, height, false);
		d[i] = new image<float[VALUES]>(width, height, false);
		l[i] = new image<float[VALUES]>(width, height, false);
		r[i] = new image<float[VALUES]>(width, height, false);

		for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
		for (int value = 0; value < VALUES; value++) {
		imRef(u[i], x, y)[value] = imRef(u[i+1], x/2, y/2)[value];
		imRef(d[i], x, y)[value] = imRef(d[i+1], x/2, y/2)[value];
		imRef(l[i], x, y)[value] = imRef(l[i+1], x/2, y/2)[value];
		imRef(r[i], x, y)[value] = imRef(r[i+1], x/2, y/2)[value];
		}
		}
		}
		// delete old messages and data
		delete u[i+1];
		delete d[i+1];
		delete l[i+1];
		delete r[i+1];
		delete data[i+1];
		}
		*/

		// BP
		//bp_cb(u[i], d[i], l[i], r[i], data[i], ITER, neighbor);
		bp_cb(&message[i], data[i], neighbor);
	}

	image<uchar> *out = output(message[0], data[0], each_pix, represent_xy, segCnt);

	/*delete u[0];
	delete d[0];
	delete l[0];
	delete r[0];*/
	delete message[0];
	delete data[0];

	return out;
}

image<rgb> *convert_n_load(char **argv){

	printf("loading input image.\n");

	char inFileName[100], outFileName[100];
	strcpy_s(inFileName, argv[4]);
	strcpy_s(outFileName, "ch.ppm");

	//IplImage *changeFile;
	/* Convert jpg to ppm */
	Mat changeFile = imread(inFileName, IMREAD_UNCHANGED);
	if(!changeFile.empty())
	{
		imwrite(outFileName, changeFile);
	}

	return loadPPM(outFileName);
}

/*
void blur(image<rgb> *input, image<uchar> *label){

	int width = input->width();
	int height = input->height();
		
	int left_bound,right_bound,up_bound,down_bound;
	double value;
	vector<vector<rgb> > src(19 ,vector<rgb>(19));
	vector<vector<rgb> > dst(19 ,vector<rgb>(19));
	vector<vector<rgb> > result(width ,vector<rgb>(height));


	for(int i=30; i< (width-30) ;i++){
		for(int j=30; j< (height-30) ;j++){
			value = ceil((double(imRef(label, i, j)))/255*100);
			if ( value != 0){
				left_bound = i-10;right_bound = i+10;
				up_bound = j-10;down_bound = j+10;

				if ( left_bound < 0 )			left_bound = 0;
				if ( right_bound >= width )		right_bound = width;
				if ( up_bound < 0 )				up_bound = 0;
				if ( down_bound >= height )		down_bound = height;

				for(int y=0;y<19;y++){
					for(int x=0;x<19;x++){
						src[x][y] = imRef(input,left_bound+x,up_bound+y);
					}
				}
				 
				GaussianBlur( src, dst, Size( 19, 19 ),value/20);

				for(int y=0;y<19;y++){
					for(int x=0;x<19;x++){
						result[left_bound+x][up_bound+y] = dst[x][y];
					}
				}
			}
		}
	}

	
	////////  show blurred image  //////////
	imwrite( "Gray_Image.pgm", result);
	IplImage * imagew2 = 0;
	imagew2->imageData;
	imagew2 = cvLoadImage("Gray_Image.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	cvNamedWindow("Gray_Image", CV_WINDOW_AUTOSIZE);
	cvShowImage("Gray_Image", imagew2);
	cvWaitKey(0);
	cvDestroyWindow("Gray_Image");	
}
*/

void blur(char *filename, image<uchar> *out){
	vector<Mat*> r;
	int width = out->width();
	int height = out->height();

	Mat result;
	Mat img = imread( filename, 1 );
	result = img.clone();

	int left_bound,right_bound,up_bound,down_bound;
	double label;

	for(int i=0; i< width ;i++){
		for(int j=0; j< height ;j++){
			label = ceil((double(imRef(out, i, j)))/255*100);
			if ( label != 0){
				left_bound = i-10;
				right_bound = i+10;
				up_bound = j-10;
				down_bound = j+10;

				if ( left_bound < 0 )
				left_bound = 0;

				if ( right_bound >= width )
				right_bound = width;

				if ( up_bound < 0 )
				up_bound = 0;

				if ( down_bound >= height )
				down_bound = height;

				GaussianBlur( img(Range(up_bound,down_bound),Range (left_bound,right_bound)), 
					result(Range(up_bound,down_bound),Range (left_bound,right_bound)), Size( 11, 11 ),label/20);
			}
		}
	}
	
	////////  show blurred image  //////////
	string resultFileName = "[result]" + string(filename);
	imwrite( resultFileName, result);
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Display window", result );                   // Show our image inside it.
	waitKey();
	destroyAllWindows();

}

int main(int argc, char **argv) {
	image<uchar> *out;
	image<uchar> *label;

	//int represent_xy[1000][2]; //每個segment的代表點
	vec2di each_pix; // each pixels' representative
	vec2di neighbor;
	vec2di represent_xy;
	////////  over segmentation  ////////////
	float sigma = atof(argv[1]);
	float k = atof(argv[2]);
	int min_size = atoi(argv[3]);
	int num_ccs;

	image<rgb> *input = convert_n_load(argv);
	segment_image(input, sigma, k, min_size, &num_ccs, represent_xy, each_pix, neighbor);

	////////////  bp vision  ////////////////
	// compute disparities
	label = stereo_ms(represent_xy, each_pix, neighbor);

	///////////  save label(0~255) /////////
	savePGM(label, "label.pgm");
	Mat imgLabel = imread("label.pgm");
	string imgLabelName = "[label]" + string(argv[4]);
	imwrite(imgLabelName, imgLabel);

	// show label result
	namedWindow("Blur Label", WINDOW_AUTOSIZE);
	imshow("Blur Label", imgLabel);


	////////  blur image  //////////////////
	blur(argv[4], label);

	////////  show blurred image  //////////
	/*
	imwrite("Blur_Result.jpg", result);

	IplImage * imagew2 = 0;
	imagew2->imageData;
	imagew2 = cvLoadImage("Blur_Result.jpg", CV_LOAD_IMAGE_COLOR);
	cvNamedWindow("Blur_Result", CV_WINDOW_AUTOSIZE);
	cvShowImage("Blur_Result", imagew2);
	cvWaitKey(0);
	cvDestroyWindow("Blur_Result");
	*/
	//delete out;
	return 0;
}
