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

web:http://cs.brown.edu/people/pfelzens/segment/index.html
*/

/*
Parameter Example

Segmentation parameters: sigma = 0.5, K = 500, min = 50.
Segmentation parameters: sigma = 0.5, K = 500, min = 50.
*/

#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include <cstdlib>
#include <stdlib.h>
#include <cstdio>
//#include <vector>
//#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "image.h"
#include "misc.h"
#include "filter.h"
#include "segment-graph.h"

#define vec2di vector<vector<int> >
#define vec2db vector<vector<bool> >
using namespace std;

// random color
rgb random_rgb(){ 
  rgb c;
  double r;
  
  c.r = (uchar)rand();
  c.g = (uchar)rand();
  c.b = (uchar)rand();

  return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b,
			 int x1, int y1, int x2, int y2) {
  return sqrt(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
	      square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
	      square(imRef(b, x1, y1)-imRef(b, x2, y2)));
}


int search_for_represent_index(int represent,int number, vector<int> arr_represent)
{   
	int index;
	for (int i = 0; i < number; i++)
	{
		if (represent == arr_represent[i])
		{
			index = i;
			break;
		}
	}
	return index;
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
void segment_image(image<rgb> *im, float sigma, float c, int min_size,
			  int *num_ccs, vec2di &represent_xy, vec2di &each_pix, vec2di &neighbor) {
  int width = im->width();
  int height = im->height();
  int represent_index = 0;
  vector<int> represent; // record the representative of every neighbor

  /* Initialize each_pix */
  for (int i = 0; i < width; i++)
  {
	  vector<int> cr;
	  for (int j = 0; j < height; j++)
	  {
		  cr.push_back(-1);
	  }
	  each_pix.push_back(cr);
  }

  image<float> *r = new image<float>(width, height);
  image<float> *g = new image<float>(width, height);
  image<float> *b = new image<float>(width, height);

  // smooth each color channel  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      imRef(r, x, y) = imRef(im, x, y).r;
      imRef(g, x, y) = imRef(im, x, y).g;
      imRef(b, x, y) = imRef(im, x, y).b;
    }
  }
  image<float> *smooth_r = smooth(r, sigma);
  image<float> *smooth_g = smooth(g, sigma);
  image<float> *smooth_b = smooth(b, sigma);
  delete r;
  delete g;
  delete b;
 
  // build graph
  edge *edges = new edge[width*height*4];
  int num = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x < width-1) {
	edges[num].a = y * width + x;
	edges[num].b = y * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
	num++;
      }

      if (y < height-1) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + x;
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
	num++;
      }

      if ((x < width-1) && (y < height-1)) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
	num++;
      }

      if ((x < width-1) && (y > 0)) {
	edges[num].a = y * width + x;
	edges[num].b = (y-1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
	num++;
      }
    }
  }
  delete smooth_r;
  delete smooth_g;
  delete smooth_b;
  printf( "Number of edges: %d\n", num);

  // segment
  universe *u = segment_graph(width*height, num, edges, c);
  

  // post process small components
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }

  int num_pix = width * height;
  for (int i = 0; i < num_pix; i++)
  {   
	  /*Find all representative of segment*/
	  if (i == u->find(i))
	  {   
		  represent.push_back(i);
	  }
  }
  
  /* Initialize represent_xy */
  for (int i = 0; i < u->num_sets(); i++)
  {
	  vector<int> cr;
	  for (int j = 0; j < 2; j++)
	  {
		  cr.push_back(-1);
	  }
	  represent_xy.push_back(cr);
  }
  
  for(int i = 0; i < u->num_sets(); i++)
  {
	  represent_xy[i][0] = ((represent[i]+width) % width);
	  represent_xy[i][1] = (represent[i] / width);
  }

  /*Record each pixels' representative(index)*/
  for (int i = 0; i < width; i++)
  {
	  for (int j = 0; j < height; j++)
	  {
		  each_pix[i][j] = search_for_represent_index(u->find(j*width + i), u->num_sets(), represent);
	  }
  }

  
  vec2db istaken;
  const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
  const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

  for (int i = 0; i < width; i++)
  {
	  vector<bool> nb;
	  for (int j = 0; j < height; j++)
	  {
		  nb.push_back(false);
	  }
	  istaken.push_back(nb);
  }

  neighbor.resize(u->num_sets());
  /* Go through all the pixels. */
  
  for (int i = 0; i < width; i++) {
	  for (int j = 0; j < height; j++) {
		  int nr_p = 0;

		  //new edit
		  int represent_now = each_pix[i][j];

		  /* Compare the pixel to its 8 neighbours. */
		  for (int k = 0; k < 8; k++) {
			  int x = i + dx8[k], y = j + dy8[k];

			  if (x >= 0 && x < width && y >= 0 && y < height) {

				  if (each_pix[i][j] == 1 && each_pix[x][y] == 4)
				  {
					  int a = 1;
				  }

				  if (istaken[x][y] == false && each_pix[i][j] != each_pix[x][y]) {
					  nr_p += 1;
					  //new edit
					  if (nr_p >= 2)
					  {

						  int isneighbor = 0;
						  if (neighbor[represent_now].size() == 0)
							  neighbor[each_pix[i][j]].push_back(each_pix[x][y]);
						  
						  else if (neighbor[represent_now].size()  < 4){
							  for (int z = 0; z<neighbor[represent_now].size(); z++) {
								  if (neighbor[represent_now][z] == each_pix[x][y]) {
									  isneighbor = 1;
									  break;
								  }
							  }

							  if (isneighbor == 0)
								  neighbor[each_pix[i][j]].push_back(each_pix[x][y]);
						  }
					  }
					  ///////////////////////////////////
				  }
			  }
		  }
		  if (nr_p >= 2) {
			  istaken[i][j] = true;
		  }
	  }
  }
  
  delete [] edges;
  *num_ccs = u->num_sets();
  
  delete u;

}

#endif
