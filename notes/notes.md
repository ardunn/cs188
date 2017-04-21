# CS 188 Project Notes

## 4/21/14

We use columns of X and y to determine cancer data

* X is a list of x vectors containing pixel data

* For example [a, d, t] where a is the ADC pixel intensity, d is the DWI pixel intensity, and t is the T2 pixel intensity.

* The x vectors are mapped to y scalars determined by a binary mask.

* The binary mask is an image file with 1's as cancerous regions and 0's as benign regions.

* Fabien recommends using vectorized image regions instead of individual pixels of x

* In other words, we increase the vector size (features) and reduce the number of data. We will experiment to see which is the best image subsection size.

* In addition to regular ML methods, Fabien suggests using deep learning.