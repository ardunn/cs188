# CS 188 Project Notes

## 4/21/17

We use columns of X and y to determine cancer data

* X is a list of x vectors containing pixel data

* For example [a, d, t] where a is the ADC pixel intensity, d is the DWI pixel intensity, and t is the T2 pixel intensity.

* The x vectors are mapped to y scalars determined by a binary mask.

* The binary mask is an image file with 1's as cancerous regions and 0's as benign regions.

* Fabien recommends using vectorized image regions instead of individual pixels of x

* In other words, we increase the vector size (features) and reduce the number of data. We will experiment to see which is the best image subsection size.

* In addition to regular ML methods, Fabien suggests using deep learning.

## 4/25/17

As a follow up to data discussion above, the specific technique used to map the relationship between image pixel data is [Kernel Spectral Regression](http://www.cad.zju.edu.cn/home/dengcai/Data/SR.html). 

From the [paper](https://c.ymcdn.com/sites/siim.org/resource/resmgr/siim2016abstracts/Image_Tan.pdf) presented by Dr. Scalzo and Dr. Tan:
> A Kernel Spectral Regression (KSR) was used to model the nonlinear relation between T2, high b-value DWI and ADC
images and location of prostate tumor on whole mount
