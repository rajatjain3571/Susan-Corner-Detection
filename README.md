# Susan-Corner-Detection
Detecting Corners using susan corner detection algorithm

Smallest Univalue Segment Assimilating Nucleus (SUSAN): Smith and Brady (IJCV, Vol. 23(1), pp.
45–78, 1997) presented an entirely different approach to the 1D and 2D feature detection in images, such as
edges and corners, respectively. The SUSAN corner detection algorithm does not require derivative operation;
hence, it can work well with noisy images. The main idea of the SUSAN is the use of a mask to count the num-
ber of pixels having the same brightness as the center pixel. By comparing the number of pixel having the same
brightness as the center pixel with a threshold, the detector can determine whether the center pixel is a corner.


Circular mask M consisting of 37 pixels should be used. The central pixel of the mask is called a nucleus. Then
intensities of all pixels within a mask are compared with an intensity of a nucleus and an area of “similar” pixels is
marked. This area is called USAN (Univalue Segment Assimilating Nucleus) and it conveys the most important
information on a local structure of an image. Analyzing the size, centroid and the second moments of USAN the
exact information on a type of local structure around a nucleus is inferred, such as edges or corners. For those
regions, inverted USAN area shows strong peaks –thus the term SUSAN – i.e. the smallest USAN. This approach
has an additional advantage of not using any derivatives which are cumbersome to use in the presence of noise.


Place a circular mask around a pixel (i.e., nucleus)
• Calculate the number n(r 0 ) of pixels within the circular mask which have similar brightness to the
nucleus in accordance with Equation 3.1. Such pixels constitute the USAN.
• Compute the strength of a corner from Equation 3.2.
• Use non-max suppression to find corners.
