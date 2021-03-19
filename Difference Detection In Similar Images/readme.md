This project provide an approach to detect differences in two similar images which may be different in perspectives，gray levels or scales.
To correctly detect the differences, it is necessary to find the transformations between the two images.
So, in this project I used SIFT to find the Homography between the two images. With Homography, we can compare the corresponding points between the two images.
