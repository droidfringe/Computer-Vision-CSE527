# Computer-Vision-CSE527
Computer Vision Fall 2017 homeworks <br />
HW1 - Histograms, Filters, Deconvolution, Blending <br />
        - Perform histogram equalization on color image <br />
        - Do highpass and low pass filtering in frequency domain <br />
        - Given convolution mask and a blurred image, recover original image <br />
        - Perform Laplacian pyramid blending of two images <br />
HW2 - Panorama stitching <br />
        - Create panorama usng homographies and perspective warping on a common plane <br />
            - Compute corresponding points using SIFT featues between 2 images <br />
            - Estimate homography using corresponding points using RANSAC <br />
            - Flatten one image onto image plane of other using the computed homography <br />
        - Using cylindrical warping <br />
            - Transform image to cylindrical coordinates <br />
            - Use similar method to create cylindrical panorama <br />
HW3 - Detection and tracking <br />
        - Implement CAMShift, Kalman Filter, Particle Filter, Optical FLow trackers <br />
HW4 - Segmentation using SLIC and Graph cut <br />
        - Input: an image and sparse markings for foreground and background <br />
        - Calculated SLIC over image <br />
        - Calculated color histograms for all superpixels <br />
        - Calculated color histograms for FG and BG using provided sparse markings <br />
        - Constructd a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term) <br />
        - Used graph-cut algorithm to get the final segmentation <br />
        - Created interactive UI for specifying background and foreground, and showed segmentation results <br />
HW5 - Structured Light 3d Scanner <br />
        - Computed 2d-2d correspondence between projector and camera using binary code for each pixel <br />
        - Used 2d-2d correspondences to find 3d points <br />
        - Also added color to the estimated 3d point cloud  <br />
HW6 - Train CNN on MNIST <br />
        - Train an MNIST CNN classifier on just the digits: 1, 4, 5 and 9 <br />
        - Freeze first 4 layers of this network and train on digits 0, 2, 3, 6, 8 <br />
        - Use dropout and visualize learned conv layer filters <br />