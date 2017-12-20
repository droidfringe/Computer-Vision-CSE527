# Computer-Vision-CSE527
Computer Vision Fall 2017 homeworks <br />
HW1 - Histograms, Filters, Deconvolution, Blending
        - Perform histogram equalization on color image
        - Do highpass and low pass filtering in frequency domain
        - Given convolution mask and a blurred image, recover original image
        - Perform Laplacian pyramid blending of two images
HW2 - Panorama stitching
        - Create panorama usng homographies and perspective warping on a common plane
            - Compute corresponding points using SIFT featues between 2 images
            - Estimate homography using corresponding points using RANSAC
            - Flatten one image onto image plane of other using the computed homography
        - Using cylindrical warping
            - Transform image to cylindrical coordinates
            - Use similar method to create cylindrical panorama
HW3 - Detection and tracking
        - Implement CAMShift, Kalman Filter, Particle Filter, Optical FLow trackers
HW4 - Segmentation using SLIC and Graph cut
        - Input: an image and sparse markings for foreground and background
        - Calculated SLIC over image
        - Calculated color histograms for all superpixels
        - Calculated color histograms for FG and BG using provided sparse markings
        - Constructd a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
        - Used graph-cut algorithm to get the final segmentation
        - Created interactive UI for specifying background and foreground, and showed segmentation results
HW5 - Structured Light 3d Scanner
        - Computed 2d-2d correspondence between projector and camera using binary code for each pixel
        - Used 2d-2d correspondences to find 3d points
        - Also added color to the estimated 3d point cloud 
HW6 - Train CNN on MNIST
        - Train an MNIST CNN classifier on just the digits: 1, 4, 5 and 9
        - Freeze first 4 layers of this network and train on digits 0, 2, 3, 6, 8
        - Use dropout and visualize learned conv layer filters