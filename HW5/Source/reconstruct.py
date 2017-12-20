# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_color = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))
    #cv2.imshow('mask', proj_mask.astype(np.float32))
    #cv2.waitKey(0)
    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2)), (0,0), fx=scale_factor,fy=scale_factor)
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        # print(bit_code)
        # TODO: populate scan_bits by putting the bit_code according to on_mask
        scan_bits += bit_code*on_mask

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    #print binary_codes_ids_codebook.__class__
    correspondenceImg = np.zeros((h,w,3), dtype=np.uint8)
    #print correspondenceImg.shape
    camera_points = []
    camera_points_color = []
    projector_points = []
    matchCount = 0
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            p_x, p_y = binary_codes_ids_codebook[scan_bits[y,x]]
            if p_x >= 1279 or p_y >= 799: # filter
                continue
            matchCount += 1
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
            c_x, c_y = x/2.0, y/2.0
            camera_points.append([c_x, c_y])
            camera_points_color.append(ref_color[y,x,:])
            projector_points.append([p_x, p_y])
            correspondenceImg[y,x,2] = np.uint8((p_x/1280.0)*255)
            correspondenceImg[y,x,1] = np.uint8((p_y/800.0)*255)

    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    print matchCount
    # cv2.namedWindow('correnpondence', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('correnpondence', correspondenceImg)
    # cv2.waitKey(0)
    cv2.imwrite('correspondence.jpg', correspondenceImg)
    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # print camera_K
    # print camera_d
    camera_points_in = np.array([camera_points], dtype=np.float32)
    camera_points_normalized = cv2.undistortPoints(camera_points_in, camera_K, camera_d)
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    projector_points_in = np.array([projector_points], dtype=np.float32)
    projector_points_normalized = cv2.undistortPoints(projector_points_in, projector_K, projector_d)
    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    '''
    camera_proj_matrix = np.zeros((3,4))
    camera_proj_matrix[0:3,0:3] = camera_K
    projector_intrinsics = np.zeros((3,4))
    projector_intrinsics[0:3,0:3] = projector_K
    projector_extrinsics = np.zeros((4,4))
    projector_extrinsics[0:3,0:3] = projector_R
    projector_extrinsics[0:3,3] = projector_t.T
    projector_extrinsics[3,3] = 1.0
    projector_proj_matrix = np.matmul(projector_intrinsics, projector_extrinsics)
    '''
    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    camera_proj_matrix = np.zeros((3,4))
    camera_proj_matrix[0:3,0:3] = np.eye(3)
    projector_proj_matrix = np.zeros((3,4))
    projector_proj_matrix[0:3,0:3] = projector_R
    projector_proj_matrix[0:3,3] = projector_t.T
    
    points_homogeneous = cv2.triangulatePoints(camera_proj_matrix, projector_proj_matrix, camera_points_normalized, projector_points_normalized)
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    points_3d = cv2.convertPointsFromHomogeneous(points_homogeneous.T)
	# TODO: name the resulted 3D points as "points_3d"
    mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
    #points_3d_filtered = points_3d
    
    points_3d_filtered = []
    points_3d_filtered_with_color = []
    for i in range(points_3d.shape[0]):
        if(mask[i]):
            points_3d_filtered.append(points_3d[i,:,:])
            points_3d_filtered_with_color.append(np.zeros((1,6), dtype=np.float32))
            points_3d_filtered_with_color[-1][0,0:3] = points_3d[i,0,:]
            points_3d_filtered_with_color[-1][0,3:6] = 255*camera_points_color[i]
    write_3d_points_with_color(points_3d_filtered_with_color)
    return points_3d_filtered
    '''
    points_3d_filtered = np.array(points_3d_filtered)
    #rv,_ = cv2.Rodrigues(projector_R)
    rv,_ = cv2.Rodrigues(np.eye(3))
    tv = np.zeros((3,1), dtype=np.float32)
    print points_3d_filtered.shape
    proj_pts,_ = cv2.projectPoints(points_3d_filtered, rv, tv, camera_K, camera_d)

    min_z = np.min(points_3d_filtered[:,:,2])
    max_z = np.max(points_3d_filtered[:,:,2])

    depthImg = np.zeros_like(correspondenceImg)
    for i in range(proj_pts.shape[0]):
        #depthImg[np.int(proj_pts[i,0,1]), np.int(proj_pts[i,0,0]), 0] = np.uint8(255*(points_3d_filtered[i,0,2] - min_z)/(max_z - min_z))
        depthImg[np.int(proj_pts[i,0,1]), np.int(proj_pts[i,0,0]), 2] = np.uint8(255*(max_z - points_3d_filtered[i,0,2])/(max_z - min_z))
        depthImg[np.int(proj_pts[i,0,1]), np.int(proj_pts[i,0,0]), 1] = np.uint8(255*(points_3d_filtered[i,0,2] - min_z)/(max_z - min_z))

    cv2.imwrite('depthImgProf.png',depthImg)
    return points_3d_filtered
    '''
	
def write_3d_points(points_3d):

    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    #print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))

    return points_3d#, camera_points, projector_points

def write_3d_points_with_color(points_3d_with_color):

    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    #print(points_3d.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
        for p in points_3d_with_color:
            f.write("%d %d %d %d %d %d\n"%(p[0,0],p[0,1],p[0,2],p[0,5],p[0,4],p[0,3]))

    return points_3d_with_color#, camera_points, projector_points


if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)