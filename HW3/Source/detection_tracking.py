import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('D:\Softwares\Programming\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices


def skeleton_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x = c + w/2
    pt_y = r + h/2
    pt = (pt_x, pt_y)
    output.write("%d,%d,%d\n" %(frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]


def particle_filter_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x = c + w/2
    pt_y = r + h/2
    pt = (pt_x, pt_y)
    output.write("%d,%d,%d\n" %(frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    frame_hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)
    n_particles = 200
    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
    #f0 = particleevaluator(back_proj, pos) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
    im_h, im_w, channels = frame.shape
    stepsize = 6
    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector
        # Particle motion model: uniform step (TODO: find a better motion model)
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

        # Clip out-of-bounds particles
        #print particles.T
        particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)
        #print particles.T
        #print weights
        frame_hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)
        f = particleevaluator(back_proj, particles.T) # Evaluate particles
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
        #print 'After update'
        #print weights

        #if frameCounter%5 == 0:
        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights
        # write the result to the output file

        # For display purposes
        # frame2 = frame.copy()
        # cv2.circle(frame2, (int(pos[0]), int(pos[1])), 5, (255, 0, 0), -1)
        # for i in particles:
            # cv2.circle(frame2, (i[0],i[1]), 1, (0, 0, 255))
        # cv2.imshow('window', frame2)
        # cv2.waitKey(200)
        output.write("%d,%d,%d\n" % (frameCounter,pos[0],pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()



# Ref - https://docs.opencv.org/3.2.0/db/df8/tutorial_py_meanshift.html
def camshift_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x = c + w/2
    pt_y = r + h/2
    output.write("%d,%d,%d\n" %(frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    # print roi_hist.shape
    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    # Set terminating condition for camshift
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()
        frame_hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        confidence_map = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(confidence_map, track_window, term_crit)
        # print ret, track_window
        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        # print pts
        pt = (pts[0] + pts[2])/2
        pt_x = pt[0]
        pt_y = pt[1]
        #For displaying result
        # frame2 = frame.copy()
        # cv2.polylines(frame2,[pts],True, (0,0,255))
        # cv2.circle(frame2, (int(pt_x), int(pt_y)), 5, (255, 0, 0), -1)
        # cv2.imshow('window', frame2)
        # cv2.waitKey(100)
        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter, pt_x, pt_y)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()



def kalman_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    # output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    pt_x = c + w/2
    pt_y = r + h/2
    output.write("%d,%d,%d\n" % (frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # Initialise Kalman Filter
    kalman = cv2.KalmanFilter(4,2,0)
    state = np.array([pt_x,pt_y,0,0], dtype='float64') # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-4 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-2 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        prediction = kalman.predict()
        c,r,w,h = detect_one_face(frame)
        if((c,r,w,h) == (0,0,0,0)):
            posterior = prediction
        else:
            measurement = np.array([c + w/2,r + h/2], dtype='float64')
            posterior = kalman.correct(measurement)
        pt_x = posterior[0]
        pt_y = posterior[1]
        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()
        #print frameCounter
        #print prediction[0], prediction[1]
        #print measurement
        #print posterior[0], posterior[1]

        # For displaying result
        # frame2 = frame.copy()
        # if((c,r,w,h) != (0,0,0,0)):
            # cv2.rectangle(frame2,(c,r),(c+w,r+h),(0,0,255))
        # cv2.circle(frame2, (int(pt_x), int(pt_y)), 5, (255, 0, 0), -1)
        # cv2.imshow('window', frame2)
        # cv2.waitKey(100)
        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,pt_x,pt_y)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


'''
def optical_flow_tracker_gfd(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x = c + w/2
    pt_y = r + h/2
    pt = (pt_x, pt_y)
    output.write("%d,%d,%d\n" %(frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_roi = frame_gray[r:r+h, c:c+w]
    interest_points = cv2.goodFeaturesToTrack(face_roi, 50, 0.1, 5)
    
    # print interest_points
    # for i in range(len(interest_points)):
        # interest_points[i][0][0] += r
        # interest_points[i][0][1] += c
    # #for i in range(len(interest_points)):
    # print interest_points

    face_roi2 = face_roi.copy()
    for i in range(len(interest_points[0])):
        print interest_points[0][i], interest_points[0][i][0], interest_points[0][i][1]
        cv2.circle(face_roi2, (int(interest_points[0][i][1]),int(interest_points[0][i][0])), 1, (0, 0, 0))
    cv2.imshow('window', face_roi2)
    cv2.waitKey(0)
    interest_points_next = np.zeros_like(interest_points)
    
    while(1):
        interest_points_next = ()
        frame_prev = frame_gray.copy()
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print 'Interst points - ', interest_points
        interest_points_next = cv2.calcOpticalFlowPyrLK(frame_prev, frame_gray, interest_points, interest_points_next)
        #print interest_points_next
        # write the result to the output file
        frame2 = frame.copy()
        #cv2.rectangle(frame2,(c,r),(c+w,r+h),(0,0,255))
        #cv2.circle(frame2, (int(pos[0]), int(pos[1])), 5, (255, 0, 0), -1)
        for i in range(len(interest_points_next[0])):
            print interest_points_next[0][i][0][0], interest_points_next[0][i][0][1]
            cv2.circle(frame2, (int(interest_points_next[0][i][0][1]),int(interest_points_next[0][i][0][0])), 1, (0, 0, 0))
        cv2.imshow('window', frame2)
        cv2.waitKey(100)
        output.write("%d,%d,%d\n" % (frameCounter,0,0)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        #interest_points = np.array(interest_points_next)

    output.close()
'''

def optical_flow_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x = c + w/2
    pt_y = r + h/2
    pt = (pt_x, pt_y)
    output.write("%d,%d,%d\n" %(frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1
    
    while(1):
        frame_prev = frame.copy()
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        c,r,w,h = detect_one_face(frame)
        if((c,r,w,h) != (0,0,0,0)):
            pt_x = c + w/2
            pt_y = r + h/2
            cp,rp,wp,hp = c,r,w,h
        else:
            frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            points_to_track = []
            for i in range(cp+wp/2,cp+3*wp/2,10):
                for j in range(rp+hp/2,rp+3*hp/2,10):
                    curr_pt = np.zeros(2,dtype=np.float32)
                    curr_pt[0], curr_pt[1] = 1.0*i, 1.0*j
                    points_to_track.append(curr_pt)
            #points_next_frame = []
            points_to_track = np.array(points_to_track, dtype=np.float32)
            print points_to_track
            points_next_frame = points_to_track.copy()
            points_next_frame, st, err = cv2.calcOpticalFlowPyrLK(frame_prev_gray, frame_gray, points_to_track, None)
            frame2 = frame.copy()
            frame3 = frame.copy()
            for pos in points_next_frame:
                cv2.circle(frame2,(int(pos[0]), int(pos[0])), 2, (255, 0, 0))
            cv2.imshow('current frame',frame2)
            for pos in points_to_track:
                cv2.circle(frame3,(int(pos[0]), int(pos[1])), 2, (0, 0, 255))
            cv2.imshow('previous frame',frame3)
            cv2.waitKey(0)
            print frameCounter, points_to_track, points_next_frame
            #for i in range(len(points_next_frame)):
            
        output.write("%d,%d,%d\n" % (frameCounter,pt_x,pt_y)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        #interest_points = np.array(interest_points_next)

    output.close()



def optical_flow_tracker2(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x = c + w/2
    pt_y = r + h/2
    pt = (pt_x, pt_y)
    output.write("%d,%d,%d\n" %(frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100, qualityLevel = 0.1, minDistance = 2)
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (7,7), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    while(1):
        frame_prev = frame.copy()
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        c,r,w,h = detect_one_face(frame)
        if((c,r,w,h) != (0,0,0,0)):
            pt_x = c + w/2
            pt_y = r + h/2
            cp,rp,wp,hp = c,r,w,h
        else:
            frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #points_to_track = cv2.goodFeaturesToTrack(frame_prev_gray, mask = None, **feature_params)
            #points_next_frame, st, err = cv2.calcOpticalFlowPyrLK(frame_prev_gray, frame_gray, points_to_track, None, **lk_params)
            face_prev_gray = frame_prev_gray[rp:rp+hp,cp:cp+wp]
            face_curr_gray = frame_gray[rp:rp+hp,cp:cp+wp]
            points_to_track = cv2.goodFeaturesToTrack(face_prev_gray, mask = None, **feature_params)
            #print face_prev_gray.shape, face_curr_gray.shape, len(points_to_track)
            points_next_frame, st, err = cv2.calcOpticalFlowPyrLK(face_prev_gray, face_curr_gray, points_to_track, None, **lk_params)
            
            points_next_frame = points_next_frame[st == 1]
            frame2 = frame.copy()
            frame3 = frame_prev.copy()
            for pos in points_next_frame:
                #print pos
                cv2.circle(frame2,(int(cp + pos[0]), int(rp + pos[1])), 2, (255, 0, 0))
            pt = np.sum(points_next_frame,axis=0)/(1.0*points_next_frame.shape[0])
            pt_x, pt_y = cp + pt[0], rp + pt[1]
            cv2.circle(frame2,(int(pt_x), int(pt_y)), 2, (0, 255, 0))
            cv2.imshow('current frame',frame2)
            #print points_to_track
            for pos in points_to_track:
                #print pos
                cv2.circle(frame3,(int(cp + pos[0][0]), int(rp + pos[0][1])), 2, (0, 0, 255))
            cv2.imshow('previous frame',frame3)
            cv2.waitKey(100)

        output.write("%d,%d,%d\n" % (frameCounter,pt_x,pt_y)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        #interest_points = np.array(interest_points_next)

    output.close()


def optical_flow_tracker3(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x = c + w/2
    pt_y = r + h/2
    pt = (pt_x, pt_y)
    output.write("%d,%d,%d\n" %(frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100, qualityLevel = 0.1, minDistance = 2)
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (7,7), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    face_gray = cv2.cvtColor(frame[r:r+h,c:c+w], cv2.COLOR_BGR2GRAY)
    points_to_track = cv2.goodFeaturesToTrack(face_gray, mask = None, **feature_params)
    while(1):
        frame_prev = frame.copy()
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        c,r,w,h = detect_one_face(frame)
        if((c,r,w,h) != (0,0,0,0)):
            pt_x = c + w/2
            pt_y = r + h/2
            cp,rp,wp,hp = c,r,w,h
        else:
            frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #points_to_track = cv2.goodFeaturesToTrack(frame_prev_gray, mask = None, **feature_params)
            #points_next_frame, st, err = cv2.calcOpticalFlowPyrLK(frame_prev_gray, frame_gray, points_to_track, None, **lk_params)
            face_prev_gray = frame_prev_gray[rp:rp+hp,cp:cp+wp]
            face_curr_gray = frame_gray[rp:rp+hp,cp:cp+wp]
            #points_to_track = cv2.goodFeaturesToTrack(face_prev_gray, mask = None, **feature_params)
            #print face_prev_gray.shape, face_curr_gray.shape, len(points_to_track)
            #points_next_frame, st, err = cv2.calcOpticalFlowPyrLK(face_prev_gray, face_curr_gray, points_to_track, None, **lk_params)
            points_next_frame, st, err = cv2.calcOpticalFlowPyrLK(face_prev_gray, face_curr_gray, points_to_track, None, **lk_params)
            
            points_next_frame = points_next_frame[st == 1]
            frame2 = frame.copy()
            frame3 = frame_prev.copy()
            # for pos in points_next_frame:
                # print pos
                # cv2.circle(frame2,(int(cp + pos[0]), int(rp + pos[1])), 2, (255, 0, 0))
            pt = np.sum(points_next_frame,axis=0)/(1.0*points_next_frame.shape[0])
            pt_x, pt_y = cp + pt[0], rp + pt[1]
            #cv2.circle(frame2,(int(pt_x), int(pt_y)), 2, (0, 255, 0))
            #cv2.imshow('current frame',frame2)
            #print points_to_track
            # for pos in points_to_track:
                # #print pos
                # cv2.circle(frame3,(int(cp + pos[0][0]), int(rp + pos[0][1])), 2, (0, 0, 255))
            # cv2.imshow('previous frame',frame3)
            # cv2.waitKey(500)
        # frame3 = frame.copy()
        # cv2.circle(frame3,(int(pt_x), int(pt_y)), 2, (0, 0, 255))
        # cv2.imshow('window', frame3)
        # cv2.waitKey(200)
        output.write("%d,%d,%d\n" % (frameCounter,pt_x,pt_y)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()




if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camshift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        particle_filter_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        optical_flow_tracker3(video, "output_of.txt")

'''
For Kalman Filter:

# --- init

state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state


# --- tracking

prediction = kalman.predict()

# ...
# obtain measurement

if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)

# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:

# --- init

# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200

init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


# --- tracking

# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)

f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''
