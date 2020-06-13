from model_detector.modelDetector import *
from modelTracker import *
import os
import imutils
from imutils.video import FPS
import sys
import requests
import time
import imagezmq

######### INITIALIZE PARAMETERS  ###########
# Path to Video
(H_frame,W_frame) = (None,None)
url = "http://192.168.1.86:8080/shot.jpg"

#CONFIG INPUT PARAMETERS
nbr_frames_tracking=10
methodTracker = "dlib_correlation"

#Model detector/tracker
personTracker = personTracker(maxNbrFramesDisappeared=50,maxDistance=50)
modelDetector_SSD = model_detector(path_protocol=path_protocol_model_SSD,
                                   path_weight=path_weight_model_SSD,
                                   classes=CLASSES_MobileNet_SSD)
modelTracker=modelTracker(model_detector=modelDetector_SSD, frame_size=frameSize(W_frame,H_frame),
                          personTracker=personTracker,methodTracker=methodTracker, confidence=0.2)

#Results
total_In = [0] # use "list" for passing by reference
total_Out = [0]
total = [0]

#total nbr of frames
totalFrames = [0] # use "list" for passing by reference in modelTracker.run() function

# Writing output video
save_video=True
writer=None
#################END#######################

##### LOADING AND PROCESSING VIDEO ###########
print("[INFO] starting streaming webcam thread...")

# Loop frame
while True:
    img_resp= requests.get(url)
    img_arr= np.array(bytearray(img_resp.content),dtype=np.uint8)
    frame = cv2.imdecode(img_arr,-1)
    if frame is not None:
        # resize the frame to have a maximum width of 512 pixels,
        # then convert the frame from BGR to RGB
        frame = imutils.resize(frame, width=512)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get size of frame
        if W_frame is None or H_frame is None:
            (H_frame, W_frame) = frame.shape[:2]
            modelTracker.set_size_frame(H_frame, W_frame)
            print("Height Frame:", H_frame,"Width Frame:",W_frame)

        # Initialize output writer
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter('video.avi', fourcc, 24,
                                    (W_frame, H_frame), True)

        #Run model in each frame
        modelTracker.run_model(frame,totalFrames,total_In,total_Out,nbr_frames_tracking)
        #print('Total In: ',total_In,'Total Out: ',total_Out)

        if writer is not None:
            writer.write(frame)

        # show the frame and update the FPS counter
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    else:
        break

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# Cleanup
cv2.destroyAllWindows()
sys.exit(1)





