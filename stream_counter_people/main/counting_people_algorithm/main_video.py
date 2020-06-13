from stream_counter_people.main.counting_people_algorithm.modelTracker import *
from stream_counter_people.main.counting_people_algorithm.model_detector.modelDetector import *
import os
import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
import sys
import time
import imagezmq

######### INITIALIZE PARAMETERS  ###########
# Path to Video
(H_frame,W_frame) = (None,None)
source_video="input/example_01.mp4"
path_video = os.path.join(os.getcwd(),source_video)

#CONFIG INPUT PARAMETERS
nbr_frames_tracking=10
methodTracker = "csrt"

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
save_video=False
writer=None
#################END#######################

##### LOADING AND PROCESSING VIDEO ###########
print("[INFO] starting video file thread...")
fvs = FileVideoStream(path_video).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()

# Loop frame
while fvs.more():
    frame = fvs.read()
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
        if save_video == True:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter('video.avi', fourcc, 24,
                                        (W_frame, H_frame), True)

        #Run model in each frame
        modelTracker.run_model(frame,totalFrames,total_In,total_Out,nbr_frames_tracking)
        #print('Total In: ',total_In,'Total Out: ',total_Out)
        if save_video == True:
            if writer is not None:
                writer.write(frame)

        # show the frame and update the FPS counter
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        fps.update()
    else:
        break
# stop the timer and display FPS information
fps.stop()
print("[INFO] Total time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if save_video == True:
    if writer is not None:
        writer.release()

# Cleanup
cv2.destroyAllWindows()
fvs.stop()
sys.exit(1)





