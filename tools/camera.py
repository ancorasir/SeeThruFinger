import cv2  # OpenCV library 
import time # time library
from threading import Thread # library for multi-threading

class WebcamStream:
    # initialization method 
    def __init__(self, stream_id=4):
        self.stream_id = stream_id # default is 0 for main camera 
        self.image_w = 1280
        self.image_h = 720
        # opening video capture stream 
        self.vcap = cv2.VideoCapture(self.stream_id)
        self.vcap.set(3,self.image_w) #设置分辨率
        self.vcap.set(4,self.image_h)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False 
        self.stopped = True
        # thread instantiation  
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background 
        self.start()
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t.start()
    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()
    # method to return latest read frame 
    def read(self):
        return self.frame
    # method to stop reading frames 
    def stop(self):
        self.stopped = True

# webcam_stream = WebcamStream(stream_id=4) # 0 id for main camera
# # processing frames in input stream
# num_frames_processed = 0 
# start = time.time()
# for i in range(300):
#     frame = webcam_stream.read()
#     print(frame.shape)
#     # adding a delay for simulating video processing time 
    
# end = time.time()
# print('fps:',300/(end-start))
# webcam_stream.stop()