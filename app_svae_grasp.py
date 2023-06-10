import cv2,os,sys, time
import numpy as np
from datetime import datetime
sys.path.append(sys.path[0]+"./tracker")
sys.path.append(sys.path[0]+"./tracker/model")
sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")
import rtde_control
import rtde_receive
from threading import Thread
from tools.camera import WebcamStream
from tools.DHgripper import DHgripper
import matplotlib.pyplot as plt
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# calibration with wider gripper width
dst = np.zeros((4, 2), dtype = "float32")
dst[0] = [0.195, -0.605] #, 0.3214800275939105]
dst[1] = [0.195, -1.005]
dst[2] = [-0.205, -1.005]
dst[3] = [-0.205, -0.63]
rect = np.zeros((4, 2), dtype = "float32") #(v,u)
rect[0] = [664,98]
rect[1] = [1113,114]
rect[2] = [1120,603               ]
rect[3] = [671,627]

M = cv2.getPerspectiveTransform(rect, dst)

class Robot():
    def __init__(self):
        super().__init__()
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
    def move(self,target_pose):
        self.rtde_c.moveL(target_pose, 0.1, 0.5)

def sweep():
    global model, dir, cam9, gripper
    timestamps = []
    images = []
    masks = []
    i = 0
    gripper.Position(1000)
    robot.rtde_c.moveL(home, 0.1, 0.5, False)
    robot.rtde_c.moveL(home2, 0.05, 0.3, True)
    s = time.time()
    print('Start sweeping')
    while abs(robot.rtde_r.getActualTCPPose()[5]-home2[5])>0.002:
        timestamps.append(time.time()-s)
        color_image = cam9.read()  
        img_reduced = color_image[::2,::2,:]
        images.append(img_reduced)
        hh = time.time()
        mask, logit = model.xmem.track(img_reduced)
        masks.append(mask)  
        i += 1
        print('Sweeping %s frames'%i) 
    masks = np.array(masks)
    images = np.array(images)
    print(images.shape)
    inpainted_frames = model.baseinpainter.inpaint(images[:,:,320:600,:], masks[:,:,320:600], ratio=1)
    out = cv2.VideoWriter(dir+'/inpaint_720-360_cam9_%s.mov'%(time.time()), cv2.VideoWriter_fourcc(*'jpeg'), len(timestamps)/timestamps[-1], (280,360))
    out1 = cv2.VideoWriter(dir+'/original_720-360_cam9_%s.mov'%(time.time()), cv2.VideoWriter_fourcc(*'jpeg'), len(timestamps)/timestamps[-1], (640,360))
    for i in range(len(timestamps)):
        painted_image = mask_painter(images[i], (masks[i]==1).astype('uint8'), mask_color=1+1)
        data = np.concatenate([painted_image,inpainted_frames[i]],1)
        out.write(inpainted_frames[i])
        out1.write(images[i])
    out.release()
    out1.release()
    return inpainted_frames, color_image

from forcesensor.base_forcesensor import BaseForceSensor
from tracker.base_tracker import BaseTracker
class FT_stream:
    # initialization method 
    def __init__(self, svae_checkpoint, xmem,cam):
        self.forcesensor = BaseForceSensor(svae_checkpoint, device="cuda:0")
        self.xmem = xmem
        self.cam = cam
        self.fts = [np.zeros([1,6])]
        self.images = []
        self.timestamps = []
        self.masks = []
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
        s=time.time()
        while True :
            if self.stopped is True :
                break
            self.timestamps.append(time.time()-s)
            color_image = self.cam.read()  
            img_reduced = color_image[::2,::2,:]
            self.images.append(img_reduced)
            mask, logit = self.xmem.track(img_reduced)
            self.masks.append(mask)
            ft = self.forcesensor.measure( mask[:,140:500], 9 )
            self.fts.append( ft.reshape([1,6]) ) 
    # method to return latest read frame 
    def read(self):
        return self.fts[-1][0]
    # method to stop reading frames 
    def stop(self):
        self.stopped = True

##################################################################################
# Initialize device
gripper = DHgripper() 
robot = Robot()
cam9 = WebcamStream(0)

home = [0.0,-0.65,0.614,0,3.12,0.36]
home2 = [0.0,-0.633,0.614,0,3.14,-0.07]
grasp_above = [0.0,-0.86,0.37,0,3.1415,0]

robot.move(home)

for i in range(100):
    color_image = cam9.read()
    cv2.imshow("Frame", color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
###################################################################################
# Load model for segmentation, inpaint, and force measurement
# issue: https://github.com/PyAV-Org/PyAV/issues/978
from track_anything import TrackingAnything_2
from track_anything import parse_augment
import requests,json,torch,torchvision,time  
from tools.painter import mask_painter
folder ="./checkpoints"
SAM_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
xmem_checkpoint = "./checkpoints/XMem-s012.pth"
e2fgvi_checkpoint = "./checkpoints/E2FGVI-HQ-CVPR22.pth"
svae_checkpoint = "./forcesensor/AmphibiousSoftFinger/lightning_logs/version_9/checkpoints/epoch=49-step=25000.ckpt"

# initialize sam, xmem, e2fgvi models
args = parse_augment()
model = TrackingAnything_2(xmem_checkpoint, e2fgvi_checkpoint, args)
template_mask = np.load("cam9_template.npy")/255
# template_mask = np.load("cam10_template.npy")/255
mask, logit = model.xmem.track(color_image[::2,::2,:], template_mask)
for i in range(30):
    mask, logit = model.xmem.track(color_image[::2,::2,:])

from ultralytics import RTDETR
detector = RTDETR("rtdetr-l.pt")
##########################################################################
# object detection and grasp
dir = './grasp-' + datetime.now().strftime("%m%d-%H%M%S")
os.mkdir(dir)

xmem = BaseTracker(xmem_checkpoint, device="cuda:0")
mask, logit = xmem.track(color_image[::2,::2,:], template_mask)

ft_stream = FT_stream(svae_checkpoint, xmem, cam9)
while True:
    inpainted_frames, last_frame = sweep()
    last_inpainted_frame = np.ascontiguousarray(inpainted_frames[-1,:,:,:])
    res = detector.predict(last_inpainted_frame)
    res_plotted = res[0].plot()
    plt.imshow(res_plotted[:,:,::-1])
    plt.show()
    decision = input("Press Enter to continue...")
    if decision =="q":
        break

    if len(res[0].boxes.xywh) ==0:
        print('Detect Nothing to pick!')
        break
    for j in range(len(res[0].boxes.conf)):
        v, u, w,h = res[0].boxes.xywh[j].cpu().numpy()
        if w>200 or h>330:
            continue
        u = u*2
        v = (v + 320)*2
        target = np.matmul(M,np.array([v,u,1]))
        print(target)
        x,y = target[:2]/target[2]
        x = x*0.85
        y = y *0.97#/ ( 1 + 0.06*(v-680)/450 )
        robot.rtde_c.moveL([x,y,0.40,0,3.1415,0],0.1, 0.5)
        robot.rtde_c.moveL([x,y,0.325,0,3.1415,0],0.1, 0.5)
        i = 0
        s = time.time()
        gripper_move_count = 0
        for i in range(600):
            ft = ft_stream.read()
            time.sleep(0.03)
            print('Frame %s ft: '%i,ft)
            current_q = robot.rtde_r.getActualQ()
            if np.abs(ft[5])>0.05:
                current_q[5] = current_q[5] + ft[5]
                robot.rtde_c.moveJ(current_q)
                continue
            if abs(ft[0]) < 3 and gripper_move_count<240:
                gripper_move_count += 1
                gripper.Position(1000-gripper_move_count*3)
            else:                
                break
        # lift gripper
        robot.rtde_c.moveL([x,y,0.48,0,3.1415,0],0.1, 0.5)
        if abs(ft[0])<0.5:
            gripper.Position(1000)
            continue
        # move to the drop box
        robot.move([0.33, -1, 0.48, 0, 3.077, -0.63])
        gripper.Position(1000)

######################################################################
# Save data and video
ft_stream.stop()
print('Start to writing videos and data...')
out1= cv2.VideoWriter(dir+'/original_grasp_720-360_cam9.mov', cv2.VideoWriter_fourcc(*'jpeg'), 28, (640, 360)) #26
out2 = cv2.VideoWriter(dir+'/mask_720-360_cam9.mov', cv2.VideoWriter_fourcc(*'jpeg'), 28, (640, 360),0)
out3 = cv2.VideoWriter(dir+'/painted_720-360_cam9.mov', cv2.VideoWriter_fourcc(*'jpeg'), 28, (640, 360))
for i in range(len(ft_stream.timestamps)):
    painted_image = mask_painter(ft_stream.images[i], (ft_stream.masks[i]==1).astype('uint8'), mask_color=1+1)
    out2.write(ft_stream.masks[i]*255)
    out1.write(ft_stream.images[i])
    out3.write(painted_image)


out1.release()
out2.release()
out3.release()

np.save(dir+'/images.npy', ft_stream.images)
np.save(dir+'/masks.npy', ft_stream.masks)
np.save(dir+'/fts.npy',ft_stream.fts)
np.save(dir+'/timestamps',ft_stream.timestamps)
print("Saved data in %s"%dir)
