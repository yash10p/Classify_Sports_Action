import cv2
import numpy as np
import os

data_dir = os.path.join(os.getcwd(), "NNFL_Training_Set")


encoding = {"Golf-Swing-Back": 0, "Golf-Swing-Front": 0, "Golf-Swing-Side": 0,\
            "Kicking-Front": 1, "Kicking-Side": 1, "Lifting": 2,\
            "Riding-Horse": 3, "Running": 4, "SkateBoarding": 5, \
            "Swing-Bench": 6, "Swing-SideAngle": 7, "Walking": 8}
            
classes = ["Golf-Swing-Back", "Golf-Swing-Front", "Golf-Swing-Side",\
           "Kicking-Front", "Kicking-Side", "Lifting", "Riding-Horse",\
           "Running", "SkateBoarding", "Swing-Bench", "Swing-SideAngle",\
           "Walking"]
           
for cl in classes:
    cl_path = os.path.join(data_dir, cl)
        
    cl_data = [os.path.join(cl_path, inst) for inst in os.listdir(cl_path) if not inst.startswith(".")]
    
    for data in cl_data:
        vid = [os.path.join(data, x) for x in os.listdir(data) if x.endswith(".avi")]
        
        cap = cv2.VideoCapture(vid[0])
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        count = 0
        
        while(ret):
            ret, frame2 = cap.read()
            if not ret:
                break
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2',bgr)
            cv2.imwrite(os.path.join(data, 'frame{}.jpg'.format(count)),bgr)
            count+=1
            prvs = next
        cap.release()
        cv2.destroyAllWindows()
