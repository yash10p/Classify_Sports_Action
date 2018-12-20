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
        
        path = os.path.join(data, "rgb")
        os.makedirs(path)
        ret = True
        count = 0
        while(ret):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(path, 'frame{}.jpg'.format(count)),frame)
            count += 1
        cap.release()
        cv2.destroyAllWindows()
