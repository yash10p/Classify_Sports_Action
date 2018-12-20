import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import sequence

cur_dir = os.getcwd()

encoding = {"Golf-Swing-Back": 0, "Golf-Swing-Front": 0, "Golf-Swing-Side": 0,\
            "Kicking-Front": 1, "Kicking-Side": 1, "Lifting": 2,\
            "Riding-Horse": 3, "Running": 4, "SkateBoarding": 5, \
            "Swing-Bench": 6, "Swing-SideAngle": 7, "Walking": 8}
            
classes = ["Golf-Swing-Back", "Golf-Swing-Front", "Golf-Swing-Side",\
           "Kicking-Front", "Kicking-Side", "Lifting", "Riding-Horse",\
           "Running", "SkateBoarding", "Swing-Bench", "Swing-SideAngle",\
           "Walking"]
           
mapped_classes = ["GolfSwing", "Kicking", "Lifting", "RidingHorse", "Running", "SkateBoarding",\
                  "Swing-Bench", "Swing-Side", "Walking"]

model = load_model('my_model.h5')

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))
    

def avg(fr):
    
    for i in range(len(fr)):
        init = False
        n = len(fr[i])
        for j in range(len(fr[i])):
            if init == False:
                av = np.zeros(fr[i][j].shape)
                init = True
            else:
                av = av + fr[i][j]/n
        fr[i] = av
    
    return fr

def preprocess(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #img = img.astype('float32')
    #img = img/255.0
    #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #apply binary threshold

    resz_img = cv2.resize(img, (64, 64)) #resize it to 25*25 image

    return resz_img
    
    
def stack_n_frames(frames, n=20):
    x = None
    
    fr = list(split(frames, n))
    fr = avg(fr)
    
    for f in fr:
            
        frame = f.reshape(1, 64, 64, 3)
        if(x is None):
            x = frame
        else:
            x = np.concatenate((x, frame), axis=0)
            
    x = x.reshape(x.shape[0], 64, 64, 3)
    return x

def get_frames_from_video(vid):
    
    vidcap = cv2.VideoCapture(vid)
    
    frames = []
    
    success = True
    
    while success:
      success,image = vidcap.read()
      if not success:
        break
      image = preprocess(image)
      frames.append(image)
      #print 'Read a new frame: ', success
      
    return frames
    
def get_flow_frames_from_video(vid):
    vidcap = cv2.VideoCapture(vid)

    frames = []

    success, image = vidcap.read()
                    
    prvs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image)
    hsv[...,1] = 255
    init = True
    
    count = 1
    success = True
    
    while success:
        success,img = vidcap.read()
        
        if not success:
            break
        next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        image = preprocess(bgr)
        frames.append(image)
      
    return frames
    
def predict(path_to_vid):
    flow = []
    rgb = []
    
    flow_frames = get_flow_frames_from_video(path_to_vid)
    rgb_frames = get_frames_from_video(path_to_vid)
    
    flow.append(stack_n_frames(flow_frames))
    flow = np.array(flow)
    flow = sequence.pad_sequences(flow, 20)
    
    rgb.append(stack_n_frames(rgb_frames))
    rgb = np.array(rgb)
    rgb = sequence.pad_sequences(rgb, 20)
    
    preds = model.predict([flow, rgb])
    
    pred = np.argmax(preds[0])
    
    return mapped_classes[pred]


if __name__ == "__main__":
    # parse arguments
    
    if len(sys.argv) != 2:
        print "usage: python test_directory"
        exit()
        
    test_dir = sys.argv[1]
    
    vid_names = [ x for x in os.listdir(test_dir) if x.endswith(".avi")]
    sorted(vid_names)
    
    for vid in vid_names:
        pred = predict(os.path.join(test_dir, vid))
        print vid, pred
    
