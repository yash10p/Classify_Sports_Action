import os
import numpy as np
import cv2
from keras.utils import to_categorical

data_dir = os.path.join(os.getcwd(), "NNFL_Training_Set")


encoding = {"Golf-Swing-Back": 0, "Golf-Swing-Front": 0, "Golf-Swing-Side": 0,\
            "Kicking-Front": 1, "Kicking-Side": 1, "Lifting": 2,\
            "Riding-Horse": 3, "Running": 4, "SkateBoarding": 5, \
            "Swing-Bench": 6, "Swing-SideAngle": 7, "Walking": 8}
            
classes = ["Golf-Swing-Back", "Golf-Swing-Front", "Golf-Swing-Side",\
           "Kicking-Front", "Kicking-Side", "Lifting", "Riding-Horse",\
           "Running", "SkateBoarding", "Swing-Bench", "Swing-SideAngle",\
           "Walking"]
           
mapped_classes = ["Golf-Swing", "Kicking", "Lifting", "Riding-Horse", "Running", "SkateBoarding",\
                  "Swing-Bench", "Swing-Side", "Walking"]

print classes

def preprocess(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    #img = img.astype('float32')
    #img = img/255.0
    #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #apply binary threshold

    resz_img = cv2.resize(img, (128, 128)) #resize it to 25*25 image

    return resz_img


def load_data():
    X = []
    Y = []
    for cl in classes:
        cl_path = os.path.join(data_dir, cl)
        
        cl_data = [os.path.join(cl_path, inst) for inst in os.listdir(cl_path) if not inst.startswith(".")]
        
        cl_data = [os.path.join(ci, "flow") for ci in cl_data]
        
        print cl, len(cl_data)
        fr = []
        for data in cl_data:
            frames_path = [ os.path.join(data, "frame{}.jpg".format(i)) for i in range(19)]
            fr.append(len(frames_path))
            x_train = None
            
            Y.append(to_categorical(encoding[cl], len(mapped_classes)))
            
            for frame in frames_path:
                #print frame
                img = cv2.imread(frame)
                
                img = preprocess(img)
                img = img.reshape(1, 128, 128,3)
                if(x_train is None):
                    x_train = img
                else:
                    x_train = np.concatenate((x_train, img), axis=0)
            x_train = x_train.reshape(19, 128, 128,3)
            X.append(x_train)
        print min(fr)
            
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, Y = load_data()
    print X.shape
    print X[0].shape
    print Y[0]
    print len(Y)
