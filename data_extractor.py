#------------------import libraries
import cv2
import numpy as np
import glob

#-----let's define a function to extract the all features in a folder. This function receives the folder, number of the features, a model to extract the data and the saved file name then it tries to extrac data from all 4 consecutive frames. Finally, it concatenates them in one matix and saves the matrix 
def data_ext(folder, number_of_feature,data_extractor, saved_file):
    HF=np.zeros([number_of_feature,0])
    datanum=0
    loc=folder+"/*.avi"
    for vid in glob.glob(loc):
            for rest in range(0,4):
                cap = cv2.VideoCapture(vid)
                i=0
                f_num=0
                while(True):
                    # Capture frame-by-frame
                        ret, frame = cap.read()
                        if ret==True:
                            f_num+=1
                            if f_num-int(f_num/4)*4==rest:
                                i+=1
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                                if i==1:
                                    p1=np.array(frame)
                                elif i==2:
                                    p2=np.array(p1)
                                    p1=np.array(frame)
                                elif i==3:
                                    p3=np.array(p2)
                                    p2=np.array(p1)
                                    p1=np.array(frame)
                                else:
                                    p4=np.array(p3)
                                    p3=np.array(p2)
                                    p2=np.array(p1)
                                    p1=np.array(frame)
                                    FEAT=data_extractor(p1,p2,p3,p4)
                                    print(FEAT.shape)
                                    HF=np.column_stack((HF,FEAT))
                                                            #print(i)
                         #cv2.imshow('frame',frame)
                         # k= cv2.waitKey(1) 
                        else:
                                break
    HF=HF.transpose()
    np.save(saved_file,HF)
    return HF
