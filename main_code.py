# import needed libraries
from data_extractor import data_ext
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import numpy as np
import time
from sklearn.externals import joblib
from haarfeature import features
from new_dataset import resize
from find_features import find_feature

svc=SVC(probability=True, kernel='linear')



# -------------------define the places where all positive movies (fall movies) and negative movies (not-fall movies) are---------------------------------
positive_file_folder="fall"
negative_file_folder="not-fall"
#---------------------define the number of features should be used for extracting data--------------------------------------------------------
n_feature=10

#-------------------------resize data set with the defined function named resize which receives the original files path and resize the movies in the directory to 15*20 movies in the new directory path which also should be received as one of the inputs------------------------------
resize(positive_file_folder,'resized_fall')
resize(negative_file_folder,'resized_not_fall')




#--------------(!!!! time consuming !!!) extract all available features (if the features have extracted previously then comment first two lines and uncomment the other two line (Be careful about the namess))--------------------
positive_features=data_ext('resized_fall',700272,features,"positive_feature.npy")
negative_features=data_ext('resized_not_fall',700272,features,"negative_feature.npy")

# positive_features=np.load("positive_feature.npy")
# negative_features=np.load("negative_feature.npy")





#---------------label features----------------------
positive_label=np.ones([positive_features.shape[0],1])
negative_label=np.zeros([negative_features.shape[0],1])


#---------------concatenate all features----------------
all_features=np.row_stack((positive_features,negative_features))

del positive_features, negative_features

labels=np.row_stack((positive_label,negative_label))

del positive_label, negative_label

#-------------test and train data -----------------------------------
X, X_test, Y, y_test =  train_test_split(all_features, labels, test_size=0.4)
del all_features, labels

#-------------------Adaboost-----------------------------------------
abc = AdaBoostClassifier(n_estimators=n_feature,learning_rate=1)
model = abc.fit(X, Y)
#-----------------find accuracy----------------------------------------
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#------------------save model-------------------------------------------
joblib.dump(model,"fall_model.sav")
#------------------find which feature have been chosen by Adaboost-------
feature_importances=model.feature_importances_
#-----------------write a new code-------------
	#----------first the features should be labled (if your data extraction scenario is not same as your previous scenario, you should first save a new matrix named new_samples_clue.py using features function in haarfeature_with_clue.py. Be careful to change the data extraction scenario in this folder too.
#first method
from haarfeature_with_clue
data=np.load('new_samples_clue.npy')
file = open("final_model.py","w") 

file.write("from haarfeature import move_frame, feat12, feat21, feat13,featmul \n") 
file.write("import numpy as np \n") 
file.write("def fall_model(a,b,c,d):\n")
file.write("    U1=move_frame(b,0)\n")
file.write("    D1=move_frame(b,1)\n")
file.write("    R1=move_frame(b,2)\n")
file.write("    L1=move_frame(b,3)\n")
file.write("    U2=move_frame(c,0)\n")
file.write("    D2=move_frame(c,1)\n")
file.write("    R2=move_frame(c,2)\n")
file.write("    L2=move_frame(c,3)\n")
file.write("    U3=move_frame(d,0)\n")
file.write("    D3=move_frame(d,1)\n")
file.write("    R3=move_frame(d,2)\n")
file.write("    L3=move_frame(d,3)\n")
file.write("    dU10=np.bitwise_xor(U1,a)\n")
file.write("    dD10=np.bitwise_xor(D1,a)\n")
file.write("    dR10=np.bitwise_xor(R1,a)\n")
file.write("    dL10=np.bitwise_xor(L1,a)\n")
file.write("    delta10=np.bitwise_xor(b,a)\n")
file.write("    dU20=np.bitwise_xor(U2,a)\n")
file.write("    dD20=np.bitwise_xor(D2,a)\n")
file.write("    dR20=np.bitwise_xor(R2,a)\n")
file.write("    dL20=np.bitwise_xor(L2,a)\n")
file.write("    delta20=np.bitwise_xor(c,a)\n")
file.write("    dU30=np.bitwise_xor(U3,a)\n")
file.write("    dD30=np.bitwise_xor(D3,a)\n")
file.write("    dR30=np.bitwise_xor(R3,a)\n")
file.write("    dL30=np.bitwise_xor(L3,a)\n")
file.write("    delta30=np.bitwise_xor(d,a)\n")
file.write("    dU21=np.bitwise_xor(U2,b)\n")
file.write("    dD21=np.bitwise_xor(D2,b)\n")
file.write("    dR21=np.bitwise_xor(R2,b)\n")
file.write("    dL21=np.bitwise_xor(L2,b)\n")
file.write("    delta21=np.bitwise_xor(c,b)\n")
file.write("    dU31=np.bitwise_xor(U3,b)\n")
file.write("    dD31=np.bitwise_xor(D3,b)\n")
file.write("    dR31=np.bitwise_xor(R3,b)\n")
file.write("    dL31=np.bitwise_xor(L3,b)\n")
file.write("    delta31=np.bitwise_xor(d,b)\n")
file.write("    dU32=np.bitwise_xor(U3,c)\n")
file.write("    dD32=np.bitwise_xor(D3,c)\n")
file.write("    dR32=np.bitwise_xor(R3,c)\n")
file.write("    dL32=np.bitwise_xor(L3,c)\n")
file.write("    delta32=np.bitwise_xor(d,c)\n")
feat_count=1
for i in range(0,int(feature_importances.shape[0])):
	if feature_importances[i]>0:
		new_line=find_feature(data[i,:])
		feat_num='    '+'feat_{}='.format(feat_count)
		file.write(feat_num+new_line)
		feat_count+=1
file.write("    feat=np.array([feat_1,feat_2,feat_3,feat_4,feat_5,feat_6,feat_7,feat_8,feat_9,feat_10])\n")
file.write("    feat=feat.transpose()\n")
file.write("    return feat\n")
file.close()

from final_model import fall_model
#--------------extract features--------------------
trained_positive_features=data_ext(positive_file_folder,10,fall_model,"trained_positive_feature.npy")
trained_negative_features=data_ext(negative_file_folder,10,fall_model,"trained_negative_feature.npy")

#---------------label features----------------------
trained_positive_label=np.ones([trained_positive_features.shape[0],1])
trained_negative_label=np.zeros([trained_negative_features.shape[0],1])


#---------------concatenate all features----------------
trained_all_features=np.row_stack((trained_positive_features,trained_negative_features))

del trained_positive_features, trained_negative_features

trained_labels=np.row_stack((trained_positive_label,trained_negative_label))

del trained_positive_label, trained_negative_label

#-------------test and train data -----------------------------------
trained_X, trained_X_test, trained_Y, trained_y_test =  train_test_split(trained_all_features, trained_labels, test_size=0.4)
del trained_all_features, trained_labels

#-------------------Adaboost-----------------------------------------
abc = AdaBoostClassifier(n_estimators=n_feature,learning_rate=1)
trained_model = abc.fit(X, y)
#-----------------find accuracy----------------------------------------
trained_y_pred = trained_model.predict(trained_X_test)
print("{}'s Accuracy:".format(n_number),metrics.accuracy_score(trained_y_test, trained_y_pred))
#------------------save model-------------------------------------------
joblib.dump(model,"trained_fall_model.sav")





#-------------------multi_scale----------------------------

cap = cv2.VideoCapture(test_video)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
v_out = cv2.VideoWriter('result.avi'.format(n_feat),fourcc, 20.0, (360,480),False)
frame_n=0
f_name=0
rest=0
count=0
while True:
    ret, frame=cap.read()
    frame_n+=1
    if ret==True:
        f_name+=1
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        L=frame.shape[1]
        H=frame.shape[0]
        if f_name-int(f_name/4)*4==rest:
            count+=1
            if frame_n==rest+4:
                img1=frame[int(H/2-240):int(H/2+240),int(L/2-180):int(L/2+180)]

            elif frame_n==rest+8:
                img2=np.array(img1)
                img1=frame[int(H/2-240):int(H/2+240),int(L/2-180):int(L/2+180)]

            elif frame_n==rest+12:
                img3=np.array(img2)
                img2=np.array(img1)
                img1=frame[int(H/2-240):int(H/2+240),int(L/2-180):int(L/2+180)]

            else:
                img4=np.array(img3)
                img3=np.array(img2)
                img2=np.array(img1)

                img1=frame[int(H/2-240):int(H/2+240),int(L/2-180):int(L/2+180)]
                resized_img1=cv2.resize(img1,(15,20),interpolation=cv2.INTER_AREA)
                resized_img2=cv2.resize(img2,(15,20),interpolation=cv2.INTER_AREA)
                resized_img3=cv2.resize(img3,(15,20),interpolation=cv2.INTER_AREA)
                resized_img4=cv2.resize(img4,(15,20),interpolation=cv2.INTER_AREA)
                feat=data_ext(resized_img4,resized_img3,resized_img2,resized_img1)  
                feat=feat.reshape(1,-1)   
                out=model.predict(feat)
                if out==1:
                    img4=cv2.putText(img4, "fall", (int(L/2), int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA) 
                # print(img4.shape)
                v_out.write(img4)
    else:
        break

v_out.release()
