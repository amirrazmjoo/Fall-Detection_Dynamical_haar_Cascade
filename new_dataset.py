# this code tries to convert the fall and not-fall videos to the desired form which is 20*15-sized-frame videos, then it saves the new movies in another folder

import numpy as np 
import cv2
import glob

def resize(first_folder,target_folder):

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')

	vid_num=0
	loc=first_folder+"/*.avi"
	for vid in glob.glob(loc):
		print(vid)
		vid_num+=1
		# print(vid_num)
		out = cv2.VideoWriter('{}/video_{}.avi'.format(target_folder,vid_num),fourcc, 20.0, (15,20),True)
		cap = cv2.VideoCapture(vid)
		while True:
			
			ret, frame=cap.read()
			if ret==True:
				resizedfram=cv2.resize(frame,(15,20),interpolation=cv2.INTER_AREA)
				# print(resizedfram.shape)
				out.write(resizedfram)
			else:
				out.release()
				break
	return 