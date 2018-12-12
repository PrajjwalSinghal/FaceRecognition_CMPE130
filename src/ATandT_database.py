import numpy as np
import cv2
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt

face_cascades = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


#Switch different recognizer for different algorithm
recognizer = cv2.face.FisherFaceRecognizer_create()
#cv2.face.EigenFaceRecognizer_create()
#cv2.face.FisherFaceRecognizer_create()
#cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")


lables = {"person_name":1}
with open("lables.pickle", 'rb') as f:
	og_lables = pickle.load(f)
	lables = {v:k for k,v in og_lables.items()}

	Train_images = 0
unknown_images = 0
for outer_range in range(1,41):
	Train_images = 0
	unknown_images = 0
	path_name_outer = "att_faces" + "/" + "s" + str(outer_range) + "/"
	for inner_range in range(1,11):
		path_name = path_name_outer + str(inner_range) + ".pgm"
		#print(path_name)
		frame_test = cv2.imread(path_name, 0)
		size = (100,100)
		id_, conf = recognizer.predict(cv2.resize(frame_test,size))
		name = lables[id_]
		if inner_range < 6:
			dir_name = "s"+str(outer_range)
			if name == dir_name:
				Train_images += 1
		if inner_range > 5:
			dir_name = "s"+str(outer_range)
			if name == dir_name:
				unknown_images += 1
	print("s"+str(outer_range))
	print("Trained Images", Train_images ,5)	
	print("unknown Images", unknown_images ,5)	