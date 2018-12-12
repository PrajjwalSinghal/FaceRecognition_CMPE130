import cv2
import os

import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascades = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#Switch different recognizer for different algorithm
recognizer = cv2.face.FisherFaceRecognizer_create()
# cv2.face.EigenFaceRecognizer_create()
#cv2.face.FisherFaceRecognizer_create()
#cv2.face.LBPHFaceRecognizer_create()

current_id = 0
lable_ids = {}
y_lables = []
x_train = []


for root, dirs, files in os.walk(image_dir):
	for files in files:
		if files.endswith("png") or files.endswith("jpg") or files.endswith("pgm"):
			path = os.path.join(root, files)
			lable = os.path.basename(root).replace(" ", "-").lower()
			#print(lable,path)
			if not lable in lable_ids:
				lable_ids[lable] = current_id
				current_id += 1

			id_ = lable_ids[lable]
			#print(lable_ids)
			#y_lables.append(lable) # some number for lable
			#x_train.append(path) # verify this image, turn it into a numpy array
			pil_image = Image.open(path).convert("L") #L converts it to grayscale
			size = (550, 550)
			final_image = pil_image #.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8") #converting it to an array
			#print(image_array)
			faces = face_cascades.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			#x_train.append(image_array)
			size = (100,100)
			#for (x,y,w,h) in faces:
			#	roi = image_array[y:y+h, x:x+w]
			#	
			x_train.append(cv2.resize(image_array,size))
			y_lables.append(id_)

#print(y_lables)
#print(x_train)

with open("lables.pickle", 'wb') as f:
	pickle.dump(lable_ids, f)

recognizer.train((x_train), np.array(y_lables))
recognizer.save("trainer.yml")
