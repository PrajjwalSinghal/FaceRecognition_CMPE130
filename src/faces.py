import numpy as np
import cv2
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
################
plt.ion()
fig = plt.figure()
plt.axis([0,1000,0,1])
i = 0
x_t = list()
y_t = list()
#################
face_cascades = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')



recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read("trainer.yml")


lables = {"person_name":1}
with open("lables.pickle", 'rb') as f:
	og_lables = pickle.load(f)
	lables = {v:k for k,v in og_lables.items()}
cap = cv2.VideoCapture(0)
myint = 0

'''
celeb_name = "peter-dinklage" #change the name to check for different person
trained_images = 0
unknown_images = 0
for x_name in range(1,13):
	path_name = "."
	if x_name < 7:
		path_name = "TestImages" + "/" + celeb_name + "/" + str(x_name) + ".jpg"
	if x_name > 6:
		path_name = "TestImages" + "/" + celeb_name + "/" + str(x_name) + ".jpeg"
	frame_test = cv2.imread(path_name,0)
	frame_test_faces = face_cascades.detectMultiScale(frame_test, scaleFactor=1.5, minNeighbors=5, minSize=(1,1))
	for(x,y,w,h) in frame_test_faces:
		roi_gray = frame_test[y:y+h, x:x+h]
		size = (550,550)
		id_, conf = recognizer.predict(cv2.resize(roi_gray,size))
		name = lables[id_]
		print(x_name)
		if x_name < 7:
			if name == celeb_name:
				trained_images += 1
		
		if x_name > 6:
			if name == celeb_name:
				unknown_images += 1	
		
		
		print(name, id_)
print("trained_images", trained_images)
print("unknown_images", unknown_images)
'''


i = 0
while(True):
	#Capture Frame-by-Frame
	ret, frame = cap.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascades.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(1,1))
	
	for(x,y,w,h) in faces:
		#print(x,y,w,h)
		#img_item = str(myint)+".png"
		#myint += 1
		#cv2.imwrite(img_item, frame)
		roi_gray = gray[y:y+h, x:x+h]
		roi_color   = frame[y:y+h, x:x+h]
		size = (100,100)
		id_, conf = recognizer.predict(cv2.resize(roi_gray,size))
		
		#if conf < 50:
			#print(id_)
			#print(lables[id_])
		font = cv2.FONT_HERSHEY_SIMPLEX
		name = lables[id_]
		color = (255, 255, 255)
		stroke = 2;
		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		#else:
		#	font = cv2.FONT_HERSHEY_SIMPLEX
		#	name = 'unkown'
		#	color = (255, 255, 255)
		#	stroke = 2;
		#	cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		#print(conf)
		i += 1
		plt.scatter(i, conf)
		plt.show()
		plt.pause(0.0001)

		color = (255, 0, 0) #BGR
		stroke = 2
		end_cord_x = x + w;
		end_cord_y = y + h;
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
		

	#Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

#When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
