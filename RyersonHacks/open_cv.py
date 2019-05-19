import numpy as np
import cv2
from keras.preprocessing import image
import pandas as pd
import datetime
from shutil import copyfile

#-----------------------------
#opencv initialization
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("model.json", "r").read())
model.load_weights('./facial_expression_model_weights.h5') #load weights

#-------------------------------------------------------------------
# Define class
class Response():
	angry = 0
	disgust = 0
	fear = 0
	happy = 0
	sad = 0
	surprise = 0
	neutral = 0

#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
today = str(datetime.date.today())

count_angry = 0
count_disgust= 0
count_fear = 0
count_happy = 0
count_sad = 0
count_surprise= 0
count_neutral = 0

resp = Response()
df = pd.DataFrame(columns = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])

while(True):

	try:
		ret, img = cap.read()
		#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		#print(faces) #locations of detected faces

		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
			
			predictions = model.predict(img_pixels) #store probabilities of 7 expressions
			
			#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])
			
			emotion = emotions[max_index]

			new_emotion_count = getattr(resp, emotion) + 1
			setattr(resp, emotion, new_emotion_count)
			
			#write emotion text above rectangle
			cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

			if emotion == 'angry':
				count_angry += 1
			elif emotion == 'disgust':
				count_disgust += 1
			elif emotion == 'fear':
				count_fear += 1
			elif emotion == 'happy':
				count_happy += 1
			elif emotion == 'sad':
				count_sad += 1
			elif emotion == 'surprise':
				count_surprise += 1
			elif emotion == 'neutral':
				count_neutral += 1

			#process on detected face end
			#-------------------------
			cv2.imshow('img',img)

		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break
	except:
		break

df.loc[today,'Anger'] = count_angry
df.loc[today, 'Disgust'] = count_disgust
df.loc[today, 'Fear'] = count_fear
df.loc[today, 'Happy'] = count_happy
df.loc[today, 'Sad'] = count_sad	
df.loc[today, 'Surprise'] = count_surprise
df.loc[today, 'Neutral'] = count_neutral

df.to_excel('/Users/lauradang/RyersonHacks/data/emotions.xlsx')

#kill open cv things		
cap.release()
cv2.destroyAllWindows()