import cv2
import os
import numpy as np

dataPath = ''
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

#Label for each class of person
labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir

	for fileName in os.listdir(personPath):
		print('Rreading image: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
	label = label + 1


# Create EigenFace Mocel
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Model training
print("Training .  .  .")
face_recognizer.train(facesData, np.array(labels))

# Save the model
face_recognizer.write('EigenFaceCustomModel(600).xml')
print("Training completed")