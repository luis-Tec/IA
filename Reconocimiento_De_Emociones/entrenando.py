import cv2
import os
import numpy as np
import time

"""Module for train the model with own images"""

dataPath = 'C:/Users/angel/Desktop/Semestre/Inteligencia Artificial/Tareas/IA/Reconocimiento_De_Emociones/Data'  # Path about of the different emotions


def getModel(method, facesData, labels):
    """Select and take time of training of the model. Finally, get model"""
    print(method)

    if method == 'EigenFaces':
        emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces':
        emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH':
        emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Training the faces recognition
    print("Training ( " + method + " )...")
    startTime = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    trainingTime = time.time() - startTime
    print("Training time ( " + method + " ): ", trainingTime)

    # Save the model
    emotion_recognizer.write("Model/model" + method + ".xml")


def main():
    """Main function for train the model"""
    emotionsList = os.listdir(dataPath)
    print('Emotions List: ', emotionsList)

    labels = []
    facesData = []
    label = 0

    for nameDir in emotionsList:
        emotionsPath = dataPath + '/' + nameDir

        for fileName in os.listdir(emotionsPath):
            # print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(emotionsPath + '/' + fileName, 0))
        # image = cv2.imread(emotionsPath+'/'+fileName,0)
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
        label = label + 1
    #
    # getModel('EigenFaces', facesData, labels)
    # getModel('FisherFaces', facesData, labels)
    getModel('LBPH', facesData, labels)


main()
