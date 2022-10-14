import cv2
import os
import imutils

"""Module used for capture faces in video and it convert in images"""

# List of carpet name of the emotion to capture in different videos
emotionList = ['Enojo', 'Felicidad', 'Sorpresa', 'Tristeza', 'Neutral']
# Carpet name of the emotion to capture in realtime video.
emotionName = 'Enojo'
# emotionName = 'Felicidad'
# emotionName = 'Sorpresa'
# emotionName = 'Tristeza'
# emotionName = 'Neutral'

dataPath = 'C:/Users/angel/Desktop/Semestre/Inteligencia Artificial/Tareas/IA/Reconocimiento_De_Emociones/Data'  # Path to save images of the videos
dataPathVideo = 'C:/Users/angel/Desktop/Semestre/Inteligencia Artificial/Tareas/IA/Reconocimiento_De_Emociones/Videos'  # Path to videos


def captureVideoRealtime():
    """Capture video in realtime and take images of video"""

    # Cambia a la ruta donde hayas almacenado Data
    emotionsPath = dataPath + '/' + emotionName  # Path of emotions in dataPath

    if not os.path.exists(emotionsPath):
        print('Created Carpet: ', emotionsPath)
        os.makedirs(emotionsPath)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:

        ret, frame = cap.read()
        if not ret: break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(emotionsPath + '/rostro_{}.jpg'.format(count), rostro)
            count = count + 1
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 200:
            break
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def captureVideos(i):
    """Capture 10 videos for each emotion in format 'emotionIndexVideo' for example: Felicidad9"""
    contImagesToSave = 500
    emotionsPath = dataPath + '/' + emotionList[i]  # Path of emotions in dataPath

    if not os.path.exists(emotionsPath):
        print('Created Carpet: ', emotionsPath)
        os.makedirs(emotionsPath)

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    count2 = contImagesToSave
    for cont in range(0, 10):

        cap = cv2.VideoCapture(dataPathVideo + '/' + emotionList[i] + str(cont) + '.mp4')
        print(dataPathVideo + '/' + emotionList[i] + str(cont) + '.mp4')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                count2 += contImagesToSave
                break
            if cont != 5 and cont != 3: frame = cv2.rotate(frame, cv2.ROTATE_180)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(emotionsPath + '/rostro_{}.jpg'.format(count), rostro)
                count = count + 1
            cv2.imshow('frame', frame)

            k = cv2.waitKey(1)
            if k == 27 or count >= count2:
                count2 += contImagesToSave
                break
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function for capture faces"""

    print("Select capture: ")
    print("1. Video in Realtime")
    print("2. Video in PC")
    option = input("Option: ")
    if option == "1":
        captureVideoRealtime()
    else:
        i = 0
        while i < len(emotionList):
            captureVideos(i)
            i += 1


main()
