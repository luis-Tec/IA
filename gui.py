from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import os
import pickle

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")
#Load models data

#person_recognition
dataPath = 'C:/Users/Luis/Documents/GitHub/IA/Reconocimiento_De_Personas/Dataset/Training/Personas(light) - 150px'
imagePathsPerson = os.listdir(dataPath)
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('C:/Users/Luis/Documents/GitHub/IA/Reconocimiento_De_Personas/EigenFaceCustomModel(light-150px).xml')

#facial_recognition
face_cascade = cv2.CascadeClassifier(
    'C:/Users/Luis/Documents/GitHub/IA/Reconocimiento_De_Rostros/haarcascade_frontalface_alt2.xml'
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(
    'C:/Users/Luis/Documents/GitHub/IA/Reconocimiento_De_Rostros/face-trainner.yml'
)
labels = {}
with open(
        "C:/Users/Luis/Documents/GitHub/IA/Reconocimiento_De_Rostros/face-labels.pickle",
        "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

#emotion_recognition
emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
emotion_recognizer.read(
    'C:/Users/Luis/Documents/GitHub/IA/Reconocimiento_De_Emociones/modelFisherFaces.xml'
)
imagePathsEmotion = os.listdir(
    'C:/Users/Luis/Documents/GitHub/IA/Reconocimiento_De_Emociones/Dataset')


def loadVideo():
    global cap
    if selected.get() == 1:
        path_video = filedialog.askopenfilename(
            filetypes=[("all video format",
                        ".mp4"), ("all video format", ".avi")])
        if len(path_video) > 0:
            btnEnd.configure(state="active")
            rad1.configure(state="disabled")
            rad2.configure(state="disabled")
            #pathInputVideo = "..." + path_video[-20:]
            cap = cv2.VideoCapture(path_video)
            view()
    if selected.get() == 2:
        btnEnd.configure(state="active")
        rad1.configure(state="disabled")
        rad2.configure(state="disabled")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        view()

def view():
    global cap
    global root
    global centerVideo
    ret, frame = cap.read()
    print(centerVideo)
    if ret == True:
        frame = imutils.resize(frame, height=600)
        frame = load_model(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, view)
    else:
        centerVideo = FALSE
        root.eval('tk::PlaceWindow . center')
        lblVideo.image = ""
        rad1.configure(state="active")
        rad2.configure(state="active")
        selected.set(0)
        btnEnd.configure(state="disabled")
        cap.release()
    if centerVideo == 0:
        root.eval('tk::PlaceWindow . center')
        centerVideo = 1

def load_model(frame):
    if (selectedModel.get() == 0):
        return facial_recognition(frame)
    elif (selectedModel.get() == 1):
        return person_recognition(frame)
    elif (selectedModel.get() == 2):
        return emotion_recognition(frame)


def person_recognition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3,
                    (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # EigenFaces
        if result[1] < 7500:
            cv2.putText(frame, '{}'.format(imagePathsPerson[result[0]]),
                        (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Unknown', (x, y - 20), 2, 0.8, (0, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame


def facial_recognition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.5,
                                          minNeighbors=5)

    for (x, y, w, h) in faces:
        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
    return frame


def emotion_recognition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.5,
                                          minNeighbors=5)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3,
                    (255, 255, 0), 1, cv2.LINE_AA)

        # FisherFace
        if result[1] < 500:
            cv2.putText(frame, '{}'.format(imagePathsEmotion[result[0]]),
                        (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame


def clear():
    lblVideo.image = ""
    rad1.configure(state="active")
    rad2.configure(state="active")
    selected.set(0)
    cap.release()
    root.eval('tk::PlaceWindow . center')

cap = None
centerVideo = 0
root = Tk()
lblInfo1 = Label(root, text="Prototipo", font="bold")
lblInfo1.grid(column=0, row=0, columnspan=4)

selectedModel = IntVar()

radModel1 = Radiobutton(root,
                        text="Detector facial",
                        width=26,
                        value=0,
                        state=NORMAL,
                        variable=selectedModel)
radModel2 = Radiobutton(root,
                        text="Reconocimiento de personas",
                        width=26,
                        value=1,
                        variable=selectedModel)
radModel3 = Radiobutton(root,
                        text="Reconocimiento de emociones",
                        width=26,
                        value=2,
                        variable=selectedModel)
radModel1.grid(column=0, row=1)
radModel2.grid(column=1, row=1)
radModel3.grid(column=2, row=1)

selected = IntVar()
rad1 = Radiobutton(root,
                   text="Elegir video",
                   width=20,
                   value=1,
                   variable=selected,
                   command=loadVideo)
rad2 = Radiobutton(root,
                   text="Video en directo",
                   width=20,
                   value=2,
                   variable=selected,
                   command=loadVideo)
rad1.grid(column=0, row=2)
rad2.grid(column=1, row=2)

lblVideo = Label(root)
lblVideo.grid(column=0, row=4, columnspan=4)
btnEnd = Button(root, text="Terminar", state="disabled", command=clear)
btnEnd.grid(column=0, row=5, columnspan=4, pady=10)
root.eval('tk::PlaceWindow . center')
root.mainloop()