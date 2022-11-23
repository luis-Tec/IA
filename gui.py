from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import os
import pickle
from datetime import datetime

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")
# Main Path // ** Its necessary configure **
urlMain = 'C:/Users/angel/Desktop/Semestre/Inteligencia Artificial/Tareas/IA/'

"""   Load models data   """
# var for reports
nameModel = ''
jsonReport = {}
inFrame = []
timeFace = []

# person_recognition
# dataPath = urlMain + 'Reconocimiento_De_Personas/Dataset/Training/Personas(light) - 150px'
# imagePathsPerson = os.listdir(dataPath)
imagePathsPerson = ["", ""]
face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer.read(urlMain + 'Reconocimiento_De_Personas/EigenFaceCustomModel(light-150px).xml')

# facial_recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(
    urlMain + 'Reconocimiento_De_Rostros/face-trainner.yml'
)
labels = {}
with open(
        urlMain + "Reconocimiento_De_Rostros/face-labels.pickle",
        "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# emotion_recognition
emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
emotion_recognizer.read(
    urlMain + 'Reconocimiento_De_Emociones/modelFisherFaces.xml'
)
imagePathsEmotion = os.listdir(
    urlMain + 'Reconocimiento_De_Emociones/Data')


# functions

# load video with path or realtime
def loadVideo():
    global cap, jsonReport
    jsonReport = {}
    if selected.get() == 1:
        path_video = filedialog.askopenfilename(
            filetypes=[("all video format",
                        ".mp4"), ("all video format", ".avi")])
        if len(path_video) > 0:
            btnEnd.configure(state="active")
            btnEnd2.configure(state="active")
            rad1.configure(state="disabled")
            rad2.configure(state="disabled")
            # pathInputVideo = "..." + path_video[-20:]
            cap = cv2.VideoCapture(path_video)
            view()
    if selected.get() == 2:
        btnEnd.configure(state="active")
        btnEnd2.configure(state="active")
        rad1.configure(state="disabled")
        rad2.configure(state="disabled")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        view()


# Show the video
def view():
    global cap
    global root
    global centerVideo
    ret, frame = cap.read()
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
        btnEnd2.configure(state="disabled")
        cap.release()
    if centerVideo == 0:
        root.eval('tk::PlaceWindow . center')
        centerVideo = 1


# load the model selected
def load_model(frame):
    global nameModel
    if selectedModel.get() == 0:
        nameModel = 'facial recognition'
        return facial_recognition(frame)
    elif selectedModel.get() == 1:
        nameModel = 'person recognition'
        return person_recognition(frame)
    elif selectedModel.get() == 2:
        nameModel = 'emotion recognition'
        return emotion_recognition(frame)


def person_recognition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    reportsFrame = []

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3,
                    (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = frame[y:y+h, x:x+w]

        # EigenFaces
        if result[1] < 7500:
            cv2.putText(frame, '{}'.format(imagePathsPerson[result[0]]),
                        (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            reportsFrame.append(imagePathsPerson[result[0]])
            cv2.imwrite('Reports/images/' + imagePathsPerson[result[0]] + '.jpg', roi_color)
        else:
            cv2.putText(frame, 'Unknown', (x, y - 20), 2, 0.8, (0, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame


def facial_recognition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray,
                                         scaleFactor=1.5,
                                         minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y + h, x:x + w]
        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
        cv2.imwrite('Reports/images/rostro.jpg', roi_color)
    return frame


def emotion_recognition(frame):
    global inFrame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    reportsFrame = []
    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3,
                    (255, 255, 0), 1, cv2.LINE_AA)
        roi_color = frame[y:y+h, x:x+w]

        # FisherFace
        if result[1] < 500:
            cv2.putText(frame, '{}'.format(imagePathsEmotion[result[0]]),
                        (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            reportsFrame.append(imagePathsEmotion[result[0]])
            cv2.imwrite('Reports/images/' + imagePathsEmotion[result[0]] + '.jpg', roi_color)
        else:
            cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            reportsFrame.append('No identificado')
            cv2.imwrite('Reports/images/No identificado.jpg', roi_color)
    addReports(reportsFrame)
    inFrame = reportsFrame.copy()
    return frame


# add or addition reports to json global
def addReports(lista):
    global jsonReport, inFrame
    for data in lista:
        if data not in inFrame:  # If not is in frame, can add it
            if data in jsonReport:
                jsonReport[data] = jsonReport[data] + 1
            else:
                jsonReport[data] = 1


# save reports of each model
def saveReports():
    global jsonReport
    # get string of current date and time
    now = datetime.now()
    strDateNow = str(now.day) + '-' + str(now.month) + '-' + str(now.year) + ' ' + str(now.hour) + '.' + str(
        now.minute) + '.' + str(now.second)
    # Create file with name of model and current datetime.
    file = open(urlMain + 'Reports/' + nameModel + ' ' + strDateNow + '.txt', 'w')
    # also, add reports in file
    file.write('Reports for ' + nameModel + '\n')
    file.write('Current Datetime: ' + strDateNow + os.linesep)
    for key in jsonReport:
        file.write(key + ' = ' + str(jsonReport[key]) + '\n')
    file.close()


# Clear or stop the recognition
def clear():
    lblVideo.image = ""
    rad1.configure(state="active")
    rad2.configure(state="active")
    selected.set(0)
    cap.release()
    root.eval('tk::PlaceWindow . center')


# first view or main view
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
btnEnd2 = Button(root, text="Guardar Reporte", state="disabled", command=saveReports)
btnEnd2.grid(column=0, row=6, columnspan=4, pady=10)
root.eval('tk::PlaceWindow . center')
root.mainloop()
