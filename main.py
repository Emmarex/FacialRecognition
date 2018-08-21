import cv2
import numpy as np 
import os
from PIL import Image

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
facial_recogniser = cv2.face.LBPHFaceRecognizer_create()

def init_cam():
    cap = cv2.VideoCapture(0)
    cap.set(0,4000) #width
    cap.set(1,500) #height

    while(True):
        count = 0
        ret , frame = cap.read()
        frame = cv2.flip(frame,1) #flip camera vertically
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            # scaleFactor=1.4,
            minNeighbors=5,
            minSize=(100,100)
        )

        # print("Number of faces ",len(faces))
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
            count += 1
            roi_color = frame[y:y+h, x:x+w]
            # Save faces
            filepath = 'Dataset/user'+str(count)+'.jpg'
            cv2.imwrite(filepath,roi_color)

        cv2.imshow('Attendance Management System', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27: #ESC Key
            break;

    cap.release()
    cv2.destroyAllWindows()


def get_and_format_images():
    imagePaths = [os.path.join('Dataset',f) for f in os.listdir('Dataset')]
    faceSamples = []
    ids = []
    for image_path in imagePaths:
        PIL_img = Image.open(image_path).convert('L') #Convert Image to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(image_path)[-1].split('.')[0].replace("user",""))
        faces = faceCascade.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
        
    return faceSamples,ids

def train_machine():
    faces,user_ids = get_and_format_images()
    facial_recogniser.train(faces,np.array(user_ids))
    # #save model 
    facial_recogniser.write('Trainer/facial_trainer.yml')
    print("Training Succesfull. {0} faces Trained".format(len(np.unique(user_ids))))

def facial_detection():
    facial_recogniser.read('Trainer/facial_trainer.yml')
    cam = cv2.VideoCapture(0)
    cam.set(3,1000)
    cam.set(4,1000)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while(True):
        count = 0
        ret , frame = cam.read()
        frame = cv2.flip(frame,1) #flip camera vertically
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.0,
            minNeighbors=6,
            minSize=(100,100)
        )

        predicted_users = list()
        # print("Number of faces ",len(faces))
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
            
            id, confidence = facial_recogniser.predict(gray[y:y+h,x:x+w])

            if(confidence < 100):
                predicted_users.append(id)
                print(" {0}".format(id))
                print(" {0}%".format(round(100 - confidence)))
            else:
                print(" {0}%".format(round(100 - confidence)))
            
            count += 1
            roi_color = frame[y:y+h, x:x+w]
            # Save faces
            filepath = 'Dataset/user'+str(count)+'.jpg'
            cv2.imwrite(filepath,roi_color)

        cv2.imshow('Attendance Management System', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27: #ESC Key
            print(predicted_users)
            break;

    cam.release()
    cv2.destroyAllWindows()


init_cam()