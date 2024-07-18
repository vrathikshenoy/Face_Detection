import threading
import cv2
from  deepface import DeepFace
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf


cap = cv2.VideoCapture(1 , cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0


face_match = False

referance_img =cv2.imread('elon.jpg')


def check_face_match(frame):
    global face_match
    try:
        if DeepFace.verify(frame, referance_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError as e:
        print(e)
        face_match = False
        pass

while True:
    ret , frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face_match, args= (frame.copy(),)).start()
               
            except ValueError as e :
                print(e)
                pass

        counter += 1

        if face_match:
            cv2.putText(frame, 'Face Matched', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, 'Face Not Matched', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  

        cv2.imshow('Video', frame)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cv2.destroyAllWindows()


