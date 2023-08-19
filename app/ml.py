

import numpy as np
import cv2
import sklearn
import pickle
from django.conf import settings
import os

STATIC_DIR=settings.STATIC_DIR

#face det
face_detector_model=cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR,'./models/deploy.prototxt.txt'),
 
os.path.join(STATIC_DIR,'./models/res10_300x300_ssd_iter_140000.caffemodel'))

#feature extravtion
face_feature_model=cv2.dnn.readNetFromTorch(os.path.join(STATIC_DIR,'./models/openface.nn4.small2.v1.t7'))
#face recogn
face_recognition_model=pickle.load(open(os.path.join(STATIC_DIR,'./models/machinlearning_face_person_identity.pkl'),mode='rb'))
#emotion recog model
emotion_recognition_model=pickle.load(open(os.path.join(STATIC_DIR,'./models/machinlearning_face_person_emotion.pkl'),mode='rb'))




def pipelinemodel(path):  
    #pipeline model
    img=cv2.imread(path)
    image=img.copy()
    h,w=img.shape[:2]
    #face detection
    img_blob=cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),swapRB=False,crop=False)
    face_detector_model.setInput(img_blob)
    detection=face_detector_model.forward()
    #machine result
    ml_results=dict(
        face_detect_score=[],
        face_name=[],
        face_name_score=[],
        emotion_name=[],
        emotion_name_score=[],
        count=[]
        
    )
    count=1
    if len(detection)>0:
        for i,conf in enumerate(detection[0,0,:,2]):
            if conf>0.5:
                box=detection[0,0,i,3:7]*np.array([w,h,w,h])
                startx,starty,endx,endy=box.astype(int)
                cv2.rectangle(image,(startx,starty),(endx,endy),(0,245,0))
                #feature extraction
                face_roi=img[starty:endy,startx:endx]
                face_blob=cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),swapRB=True,crop=True)
                face_feature_model.setInput(face_blob)
                vectors=face_feature_model.forward()
                #pedict with ml
                face_name=face_recognition_model.predict(vectors)[0]
                face_score=face_recognition_model.predict_proba(vectors).max()
                #print(face_name)
                #print(face_score)

                #Emotion
                emotion_name=emotion_recognition_model.predict(vectors)[0]
                emotion_score=emotion_recognition_model.predict_proba(vectors).max()

                text_face='{} : {:.0f} %'.format(face_name,face_score*100)
                text_emotion='{} : {:.0f} %'.format(emotion_name,emotion_score*100)
                cv2.putText(image,text_face,(startx,starty-10),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,230),2)
                cv2.putText(image,text_emotion,(startx,endy),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,230),2)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/process.jpg'),image)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'ml_output/roi_{}.jpg'.format(count)),face_roi)
            # print(emotion_name,emotion_score)
                ml_results['count'].append(count)
                ml_results['face_detect_score'].append(conf)
                ml_results['face_name'].append(face_name)
                ml_results['face_name_score'].append(face_score)
                ml_results['emotion_name'].append(emotion_name)
                ml_results['emotion_name_score'].append(emotion_score)
                count+=1
    
    return ml_results




