# importation du module os (système d’exploitation)
import os
#importation de la bibliotheque open cv(pour le traitement d'images en temps réel)
import cv2
#importation de la bibliothèque numpy (pour d’effectuer des calculs numériques)
import numpy as np
#importation de la biblipthèque pillow(Python Imaging Library)(pour le traitement d’image.offre un accès rapide aux données contenues dans une image)
from PIL import Image as img
#importation du module pickle(implémente des protocoles binaires de sérialisation et dé-sérialisation d'objets Python.(convertie en flux d'octets)
import pickle
#importation de la bibliothèque SQLite (propose un moteur de base de données relationnelle accessible par le langage SQL)
import sqlite3
#importation de la date
from datetime import date
#importation du temps
from datetime import datetime
#importation de la bibliothèque smtplib (pour envoyer un mail en SMTP)
import smtplib
#definition du temps du moment
time = datetime.now()
#definition de la date d'aujourd'hui
date = date.today()
#classificateur pré-entraîné pour le visage
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#entraînez l'ensemble de données qui génère le fichier trainningData.yml que nous utiliserons pour la reconnaissance faciale
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainningData.yml')
id=0
#definition des fonts pour le text
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 0)

#definition du fonction get profile qui prend l'id et retourne les infos contenus dans la base
def getProfile(id):
    #connection à la base FaceBase
    conn=sqlite3.connect("FaceBase.db")
    #creation du curseur
    cursor=conn.cursor()
    #definition de la requete select qui va retourner tous les infos
    cmd="SELECT * FROM People WHERE ID="+str(id)
    #execution de la requete
    cursor.execute(cmd)
    #initialisation du variable profile
    profile=None
    #parcour du curseur
    for row in cursor:
        #affectation des infos du curseur vers la variable profile
        profile=row
    #fermer la connection   
    conn.close()
    #retourner profile
    return profile

# WEBCAM INITIALIZATION
cam=cv2.VideoCapture(0)

while True:
    ret,img=cam.read()
     #Transformer une image en niveau de gris
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detectation du visage
    faces=faceDetect.detectMultiScale(gray,scaleFactor=1.2,minSize=(100,100),flags=cv2.cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        #méthode utilisée pour dessiner un rectangle sur n'importe quelle image
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #predict renvoie l'identifiant sur lequel vous l'avez entraîné
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        #si la variable profile n'est pas vide
        if(profile!=None):
            #afficher un text qui contient le nom
            cv2.putText(img,"nom: "+str(profile[1]),(x,y+h+30), fontface, fontscale, fontcolor)
            #afficher un text qui contient le prenom
            cv2.putText(img,"prenom: "+str(profile[2]),(x,y+h+60), fontface, fontscale, fontcolor)
            #afficher un text qui contient l'email
            cv2.putText(img,"email: "+str(profile[3]),(x,y+h+90), fontface, fontscale, fontcolor)
        else:
            #afficher un text qui contient unknown
            cv2.putText(img,"unkown",(50, 450) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        #affiche le nom du window et l'image qui doit être affichée.
        cv2.imshow("Face",img)
        #attend jusqu'à 1 milliseconde que l'utilisateur appuie sur une touche
        cv2.waitKey(1)
while True:
    ret, frame = cam.read()
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        results = person_saved_model.predict(face)
        
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            #initialisation d'une variable
            display_string = str(confidence) + '% FACE MATCHING!!!'
            #afficher un text qui contient la variable display_string
            cv2.putText(img, display_string, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 2)
        
        #IF CONFIDENCE SCORE WILL BE under 90% EMAIL WILL BE SEND.
        if confidence > 80:
            #afficher un text qui contient Face Unmatched
            cv2.putText(img, "FACE UNMATCHED!!!", (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            #affiche le nom du window et l'image qui doit être affichée.
            cv2.imshow('FACE RECOGNITION', img)
            sender_email = "taissir.guesmi@hotmail.fr"
            rec_email = "taissir.guesmi@hotmail.com"
            password = "tassou<3"
            message = "Taissir face not dedected at"+date+time
            #Create SMTP session
            server = smtplib.SMTP('smtp.gmail.com', 587)
            #Use TLS to add security 
            server.starttls()
            #User Authentication 
            server.login(sender_email, password)
            print("Login success")
            #Sending the Email
            server.sendmail(sender_email, rec_email, message)
            print("Email has been sent to ", rec_email)
       
    except:
        #afficher un text qui contient Face DOESN'T Found
        cv2.putText(img, "FACE DOESN'T FOUND!!!", (50, 100) , cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        #affiche le nom du window et l'image qui doit être affichée.
        cv2.imshow('FACE RECOGNITION', img)
        pass
    #attend jusqu'à 1 milliseconde que l'utilisateur appuie sur 13
    if cv2.waitKey(1) == 13:
      break
        
cam.release()
cv2.destroyAllWindows()
