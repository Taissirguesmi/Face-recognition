import cv2 
import sqlite3
import numpy as np

#face detection using CascadeClassifier 
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#La méthode CascadeClassifier dans le module cv2 prend en charge le chargement de fichiers XML haar-cascade. Ici, nous avons besoin de « haarcascade_frontalface_default.xml » pour la détection des visages.

cam=cv2.VideoCapture(0)
#(capture video) c'est a dire la premiere camera ou webcam


#nous devons développer un formateur qui entraîne le système afin de reconnaître les visages
def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID= "+str(Id)
    cursor=conn.cursor()                                        
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if (isRecordExist==1):
        cmd="UPDATE People SET Nom "+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO People(ID,Nom)VALUES(?,?)"
    cursor.execute(cmd,(str(Id),str(Name)))
    conn.commit()
    conn.close()
    

id=input("enter user id") #entrer l'id
#Ce programme demande à l’utilisateur d’entrer l’id unique et le nom de cet utilisateur.
name=input("enter your name") #entre ton nom
insertOrUpdate(id,name)#mise a jour d l'id et le nom apres l'insertion 
sampleNum=0
#Après la saisie, les détails sont stockés dans la base de données.

while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Après cela, la caméra s’ouvrira, puis elle convertira l’image colorée (RVB) en image en niveaux de gris,puis elle détectera le visage,
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        #puis elle stockera les images avec différentes expressions dans un dossier nommé « dataset » avec le nom de fichier « user» avec son identifiant
        #et avec le numéro d’image 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Face",img)
    cv2.waitKey(100)
    if(sampleNum>20):               
#Le programme ci-dessus stockera jusqu’à 20 images. Après avoir stocké 20 images
        cam.release()
        #la caméra sera libérée après cela,
        cv2.destroyAllWindows()
        #les fenêtres seront fermées (fenêtre de sortie)
        break



