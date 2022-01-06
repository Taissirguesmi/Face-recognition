
import os#The module OS en Python fournit un moyen d’utiliser les fonctionnalités dépendantes du système d’exploitation.
import cv2 #cv2 load image
import numpy as np #NumPy est une bibliothèque pour langage de programmation Python,
#destinée à manipuler des matrices ou tableaux multidimensionnels ainsi que des fonctions mathématiques opérant sur ces tableaux

from PIL import Image#PIL est le Python#Imaging library qui fournit à l’interpréteur python des capacités d’édition d’images.

recognizer=cv2.face.LBPHFaceRecognizer_create()#LBPH Algorithm 

path='dataSet' #destination

def getImagesWithID(path):#fonction made to get
#the id of the image and it takes the path of the image a s a parmeter
    
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]#liste vide pour les images
    IDs=[]#liste pour les id
    for imagePath in imagePaths:       
        faceImg=Image.open(imagePath).convert('L'); 
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print (ID) #afficher l'id
        IDs.append(ID)#ajouter l'id a IDs
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs,faces

IDs,faces=getImagesWithID(path)
#montrer les images
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows() #fermer la fenetre

        
        
    
