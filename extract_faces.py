import numpy as np
import cv2
import tensorflow as tf
import time
import sys
import os
import shutil
import matplotlib.pyplot as plt
from tensorflow import keras
import keras.backend as K
from keras.layers import Lambda,Dense,Flatten,Conv2D,MaxPool2D,Input,Embedding
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import RMSprop,SGD
from keras.models import Sequential,Model
from imutils import paths
from PIL import Image
from numpy import asarray,savez_compressed,load,expand_dims
from mtcnn.mtcnn import MTCNN
from vidgear.gears import VideoGear
from os import listdir
from os.path import isdir
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.svm import SVC
from pathlib import Path

def extract_face(path,image,required_size=(160, 160)):
    """
      This function extract multiple faces from a given image or path of the image using MTCNN 
      and returns array of faces.
    """
    if path!='':
        image=Image.open(path)
    image=image.convert('RGB')
    pixels=asarray(image)
    detector=MTCNN()
    results=detector.detect_faces(pixels)
    face_array=[]
    for i in range(len(results)): # multiple faces
        x1,y1,width,height=results[i]['box']
        x1,y1=abs(x1),abs(y1)
        x2,y2=x1+width,y1+height
        face=pixels[y1:y2,x1:x2]
        image=Image.fromarray(face)
        image=image.resize(required_size)
        face_array.append(asarray(image))
    return face_array
