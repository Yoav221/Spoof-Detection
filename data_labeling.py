import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from const import *
from time import time
from functions import *
import os
# TODO: loop through each image in the directory and extract its name and path
# detect the face properties of each image
# extract properties
# preprocces them (normalize)
# save to text file with the same name of the image

os.lisdir("Data")