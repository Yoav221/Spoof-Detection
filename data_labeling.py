import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from const import *
from time import time
from functions import *
import os


IMAGES_PATH = "DataSet/labeling_test"
images = os.listdir(IMAGES_PATH)
CLASS_ID = 1

detector = FaceDetector()

img_names_list = []
for img in images:
    img_name = img.split(".")[0]
    if img_name != "":
        img_names_list.append(img_name)

for img_name in img_names_list:
    img = cv2.imread(f"{IMAGES_PATH}/{img_name}.jpg")
    img, bboxes = detector.findFaces(img)

    info_list = []

    # check if any face is detected
    if bboxes:
        # loop through each bounding box
        for bbox in bboxes:
            # bbox contains 'id', 'bbox', 'score', 'center'
            x, y, w, h = bbox["bbox"]
            # check the score
            score = bbox["score"][0]
            # make sure that we're capturing real face
            if score > CONFIDENCE:
                # changing the size of the bounding box to larger one
                x, y, w, h = offset(x, y, w, h)
                # avoid potential error
                x, y, w, h = avoid_negative(x, y, w, h)
                # normalization
                x_center_n, y_center_n, w_n, h_n = normalize_values(img, x, y, w, h)
                # avoid potential error
                x_center_n, y_center_n, w_n, h_n = avoid_values_above_1(x_center_n, y_center_n, w_n, h_n)
                # Append the properties to the list
                info_list.append(f'{CLASS_ID} {x_center_n} {y_center_n} {w_n} {h_n}\n')

            # save the properties in text file
            save_properties_in_txt(info_list, img_name, output_data_dir=IMAGES_PATH)
            print(f"The information of '{img_name}.jpg' is saved inside the text file!")