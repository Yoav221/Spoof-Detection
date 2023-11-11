import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from const import *
from time import time
from functions import *
import os

IMAGES_PATH = "DataSet/DataCollect"
CLASS_ID = 1  # 1 for REAL, 0 for FAKE


class DataLabel:

    def __init__(self, images_path=IMAGES_PATH, output_dir=IMAGES_PATH, class_id=CLASS_ID):

        self.images_path = images_path
        self.output_dir = output_dir
        self.class_id = class_id

        self.detector = FaceDetector()
        self.img_names_list = []
        self.count_total_img_info = 0

    def create_img_names_list(self):
        images = os.listdir(self.images_path)
        img_names_list = []
        for img in images:
            img_name = img.split(".")[0]
            if img_name != "":
                self.img_names_list.append(img_name)

    def save_properties_in_txt(self, info_list, img_name):
        for info in info_list:
            f = open(f"{self.output_dir}/{img_name}.txt", 'a')
            f.write(info)
            f.close()

    def label(self):
        # loop over every image
        for img_name in self.img_names_list:
            img = cv2.imread(f"{self.images_path}/{img_name}.jpg")
            # find face in each image
            img, bboxes = self.detector.findFaces(img)

            info_list = []

            # check if any face is detected
            if bboxes:
                # loop through each bounding box
                for bbox in bboxes:
                    # bbox contains 'id', 'bbox', 'score', 'center'
                    x, y, w, h = bbox["bbox"]
                    # check if score above confidence
                    score = bbox["score"][0]
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
                        info_list.append(f'{self.class_id} {x_center_n} {y_center_n} {w_n} {h_n}\n')

                    # save the properties in text file
                    self.save_properties_in_txt(info_list, img_name)
                    print(f"The information of '{img_name}' is saved!")
                    self.count_total_img_info += 1


if __name__ == '__main__':
    l = DataLabel()
    l.create_img_names_list()
    l.label()
    print("Total: {l.count_total_img_info}")
