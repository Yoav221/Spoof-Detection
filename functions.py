import os
import random
import shutil
from itertools import islice
from const import *
import cv2
import time

# --------- Main ---------

def saving_prediction_video(cap, prediction_video_path=PREDICTION_VIDEO_PATH):
    # prediction video details
    counter_video_name = 1
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    # saving the prediction video
    while True:
        video_prediction_file_name = prediction_video_path + f"test_video_{counter_video_name}.avi"
        if not os.path.exists(video_prediction_file_name):
            video_prediction = cv2.VideoWriter(video_prediction_file_name, cv2.VideoWriter_fourcc(*'MJPG'), 24, size)
            break
        counter_video_name += 1
    return video_prediction


# --------- Data Collection Functions ---------

# Initialize the webcam
def initialize_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    return cap


def extract_time_now():
    time_now = time.time()
    time_now = str(time_now).split('.')
    time_now = time_now[0] + time_now[1]
    return time_now


def offset(x, y, w, h):
    offset_w = (WIDTH_Percentage / 100) * w
    x = int(x - offset_w)
    w = int(w + offset_w * 2)
    offset_h = (HEIGHT_Percentage / 100) * h
    y = int(y - offset_h * 3)
    h = int(h + offset_h * 3.5)
    return x, y, w, h


# --------- SplitData Functions ---------


def create_directories(SPLIT_DATA_PATH):
    try:
        shutil.rmtree(SPLIT_DATA_PATH)
    except OSError as e:
        os.mkdir(SPLIT_DATA_PATH)

    os.makedirs(f"{SPLIT_DATA_PATH}/train/images", exist_ok=True)
    os.makedirs(f"{SPLIT_DATA_PATH}/train/labels", exist_ok=True)
    os.makedirs(f"{SPLIT_DATA_PATH}/validation/images", exist_ok=True)
    os.makedirs(f"{SPLIT_DATA_PATH}/validation/labels", exist_ok=True)
    os.makedirs(f"{SPLIT_DATA_PATH}/test/images", exist_ok=True)
    os.makedirs(f"{SPLIT_DATA_PATH}/test/labels", exist_ok=True)


def get_shuffle_names():
    names_list = os.listdir("DataSet/all")
    unique_names = []
    for name in names_list:
        unique_names.append(name.split('.')[0])

    unique_names = list(set(unique_names))
    unique_names = unique_names[1:]

    # Shuffle:
    random.shuffle(unique_names)
    return unique_names


# Put the remaining images in training
def get_remaining_to_train(total_img_count, train_length, val_length, test_length):
    if total_img_count != (train_length + val_length + test_length):
        remaining = total_img_count - (train_length + val_length + test_length)
        train_length += remaining
        return train_length


# Copy the files
def copy_files(output, input=INPUT_DATA_PATH, output_file=SPLIT_DATA_PATH):
    list_of_sets = ["train", "validation", "test"]
    for i, out in enumerate(output):
        for file_name in out:
            shutil.copy(f"{input}/{file_name}.jpg", f"{output_file}/{list_of_sets[i]}/images/{file_name}.jpg")
            shutil.copy(f"{input}/{file_name}.txt", f"{output_file}/{list_of_sets[i]}/labels/{file_name}.txt")
