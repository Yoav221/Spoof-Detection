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


def create_video_for_fake_training(cap, path="Videos for fake training/"):
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    counter_video_name = 1
    while True:
        file_name = f"{path}video_for_fake_{counter_video_name}.avi"
        if not os.path.exists(file_name):
            result = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), 24, size)
            break
        counter_video_name += 1
    return result

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


def avoid_negative(x, y, w, h):
    if x < 0: x = 0
    if y < 0: y = 0
    if w < 0: w = 0
    if h < 0: h = 0
    return x, y, w, h


def get_blurriness_list(img, x, y, w, h, blur_list, blur_thresh=BLUR_THRESH):
    img_face = img[y:y + h, x:x + w]
    blur_value = int(cv2.Laplacian(img_face, cv2.CV_64F).var())
    if blur_value > blur_thresh:
        blur_list.append(True)
    else:
        blur_list.append(False)
    return blur_list, blur_value


def normalize_values(img, x, y, w, h, floating_point=FLOATING_POINT):
    # Finding midpoints and width/height of the image:
    im_h, im_w, _ = img.shape
    x_center, y_center = x + w / 2, y + h / 2
    # Normalize:
    x_center_n = round(x_center / im_w, floating_point)
    y_center_n = round(y_center / im_h, floating_point)
    w_n = round(w / im_w, floating_point)
    h_n = round(h / im_h, floating_point)
    return x_center_n, y_center_n, w_n, h_n


def avoid_values_above_1(x_center_n, y_center_n, w_n, h_n):
    if x_center_n > 1: x_center_n = 1
    if y_center_n > 1: y_center_n = 1
    if w_n > 1: y_center_n = 1
    if h_n > 1: h_n = 1
    return x_center_n, y_center_n, w_n, h_n


def save_properties_in_txt(info_list, time_now, output_data_dir=OUTPUT_DATA_DIR):
    for info in info_list:
        f = open(f"{output_data_dir}/{time_now}.txt", 'a')
        f.write(info)
        f.close()

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
        if name.split('.')[0] != "":
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
