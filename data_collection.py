import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from const import *
from time import time
from functions import *

# constants
CLASS_ID = int(input("Are you collecting: Fake (0) or Real (1)? "))  # 0 is fake, 1 is real
SAVE = input("Would you like to save the data? (y or any other letter...) ")
SAVE_FAKE_VIDEOS = False

cap = initialize_webcam()
detector = FaceDetector()

# creating fake training video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
counter_video_name = 1
while SAVE_FAKE_VIDEOS:
    file_name = f"Videos for fake training/video_for_fake_{counter_video_name}.avi"
    if not os.path.exists(file_name):
        result = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), 24, size)
        break
    counter_video_name += 1

while True:
    # read the current frame from the webcam
    success, img = cap.read()
    img_out = img.copy()

    # write frame to the fake training video
    if success and SAVE_FAKE_VIDEOS:
        result.write(img)

    # detect faces in the image
    img, bboxes = detector.findFaces(img, draw=False)

    blur_list = []
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

                # Avoid values below 0 (avoid potential error during live cam)
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # blurriness
                img_face = img[y:y + h, x:x + w]
                blur_value = int(cv2.Laplacian(img_face, cv2.CV_64F).var())
                if blur_value > BLUR_THRESH:
                    blur_list.append(True)
                else:
                    blur_list.append(False)

                # Finding midpoints and width/height of the image:
                im_h, im_w, _ = img.shape
                x_center, y_center = x + w / 2, y + h / 2

                # Normalization:
                x_center_n = round(x_center / im_w, FLOATING_POINT)
                y_center_n = round(y_center / im_h, FLOATING_POINT)
                w_n = round(w / im_w, FLOATING_POINT)
                h_n = round(h / im_h, FLOATING_POINT)

                # Avoid values above 1 (avoid potential error)
                if x_center_n > 1: x = 1
                if y_center_n > 1: y = 1
                if w_n > 1: w = 1
                if h_n > 1: h = 1

                # Append the properties to the list
                info_list.append(f'{CLASS_ID} {x_center_n} {y_center_n} {w_n} {h_n}\n')

                # Drawing
                cv2.rectangle(img_out, (x, y, w, h), color=(255, 0, 0), thickness=3)
                cvzone.putTextRect(img_out, f'Score: {int(score * 100)}% ,Blur: {blur_value}', (x, y - 20),
                                   scale=1, thickness=1)

        # Saving the Data
        if SAVE == 'y':
            if all(blur_list) and blur_list != []:  # if there is at least one face that blur > thresh
                time_now = extract_time_now()
                # save the images
                cv2.imwrite(f"{OUTPUT_DATA_DIR}/{time_now}.jpg", img)
                # save the labels with the properties in the text file
                for info in info_list:
                    f = open(f"{OUTPUT_DATA_DIR}/{time_now}.txt", 'a')
                    f.write(info)
                    f.close()

    cv2.imshow("Image", img_out)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
