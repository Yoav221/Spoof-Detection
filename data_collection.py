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

if SAVE_FAKE_VIDEOS:
    result = create_video_for_fake_training(cap)

while True:
    # read the current frame from the webcam
    success, img = cap.read()
    print(img)
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
                # avoid potential error
                x, y, w, h = avoid_negative(x, y, w, h)
                # blurriness
                print(x, y, w, h)
                blur_list, blur_value = get_blurriness_list(img, x, y, w, h, blur_list=blur_list)
                # normalization
                x_center_n, y_center_n, w_n, h_n = normalize_values(img, x, y, w, h)
                # avoid potential error
                x_center_n, y_center_n, w_n, h_n = avoid_values_above_1(x_center_n, y_center_n, w_n, h_n)
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
                # save the image
                cv2.imwrite(f"{OUTPUT_DATA_DIR}/{time_now}.jpg", img)
                # save the properties in text file
                save_properties_in_txt(info_list, time_now)

    cv2.imshow("Image", img_out)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
