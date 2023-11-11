import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from const import *
from time import time
from functions import *

SAVE = input("Would you like to save the data? (y or any other letter...) ")
SAVE_FAKE_VIDEO = False


class DataCollect:

    def __init__(self, save_fake_video=SAVE_FAKE_VIDEO, save=SAVE, output_data_dir=OUTPUT_DATA_DIR,
                 confidence=CONFIDENCE):
        self.save_fake_video = save_fake_video
        self.save = save
        self.output_data_dir = output_data_dir
        self.confidence = confidence

        self.cap = initialize_webcam()
        self.detector = FaceDetector()

    def create_video_for_fake_training(self, path="Videos for fake training/"):
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        size = (frame_width, frame_height)
        counter_video_name = 1
        while True:
            file_name = f"{path}video_for_fake_{counter_video_name}.avi"
            if not os.path.exists(file_name):
                result = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), 24, size)
                break
            counter_video_name += 1
        return result

    def collect(self):

        if self.save_fake_video:
            result = self.create_video_for_fake_training(self.cap)

        while True:
            # read the current frame from the webcam
            success, img = self.cap.read()
            # img for saving, img_out for displaying
            img_out = img.copy()

            # write frame to the fake training video
            if success and self.save_fake_video:
                result.write(img)

            # detect faces in the image
            img, bboxes = self.detector.findFaces(img, draw=False)

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
                    if score > self.confidence:
                        # changing the size of the bounding box to larger one
                        x, y, w, h = offset(x, y, w, h)
                        # avoid potential error
                        x, y, w, h = avoid_negative(x, y, w, h)
                        # blurriness
                        blur_list, blur_value = get_blurriness_list(img, x, y, w, h, blur_list=blur_list)
                        # drawing
                        cv2.rectangle(img_out, (x, y, w, h), color=(255, 0, 0), thickness=3)
                        cvzone.putTextRect(img_out, f'Score: {int(score * 100)}% ,Blur: {blur_value}', (x, y - 20),
                                           scale=1, thickness=1)

                # Saving the Data
                if self.save == 'y':
                    if all(blur_list) and blur_list != []:  # if there is at least one face that blur > thresh
                        time_now = extract_time_now()
                        # save the image
                        cv2.imwrite(f"{OUTPUT_DATA_DIR}/{time_now}.jpg", img)

            cv2.imshow("Image", img_out)

            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to quit
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    collector = DataCollect()
    collector.collect()
