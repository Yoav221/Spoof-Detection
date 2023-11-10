WIDTH_Percentage = 10
HEIGHT_Percentage = 20
CONFIDENCE = 0.8

CAM_WIDTH, CAM_HEIGHT = 640, 480
FLOATING_POINT = 6
BLUR_THRESH = 35 # Larger is more focus

# Directories:
OUTPUT_DATA_DIR = 'DataSet/DataCollect'
PREDICTION_VIDEO_PATH = 'Prediction videos/'
# Consts for Split Data:
INPUT_DATA_PATH = "DataSet/all"
SPLIT_DATA_PATH = "DataSet/SplitData"
split_ratio = {'train': 0.7, 'validation': 0.2, 'test': 0.1}
CLASSES = ['fake', 'real']


DATA_YAML = f"path: /Users/yoav/PycharmProjects/Spoofing Detection/DataSet/SplitData\n\
train: train/images\n\
val: validation/images\n\
test: test/images\n\
\n\
nc: {len(CLASSES)}\n\
names: {CLASSES}"