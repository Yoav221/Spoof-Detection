import os
import random
import shutil
from itertools import islice
from const import *
from functions import *

# Create directories and get the unique names:
create_directories(SPLIT_DATA_PATH)
unique_names = get_shuffle_names()

# Calculate the length of train/val/test set:
total_img_count = len(unique_names)
train_length = int(total_img_count * split_ratio["train"])
val_length = int(total_img_count * split_ratio["validation"])
test_length = int(total_img_count * split_ratio["test"])

# Get the remaining images to the train set:
train_length = get_remaining_to_train(total_img_count, train_length, val_length, test_length)

# Split the list
length_to_split = [train_length, val_length, test_length]
input_iterator = iter(unique_names)
output = [list(islice(input_iterator, elem)) for elem in length_to_split]
print(f"Total images: {total_img_count}\nTrain = {len(output[0])}, Validation = {len(output[1])}, Test = {len(output[2])}")


# Copy the files to the relevant directory:
copy_files(output)
print("Split process completed.")

# Create the data.yaml file for the YOLO Training
f = open(f"DataSet/SplitData/data.yaml", 'a')
f.write(DATA_YAML)
f.close()
print("Data.yaml created.")

