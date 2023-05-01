import csv
import random
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time
import math
import os
import numpy

start_time = time.time()


# Convert H.M.S format to bulk seconds
def time_stamp(time_str):
    hours, minutes, seconds = time_str.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


# Function to add zeroes in front of numbers in filenames to maintain order in directory
def add_zeros(variable):
    variable = str(variable)
    digit_count = len(variable)
    missing_zeroes = 9 - digit_count
    variable = ("0" * missing_zeroes) + variable
    return variable


# Some variables
name_counter_frame, name_counter_diff, agg_video_count, agg_diff_count = 0, 0, 0, 0
clip_counter, nonagg_video_count, nonagg_diff_count = 0, 0, 0
nonagg_video, agg_video = 0, 0
output_list_video, output_list_frame, output_list_diff = [], [], []
difference_interval = 10

# Directory paths
original_video_save_PATH = "C:\\Users\\Harry\\PigAggressionData\\original_videos\\"
clipped_video_save_PATH = "C:\\Users\\Harry\\PigAggressionData\\clipped_videos_train\\"
extracted_frame_save_PATH = "C:\\Users\\Harry\\PigAggressionData\\extracted_frames_train\\"
frame_save_PATH_train_agg = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets\\train\\true_aggression\\"
frame_save_PATH_train_nonagg = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets\\train\\false_aggression\\"
frame_save_PATH_val_agg = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets\\validation\\true_aggression\\"
frame_save_PATH_val_nonagg = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets\\validation\\false_aggression\\"
frame_save_PATH_test_agg = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets\\test\\true_aggression\\"
frame_save_PATH_test_nonagg = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets\\test\\false_aggression\\"

output_list_frame = os.listdir(extracted_frame_save_PATH)

# Extract image difference, save to directory
frame_count_total = len(output_list_frame)
for i in output_list_frame:
    image1 = cv2.imread(extracted_frame_save_PATH + i)
    index_current_frame = output_list_frame.index(i)
    try:
        next_image2 = output_list_frame[index_current_frame + difference_interval]
    except:
        break
    image2 = cv2.imread(extracted_frame_save_PATH + next_image2)
    current_image_trim_label, next_image_trim_label = i.split("frame")[0], next_image2.split("frame")[0]
    if next_image_trim_label == current_image_trim_label:
        image2 = cv2.imread(extracted_frame_save_PATH + next_image2)
        pre_diff = 0 + cv2.absdiff(image1, image2)
        diff = cv2.cvtColor(pre_diff, cv2.COLOR_BGR2GRAY)
        diff[0, 0] = 255
        diff[223, 223] = 0
        file_name_diff = add_zeros(name_counter_diff) + "_diff_" + str(i)
        output_list_diff.append(file_name_diff)
        if "non" in file_name_diff:
            nonagg_diff_count += 1
            if "val" in file_name_diff:
                cv2.imwrite(frame_save_PATH_val_nonagg + file_name_diff, diff)
            elif "test" in file_name_diff:
                cv2.imwrite(frame_save_PATH_test_nonagg + file_name_diff, diff)
            else:
                cv2.imwrite(frame_save_PATH_train_nonagg + file_name_diff, diff)
        elif "agg" in file_name_diff:
            agg_diff_count += 1
            if "val" in file_name_diff:
                cv2.imwrite(frame_save_PATH_val_agg + "_" + file_name_diff, diff)
            elif "test" in file_name_diff:
                cv2.imwrite(frame_save_PATH_test_agg + file_name_diff, diff)
            else:
                cv2.imwrite(frame_save_PATH_train_agg + "_" + file_name_diff, diff)
        else:
            print("No condition met")
    else:
        pass

end_time = time.time()
print("Time taken to run script: " + str(end_time - start_time))
