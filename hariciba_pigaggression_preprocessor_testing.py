import csv
import os

import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time

start_time = time.time()


def time_stamp(time_str):
    hours, minutes, seconds = time_str.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def add_zeros(variable):
    variable = str(variable)
    digit_count = len(variable)
    missing_zeroes = 9 - digit_count
    variable = ("0" * missing_zeroes) + variable
    return variable


name_counter_frame, name_counter_diff, agg_frame_count, agg_diff_count = 0, 0, 0, 0
clip_counter, nonagg_frame_count, nonagg_diff_count = 0, 0, 0
output_list_video, output_list_frame, output_list_diff = [], [], []
difference_interval = 1

original_video_save_PATH = "C:\\Users\\Harry\\PigAggressionData\\original_videos\\"
clipped_video_save_PATH = "C:\\Users\\Harry\\PigAggressionData\\clipped_videos_train\\"
extracted_frame_save_PATH = "C:\\Users\\Harry\\PigAggressionData\\extracted_frames_test\\"
frame_save_PATH_test = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets_test\\unlabeled\\"

output_list_first = os.listdir("C:\\Users\\Harry\\PigAggressionData\\clipped_videos_train\\")
for video in output_list_first:
    if "test" in video:
        output_list_video.append(video)
    else:
        pass

# Extract frames
for i in output_list_video:
    video_capture = cv2.VideoCapture(clipped_video_save_PATH + i)
    proceed, image = video_capture.read()
    file_name_frame_fix = str(i)
    file_name_frame_fix = file_name_frame_fix.removesuffix(".mp4")
    file_name_frame_fix = file_name_frame_fix.removesuffix(".mp4")
    while proceed:
        file_name_frame_final = file_name_frame_fix + "frame" + add_zeros(name_counter_frame) + ".jpg"
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropped_image = grayscale_image[54:666, 96:1184]
        cv2.imwrite(extracted_frame_save_PATH + file_name_frame_final, cropped_image)
        output_list_frame.append(file_name_frame_final)
        proceed, image = video_capture.read()
        name_counter_frame += 1

for i in output_list_frame:
    if "non" in i:
        nonagg_frame_count += 1
    elif "agg" in i:
        agg_frame_count += 1
    else:
        pass

subclasses = {"headbiting": 0, "parallelpressing": 0, "inverseparallelpressing": 0, "chasing": 0,
              "headtoheadknocking": 0, "headtobodyknocking": 0, "mounting": 0, "mobile": 0, "immobile": 0}
for i in output_list_frame:
    subclass_id = str(i.split("ssive")[1])
    subclass_id2 = str(subclass_id.split("_")[0])
    if subclass_id2 in subclasses:
        subclasses[subclass_id2] += 1

print(str(subclasses))
print("nonagg frame count: " + str(nonagg_frame_count))
print("agg frame count: " + str(agg_frame_count))

for i in output_list_frame:
    image1 = cv2.imread(extracted_frame_save_PATH + i)
    index_current_frame = output_list_frame.index(i)
    try:
        next_image = output_list_frame[index_current_frame + difference_interval]
    except:
        print("List index out of range. Next image set to current.")
        next_image = i
    image2 = cv2.imread(extracted_frame_save_PATH + next_image)
    image1 = cv2.resize(image1, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    image2 = cv2.resize(image2, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    pre_diff = 255 - cv2.absdiff(image1, image2)
    diff = cv2.cvtColor(pre_diff, cv2.COLOR_BGR2GRAY)
    diff[0, 0] = 0
    diff[223, 223] = 255
    file_name_diff = add_zeros(name_counter_diff) + "_diff_" + str(i)
    output_list_diff.append(file_name_diff)
    name_counter_diff += 1
    cv2.imwrite(frame_save_PATH_test + file_name_diff, diff)

end_time = time.time()
print("Time taken to run script: " + str(end_time - start_time))
