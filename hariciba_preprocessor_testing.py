import csv
import os
import math

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
clipped_video_save_PATH = "C:\\Users\\Harry\\PigAggressionData\\clipped_videos_train_new\\"
extracted_frame_save_PATH = "C:\\Users\\Harry\\PigAggressionData\\extracted_frames_test\\"
frame_save_PATH_test = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets_test\\unlabeled\\"

output_list_first = os.listdir("C:\\Users\\Harry\\PigAggressionData\\clipped_videos_train_new\\")
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

# Extract image difference, save to directory
frame_count_total = len(output_list_frame)
clip_frame_holder = []
diff_holder = []
for i in output_list_frame:
    index_current_frame = output_list_frame.index(i)
    if index_current_frame + difference_interval < frame_count_total:
        image2 = output_list_frame[index_current_frame + difference_interval]
        current_image_trim_label, next_image_trim_label = i, image2
        current_image_trim_label = current_image_trim_label.split("frame")[0]
        next_image_trim_label = next_image_trim_label.split("frame")[0]
    else:
        print("Ending")
        next_image_trim_label = "DONE"
    if next_image_trim_label == current_image_trim_label:
        clip_frame_holder.append(i)
    else:
        skip_counter = 1
        repeater = len(clip_frame_holder) + 1
        for image in clip_frame_holder:
            skip_counter += 1
            index_current_image = clip_frame_holder.index(image)
            try:
                next_image = clip_frame_holder[index_current_image + difference_interval]
            except:
                continue
            if skip_counter % 2 == 0:
                image1 = cv2.imread(extracted_frame_save_PATH + image)
                image2 = cv2.imread(extracted_frame_save_PATH + next_image)
                pre_diff = 0 + cv2.absdiff(image1, image2)
                pre_diff = cv2.cvtColor(pre_diff, cv2.COLOR_BGR2GRAY)
                pre_diff = pre_diff #* blend_weight
                diff_holder.append(pre_diff)
            else:
                pass
        skip_counter = -1
        for k in diff_holder:
            skip_counter += 1
            if skip_counter == 0:
                diff = k
            else:
                diff = k + diff
        diff[0, 0] = 255
        diff[223, 223] = 0
        diff[diff > 255] = 255
        diff[diff <= 51] = 0
        file_name_diff = add_zeros(name_counter_diff) + "_diff_" + str(i)
        file_name_diff.removesuffix(".jpg")
        output_list_diff.append(file_name_diff)
        clip_frame_holder, diff_holder = [], []
        counter = 0
        for repeat in range(repeater):
            counter += 1
            cv2.imwrite(frame_save_PATH_test + file_name_diff + add_zeros(counter) + ".jpg", diff)
end_time = time.time()
print("Time taken to run script: " + str(end_time - start_time))
