import csv
import random
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time
import math
import os

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
difference_interval = 1

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

# Trim videos
annotations_file = open("annotated_csv\\3dayscombined_forHarry_cleaned.csv", 'r')
csv_annotations = csv.reader(annotations_file, delimiter=',')
for each_line in csv_annotations:
    clip_counter += 1
    subclass_label = each_line[4]
    trim_label = each_line[3]
    file_name = str(each_line[0])
    pig_id, pen_id = str(each_line[5]), str(each_line[6])
    trim_start = each_line[1]
    trim_start = time_stamp(trim_start)
    trim_end = each_line[2]
    trim_end = time_stamp(trim_end)
    out_file = add_zeros(clip_counter) + str(trim_label) + str(subclass_label) + "_"\
               + pig_id + pen_id + "_" + str(file_name)
    out_file = out_file.replace(" ", "")
    out_file = out_file.removesuffix(".mp4")
    subdivision_counter = math.floor((trim_end - trim_start)/2)
    subdivision_counter_clip = 0
    clip_tag_str = ""
    # Cut videos longer than 2 seconds into 2 second fragments (if odd, remove last second)
    for i in range(subdivision_counter):
        subdivision_counter_clip += 1
        clip_tag_str = str(subdivision_counter_clip)
        trim_end = trim_start + 2
        ffmpeg_extract_subclip(original_video_save_PATH + file_name,
                               trim_start,
                               trim_end,
                               targetname=clipped_video_save_PATH + out_file +
                                          "sub" + clip_tag_str + ".mp4")
        trim_start = trim_end
        output_list_video.append(out_file + "sub" + clip_tag_str + ".mp4")
        print("Total clip count: " + str(len(output_list_video)))
print("Finished clipping")

# Total subclasses
subclasses_clip = {"headbiting": 0, "parallelpressing": 0, "inverseparallelpressing": 0, "chasing": 0,
              "headtoheadknocking": 0, "headtobodyknocking": 0, "mounting": 0, "mobile": 0, "immobile": 0}
for i in output_list_video:
    if "nonagg" in i:
        nonagg_video += 1
    elif "agg" in i:
        agg_video += 1
    else:
        pass
    subclass_id = str(i.split("ssive")[1])
    subclass_id2 = str(subclass_id.split("_")[0])
    if subclass_id2 in subclasses_clip:
        subclasses_clip[subclass_id2] += 1
print("Agg Video: " + str(agg_video) + ", Nonagg Video: " + str(nonagg_video))
print(str(subclasses_clip))

# Shuffle, then sort videos into respective subclasses
random.shuffle(output_list_video)
headbiting, parallelpressing, headtoheadknocking, mounting, mobile, immobile, chasing, inverseparallelpressing, headtobodyknocking = [], [], [], [], [], [], [], [], []
mounting_pre, immobile_max = 0, 0
immobile_quota = agg_video * (subclasses_clip["immobile"]/(subclasses_clip["immobile"]+subclasses_clip["mobile"]))
for clip in output_list_video:
    if "mounting" in clip:
        mounting_pre += 1
    else:
        pass
for clip in output_list_video:
    if "headbiting" in clip:
        headbiting.append(clip)
    elif ("parallelpressing" in clip) and ("inverseparallelpressing" not in clip):
        parallelpressing.append(clip)
    elif "inverseparallelpressing" in clip:
        inverseparallelpressing.append(clip)
    elif "headtoheadknocking" in clip:
        headtoheadknocking.append(clip)
    elif "headtobodyknocking" in clip:
        headtobodyknocking.append(clip)
    elif "chasing" in clip:
        chasing.append(clip)
    elif "mounting" in clip:
        mounting.append(clip)
    elif "immobile" in clip:
        if immobile_max < immobile_quota:
            immobile.append(clip)
            immobile_max += 1
        else:
            pass
    elif ("mobile" in clip) and ("immobile" not in clip):
        if (len(mobile) + immobile_quota + mounting_pre) < agg_video:
            mobile.append(clip)
        else:
            pass
    else:
        print("No conditions met")
output_list_video = []


# Shuffle list, split into train/val/test, re-assimilate to grand list with new tag, rename file in directory
def shuffle_split_reassimilate_rename(subclass_list):
    random.shuffle(subclass_list)
    half_length = len(subclass_list) // 2
    val_test_split, train_split = subclass_list[:half_length], subclass_list[half_length:]
    half_length_2 = len(val_test_split) // 2
    test_split, val_split = val_test_split[:half_length_2], val_test_split[half_length_2:]
    for q in train_split:
        q_index = train_split.index(q)
        new_q = q.removesuffix(".mp4")
        new_q = new_q + "train" + ".mp4"
        train_split[q_index] = new_q
        output_list_video.append(train_split[q_index])
        os.rename(clipped_video_save_PATH + q, clipped_video_save_PATH + train_split[q_index])
    for r in val_split:
        r_index = val_split.index(r)
        new_r = r.removesuffix(".mp4")
        new_r = new_r + "val" + ".mp4"
        val_split[r_index] = new_r
        output_list_video.append(val_split[r_index])
        os.rename(clipped_video_save_PATH + r, clipped_video_save_PATH + val_split[r_index])
    for s in test_split:
        s_index = test_split.index(s)
        new_s = s.removesuffix(".mp4")
        new_s = new_s + "test" + ".mp4"
        test_split[s_index] = new_s
        output_list_video.append(test_split[s_index])
        os.rename(clipped_video_save_PATH + s, clipped_video_save_PATH + test_split[s_index])


# Apply above function to all subclasses
shuffle_split_reassimilate_rename(headbiting)
shuffle_split_reassimilate_rename(parallelpressing)
shuffle_split_reassimilate_rename(inverseparallelpressing)
shuffle_split_reassimilate_rename(mounting)
shuffle_split_reassimilate_rename(mobile)
shuffle_split_reassimilate_rename(immobile)
shuffle_split_reassimilate_rename(headtoheadknocking)
shuffle_split_reassimilate_rename(headtobodyknocking)
shuffle_split_reassimilate_rename(chasing)

# Calculate results of subclass split
subclasses = {"headbiting": 0, "parallelpressing": 0, "inverseparallelpressing": 0, "chasing": 0,
              "headtoheadknocking": 0, "headtobodyknocking": 0, "mounting": 0, "mobile": 0, "immobile": 0}
subclasses_val = {"headbiting": 0, "parallelpressing": 0, "inverseparallelpressing": 0, "chasing": 0,
              "headtoheadknocking": 0, "headtobodyknocking": 0, "mounting": 0, "mobile": 0, "immobile": 0}
subclasses_test = {"headbiting": 0, "parallelpressing": 0, "inverseparallelpressing": 0, "chasing": 0,
              "headtoheadknocking": 0, "headtobodyknocking": 0, "mounting": 0, "mobile": 0, "immobile": 0}
for i in output_list_video:
    if "val" in i:
        subclass_id = str(i.split("ssive")[1])
        subclass_id2 = str(subclass_id.split("_")[0])
        if subclass_id2 in subclasses_val:
            subclasses_val[subclass_id2] += 1
    elif "test" in i:
        subclass_id = str(i.split("ssive")[1])
        subclass_id2 = str(subclass_id.split("_")[0])
        if subclass_id2 in subclasses_test:
            subclasses_test[subclass_id2] += 1
    else:
        subclass_id = str(i.split("ssive")[1])
        subclass_id2 = str(subclass_id.split("_")[0])
        if subclass_id2 in subclasses:
            subclasses[subclass_id2] += 1
print(str(subclasses))
print(str(subclasses_val))
print(str(subclasses_test))

# Extract frames, save frame as grayscale, resized
for i in output_list_video:
    video_capture = cv2.VideoCapture(clipped_video_save_PATH + i)
    proceed, image = video_capture.read()
    file_name_frame_fix = str(i)
    file_name_frame_fix = file_name_frame_fix.removesuffix(".mp4")
    while proceed:
        file_name_frame_final = file_name_frame_fix + "frame" + add_zeros(name_counter_frame) + ".jpg"
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropped_image = grayscale_image[54:666, 96:1184]
        resized_image = cv2.resize(cropped_image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(extracted_frame_save_PATH + file_name_frame_final, resized_image)
        output_list_frame.append(file_name_frame_final)
        proceed, image = video_capture.read()
        name_counter_frame += 1

end_time_prediff = time.time()
print("Time taken to run script (prediff): " + str(end_time_prediff - start_time))

# Extract image difference, save to directory
frame_count_total = len(output_list_frame)
for i in output_list_frame:
    end_of_clip = False
    image1 = cv2.imread(extracted_frame_save_PATH + i)
    index_current_frame = output_list_frame.index(i)
    if index_current_frame + difference_interval < frame_count_total:
        next_image = output_list_frame[index_current_frame + difference_interval]
    else:
        next_image = i
        end_of_clip = True
    current_image_trim_label, next_image_trim_label = i[0:8], next_image[0:8]
    if next_image_trim_label == current_image_trim_label:
        pass
    else:
        next_image = i
        end_of_clip = True
    image2 = cv2.imread(extracted_frame_save_PATH + next_image)
    pre_diff = 255 - cv2.absdiff(image1, image2)
    diff = cv2.cvtColor(pre_diff, cv2.COLOR_BGR2GRAY)
    diff[0, 0] = 0
    diff[223, 223] = 255
    file_name_diff = add_zeros(name_counter_diff) + "_diff_" + str(i)
    output_list_diff.append(file_name_diff)
    name_counter_diff += 1
    if end_of_clip:
        pass
    elif "non" in file_name_diff:
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
print("Difference Count: " + str(name_counter_diff))

end_time = time.time()
print("Time taken to run script: " + str(end_time - start_time))
