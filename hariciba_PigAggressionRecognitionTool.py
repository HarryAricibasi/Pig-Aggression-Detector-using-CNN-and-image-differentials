import csv
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import os
import math
import time
import cv2
import PySimpleGUI as psg

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


def make_dir_if_missing(builtinfolder):
    if not os.path.exists(builtinfolder):
        os.makedirs(builtinfolder)


print("Please wait for user interface to load")
time.sleep(1)
psg.theme("SystemDefaultForReal")
layout = [
        [psg.Text('\nPig Aggression Recognition Tool (PART)\n', font=('Arial Bold', 20),
                  expand_x=True, justification='center')],
        [psg.Text('Input folder   ', font=('Arial Bold', 12)), psg.Input(default_text="input_video_default_folder", enable_events=True, key='-IN-',
                  font=('Arial Bold', 12), expand_x=True), psg.FolderBrowse(initial_folder="input_video_default_folder", button_color="Teal", key='-in-')],
        [psg.Text('Output folder', font=('Arial Bold', 12)), psg.Input(default_text="output_default_folder", enable_events=True, key='-OUT-',
                  font=('Arial Bold', 12), expand_x=True), psg.FolderBrowse(initial_folder="output_default_folder", button_color="Teal", key='-out-')],
        [psg.Text('', font=('Arial Bold', 2))],
        [psg.Button("", image_filename='custom_assets\\circle.png', border_width=0, key='-BUTTON-')],
        [psg.Help(button_color='Orange', key='-help-'), psg.Push(), psg.Button('About', button_color="Teal", key="-about-"), psg.Exit(button_color='Red', key="-exit-")]
        ]
window = psg.Window('Pig Aggression Recognition Tool (PART)', layout, resizable=True,
                    element_justification='c', finalize=True)
window['-BUTTON-'].bind('<Enter>', '+MOUSE OVER+')
window['-BUTTON-'].bind('<Leave>', '+MOUSE AWAY+')
window['-help-'].bind('<Enter>', '+MOUSE OVER+')
window['-help-'].bind('<Leave>', '+MOUSE AWAY+')
window['-exit-'].bind('<Enter>', '+MOUSE OVER+')
window['-exit-'].bind('<Leave>', '+MOUSE AWAY+')
window['-about-'].bind('<Enter>', '+MOUSE OVER+')
window['-about-'].bind('<Leave>', '+MOUSE AWAY+')
window['-out-'].bind('<Enter>', '+MOUSE OVER+')
window['-out-'].bind('<Leave>', '+MOUSE AWAY+')
window['-in-'].bind('<Enter>', '+MOUSE OVER+')
window['-in-'].bind('<Leave>', '+MOUSE AWAY+')
while True:
    event, values = window.read()
    if event in (psg.WIN_CLOSED, "-exit-"):
        sys.exit()
    elif event == "-BUTTON-":
        input_folder_address = values["-IN-"]
        if input_folder_address == "":
            input_folder_address = "input_video_default_folder\\"
        else:
            pass
        output_folder_address = values["-OUT-"]
        if output_folder_address == "":
            output_folder_address = "output_default_folder\\"
        else:
            pass
        break
    elif event == "-about-":
        psg.Popup("Uses a novel image differential approach with a convolutional neural network.\n"
                  "\nCreated by Harry Aricibasi. \nharryaricibasi@gmail.com",
                  title="About")
    elif event == "-help-":
        psg.Popup(
                "Please select a folder in which there are .mp4 files.\n"
                "\nVideos must be top-down footage of pigs in an enclosure.\n"
                "\nEnsure there is nothing else going on in the frame.\n"
                "\nTerminal will open with progress updates and close when finished.\n"
                "\nIf multiple videos present, each will be processed separately.\n"
                "\nVideo outputs will be saved as *video_name*_prediction_video_output.mp4 in selected output folder.\n"
                "\nText outputs will be saved as *video_name*_prediction_text_output.csv in selected output folder.\n",
                title="Help")
    if event =='-BUTTON-+MOUSE OVER+':
        window['-BUTTON-'].update(image_filename='custom_assets\\circle2.png')
    if event == '-BUTTON-+MOUSE AWAY+':
        window['-BUTTON-'].update(image_filename='custom_assets\\circle.png')
    if event =='-help-+MOUSE OVER+':
        window['-help-'].update(button_color="Yellow")
    if event == '-help-+MOUSE AWAY+':
        window['-help-'].update(button_color="Orange")
    if event =='-exit-+MOUSE OVER+':
        window['-exit-'].update(button_color="Pink")
    if event == '-exit-+MOUSE AWAY+':
        window['-exit-'].update(button_color="Red")
    if event =='-about-+MOUSE OVER+':
        window['-about-'].update(button_color="LightBlue")
    if event == '-about-+MOUSE AWAY+':
        window['-about-'].update(button_color="Teal")
    if event =='-out-+MOUSE OVER+':
        window['-out-'].update(button_color="LightBlue")
    if event == '-out-+MOUSE AWAY+':
        window['-out-'].update(button_color="Teal")
    if event =='-in-+MOUSE OVER+':
        window['-in-'].update(button_color="LightBlue")
    if event == '-in-+MOUSE AWAY+':
        window['-in-'].update(button_color="Teal")
window.close()

print("Initiating program")
time.sleep(1)
name_counter_frame, name_counter_diff = 1, 1
output_list_video, output_list_frame, output_list_diff = [], [], []
difference_interval = 1

make_dir_if_missing("extracted_frames_unified")
make_dir_if_missing("pig_behaviour_classifier_datasets_test\\unlabeled\\")
make_dir_if_missing("output_default_folder")
make_dir_if_missing("input_video_default_folder")

clipped_video_save_PATH = str(input_folder_address) + "\\"
extracted_frame_save_PATH = "extracted_frames_unified\\"
frame_save_PATH_test = "pig_behaviour_classifier_datasets_test\\unlabeled\\"

print("Acquiring videos")
time.sleep(1)
output_list_first = list(set(os.listdir(clipped_video_save_PATH)))
for video in output_list_first:
    if ".mp4" not in video:
        output_list_first.remove(video)
output_list_first.sort()
for video in output_list_first:
    output_list_video.append(video)


def matrix_print(text):
    text = text + " for video: " + video + " (" + (str(output_list_video.index(video) + 1) + "/" + str(len(output_list_video))) + ")\n"
    for character in text:
        sys.stdout.write(character)
        sys.stdout.flush()
        time.sleep(0.0)
    time.sleep(1)


# Extract frames
for video in output_list_video:
    matrix_print("\nAnalyzing video properties")
    video_capture = cv2.VideoCapture(clipped_video_save_PATH + video)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    name_counter_frame, name_counter_diff = 1, 1
    output_list_frame, output_list_diff = [], []
    proceed, image = video_capture.read()
    file_name_frame_fix = str(video)
    file_name_frame_fix = file_name_frame_fix.removesuffix(".mp4")
    matrix_print("Extracting frames")
    while proceed:
        file_name_frame_final = file_name_frame_fix + "frame" + add_zeros(name_counter_frame) + ".jpg"
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(extracted_frame_save_PATH + file_name_frame_final, grayscale_image)
        output_list_frame.append(file_name_frame_final)
        proceed, image = video_capture.read()
        name_counter_frame += 1

    matrix_print("Producing image differentials")
    # Extract image difference, save to directory
    clip_frame_holder = []
    diff_holder = []
    set_counter = 0
    blend_length = int(fps) * 2
    for i in output_list_frame:
        if set_counter < blend_length:
            clip_frame_holder.append(i)
            set_counter += 1
        else:
            set_counter = 1
            skip_counter = 0
            for image in clip_frame_holder:
                skip_counter += 1
                index_current_image = clip_frame_holder.index(image)
                try:
                    next_image = clip_frame_holder[index_current_image + difference_interval]
                except:
                    next_image = clip_frame_holder[index_current_image]
                if skip_counter % 2 == 0:
                    image1 = cv2.imread(extracted_frame_save_PATH + image)
                    image2 = cv2.imread(extracted_frame_save_PATH + next_image)
                    resized_image1 = cv2.resize(image1, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                    resized_image2 = cv2.resize(image2, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                    pre_diff = 0 + cv2.absdiff(resized_image1, resized_image2)
                    pre_diff = cv2.cvtColor(pre_diff, cv2.COLOR_BGR2GRAY)
                    pre_diff = pre_diff  # * blend_weight
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
            clip_frame_holder.append(i)
            cv2.imwrite(frame_save_PATH_test + file_name_diff, diff)
            name_counter_diff += 1

    # Some variables
    aggression_threshold = 0.5  # Default: 0.5
    predict_test_count, aggressive_count, nonaggressive_count, error_count = 0, 0, 0, 0
    agg_avg_prob, nonagg_avg_prob, combined_avg_prob, test_nonagg, test_agg = 0, 0, 0, 0, 0
    file_name_iterator, prediction_labels_iterator = 0, 0
    file_names, prediction_labels = [], ["placeholder"]
    label, file_names_clip = "", ""
    pred_dir = "pig_behaviour_classifier_datasets_test"
    model_dir = "custom_assets\\cp-34.hdf5"

    matrix_print("Loading model")
    model = keras.models.load_model(model_dir, compile=True)

    matrix_print("Rescaling differentials")
    # Rescale images (same as training)
    image_rescaler = ImageDataGenerator(rescale=1./255)

    test_data_gen = image_rescaler.flow_from_directory(pred_dir,
                                                       target_size=(224, 224),
                                                       batch_size=32,
                                                       shuffle=False,
                                                       class_mode=None,
                                                       color_mode='grayscale')

    matrix_print("Generating sigmoid output")
    # Generate predictions (0-1)
    probabilities = model.predict(test_data_gen)
    probabilities = np.reshape(probabilities, -1)

    for name in test_data_gen.filenames:
        file_names.append(name)

    matrix_print("Classifying using threshold")
    # Use threshold to classify predictions (Aggressive - Non-aggressive)
    csv_labels = []
    for i in probabilities:
        file_name = file_names[file_name_iterator]
        if i >= aggression_threshold:
            label = "Aggressive"
            aggressive_count += 1
            agg_avg_prob += i
            combined_avg_prob += i
            csv_labels.append(label)
            for z in range(blend_length):
                prediction_labels.append(label)
        elif i < aggression_threshold:
            label = "Non-Aggressive"
            nonaggressive_count += 1
            nonagg_avg_prob += i
            combined_avg_prob += i
            csv_labels.append(label)
            for z in range(blend_length):
                prediction_labels.append(label)
        else:
            pass
        predict_test_count += 1
        file_name_iterator += 1
    prediction_labels.pop(0)

    matrix_print("Calculating behaviour proportions")
    for label in prediction_labels:
        if "Non" in label:
            test_nonagg += 1
        else:
            test_agg += 1
    annotated_total = test_nonagg + test_agg
    percentage_agg_annotated = str(int((round((test_agg / annotated_total), 2) * 100))) + "%"
    percentage_nonagg_annotated = str(int((round((test_nonagg / annotated_total), 2) * 100))) + "%"

    matrix_print("Initializing video output generator")
    # Initialize video output generator
    pathOut = output_folder_address + "\\" + video.split(".")[0] + '_prediction_video_output.mp4'
    size = width, height
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    matrix_print("Building video output")
    # Build video output
    images, file_names_list, output_image_list = [], [], []
    next_diff = 0
    extracted_frames_list = list(set(os.listdir('extracted_frames_unified')) - {'desktop.ini'})
    extracted_frames_list.sort()
    for file_name in extracted_frames_list:
        file_names_list.append(file_name)
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_size_main = 0.5
    font_size_big = 0.7

    matrix_print("Correcting video ending")
    if len(prediction_labels) < len(file_names_list):
        for i in range(len(file_names_list) - len(prediction_labels)):
            prediction_labels.append(prediction_labels[-1])
    else:
        pass

    probabilities_2 = np.array([])
    for i in probabilities:
        for repeater in range(blend_length):
            probabilities_2 = np.append(probabilities_2, i)

    if len(probabilities_2) < len(file_names_list):
        for i in range(len(file_names_list) - len(probabilities_2)):
            probabilities_2 = np.append(probabilities_2, 0.5)
    else:
        pass

    matrix_print("Adding interface elements")

    for i in file_names_list:
        prediction_label = prediction_labels[prediction_labels_iterator]
        file_name_index = file_names_list.index(i)
        img = cv2.imread(os.path.join('extracted_frames_unified', i))
        rectangle_coords_x, rectangle_coords_y = 505, 72
        blend_overlay = cv2.imread(frame_save_PATH_test + output_list_diff[next_diff])
        x, y, w, h = width - 224, 0, 224, 224
        img = cv2.rectangle(img, (0, 0), (0 + rectangle_coords_x, 0 + rectangle_coords_y), (0, 0, 0), -1)
        img[y:y+h, x:x+w] = blend_overlay
        img = cv2.putText(img,
                          str(i),
                          (5, 15),
                          fontScale=font_size_main,
                          fontFace=font,
                          color=(255, 255, 255))
        img = cv2.putText(img,
                          "Overall Behaviour: Aggressive: " + percentage_agg_annotated + ", Non-Aggressive: " + percentage_nonagg_annotated,
                          (5, 60),
                          fontScale=font_size_main,
                          fontFace=font,
                          color=(255, 255, 255))
        if prediction_label == "Aggressive":
            img = cv2.putText(img,
                              str(prediction_label),
                              (5, 40),
                              fontScale=font_size_big,
                              fontFace=font,
                              color=(15, 15, 255))
        elif prediction_label == "Non-Aggressive":
            img = cv2.putText(img,
                              str(prediction_label),
                              (5, 40),
                              fontScale=font_size_big,
                              fontFace=font,
                              color=(20, 255, 20))
        else:
            pass
        out.write(img)
        prediction_labels_iterator += 1
        if prediction_labels_iterator % blend_length == 0:
            next_diff += 1
        if next_diff > 29:
            next_diff = 29

    out.release()

    matrix_print("Writing to csv output")
    csv_filename = output_folder_address + "\\" + video.split(".")[0] + "_prediction_text_output.csv"
    f = open(csv_filename, "w+")
    f.close()
    timestamp = 0
    with open(csv_filename, 'w') as csvfile:
        fieldnames = ['start time', 'end time', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for item in csv_labels:
            minutes, seconds = math.floor(timestamp / 60), timestamp % 60
            minutes2, seconds2 = math.floor((timestamp + 2) / 60), (timestamp + 2) % 60
            writer.writerow({'start time': str(minutes) + ":" + str(seconds), 'end time':  str(minutes2) + ":" + str(seconds2), 'label': item})
            timestamp += 2
    csvfile.close()

    for frame in os.listdir(extracted_frame_save_PATH):
        os.remove(os.path.join(extracted_frame_save_PATH, frame))
    for diff in os.listdir(frame_save_PATH_test):
        os.remove(os.path.join(frame_save_PATH_test, diff))

end_time = time.time()
matrix_print("Time taken to run script: " + str(round(end_time - start_time)) + " seconds")

matrix_print("Complete")

layout = [
        [psg.Text('\nPig Aggression Recognition Tool (PART)\n', font=('Arial Bold', 20),
                  expand_x=True, justification='center')],
        [psg.Text('Task completed.\n Your output files are in: ' + str(output_folder_address), font=('Arial Bold', 14),
                  expand_x=True, justification='center')],
        [psg.Exit(button_color='Red')]]
window = psg.Window('Pig Aggression Recognition Tool (PART)', layout, resizable=True,
                    element_justification='c')
while True:
    event, values = window.read()
    if event in (psg.WIN_CLOSED, 'Exit'):
        sys.exit()
    else:
        break

window.close()
