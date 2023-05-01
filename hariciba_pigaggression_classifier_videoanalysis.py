import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import time
import cv2

start_time = time.time()

fps = 25.0
aggression_threshold = 0.5  # Default: 0.5
predict_test_count, aggressive_count, nonaggressive_count, error_count = 0, 0, 0, 0
agg_avg_prob, nonagg_avg_prob, combined_avg_prob, test_nonagg, test_agg = 0, 0, 0, 0, 0
file_name_iterator, prediction_labels_iterator = 0, 0
file_names, prediction_labels = [], ["placeholder"]
label, file_names_clip = "", ""
pred_dir = 'C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets_test\\'
model_dir = "classification_checkpoints\\Pig_Behaviour_Classifier_ID_7142\\checkpoints\\cp-326.hdf5"

model = keras.models.load_model(model_dir, compile=True)

image_rescaler = ImageDataGenerator(rescale=1./255)

test_data_gen = image_rescaler.flow_from_directory(pred_dir,
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   shuffle=False,
                                                   class_mode=None,
                                                   color_mode='grayscale')

probabilities = model.predict(test_data_gen)
probabilities = np.reshape(probabilities, -1)

for name in test_data_gen.filenames:
    file_names.append(name)

for i in probabilities:
    file_name = file_names[file_name_iterator]
    if i >= aggression_threshold:
        label = "Aggressive"
        aggressive_count += 1
        agg_avg_prob += i
        combined_avg_prob += i
        prediction_labels.append(label)
    elif i < aggression_threshold:
        label = "Non-Aggressive"
        nonaggressive_count += 1
        nonagg_avg_prob += i
        combined_avg_prob += i
        prediction_labels.append(label)
    else:
        pass
    predict_test_count += 1
    file_name_iterator += 1
prediction_labels.pop(0)
print("Number of prediction labels: " + str(len(prediction_labels)))

aggressive_percentage = round((aggressive_count / (aggressive_count + nonaggressive_count)) * 100)
nonaggressive_percentage = 100 - aggressive_percentage
agg_avg_prob_percent = int(round(agg_avg_prob / aggressive_count, 2) * 100)
nonagg_avg_prob_percent = int(round(nonagg_avg_prob / nonaggressive_count, 2) * 100)
combined_average_probability = int(round((agg_avg_prob_percent + nonagg_avg_prob_percent) / 2))
print("Original Total predictions: " + str(predict_test_count))
print("Original Aggressive predictions count: " + str(aggressive_count))
print("Original Nonaggressive predictions count: " + str(nonaggressive_count))
print("Original Aggressive prediction percentage: " + str(aggressive_percentage) + "%")
print("Original Nonaggressive prediction percentage: " + str(nonaggressive_percentage) + "%" + '\n')
print("Aggressive average probability: " + str(agg_avg_prob_percent))
print("Nonaggressive average probability: " + str(nonagg_avg_prob_percent))
print("Combined average probability: " + str(combined_average_probability) + '\n')

smooth_threshold = 1
prediction_labels_temp, probabilities_temp = [], []
prediction_label_counter = 0
for i in probabilities:
    prediction_label_counter += 1
    probabilities_temp.append(i)
    if prediction_label_counter % smooth_threshold == 0:
        if (sum(probabilities_temp) / len(probabilities_temp)) >= aggression_threshold:
            for x in range(smooth_threshold):
                prediction_labels_temp.append("Aggressive")
        else:
            for x in range(smooth_threshold):
                prediction_labels_temp.append("Non-Aggressive")
        probabilities_temp = []
    else:
        pass
list_difference = len(prediction_labels) - len(prediction_labels_temp)
if list_difference == 0:
    pass
else:
    probabilities_temp = []
    for i in range(list_difference):
        probabilities_temp.append(i)
    if (sum(probabilities_temp) / len(probabilities_temp)) >= aggression_threshold:
        for x in range(list_difference):
            prediction_labels_temp.append("Aggressive")
    else:
        for x in range(list_difference):
            prediction_labels_temp.append("Non-Aggressive")
    probabilities_temp = []
prediction_labels = prediction_labels_temp
print("Number of updated prediction labels: " + str(len(prediction_labels)))

TP, TN, FP, FN = 0, 0, 0, 0
subclasses = {"headbiting": 0, "parallelpressing": 0, "inverseparallelpressing": 0, "chasing": 0,
              "headtoheadknocking": 0, "headtobodyknocking": 0, "mounting": 0, "mobile": 0, "immobile": 0}
subclasses_true = {"headbiting": 0, "parallelpressing": 0, "inverseparallelpressing": 0, "chasing": 0,
                   "headtoheadknocking": 0, "headtobodyknocking": 0, "mounting": 0, "mobile": 0, "immobile": 0}
for i in file_names:
    subclass_id_pre = str(i.split("ssive")[1])
    subclass_id = str(subclass_id_pre.split("_")[0])
    if subclass_id in subclasses:
        subclasses[subclass_id] += 1
    else:
        pass
    index_updated = file_names.index(i)
    if "nonaggressive" in i:
        if prediction_labels[index_updated] == "Non-Aggressive":
            TN += 1
            subclasses_true[subclass_id] += 1
        else:
            FP += 1
    else:
        if prediction_labels[index_updated] == "Aggressive":
            TP += 1
            subclasses_true[subclass_id] += 1
        else:
            FN += 1

for key in subclasses_true:
    subclass_recall = subclasses_true[key] / subclasses[key]
    subclass_recall_percent = int(round(subclass_recall, 2) * 100)
    subclasses_true[key] = str(subclass_recall_percent) + "%"

for key in subclasses:
    if subclasses[key] == 1:
        subclasses_true[key] = "N/A"
    else:
        pass

subclasses_true_1 = str(dict(list(subclasses_true.items())[6:]))
subclasses_true_2 = str(dict(list(subclasses_true.items())[:6]))
subclasses_true_1 = subclasses_true_1.replace(": ", ":")
subclasses_true_2 = subclasses_true_2.replace(": ", ":")
subclasses_true_1 = subclasses_true_1.replace("'", "")
subclasses_true_2 = subclasses_true_2.replace("'", "")

updated_total = TP + TN + FP + FN
UPACC, UPTPR, UPTNR, UPPPV, UPNPV = str(int(round((TP + TN) / updated_total, 4) * 100)), \
                                    str(int(round(TP / (TP + FN), 4) * 100)), \
                                    str(int(round(TN / (TN + FP), 4) * 100)), \
                                    str(int(round(TP / (TP + FP), 4) * 100)), \
                                    str(int(round(TN / (TN + FN), 4) * 100))
UPF1 = int(round(2 * (float(UPPPV) * float(UPTPR)) / (float(UPPPV) + float(UPTPR)), 4))
UPF2 = int(round(5 * (float(UPPPV) * float(UPTPR)) / (4 * (float(UPPPV)) + float(UPTPR)), 4))
mcc_numerator = (TN * TP) - (FP * FN)
mcc_denominator = ((TN + FN) * (FP + TP) * (TN + FP) * (FN + TP)) ** 0.5
UPMCC = round((mcc_numerator / mcc_denominator), 4)
UPEVALLIST = "Accuracy:" + UPACC + "%, " + "TPR:" + UPTPR + "%, " + "TNR:" + UPTNR + "%, " + \
             "PPV:" + UPPPV + "%, " + "NPV:" + UPNPV + "%, " + "F1:" + str(UPF1) + "%, F2:" + str(UPF2) + "%, MCC:" + str(UPMCC)

print("Updated TP: " + str(TP))
print("Updated FP: " + str(FP))
print("Updated TN: " + str(TN))
print("Updated FN: " + str(FN))
print("Updated Accuracy: " + UPACC)
print("Updated TPR: " + UPTPR)
print("Updated TNR: " + UPTNR)
print("Updated PPV: " + UPPPV)
print("Updated NPV: " + UPNPV)
print("Updated F1: " + str(UPF1))
print("Updated F2: " + str(UPF2))
print("Updated MCC: " + str(UPMCC))

for label in prediction_labels:
    if "Non" in label:
        test_nonagg += 1
    else:
        test_agg += 1
annotated_total = test_nonagg + test_agg
percentage_agg_annotated = str(int((round((test_agg / annotated_total), 2) * 100))) + "%"
percentage_nonagg_annotated = str(int((round((test_nonagg / annotated_total), 2) * 100))) + "%"
print("Number of aggressive frames (updated): " + str(test_agg))
print("Number of nonaggressive frames (updated): " + str(test_nonagg) + "\n")

pathOut = 'C:\\Users\\Harry\\PigAggressionData\\prediction_videos\\Prediction_Video.mp4'
size = 1088, 612
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

images, file_names_list, output_image_list = [], [], []
file_count = 0
for file_name in os.listdir('C:\\Users\\Harry\\PigAggressionData\\extracted_frames_test_natural'):
    file_names_list.append(file_name)
print("Number of raw frames: " + str(len(file_names_list)) + "\n")
font = cv2.FONT_HERSHEY_DUPLEX
font_size_main = 0.5
font_size_big = 0.8
for i in file_names_list:
    prediction_label = prediction_labels[prediction_labels_iterator]
    file_name_index = file_names_list.index(i)
    img = cv2.imread(os.path.join('C:\\Users\\Harry\\PigAggressionData\\extracted_frames_test_natural', i))
    img = cv2.putText(img,
                      str(i),
                      (5, 30),
                      fontScale=font_size_main,
                      fontFace=font,
                      color=(255, 255, 255))
    img = cv2.putText(img,
                      "Threshold:" + str(int(aggression_threshold * 100)) + "%",
                      (350, 100),
                      fontScale=font_size_big,
                      fontFace=font,
                      color=(255, 255, 255))
    img = cv2.putText(img,
                      "Interval:" + str(smooth_threshold),
                      (550, 100),
                      fontScale=font_size_big,
                      fontFace=font,
                      color=(255, 255, 255))
    img = cv2.putText(img,
                      str(UPEVALLIST),
                      (5, 460),
                      fontScale=font_size_main,
                      fontFace=font,
                      color=(255, 255, 255))
    img = cv2.putText(img,
                      "Behaviour-> Aggressive:" + percentage_agg_annotated + ", Non-Aggressive:" + percentage_nonagg_annotated,
                      (5, 495),
                      fontScale=font_size_main,
                      fontFace=font,
                      color=(255, 255, 255))
    img = cv2.putText(img,
                      "Avg. Probability-> Aggressive:" + str(agg_avg_prob_percent) + "%" +
                      ", Non-Aggressive:" + str(nonagg_avg_prob_percent) + "%",
                      (5, 530),
                      fontScale=font_size_main,
                      fontFace=font,
                      color=(255, 255, 255))
    img = cv2.putText(img,
                      "Non-Aggressive:" + str(subclasses_true_1),
                      (5, 590),
                      fontScale=0.4,
                      fontFace=font,
                      color=(255, 255, 255))
    img = cv2.putText(img,
                      "Aggressive:" + str(subclasses_true_2),
                      (5, 565),
                      fontScale=0.4,
                      fontFace=font,
                      color=(255, 255, 255))
    if probabilities[file_name_index] >= aggression_threshold:
        prob_color = (15, 15, 255)
    else:
        prob_color = (20, 255, 20)
    img = cv2.putText(img,
                      str(int(round(probabilities[file_name_index], 2) * 100)) + "%",
                      (350, 65),
                      fontScale=font_size_big,
                      fontFace=font,
                      color=prob_color)
    if prediction_label == "Aggressive":
        img = cv2.putText(img,
                          str(prediction_label),
                          (5, 65),
                          fontScale=font_size_big,
                          fontFace=font,
                          color=(15, 15, 255))
    elif prediction_label == "Non-Aggressive":
        img = cv2.putText(img,
                          str(prediction_label),
                          (5, 65),
                          fontScale=font_size_big,
                          fontFace=font,
                          color=(20, 255, 20))
    else:
        pass
    if "non" in i:
        img = cv2.putText(img,
                          "Actual: Non-Aggressive",
                          (5, 100),
                          fontScale=font_size_big,
                          fontFace=font,
                          color=(20, 255, 20))
    elif "agg" in i:
        img = cv2.putText(img,
                          "Actual: Aggressive",
                          (5, 100),
                          fontScale=font_size_big,
                          fontFace=font,
                          color=(15, 15, 255))
    else:
        pass
    out.write(img)
    prediction_labels_iterator += 1
    if prediction_labels_iterator > 15000:
        break
    else:
        pass

out.release()

end_time = time.time()
print("Time taken to run script: " + str(end_time - start_time))
