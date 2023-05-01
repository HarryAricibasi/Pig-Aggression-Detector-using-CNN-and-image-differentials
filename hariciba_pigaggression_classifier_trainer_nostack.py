import os
import random
import numpy
import tensorflow
import visualkeras
import matplotlib.pyplot as plt
from keras import backend
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, \
    Precision, Recall, AUC, BinaryAccuracy
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict
import time
import math
from PIL import ImageFont

# Check GPU
print(tensorflow.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

# Force CPU
#tensorflow.config.experimental.set_visible_devices([], 'GPU')

# Start timer
start_time = time.time()

# Specify paths
dataset_path = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets_blend_prev\\"
train_dir = os.path.join(dataset_path, 'train')
validation_dir = os.path.join(dataset_path, 'validation')
test_dir = os.path.join(dataset_path, 'test')

# Configure model name
tag = "blend_prev"
model_save_name = "Pig_Behaviour_Classifier_ID_" + str(tag)
print("Model name: " + str(model_save_name))

# Get number of files in each directory
total_train = sum([len(files) for a, b, files in os.walk(train_dir)])
total_val = sum([len(files) for c, d, files in os.walk(validation_dir)])

# Training variables
batch_size = 32
epochs = 100
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Original image: 1280,720
batch_count_per_epoch = math.ceil(total_train / batch_size)
optimizer_learning_rate_variable, optimizer_learning_rate_decay = 0.00001, 0.000  # Default: 0.001, 0.000
print("Batch Size: " + str(batch_size) + ", Epochs: " +
      str(epochs) + ", Image Width, Image Height: " + str(IMG_WIDTH) + ", " + str(IMG_HEIGHT))
print("Optimizer learning rate: " + str(optimizer_learning_rate_variable) + ", Optimizer decay: " +
      str(optimizer_learning_rate_decay))

# Image rescaling as feature normalization
image_rescaler = ImageDataGenerator(rescale=1./255.)

# Apply transformation to training and validation data
train_data_gen = image_rescaler.flow_from_directory(train_dir,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode="binary",
                                                    color_mode='grayscale')
val_data_gen = image_rescaler.flow_from_directory(validation_dir,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  class_mode="binary",
                                                  color_mode='grayscale')
test_data_gen = image_rescaler.flow_from_directory(test_dir,
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   shuffle=False,
                                                   class_mode='binary',
                                                   color_mode='grayscale')

# Class indices: 0 = negative, 1 = positive. Default alphabetical order in folder, A = 0, B = 1.
print("Training data class indices (0:negative, 1:positive " + str(train_data_gen.class_indices))
print("Validation data class indices (0:negative, 1:positive " + str(val_data_gen.class_indices))


# Image 90 degree rotator function to be used in augmentation
def rotate_image(image):
    return numpy.rot90(image, numpy.random.choice([-1, 0, 1, 2]))


# Image augmentation settings
image_augmentor = ImageDataGenerator(cval=255.,
                                     fill_mode="constant",
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     rescale=1./255.,
                                     preprocessing_function=rotate_image)
'''
# Apply augmentation to training data
train_data_gen = image_augmentor.flow_from_directory(train_dir,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary',
                                                     color_mode='grayscale')

# Plot and save images of augmented data
file_names = []
augmented_images_count = 50
augmented_save_path = "augmented_examples\\"
augmented_data_gen = image_augmentor.flow_from_directory(train_dir,
                                                         batch_size=1,
                                                         save_to_dir=augmented_save_path,
                                                         save_prefix=model_save_name,
                                                         save_format='jpg',
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         color_mode='grayscale')
for name in augmented_data_gen.filenames:
    file_names.append(name)
for i in range(augmented_images_count):
    img, label = augmented_data_gen.next()
    plt.imshow(img[0], cmap="gray")
print("Augmented images saved.")
'''
# Configure model checkpoints
checkpoint_path = "classification_checkpoints\\" + model_save_name + ".\\checkpoints\\cp-{epoch:02d}.hdf5"
model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=False,
    overwrite=False,
    save_freq="epoch",
    verbose=0,
    monitor='val_auc',
    mode='max')

# Configure model early stopping
model_earlystopping_callback = tensorflow.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=epochs,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False)

# Configure optimizer learning rate and decay
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=optimizer_learning_rate_variable,
                                             decay=optimizer_learning_rate_decay)

# Build model
model = Sequential()

model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[BinaryAccuracy(), Precision(), Recall(), AUC(),
                       TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])

model.summary()

# Plot model
font = ImageFont.truetype("arial.ttf", 32)
color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'coral'
color_map[Activation]['fill'] = 'lightskyblue'
color_map[Dropout]['fill'] = 'limegreen'
color_map[MaxPooling2D]['fill'] = 'plum'
color_map[Dense]['fill'] = 'red'
color_map[Flatten]['fill'] = 'yellow'
visualkeras.layered_view(model,
                         font=font,
                         color_map=color_map,
                         legend=True,
                         to_file="classification_architecture\\" + str(model_save_name) + ".png")
print("Model architecture visualization saved.")

# Train model
fitted_model = model.fit(x=train_data_gen,
                         steps_per_epoch=len(train_data_gen),
                         epochs=epochs,
                         validation_data=val_data_gen,
                         validation_steps=len(val_data_gen),
                         callbacks=[model_earlystopping_callback, model_checkpoint_callback],
                         verbose=1)

# Save model
model_save_path = "classification_models\\" + str(model_save_name) + ".h5"
saved_model = model.save(model_save_path)
print("Model saved. Version: " + str(tag))

# Acquire metrics
metric_save_path = "classification_metrics\\"
accuracy = fitted_model.history['binary_accuracy']
val_accuracy = fitted_model.history['val_binary_accuracy']
precision = fitted_model.history['precision']
val_precision = fitted_model.history['val_precision']
recall = fitted_model.history['recall']
val_recall = fitted_model.history['val_recall']
auc = fitted_model.history['auc']
val_auc = fitted_model.history['val_auc']
loss = fitted_model.history['loss']
val_loss = fitted_model.history['val_loss']
true_pos, val_true_pos = fitted_model.history['true_positives'], fitted_model.history['val_true_positives']
false_pos, val_false_pos = fitted_model.history['false_positives'], fitted_model.history['val_false_positives']
true_neg, val_true_neg = fitted_model.history['true_negatives'], fitted_model.history['val_true_negatives']
false_neg, val_false_neg = fitted_model.history['false_negatives'], fitted_model.history['val_false_negatives']
epochs_range = range(epochs)
epochs_list = []
for i in epochs_range:
    i += 1
    epochs_list.append(i)
epochs_range = epochs_list
metric_list_reusable = []
lowest_val_loss_index = val_loss.index(min(val_loss))
font_size_main, font_size_small = 32, 24
figure_width, figure_height = 24, 24
line_width = 4
midpoint_marker = int(epochs / 2)
x_axis_plot = [lowest_val_loss_index + 1]

# Print model metrics
print("Training Loss: " + str(loss))
print("Validation Loss: " + str(val_loss))
print("Training Accuracy: " + str(accuracy))
print("Validation Accuracy: " + str(val_accuracy))
print("Training Precision: " + str(precision))
print("Validation Precision: " + str(val_precision))
print("Training Recall: " + str(recall))
print("Validation Recall: " + str(val_recall))
print("Training AUC: " + str(auc))
print("Validation AUC: " + str(val_auc))
print("Training True Positives: " + str(true_pos))
print("Training False Positives: " + str(false_pos))
print("Training True Negatives: " + str(true_neg))
print("Training False Negatives: " + str(false_neg))
print("Validation True Positives: " + str(val_true_pos))
print("Validation False Positives: " + str(val_false_pos))
print("Validation True Negatives: " + str(val_true_neg))
print("Validation False Negatives: " + str(val_false_neg))


def round_metric(metric):
    for item in metric:
        if metric.index(item) == lowest_val_loss_index:
            rounded_metric_value = round(item, 4)
            metric_list_reusable.append(rounded_metric_value)
        else:
            pass


def add_datapoint_labels(x_axis, y_axis):
    for xy in zip(x_axis, y_axis):
        plt.annotate('(%s, %s)' % xy, xy=xy, xycoords='data', textcoords='data', fontsize=font_size_main)


# F1 score
def get_f1_score(precision, recall):
    f1_score_numerator = 2 * (precision * recall)
    f1_score_denominator = (precision + recall)
    if f1_score_denominator == 0:
        f1_score_numerator = 0
        f1_score_denominator = 0
    else:
        pass
    f1_score = f1_score_numerator / f1_score_denominator
    return f1_score


# F2 score
def get_f2_score(precision, recall):
    f2_score_numerator = 5 * (precision * recall)
    f2_score_denominator = ((4 * precision) + recall)
    if f2_score_denominator == 0:
        f2_score_numerator = 0
        f2_score_denominator = 0
    else:
        pass
    f2_score = f2_score_numerator / f2_score_denominator
    return f2_score


# MCC score
def get_mcc(TP, FN, TN, FP):
    mcc_numerator = (TN * TP) - (FP * FN)
    mcc_denominator = ((TN + FN) * (FP + TP) * (TN + FP) * (FN + TP))**0.5
    if mcc_denominator == 0:
        mcc_numerator = 0
        mcc_denominator = 1
    else:
        pass
    mcc = round((mcc_numerator / mcc_denominator), 4)
    return mcc


# Plot metrics
plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.ylim(0, 1)
plt.minorticks_on()
plt.plot(epochs_range, recall, label='Training Recall', linewidth=line_width)
plt.plot(epochs_range, val_recall, label='Validation Recall', linewidth=line_width)
plt.legend(loc='upper left')
plt.title('Training and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainValRecall")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.ylim(0, 1)
plt.minorticks_on()
plt.plot(epochs_range, precision, label='Training Precision', linewidth=line_width)
plt.plot(epochs_range, val_precision, label='Validation Precision', linewidth=line_width)
plt.legend(loc='upper left')
plt.title('Training and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainValPrecision")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.ylim(0, 1)
plt.minorticks_on()
plt.plot(epochs_range, accuracy, label='Training Accuracy', linewidth=line_width)
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy', linewidth=line_width)
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainValAccuracy")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.ylim(0, 1)
plt.minorticks_on()
plt.plot(epochs_range, auc, label='Training AUC', linewidth=line_width)
plt.plot(epochs_range, val_auc, label='Validation AUC', linewidth=line_width)
plt.legend(loc='upper left')
plt.title('Training and Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC of ROC')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainValAUC")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.minorticks_on()
plt.plot(epochs_range, true_pos, label='True Positives', linewidth=line_width)
plt.plot(epochs_range, false_pos, label='False Positives', linewidth=line_width)
plt.plot(epochs_range, true_neg, label='True Negatives', linewidth=line_width)
plt.plot(epochs_range, false_neg, label='False Negatives', linewidth=line_width)
plt.legend(loc='center right')
plt.title('Training Confusion Matrix')
plt.xlabel('Epochs')
plt.ylabel('Number of Predictions')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainingConfusionMatrix")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.minorticks_on()
plt.plot(epochs_range, val_true_pos, label='True Positives', linewidth=line_width)
plt.plot(epochs_range, val_false_pos, label='False Positives', linewidth=line_width)
plt.plot(epochs_range, val_true_neg, label='True Negatives', linewidth=line_width)
plt.plot(epochs_range, val_false_neg, label='False Negatives', linewidth=line_width)
plt.legend(loc='center right')
plt.title('Validation Confusion Matrix')
plt.xlabel('Epochs')
plt.ylabel('Number of Predictions')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_ValidationConfusionMatrix")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.minorticks_on()
plt.plot(epochs_range, loss, label='Training Loss', linewidth=line_width)
plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=line_width)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (Binary Crossentropy)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainValLoss")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.ylim(0, 1)
plt.minorticks_on()
f1_score_train, f1_score_val = [], []
for i in precision:
    current_index = precision.index(i)
    f1_recall = recall[current_index]
    f1_precision = i
    f1_score_item = get_f1_score(f1_precision, f1_recall)
    f1_score_train.append(f1_score_item)
for i in val_precision:
    current_index = val_precision.index(i)
    f1_recall = val_recall[current_index]
    f1_precision = i
    f1_score_item = get_f1_score(f1_precision, f1_recall)
    f1_score_val.append(f1_score_item)
plt.plot(epochs_range, f1_score_train, label='Training F1 score', linewidth=line_width)
plt.plot(epochs_range, f1_score_val, label='Validation F1 score', linewidth=line_width)
plt.legend(loc='upper left')
plt.title('Training and Validation F1 score')
plt.xlabel('Epochs')
plt.ylabel('F1 score')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainValF1")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.ylim(0, 1)
plt.minorticks_on()
f2_score_train, f2_score_val = [], []
for i in precision:
    current_index = precision.index(i)
    f2_recall = recall[current_index]
    f2_precision = i
    f2_score_item = get_f2_score(f2_precision, f2_recall)
    f2_score_train.append(f2_score_item)
for i in val_precision:
    current_index = val_precision.index(i)
    f2_recall = val_recall[current_index]
    f2_precision = i
    f2_score_item = get_f2_score(f2_precision, f2_recall)
    f2_score_val.append(f2_score_item)
plt.plot(epochs_range, f2_score_train, label='Training F2 score', linewidth=line_width)
plt.plot(epochs_range, f2_score_val, label='Validation F2 score', linewidth=line_width)
plt.legend(loc='upper left')
plt.title('Training and Validation F2 score')
plt.xlabel('Epochs')
plt.ylabel('F2 score')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainValF2")

plt.figure(figsize=(figure_width, figure_height))
plt.rc('font', size=font_size_main)
plt.grid(visible=True, which='both', axis='both', color='black', linestyle='-', linewidth=0.25)
plt.minorticks_on()
plt.ylim(-1, 1)
MCC_train, MCC_val = [], []
for i in true_pos:
    current_index = true_pos.index(i)
    MCC_true_pos = true_pos[current_index]
    MCC_true_neg = true_neg[current_index]
    MCC_false_pos = false_pos[current_index]
    MCC_false_neg = false_neg[current_index]
    MCC_item = get_mcc(MCC_true_pos, MCC_false_neg, MCC_true_neg, MCC_false_pos)
    MCC_train.append(MCC_item)
for i in val_true_pos:
    current_index = val_true_pos.index(i)
    MCC_true_pos = val_true_pos[current_index]
    MCC_true_neg = val_true_neg[current_index]
    MCC_false_pos = val_false_pos[current_index]
    MCC_false_neg = val_false_neg[current_index]
    MCC_item = get_mcc(MCC_true_pos, MCC_false_neg, MCC_true_neg, MCC_false_pos)
    MCC_val.append(MCC_item)
plt.plot(epochs_range, MCC_train, label='Training MCC', linewidth=line_width)
plt.plot(epochs_range, MCC_val, label='Validation MCC', linewidth=line_width)
plt.legend(loc='upper left')
plt.title('Training and Validation MCC')
plt.xlabel('Epochs')
plt.ylabel('Matthews Correlation Coefficient')
plt.savefig(metric_save_path + "Model_ID_" + str(tag) + "_TrainValMCC")

end_time = time.time()
print("Time taken to run script: " + str(end_time - start_time))
backend.clear_session()
