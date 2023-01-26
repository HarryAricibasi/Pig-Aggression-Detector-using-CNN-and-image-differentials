import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import time

start_time = time.time()

model = keras.models.load_model("classification_checkpoints\\Pig_Behaviour_Classifier_ID_blend_new_csv\\checkpoints\\cp-32.hdf5", compile=True)

test_dir = "C:\\Users\\Harry\\PigAggressionData\\pig_behaviour_classifier_datasets_blend_new\\test"
image_rescaler = ImageDataGenerator(rescale=1./255)
total_test = sum([len(files) for a, b, files in os.walk(test_dir)])
print("Test data count: " + str(total_test))

test_data_gen = image_rescaler.flow_from_directory(test_dir,
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   shuffle=False,
                                                   class_mode='binary',
                                                   color_mode='grayscale')

test_results = model.evaluate(x=test_data_gen)
print("Loss: " + str(test_results[0]))
print("Accuracy: " + str(test_results[1]))
print("Precision: " + str(test_results[2]))
print("Recall: " + str(test_results[3]))
print("AUC: " + str(test_results[4]))

print("True Positives: " + str(test_results[5]))
print("False Positives: " + str(test_results[7]))
print("True Negatives: " + str(test_results[6]))
print("False Negatives: " + str(test_results[8]))

print("True Negative Rate: " + str(test_results[6] / (test_results[6] + test_results[7])))
print("Negative Predictive Value: " + str(test_results[6] / (test_results[6] + test_results[8])))

end_time = time.time()
print("Time taken to run script: " + str(end_time - start_time))
