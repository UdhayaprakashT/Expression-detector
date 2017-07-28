#The Emotion Face detection Scripts
#You can modify this script as you wish
import cv2
import glob as gb
import random
import numpy as np
#Emotion list
emojis = ["neutral", "happy", "anger", "fear", "sadness", "surprise"] 
 #Initialize fisher face classifier
fisher_face = cv2.face.createFisherFaceRecognizer()

data = {}


def getFiles(emotion, image_path):
    files = gb.glob("./final_dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*1)]
    prediction = [image_path]
    return training, prediction


def makeTrainingAndValidationSet(image_path):
    training_data = []
    training_labels = []
    prediction_data = ""
    prediction_labels = []
    for emotion in emojis:
        training, prediction = getFiles(emotion, image_path)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emojis.index(emotion))
    
        for item in prediction:  #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emojis.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def runClassifier(image_path):
    training_data, training_labels, prediction_data, prediction_labels = makeTrainingAndValidationSet(image_path)
    
    print "training fisher face classifier suing the training data"
    print "size of training set is:", len(training_labels), "images"
    fisher_face.train(training_data, np.asarray(training_labels))

    print "classification prediction"
    counter = 0
    right = 0
    wrong = 0
    for image in prediction_data:
        pred = fisher_face.predict(image)
        print pred
        return pred


#         if pred == prediction_labels[counter]:
#             right += 1
#             counter += 1
#         else:
#             wrong += 1
#             counter += 1
#     return ((100*right)/(right + wrong))
#
# #Now run the classifier
# metascore = []
# for i in range(0,1):
#     right = runClassifier()
#     print "got", right, "percent right!"
#     metascore.append(right)
#
# print "\n\nend score:", np.mean(metascore), "percent right!"