import collections
import re
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
import itertools




def readTweetFile(bernieFile, clintonFile, trumpFile, obamaFile):
    trainData = []
    timeData = [] #[(Tweet, time), (Tweet, time)....]
    testData = []
    re.DOTALL

    bernieSplit = re.split(r"(\b503Bernie\w*\b \w* \w* \w*:\w*:\w* .\w* \w* \w*)", bernieFile.read())
    clintonSplit = re.split(r"(\b503Clinton\w*\b \w* \w* \w*:\w*:\w* .\w* \w* \w*)", clintonFile.read())
    trumpSplit = re.split(r"(\b503Trump\w*\b \w* \w* \w*:\w*:\w* .\w* \w* \w*)", trumpFile.read())
    obamaSplit = re.split(r"(\b503Obama\w*\b \w* \w* \w*:\w*:\w* .\w* \w* \w*)", obamaFile.read())


    #bernie = 1, clinton = 2, trump = 3, obama = 4
    trainData = []
    testData = []

    counter = 0
    for i in range(len(bernieSplit)-1):
        if "503BernieTime" not in bernieSplit[i]: #even elements are tweets
            counter = counter + 1
            if counter > 2500:
                break
            rand = random.randint(1,2)
            if rand == 2:
                testData.append((bernieSplit[i], 1))
            else:
                trainData.append((bernieSplit[i], 1))

    counter = 0
    for i in range(len(clintonSplit)-1):
        if "503ClintonTime" not in clintonSplit[i]: #even elements are tweets
            counter = counter + 1
            if counter > 2500:
                break
            rand = random.randint(1,2)
            if rand == 2:
                testData.append((clintonSplit[i], 2))
            else:
                trainData.append((clintonSplit[i], 2))

    counter = 0
    for i in range(len(trumpSplit)-1):
        if "503TrumpTime" not in trumpSplit[i]: #even elements are tweets
            counter = counter + 1
            if counter > 2500:
                break
            rand = random.randint(1,2)
            if rand == 2:
                testData.append((trumpSplit[i], 3))
            else:
                trainData.append((trumpSplit[i], 3))

    counter = 0
    for i in range(len(obamaSplit)-1):
        if "503ObamaTime" not in obamaSplit[i]: #even elements are tweets
            counter = counter + 1
            if counter > 2500:
                break
            rand = random.randint(1,2)
            if rand == 2:
                testData.append((obamaSplit[i], 4))
            else:
                trainData.append((obamaSplit[i], 4))
    return (trainData, testData)

def featureExtractor(x):
    words = x.split(' ')
    featureVector = collections.defaultdict(float)

    #unigram feature
    for word in words:
        featureVector[word] += 1

    #bigrams are much better predictors
    #bigram feature
    for i in range(0, len(words)-1):
        featureVector[(words[i], words[i+1])] += 1

    #trigrams
    for i in range(0, len(words)-2):
        featureVector[(words[i], words[i+1], words[i+2])] += 1

    return featureVector

#demarcated is a list of tweets
def parseData(demarcator, fileName):
    f = open(fileName, "r")
    demarcated = f.read().split(demarcator)
    return demarcated


# CITATION: THIS CODE IS FROM SKLEARN PYTHON WEBSITE
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


