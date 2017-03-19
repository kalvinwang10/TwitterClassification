#from sklearn import linear_model
import numpy as np
import math
import collections
import util
import copy
import random
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools



def SGDPredicts(trainData, testData, frac, lossString, penaltyString, maxValBernie, maxValClinton, maxValTrump, maxValObama, existingBernieKeys, existingClintonKeys, existingTrumpKeys, existingObamaKeys):
    #########################Our own SGD ##########################################################
    D_train = []
    Y = []
    i = 0
    random.shuffle(trainData)
    cnf_matrix = []
    if frac != -1:
        trainData = trainData[:frac]

    for (x, y) in trainData:
        featureVector = util.featureExtractor(x)
        D_train.append(featureVector)
        Y.append(y)
        i = i+1
        if frac!=-1 and i > frac*len(trainData):
            break;

    v = DictVectorizer(sparse=True)
    X = v.fit_transform(D_train)

    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(X, Y) #now our training is done
    ###############################################
    test_error = 0
    predictions = []
    y_true = []
    for (x,y) in testData:
        featureVector = util.featureExtractor(x)
        test = v.transform(featureVector)
        predictions.append(clf.predict(test)[0])
        y_true.append(y);
        if clf.predict(test) != y: #prediction is  [1]
            test_error += 1

    train_error = 0
    for (x,y) in trainData:
        featureVector = util.featureExtractor(x)

        train = v.transform(featureVector)
        if clf.predict(train) != y:
            train_error += 1

    test_error = 1.0 * test_error / len(testData)
    train_error = 1.0 * train_error / len(trainData)


    if frac == -1:
        cnf_matrix = metrics.confusion_matrix(y_true, predictions, labels=[1, 2, 3, 4]);
        bigramWeights = v.inverse_transform(clf.coef_)
        for (storedKey, storedVal) in maxValBernie:
            existingBernieKeys[storedKey] = 1;
        for (storedKey, storedVal) in maxValClinton:
            existingClintonKeys[storedKey] = 1;
        for (storedKey, storedVal) in maxValTrump:
            existingTrumpKeys[storedKey] = 1;
        for (storedKey, storedVal) in maxValObama:
            existingObamaKeys[storedKey] = 1;

        #Bernie
        for key in bigramWeights[0].keys():
            i = 0
            for (storedKey, storedVal) in maxValBernie:
                if bigramWeights[0][key] > storedVal and key not in existingBernieKeys.keys():
                    maxValBernie[i] =  (key, bigramWeights[0][key])
                    break
                i = i+1

        #Clinton
        for key in bigramWeights[1].keys():
            i = 0
            for (storedKey, storedVal) in maxValClinton:
                if bigramWeights[1][key] > storedVal and key not in existingClintonKeys.keys():
                    maxValClinton[i] =  (key, bigramWeights[1][key])
                    break
                i = i+1

        #Trump
        for key in bigramWeights[2].keys():
            i = 0
            for (storedKey, storedVal) in maxValTrump:
                if bigramWeights[2][key] > storedVal and key not in existingTrumpKeys.keys():
                    maxValTrump[i] =  (key, bigramWeights[2][key])
                    break
                i = i+1

        #Obama
        for key in bigramWeights[3].keys():
            i = 0
            for (storedKey, storedVal) in maxValObama:
                if bigramWeights[3][key] > storedVal and key not in existingObamaKeys.keys():
                    maxValObama[i] =  (key, bigramWeights[3][key])
                    break
                i = i+1

        ## Uncomment to get Data 

        # print "Most Indicative of Bernie (the words that seperate Bernie from the rest)";
        # for (key, val) in maxValBernie:
        #     print key;
        #     print val;
        #     print "\n";
        #
        # print "Most Indicative of Clinton (the words that seperate Clinton from the rest)";
        # for (key, val) in maxValClinton:
        #     print key;
        #     print val;
        #     print "\n";
        #
        # print "Most Indicative of Trump (the words that seperate Trump from the rest)";
        # for (key, val) in maxValTrump:
        #     print key;
        #     print val;
        #     print "\n";
        #
        # print "Most Indicative of Obama (the words that seperate Obama from the rest)";
        # for (key, val) in maxValObama:
        #     print key;
        #     print val;
        #     print "\n";

        # minValBernie = [(100, 100), (100,100), (100,100), (100,100), (100,100), (100,100),(100, 100), (100,100), (100,100), (100,100), (100,100), (100,100), (100,100), (100,100)]
        # minValClinton = [(100, 100), (100,100), (100,100), (100,100), (100,100), (100,100),(100, 100), (100,100), (100,100), (100,100), (100,100), (100,100), (100,100), (100,100)]
        # minValTrump = [(100, 100), (100,100), (100,100), (100,100), (100,100), (100,100),(100, 100), (100,100), (100,100), (100,100), (100,100), (100,100), (100,100), (100,100)]
        # minValObama = [(100, 100), (100,100), (100,100), (100,100), (100,100), (100,100),(100, 100), (100,100), (100,100), (100,100), (100,100), (100,100), (100,100), (100,100)]
        # #Bernie
        # for key in bigramWeights[0].keys():
        #     i = 0
        #     for (storedKey, storedVal) in minValBernie:
        #         if bigramWeights[0][key] < storedVal:
        #             minValBernie[i] =  (key, bigramWeights[0][key])
        #             break
        #         i = i+1
        #
        # #Clinton
        # for key in bigramWeights[1].keys():
        #     i = 0
        #     for (storedKey, storedVal) in minValClinton:
        #         if bigramWeights[1][key] < storedVal:
        #             minValClinton[i] =  (key, bigramWeights[1][key])
        #             break
        #         i = i+1
        #
        # #Trump
        # for key in bigramWeights[2].keys():
        #     i = 0
        #     for (storedKey, storedVal) in minValTrump:
        #         if bigramWeights[2][key] < storedVal:
        #             minValTrump[i] =  (key, bigramWeights[2][key])
        #             break
        #         i = i+1
        #
        # #Obama
        # for key in bigramWeights[3].keys():
        #     i = 0
        #     for (storedKey, storedVal) in minValObama:
        #         if bigramWeights[3][key] < storedVal:
        #             minValObama[i] =  (key, bigramWeights[3][key])
        #             break
        #         i = i+1
        #
        # # print "Most Indicative of it NOT being Bernie (the words that seperate Bernie from the rest)";
        # # for (key, val) in minValBernie:
        # #     print key;
        # #     print val;
        # #     print "\n";
        # #
        # # print "Most Indicative of it NOT being Clinton (the words that seperate Clinton from the rest)";
        # # for (key, val) in minValClinton:
        # #     print key;
        # #     print val;
        # #     print "\n";
        # #
        # # print "Most Indicative of it NOT being Trump (the words that seperate Trump from the rest)";
        # # for (key, val) in minValTrump:
        # #     print key;
        # #     print val;
        # #     print "\n";
        # #
        # # print "Most Indicative of it NOT being Obama (the words that seperate Obama from the rest)";
        # # for (key, val) in minValObama:
        # #     print key;
        # #     print val;
        # #     print "\n";

    return (train_error*100, test_error*100, frac, cnf_matrix, maxValBernie, maxValClinton, maxValTrump, maxValObama, existingBernieKeys, existingClintonKeys, existingTrumpKeys, existingObamaKeys)
