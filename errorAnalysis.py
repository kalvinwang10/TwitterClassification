import ourSGD
import numpy as np
import matplotlib.pylab as plt
import util


n = 20;
allTestError = []
allTrainError = []

nums = [];
frac = -1

bernieFile = open('bernieTweetsTimed.txt', 'r') #testFileBernie.txt ... #bernieTweetsTimed.txt
clintonFile = open('clintonTweetsTimedCleaned.txt', 'r') #testFileClinton.txt #clintonTweetsTimed.txt
obamaFile = open('obamaTweetsTimedCleaned.txt', 'r') #testFileObama.txt #obamaTweetsTimed.txt
trumpFile = open('trumpTweetsTimed.txt', 'r') #testFileTrump.txt #trumpTweetsTimed.txt

maxValBernie = [(0, 0), (0,0), (0,0), (0,0), (0,0), (0,0),(0, 0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]
maxValClinton = [(0, 0), (0,0), (0,0), (0,0), (0,0), (0,0),(0, 0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]
maxValTrump = [(0, 0), (0,0), (0,0), (0,0), (0,0), (0,0),(0, 0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]
maxValObama = [(0, 0), (0,0), (0,0), (0,0), (0,0), (0,0),(0, 0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]
existingBernieKeys = {}
existingClintonKeys = {}
existingTrumpKeys = {}
existingObamaKeys = {}

sampleSize = range(20, 1300, 10);
for i in range(0,n):
#for i in sampleSize:
    (trainData, testData) = util.readTweetFile(bernieFile, clintonFile, trumpFile, obamaFile)

    frac = -1
    #frac = i;

    if frac != -1 and i == 0:
        i = i+1

    (trainErrorPercent, testErrorPercent, num, cnf_matrix, maxValBernie, maxValClinton, maxValTrump, maxValObama, existingBernieKeys, existingClintonKeys, existingTrumpKeys, existingObamaKeys) = ourSGD.SGDPredicts(trainData, testData, frac, "hinge", "l2", maxValBernie, maxValClinton, maxValTrump, maxValObama, existingBernieKeys, existingClintonKeys, existingTrumpKeys, existingObamaKeys)
    allTestError.append(testErrorPercent)
    allTrainError.append(trainErrorPercent)
    nums.append(num)
    bernieFile.seek(0)
    clintonFile.seek(0)
    obamaFile.seek(0)
    trumpFile.seek(0)

    #need a very long string

#get interesting statistics out of this
if frac == -1:
    print "##############STATISTICS###################";
    print "Train Error:";
    print "Mean, Median, Std: ", (np.mean(allTrainError), np.median(allTrainError), np.std(allTrainError)), "in %";
    print "\n";
    print "Test Error:";
    print "Mean, Median, Std: ", (np.mean(allTestError), np.median(allTestError), np.std(allTestError)), "in %";
    print "\n";
    print "Std of difference: ", np.std(np.array(allTestError)-np.array(allTrainError)), "in %";
    print "\n";
    print "\n";
    print "\n";
    print "Bigrams with the highest weights of all n runs";
    print "\n";
    print "Most Indicative of Bernie (the words that seperate Bernie from the rest)";
    for (key, val) in maxValBernie:
        print key;
        print val;
        print "\n";

    print "Most Indicative of Clinton (the words that seperate Clinton from the rest)";
    for (key, val) in maxValClinton:
        print key;
        print val;
        print "\n";

    print "Most Indicative of Trump (the words that seperate Trump from the rest)";
    for (key, val) in maxValTrump:
        print key;
        print val;
        print "\n";

    print "Most Indicative of Obama (the words that seperate Obama from the rest)";
    for (key, val) in maxValObama:
        print key;
        print val;
        print "\n";

    print "##############STATISTICS###################";
    print "\n";
    print "Confusion Matrix from a single run";
    print cnf_matrix;
    #plots
    #################### NOT OUR CODE - CITATION: FROM PYTHON WEBSITE ################################
    # Plot normalized confusion matrix
    class_names = ['Bernie', 'Clinton', 'Trump', 'Obama']
    np.set_printoptions(precision=2) #don't know what thid does
    plt.figure()
    util.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix')
    plt.show()
    #################### NOT OUR CODE - CITATION: FROM PYTHON WEBSITE ################################

else:
    #make plots
    print "allTestError"
    print allTestError;
    print "nums";
    print nums;
    plt.scatter(nums, allTestError, color = 'blue');
    plt.show();
