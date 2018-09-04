import sys
import getopt
import os
import math
import operator
from collections import defaultdict

class NaiveBayes:
    class TrainSplit:
        """
        Set of training and testing data
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Document:
        """
        This class represents a document with a label. classifier is 'pos' or 'neg' while words is a list of strings.
        """
        def __init__(self):
            self.classifier = ''
            self.words = []

    def __init__(self):
        """
        Initialization of naive bayes
        """
        self.stopList = set(self.readFile('data/english.stop'))
        self.bestModel = False
        self.stopWordsFilter = False
        self.naiveBayesBool = False
        self.count = 0
        self.numFolds = 10
        self.freqInPosDoc = defaultdict(float)
        self.freqInNegDoc = defaultdict(float)
        self.positiveCounts = defaultdict(float)
        self.negativeCounts = defaultdict(float)
        self.posCount = 0.0
        self.negCount = 0.0
        self.vocabulary = defaultdict(float)
        self.posWords = 0.0
        self.negWords = 0.0
        self.wordCount = 0.0

        # TODO
        # Implement a multinomial naive bayes classifier and a naive bayes classifier with boolean features. The flag
        # naiveBayesBool is used to signal to your methods that boolean naive bayes should be used instead of the usual
        # algorithm that is driven on feature counts. Remember the boolean naive bayes relies on the presence and
        # absence of features instead of feature counts.

        # When the best model flag is true, use your new features and or heuristics that are best performing on the
        # training and test set.

        # If any one of the flags filter stop words, boolean naive bayes and best model flags are high, the other two
        # should be off. If you want to include stop word removal or binarization in your best performing model, you
        # will need to write the code accordingly.

    def classify(self, words):
        """
        Classify a list of words and return a positive or negative sentiment
        """
        p_naiveScore = 0.0
        n_naiveScore = 0.0
        pSumofNks = 0.0
        nSumofNks = 0.0
        pClassification = 0.0
        nClassification = 0.0

        if self.naiveBayesBool:
            
            for word in self.freqInPosDoc:
               pSumofNks += self.freqInPosDoc[word]

            for word in self.freqInNegDoc:
               nSumofNks += self.freqInNegDoc[word]
                
            pClassification = -(math.log(self.posCount) - math.log(self.posCount + self.negCount))
            nClassification = -(math.log(self.negCount) - math.log(self.posCount + self.negCount))

            for word in words:

                pWord = -(math.log(self.freqInPosDoc[word] + 20.0) - math.log(pSumofNks + (20.0*len(self.vocabulary))))
                #pWord = -(math.log(20) - math.log(pSumofNks + (20*self.vocabCount)))
                p_naiveScore += pWord

                nWord = -(math.log(self.freqInNegDoc[word] + 20.0) - math.log(nSumofNks + (20.0*len(self.vocabulary))))
                #nWord = -(math.log(20) - math.log(nSumofNks + (20*self.vocabCount)))
                n_naiveScore += nWord

            p_naiveScore += pClassification
            n_naiveScore += nClassification

            if p_naiveScore < n_naiveScore:
                return 'pos'

            return 'neg'

        if self.bestModel:
            words = words[len(words)/2+1:]
            unique = set()
            unique_word = set()
            switchSentiment = ['However','but','however', "But", "yet", "Yet", "nonetheless", "nevertheless", "Nonetheless", "Nevertheless"]

            pClassification = -(math.log(self.posCount) - math.log(self.posCount + self.negCount))
            nClassification = -(math.log(self.negCount) - math.log(self.posCount + self.negCount))

            prev = ''
            doc = words[:]
            for word in doc:
                words.append(prev + word)
                prev = word

            temp = []
            for word in words:
                if word not in unique:
                    temp.append(word)
                unique.add(word)

            words = temp[:]

            for word in words:
                if word in self.vocabulary and word not in unique_word:
                    if word in switchSentiment:
                        p_naiveScore = 0.0
                        n_naiveScore = 0.0
                    else:
                        pWord = -(math.log(self.positiveCounts[word] + 3.0) - math.log(self.posWords + len(self.vocabulary)*2.0))
                        p_naiveScore += pWord
                        nWord = -(math.log(self.negativeCounts[word] + 3.0) - math.log(self.negWords + len(self.vocabulary)*2.0))
                        n_naiveScore += nWord
                    unique_word.add(word)

            p_naiveScore += pClassification
            n_naiveScore += nClassification

            if p_naiveScore < n_naiveScore:
                return 'pos'

            return 'neg'

        else:

            if self.stopWordsFilter:
                words = self.filterStopWords(words)

            pClassification = -(math.log(self.posCount) - math.log(self.posCount + self.negCount))
            nClassification = -(math.log(self.negCount) - math.log(self.posCount + self.negCount))

            for word in words:

                if word in self.positiveCounts:
                    pWord = -(math.log(self.positiveCounts[word]) - math.log(self.posWords))
                else:
                    pWord = -(math.log(2) - math.log(self.posWords))

                p_naiveScore += pWord

                if word in self.negativeCounts:
                    nWord = -(math.log(self.negativeCounts[word]) - math.log(self.negWords))
                else:
                    nWord = -(math.log(2) - math.log(self.negWords))

                n_naiveScore += nWord

            p_naiveScore += pClassification
            n_naiveScore += nClassification

            if p_naiveScore < n_naiveScore:
                return 'pos'

            return 'neg'

    def addDocument(self, classifier, words):
        """
        Train your model on a document with label classifier (pos or neg) and words (list of strings). You should
        store any structures for your classifier in the naive bayes class. This function will return nothing
        """
        # TODO
        # Train model on document with label classifiers and words
        # Write code here

        posWordInDocument = defaultdict(float)
        negWordInDocument = defaultdict(float)

        if classifier == 'pos':
            self.posCount += 1

        else:
            self.negCount += 1

        if self.bestModel:
            prev = ''
            doc = words[:]
            for word in doc:
                words.append(prev + word)
                prev = word

            unique_set = set()
            for word in words:
                if word not in unique_set:
                    unique_set.add(word)
                    if classifier == 'pos':
                        self.positiveCounts[word] += 1
                        self.posWords += 1

                    elif classifier == 'neg':
                        self.negativeCounts[word] += 1
                        self.negWords += 1

                    self.vocabulary[word] = 1

        else:
            for word in words:
                if classifier == 'pos':
                    self.positiveCounts[word] += 1
                    posWordInDocument[word] = 1
                    self.posWords += 1

                elif classifier == 'neg':
                    self.negativeCounts[word] += 1
                    negWordInDocument[word] = 1
                    self.negWords += 1

                self.vocabulary[word] = 1

            for word in posWordInDocument:
                self.freqInPosDoc[word] += 1

            for word in negWordInDocument:
                self.freqInNegDoc[word] += 1

        #print self.freqInPosDoc[words[0]]
        pass

    def readFile(self, fileName):
        """
        Reads a file and segments.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        str = '\n'.join(contents)
        result = str.split()
        return result

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        for fileName in posDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            doc.classifier = 'pos'
            split.train.append(doc)
        for fileName in negDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            doc.classifier = 'neg'
            split.train.append(doc)
        return split

    def train(self, split):
        for doc in split.train:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            self.addDocument(doc.classifier, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            yield split

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for doc in split.test:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """
        Construct the training/test split
        """
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print '[INFO]\tOn %d-fold of CV with \t%s' % (self.numFolds, trainDir)

            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    doc.classifier = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                for fileName in negDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    doc.classifier = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                split.train.append(doc)

            posDocTest = os.listdir('%s/pos/' % testDir)
            negDocTest = os.listdir('%s/neg/' % testDir)
            for fileName in posDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                doc.classifier = 'pos'
                split.test.append(doc)
            for fileName in negDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                doc.classifier = 'neg'
                split.test.append(doc)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """
        Stop word filter
        """
        removed = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                removed.append(word)
        return removed


def test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.stopWordsFilter = stopWordsFilter
        classifier.naiveBayesBool = naiveBayesBool
        classifier.bestModel = bestModel
        accuracy = 0.0
        for doc in split.train:
            words = doc.words
            classifier.addDocument(doc.classifier, words)

        for doc in split.test:
            words = doc.words
            guess = classifier.classify(words)
            if doc.classifier == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyFile(stopWordsFilter, naiveBayesBool, bestModel, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.stopWordsFilter = stopWordsFilter
    classifier.naiveBayesBool = naiveBayesBool
    classifier.bestModel = bestModel
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print classifier.classify(testFile)


def main():
    stopWordsFilter = False
    naiveBayesBool = False
    bestModel = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f', '') in options:
        stopWordsFilter = True
    elif ('-b', '') in options:
        naiveBayesBool = True
    elif ('-m', '') in options:
        bestModel = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(stopWordsFilter, naiveBayesBool, bestModel, args[0], args[1])
    else:
        test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel)


if __name__ == "__main__":
    main()
