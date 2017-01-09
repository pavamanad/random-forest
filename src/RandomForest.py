import random
from DecisionTree import DecisionTree

class RandomForest(object):
    """
    Class of the Random Forest
    """
    def __init__(self, tree_num):
        self.tree_num = int(tree_num)
        self.forest = []
        self.recordIndexArray = []

    def train(self, records, attributes):

        for j in range (0, len(records)):
            self.recordIndexArray.append(j)

        for i in range (0, self.tree_num):
            self.forest.append(DecisionTree())
            attributesLen = random.randint(5, 15)
            ensembleAttributes = random.sample(attributes,  attributesLen)
            ensembleRecords = self.bootstrap(records)
            self.forest[i].train(records, ensembleAttributes)
        

    def predict(self, sample):
        # Your code here
        label1 = 0
        label2 = 0
        for i in range (0, self.tree_num):
            predictedlabel = self.forest[i].predict(sample)
            if(predictedlabel == "p"):
                label1+=1
            else:
                label2+=1
        if label1 > label2:
            return "p"
        return "e"
        

    def bootstrap(self, records):
        classifierRecords = []
        classifierRecordsIndex = random.sample(self.recordIndexArray,  500)
        for i in range (0,500):
            classifierRecords.append(records[classifierRecordsIndex[i]])
        return classifierRecords
