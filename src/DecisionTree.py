import math
import random

class TreeNode(object):
    def __init__(self, isLeaf=False):
        # Your code here
        print "init"

    def predict(self, sample):
        """
        This function predicts the label of given sample
        """
        # Your code here

class DecisionTree(object):
    def __init__(self):
        self.root = None
        
    def train(self, records, attributes):
        self.root = self.tree_growth(records,attributes)

    def tree_growth(self,records,attributes):
        noOfColumn = len(records[0]["attributes"])
        noOfRows = len(records)
        node={}
        maxGain = 0.0
        leftSet=[]
        rightSet=[]
        for columnNumber in attributes:
            columnValues = {}
            for row in records:
                columnValues[row["attributes"][columnNumber]] = '1'
            for value in columnValues:
                (trueRecords,falseRecords) = self.splitRecords(records,columnNumber,value)
                gain = self.gainSplit(records,trueRecords,falseRecords)
                if(gain > maxGain):
                    maxGain = gain
                    leftSet = trueRecords
                    rightSet = falseRecords
                    node["columnIndex"] = columnNumber
                    node["columnValue"] = value
        if maxGain == 0.0:
            label1=0.0
            label2=0.0
            for row in records:
                if row["label"] == "p" :
                    label1+=1
                else:
                    label2+=1
            if(label1 > label2):
                node = {"label":"p"}
            else:
                node = {"label":"e"}
        else:
            if self.stopping_cond(leftSet):
                node["left"] = {"label":leftSet[0]["label"]}
            else:
                node["left"] = self.tree_growth(leftSet, attributes)

            if self.stopping_cond(rightSet):
                node["right"] = {"label":rightSet[0]["label"]}
            else:
                node["right"] = self.tree_growth(rightSet, attributes)
        return node

    def splitRecords(self, records, columnNumber, value):
        trueRecords=[]
        falseRecords=[]
        for row in records:
            if row["attributes"][columnNumber] == value :
                trueRecords.append(row)
            else:
                falseRecords.append(row)
        return (trueRecords, falseRecords)
    
    
    def gainSplit(self,records,trueRecords,falseRecords):
        E1 = self.entropy(trueRecords)
        E2 = self.entropy(falseRecords)
        Ep = self.entropy(records)
        recordsLength = float(len(records))
        trueRecordsLength = float(len(trueRecords))
        falseRecordsLength = float(len(falseRecords))
        p1 = trueRecordsLength/recordsLength
        p2 = falseRecordsLength/recordsLength
        gain = Ep - (p1*E1 + p2*E2)
        return gain

    def entropy(self, records):
        label1=0.0
        label2=0.0
        for row in records:
            if row["label"] == "p" :
                label1+=1
            else:
                label2+=1
        recordLength = label1 + label2
        if label1 == 0.0 or label2 == 0.0:
                entropyValue = 0.0
        else : 
            pl1 = label1/recordLength
            pl2 = label2/recordLength
            entropyValue = -1.0 * ((pl1)* math.log(pl1,2) + (pl2)* math.log(pl2, 2))
        return entropyValue

    def predict(self, sample):
        currentNode = self.root
        isNotLeafNode = True
        while isNotLeafNode:
            if(currentNode.get("label") == None):
                columnIndex = currentNode["columnIndex"]
                columnValue = currentNode["columnValue"]
                if sample["attributes"][columnIndex] == columnValue:
                    currentNode = currentNode["left"]
                else:
                    currentNode = currentNode["right"]
            else:
                isNotLeafNode = False
        return currentNode["label"]

    def stopping_cond(self, records):
        firstRowLabel = records[0]["label"]
        for row in records:
            if(row["label"] != firstRowLabel):
                return False
        return True