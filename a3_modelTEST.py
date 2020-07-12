import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn, autograd
from torch import optim

import random
import torch.nn.functional as F
#from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import matplotlib.pyplot as plt

# You can implement classes and helper functions here too.

# Create samples
def make_samples(data, batchsize):
    """
    Creates tensors for input in neural network.

    Args: 
        data - pd dataframe
        batchsize - how many samples to feed to neural network at a time 

    Returns:
        List of tensors that are tuples. Features and tensors and then 
        a label of 0 or 1 for same author or not.
    """

    for_nn = [] # list of instances to run through nn [[d1, d2], 0]

    # take out column for train/test label 
    authorfeature_df = data.drop(data.columns[1], axis=1)

    num_features = len(authorfeature_df.columns) -1 # how many columns are features
    print('num features', num_features)

    # to get how many samples needed to make
    batch = range(batchsize)

    for each_index in batch:
        full = [] # [(tensor, 0/1 label)]
        doc1doc2 = [] # concatenated tensor of doc1 and doc2

        # get first doc
        doc1 = authorfeature_df.sample()
        doc1_index = doc1.index.tolist()[0] # get index as integer (turn .index return as a list then index that list)
       
        # get features --> tensor
        doc1_features = torch.tensor(pd.Index(authorfeature_df.loc[doc1_index, 1:]))
        size = doc1_features.size()
        print('doc1 size', size)
        # add to final tensor 
        doc1doc2.append(doc1_features)
        doc1_author = authorfeature_df.loc[doc1_index, 0] 

        # delete row in data
        authorfeature_df.drop(authorfeature_df.index[doc1_index], axis=0, inplace=True)
        
        # reset index (avoids accessing index larger than df shape)
        authorfeature_df.reset_index(inplace=True, drop=True) 

        # get second doc
        same_author_or_no = random.choice([0, 1]) # 0 is not same, 1 is same 
        if same_author_or_no == 0: # NOT SAME AUTHOR

            # get all rows other than author of first doc
            diff_author = authorfeature_df[authorfeature_df[0] != doc1_author]

            # pick random row 
            doc2 = diff_author.sample()
            doc2_index = doc2.index.tolist()[0] # get index as integer (turn .index return as a list then index that list)

            doc2_features = torch.tensor(pd.Index(authorfeature_df.loc[doc2_index, 1:]))
        
            authorfeature_df.drop(authorfeature_df.index[doc2_index], axis=0, inplace=True) 
            authorfeature_df.reset_index(inplace=True, drop=True) 

            # make sample ([d1, d2], label)        
            both_docs = doc1_features + doc2_features
            sample = (both_docs, 0)

            for_nn.append(sample)
        
        else:
            # same_author_or_no == 1 --> same author

            # get all rows with the same author
            same_author = authorfeature_df[authorfeature_df[0] == doc1_author]

            # pick random row
            doc2 = same_author.sample()
            doc2_index = doc2.index.tolist()[0] # get index as integer (turn .index return as a list then index that list)
      
            doc2_features = torch.tensor(pd.Index(authorfeature_df.loc[doc2_index, 1:]))

            # delete row
            authorfeature_df.drop(authorfeature_df.index[doc2_index], axis=0, inplace=True)
            authorfeature_df.reset_index(inplace=True, drop=True) 

            # make sample ([d1, d2], label)
            both_docs = doc1_features + doc2_features
            sample = (both_docs, 1)

            for_nn.append(sample)

    return for_nn
            

### EDIT THIS ENTIRE THING (VARIABLES AND SUCH)
class BinaryAuthorClassifier(nn.Module):

    # all refer to sizes/dimensions (hidden size, output size)
    def __init__(self, input_size, hidden, nonlinear, output=1): 

        # N = batch size, D_in = input dimension
        # H = hidden dimension, D_out = output dimension
        # output = 1 (should only give either 0 or 1)

        super(BinaryAuthorClassifier, self).__init__()
        #super().__init__()
        self.hidden = hidden
        self.nonlinear = nonlinear

        if hidden > 0: # has hidden dimensions
            self.linear1 = nn.Linear(input_size, hidden)
            self.linear2 = nn.Linear(hidden, output)
        else: # no hidden dimensions
            self.linear1 = nn.Linear(input_size, output)

        if nonlinear is not None:
            if nonlinear == "tanh":
                nonlinear == nn.Tanh()
            else: # only implemented two choices, would be "relu"
                nonlinear == nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        forward function --> accept Tensor of input data
        --> return Tensor of output data
        - can use Modules defined in the constructor as well 
        as arbitrary operators on Tensors
        """

        x = self.linear1(x)

        if self.nonlinear:
            y = self.nonlinear(x)
        if self.hidden:
            y = self.linear2(x)
        final = self.sigmoid(y)

        return final
        # return torch.sigmoid(self.fc3(x))
    """
    NOT SURE IF NEEDED

    def predict(self, x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)

        return torch.tensor(ans)
        """

def train_nn(traindf, batchsize, epochs, iterations, learning_rate=0.01): # default for learning rate?
    """
    Trains neural network based on selection in classifier.

    Args:
        traindf - 
        batchsize - 
        epochs - 
        iterations - 
        learning rate - 

    Returns:
        SOMETHING
    """
    # optimizer 
    """is adaptive compared to SGD where the learning rate has essentially 
    equivalent type of effect for all weights/parameters of the model
    """
    optimizer = optim.Adam(params=bac_model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    # loss function (closer to 0, the better the model)
    #criterion = nn.BCELoss() 
    loss = nn.CrossEntropyLoss()
    """
    good for classification tasks
    penalizes:
        - (greater penalty) when incorrect predictions are predicted with high confidence
        - wrong but confident predictions
        - correct but less confident predictions
    """
    # self.epochs = epochs

    for each_epoch in range(epochs):
        samples = make_samples(traindf, batchsize)
        for each_sample in samples:
            features = each_sample[0]
            labels = each_sample[1]
            
            # loss
            # optimizer
            # backward
            # step ?
            print('feature', features)
            features.size()
            print('label', labels)
            
            #features = Variable(torch.tensor(features))
            #labels = Variable(torch.tensor(labels))
            #print('new feature', features)
            #print('new label', labels)

            out = bac_model(features)
            print('out')
            print(out)

            
            calc_loss = loss(out, labels)
            optimizer.zero_grad()
            calc_loss.backward()
            optimizer.step()  
    

def test_nn(testdf, test_size):
    """
    Test neural network.

    Args:
        testdf - 
        test_size - 
    """

    samples = make_samples(testdf, test_size)
    features = []
    labels = []
    predicted = []

    for each_sample in samples:
        feature = each_sample[0]
        label = each_sample[1]
        features.append(feature)
        labels.append(label)

        output = model(each_sample)

        if output > 0.5:
            output = 1
        else:
            output = 0
        predicted.append(output)
    
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted, average='weighted')
    recall = recall_score(labels, predicted, average='weighted') 
    fscore = f1_score(labels, predicted, average='weighted')

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)    
    print('F-measure: ', fscore)   
    


                



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    parser.add_argument("--trainsize", "-trainsize", dest="trainsize", type=int, default="140", help="How many samples all in all to train model.")
    parser.add_argument("--batchsize", "-batch", dest="batchsize", type=int, default="20", help="How many samples at a time to pass to model.")
    parser.add_argument("--testsize", "-testsize", dest="testsize", type=int, default="60", help="How many samples for testing model.")

    parser.add_argument("--hiddensize", "-hidden", dest="hiddensize", type=int, default=None, help="Size of hidden layer.")
    parser.add_argument("--epochs", "-epochs", dest="epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--nonlinearity", "-nonlinearity", dest="nonlinearity", type=str, default=None, choices=["relu", "tanh"], help="Blank for no non-linearity or choose between ReLU or Tanh.")
    

    # iterations = to epochs?
    # learning rate 


    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    # implement everything you need here

    # open csv file and reset indices
    all_data = pd.read_csv(args.featurefile, header = None)
    all_data = all_data.iloc[1:]
    
    # for input size (how many features are there)
    num_columns = len(all_data.columns)-2

    train_data = all_data[all_data[1] == "train"]
    train_data.reset_index(inplace=True, drop=True) 
    test_data = all_data[all_data[1] == "test"]
    test_data.reset_index(inplace=True, drop=True) 
    print('Finished reading csv file.')

    
    train_size = args.trainsize
    batch = args.batchsize
    # calculate iterations --> total train size divided by # of samples to pass to model each time
    iterations = train_size//batch
    test_size = args.testsize
    hidden = args.hiddensize
    epochs = args.epochs
    nonlinearity = args.nonlinearity
    
    

    # make samples 
    print("Making test samples...")
    train_samples = make_samples(train_data, batch)
    print("Done making test samples.")

    # train model
    print("Training model...")
    bac_model = BinaryAuthorClassifier(num_columns, hidden, nonlinearity)
    
    train_nn(train_data, batch, epochs, iterations)
    print('everything is ok, good job kate')

