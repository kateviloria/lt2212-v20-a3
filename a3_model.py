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
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# You can implement classes and helper functions here too.

def make_samples(data, batchsize):
    """
    Creates tensors for input in neural network.

    Args: 
        data - pd dataframe
        batchsize - how many samples to feed to neural network at a time 

    Returns:
        List of tuples. Each tuple has a tensor of features and a tensor
        label of 0 or 1 for same author or not.
    """

    for_nn = [] # list of instances to feed nn

    # take out column for train/test label 
    authorfeature_df = data.drop(data.columns[1], axis=1)

    num_features = len(authorfeature_df.columns) -1 # how many columns are features

    # how many samples needed to make
    batch = range(batchsize)

    for each_index in batch:
        doc1doc2 = [] # concatenated tensor of doc1 and doc2

        # get first doc
        doc1 = authorfeature_df.sample()
        doc1_index = doc1.index.tolist()[0] # get index as integer (turn .index return as a list then index that list)
       
        # get features and make into tensor tensor
        doc1_features = torch.tensor(pd.Index(authorfeature_df.loc[doc1_index, 1:]))
        
        # add to features tensor 
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

            doc2 = diff_author.sample()
            doc2_index = doc2.index.tolist()[0] 
            doc2_features = torch.tensor(pd.Index(authorfeature_df.loc[doc2_index, 1:]))

            authorfeature_df.drop(authorfeature_df.index[doc2_index], axis=0, inplace=True) 
            authorfeature_df.reset_index(inplace=True, drop=True) 

            both_docs = torch.cat((doc1_features, doc2_features), 0) # size will be double
            label = torch.Tensor([same_author_or_no])
            sample = (both_docs, label)
           
            for_nn.append(sample)
        
        else:
            # same_author_or_no == 1 --> same author

            # get all rows with the same author as doc1
            same_author = authorfeature_df[authorfeature_df[0] == doc1_author]

            doc2 = same_author.sample()
            doc2_index = doc2.index.tolist()[0] 
      
            doc2_features = torch.tensor(pd.Index(authorfeature_df.loc[doc2_index, 1:]))

            authorfeature_df.drop(authorfeature_df.index[doc2_index], axis=0, inplace=True)
            authorfeature_df.reset_index(inplace=True, drop=True) 

            both_docs = torch.cat((doc1_features, doc2_features), 0) 
            label = torch.Tensor([same_author_or_no])
            sample = (both_docs, label)
            
            for_nn.append(sample)

    return for_nn
            

class BinaryAuthorClassifier(nn.Module):

    # all refer to sizes/dimensions (hidden size, output size)
    def __init__(self, input_size, hidden, nonlinear, output=1): 
        
        super().__init__()
        self.hidden = hidden
        self.nonlinear = nonlinear

        if hidden > 0: # has hidden dimensions
            self.linear1 = nn.Linear(input_size, hidden)
            self.linear2 = nn.Linear(hidden, output)
        else: # no hidden dimensions
            self.linear1 = nn.Linear(input_size, output)

        if self.nonlinear is not None:
            if self.nonlinear == "tanh":
                self.nonlinear = nn.Tanh()
            else: # only implemented two choices, would be "relu"
                self.nonlinear = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if self.hidden > 0:
            if self.nonlinear:
                a = self.linear1(x)
                y = self.nonlinear(a)
                z = self.linear2(y)
            else:
                y = self.linear1(x)
                z = self.linear2(y)
        else:
            z = self.linear1(x)
        
        final = self.sigmoid(z)

        return final
 

def train_nn(traindf, batchsize, epochs, iterations, learning_rate=0.01): 
   
    optimizer = optim.Adam(params=bac_model.parameters(), lr=learning_rate)
    loss = nn.BCELoss() # (closer to 0, the better the model)
  
    for each_epoch in range(epochs):
        for each_iteration in range(iterations):
            samples = make_samples(traindf, batchsize)
            for each_sample in samples:
                features = each_sample[0]
                labels = each_sample[1]
    
                out = bac_model(features)

                calc_loss = loss(out, labels)

                optimizer.zero_grad()
                calc_loss.backward()
                optimizer.step()  
    

def test_nn(testdf, test_size):

    samples = make_samples(testdf, test_size)

    all_labels = []
    predicted = []

    for each_sample in samples:
        features = each_sample[0]
        label = each_sample[1]
        all_labels.append(label)

        output = bac_model(features)

        if output > 0.5:
            output = 1
        else:
            output = 0

        predicted.append(output)
    
    accuracy = accuracy_score(all_labels, predicted)
    precision = precision_score(all_labels, predicted, average='weighted')
    recall = recall_score(all_labels, predicted, average='weighted') 
    fscore = f1_score(all_labels, predicted, average='weighted')

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

    parser.add_argument("--hiddensize", "-hidden", dest="hiddensize", type=int, default=0, help="Size of hidden layer.")
    parser.add_argument("--epochs", "-epochs", dest="epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--nonlinearity", "-nonlinearity", dest="nonlinearity", type=str, default=None, choices=["relu", "tanh"], help="Blank for no non-linearity or choose between ReLU or Tanh.")
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    # implement everything you need here

    # open csv file and reset indices
    all_data = pd.read_csv(args.featurefile, header = None)
    all_data = all_data.iloc[1:]
    
    # for input size (how many features are there)
    num_columns = len(all_data.columns)-2
    # for concatenated docs 
    double_features = num_columns*2

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
    print("Making train samples...")
    train_samples = make_samples(train_data, batch)
    print("Done making train samples.")

    # train model
    print("Training model...")
    bac_model = BinaryAuthorClassifier(double_features, hidden, nonlinearity)
    train_nn(train_data, batch, epochs, iterations)
    print('Model trained!')

    # test model
    print('Testing model...')
    test_nn(test_data, test_size)
    print('Testing finished!')