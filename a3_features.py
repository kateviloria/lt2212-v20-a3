import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need

# My Imports
import glob 
import collections
from sklearn.decomposition import TruncatedSVD
import random
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.

    files_list = []
    all_authors = [] # list of authors for array column
    author_directories = glob.glob('{}/*'.format(args.inputdir)) 
    
   
    all_tokenized = [] # all articles tokenized
    master_list = [] # list of words -> columns for array (will need for indexing)
   
    for each_author in author_directories:
        emails = glob.glob('{}/*'.format(each_author))
        for each_email in emails:
            # list of authors without 'enron_sample/'
            filename_split = each_author.split('/') 
            author = filename_split[1] # take out 'enron_sample/'
            all_authors.append(author)
            with open(each_email, "r") as textfile:
                email_content = ""
                for each_line in textfile.readlines():
                    email_content += each_line
                word_list = email_content.split(' ')
                tokenized = []
                for every_word in word_list:
                    if every_word.isalpha(): # filters out integers and punctuation
                        no_capitals = every_word.lower() # lowercases all characters of word
                        tokenized.append(no_capitals)
                        if no_capitals not in master_list:
                            master_list.append(no_capitals)
                all_tokenized.append(tokenized)
    
    # TO CHECK IF SHAPES FOR ARRAY ARE ALL GOOD
        # print('tokenized')
        # print(len(all_tokenized))
        # print('master list')
        # print(len(master_list))
        # print('author_list')
        # print(len(all_authors) )
    
    # count freq
    word_freq = [] # list of dictionaries
    for every_file in all_tokenized:
        article_dict = {} # { WORD : FREQUENCY }, dictionary per text
        # goes through list per email and uses counter to create a dictionary
        article_dict = collections.Counter(every_file) 
        word_freq.append(article_dict)
   
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.

    # MAKE ARRAY
    # number of rows -> emails
    email_rows = len(all_tokenized)
    print('have doc rows')

    # number of columns -> words in entire lexicon
    word_columns = len(master_list)
    print('have word columns')

    # make array with 0's for all the words that doesn't appear in doc
    word_array = np.zeros((email_rows, word_columns))
    print('initial final array w all zeroes done')

    # to move to next doc
    row_index = 0
    
    # fill in array with word counts
    for every_email in word_freq:
        for every_word in every_email.keys():
            word_index = master_list.index(every_word) # get index from master_list
            word_count = every_email[every_word] # get word count from dictionary
            word_array[row_index, word_index] = word_count # put count in correct index within array
        row_index += 1 # next email/row

    # CHECK SHAPE OF FIRST VEC
        # shape_of_vec = word_array.shape
        # print(shape_of_vec)
    
    # reduce dimension using Truncated SVD (Singular Value Decomposition)
    svd = TruncatedSVD(n_components=args.dims)
    svd_array = svd.fit_transform(word_array)
    print('SVD reduction done')

    # add author column 
    author_array = np.array(all_authors).T
    # to check if author length matches array
        # author_shape = author_array.shape

    # assign train/test emails
    test_size = args.testsize/100 # integer out of 100 for train/test split
    train_size = 1-test_size
    n_emails = len(all_authors)
    # make list that (based on given weights) assigns train/test for same length as emails
    assign_trainortest = random.choices(population=['train','test'], weights = [train_size, test_size], k = n_emails)
    
    # make final array (columns: author, test/train, features)
    trainortest = np.array(assign_trainortest).T # make list into array and transpose
    labels_array = np.column_stack((author_array, trainortest))
    # array that has authors, train/test labels, and features
    final_array = np.column_stack((labels_array, svd_array)) 
   
    # array to pandas dataframe
    all_data = pd.DataFrame(data=final_array[0:,0:])

    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.

    # pandas to csv
    csv_name = str(args.outputfile) + '.csv'
    data_csv = all_data.to_csv(csv_name, index=False) 

    print("Done!")