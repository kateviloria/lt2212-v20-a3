# LT2212 V20 Assignment 3  

Put any documentation here including any answers to the questions in the 
assignment on Canvas.  

__Part 1 - creating the feature table__  
The code used for a3_features.py is basically an amalgamation of my code from Assignment 1 and Assignment 2.   
Tokenized the text by:  
    - turning text file into a list of strings (each word is a string)   
    - filtering out all integers and punctuation  
    - putting all characters in lowercase   
    --> returns a list of lists  
    NOTE: decided to take out filtering through NLTK stopwords since exploring the emails, they can be very terse and I felt as if eliminating these words would also eliminate certain features that need to be picked up for identifying the author  

Using that list of lists:  
    - create a dictionary for each text file where the key is the word and the number of occurrences of the word within the text is the value  
    --> returns a list of dictionaries [ {a_word : 3, other_word : 6 }, { a_word : 1, another_word : 4]}  

Create numpy array:  
    - initial array has all zeroes   
    - insert word counts by using index of master_list (list with all of the possible words in the data) and a row index counter (for each document)  

Reduced dimensions by using:  
    Truncated Singular Value Decomposition (SVD)  

Turn array into csv file:  
    - concatenate author labels for each doc  
    - create list that randomly chooses train or test based on weights  
    - concatenate author labels and train/test list  
    - concatenate both arrays together  
    - final_array into pandas dataframe  
    - write csv file using pandas dataframe  

__Part 2 - design and train the basic model__  

__Part 3 - augment the model__  

__Part Bonus - plotting__  
