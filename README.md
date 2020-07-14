# LT2212 V20 Assignment 3  

Put any documentation here including any answers to the questions in the 
assignment on Canvas.  

##Part 1 - creating the feature table  
The code used for a3_features.py is basically an amalgamation of my code from Assignment 1 and Assignment 2.   

__How To Run:__  
```
python3 a3_features.py enron_sample datatable 2000   
python3 a3_features.py [DATA DIRECTORY] [NAME OF OUTPUT CSV FILE] [NUMBER OF FEATURES]
```  

Tokenized the text by:  
    - turning text file into a list of strings (each word is a string)   
    - filtering out all integers and punctuation  
    - putting all characters in lowercase   
    --> returns a list of lists  
    NOTE: decided to take out filtering through NLTK stopwords. While exploring the emails, they can be very terse and I felt as if eliminating these words would also eliminate certain features that need to be picked up for identifying the author.   

Using that list of lists:  
    - create a dictionary for each text file where the key is the word and the number of occurrences of the word within the text is the value  
    --> returns a list of dictionaries [ {a_word : 3, other_word : 6 }, { a_word : 1, another_word : 4]}  

Create numpy array:  
    - initial array has all zeroes   
    - insert word counts by using index of master_list (list with all of the possible words in the data) and a row index counter (for each document)  

Reduce dimensions by using:  
    Truncated Singular Value Decomposition (SVD)  

Turn array into csv file:  
    - concatenate author labels for each doc  
    - create list that randomly chooses train or test based on weights  
    - concatenate author labels and train/test list  
    - concatenate both arrays together  
    - final_array into pandas dataframe  
    - write csv file using pandas dataframe  

##Part 2 - design and train the basic model    
__How To Run:__  
```
python3 a3_model.py enron_sample datatable.csv --trainsize 140 --batchsize 20 --testsize 60 --epochs 5
```  
__Arguments (default in parentheses):__  
* csv file  
* train size (140)  
* batch size (20)  
* test size (60)  
* epochs (5)  

__Sample Creation__  
I chose to use the .sample() function in order to randomly pick which documents will be used in training or testing. Through exploring the data in enron_sample, I saw that the number of emails per author ranged from 14 to 1051. Instead of picking through a list of authors each time (giving each author an equal chance of getting chosen), I thought that it would create a better model if there was a higher chance that authors with more emails are chosen. Then there is an equal chance of whether the second document will be from the same author or not. I used the .index method in order to access the features in the dataframe and reset the indices each time I took out an entire row. This prevented the program to try and access indices that were larger than the dataframe's shape (and would return an error even though that index row existed). Both the features and label are turned into tensors and make a tuple together.  

__Optimizer__
I chose Adam since it is adaptive compared to SGD where the learning rate has essentially an equivalent type of effect for all weights/parameters of the model.

__Loss Function__ 
I chose the Binary Cross Entropy (BCE). Through reading a multitude of articles I learned that BCE is known to be good for classification tasks like the one given in this assignment. It creates a greater penalty when incorrect predictions are predicted with a high confidence. It also penalizes wrong but confident predictions and correct but less confident predictions.  

##Part 3 - augment the model   
__How To Run:__  
```
python3 a3_model.py enron_sample datatable.csv --hiddensize 10 --nonlinearity relu
```  
__Arguments (default in parentheses):__  
* hidden size (None)
* non-linearity (None) choices = [relu, tanh]  

__Non-linearities:__  
* ReLU (rectified linear unit function element-wise)  
* Tanh (hyperbolic tangent function)  

"Try out a couple of hidden layer sizes with each non-linearity and without any non-linearity and without any non-linearity. Discuss results."


