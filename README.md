# LT2212 V20 Assignment 3  

Put any documentation here including any answers to the questions in the 
assignment on Canvas.  

## Part 1 - creating the feature table  
The code used for a3_features.py is basically an amalgamation of my code from Assignment 1 and Assignment 2.   

__How To Run:__  
```bash
python3 a3_features.py enron_sample datatable 2000   
python3 a3_features.py [DATA DIRECTORY] [NAME OF OUTPUT CSV FILE] [FINAL NUMBER OF FEATURES]
```  

_**Tokenized the text by:**_    
    - turning text file into a list of strings (each word is a string)   
    - filtering out all integers and punctuation  
    - putting all characters in lowercase   
    --> returns a list of lists  
    NOTE: decided to take out filtering through NLTK stopwords. While exploring the emails, they can be very terse and I felt as if eliminating these words would also eliminate certain features that need to be picked up for identifying the author.   

_**Using that list of lists:**_   
    - create a dictionary for each text file where the key is the word and the number of occurrences of the word within the text is the value  
    --> returns a list of dictionaries [ {a_word : 3, other_word : 6 }, { a_word : 1, another_word : 4}]  

_**Create numpy array:**_    
    - initial array has all zeroes   
    - insert word counts by using index of master_list (list with all of the possible words in the data) and a row index counter (for each document)  

_**Reduce dimensions by using:**_  
    - Truncated Singular Value Decomposition (SVD)  

_**Turn array into csv file:**_  
    - concatenate author labels for each doc  
    - create list that randomly chooses train or test based on weights  
    - concatenate author labels and train/test list  
    - concatenate both arrays together  
    - final_array into pandas dataframe  
    - write csv file using pandas dataframe  

## Part 2 - design and train the basic model    
__How To Run:__  
```bash
python3 a3_model.py datatable.csv --trainsize 140 --batchsize 20 --testsize 60 --epochs 5
```  
__Arguments (default in parentheses):__  
* .csv file  
* train size (140)  
* batch size (20)  
* test size (60)  
* epochs (5)  

__Sample Creation__  
I chose to use the .sample() function in order to randomly pick which documents will be used in creating the samples for training and testing. Through exploring the data in enron_sample, I saw that the number of emails per author ranged from 14 to 1051. Instead of picking through a list of authors each time (giving each author an equal chance of getting chosen), I thought that it would create a better model if there was a higher chance that authors with more emails are chosen. Then there is an equal chance of whether the second document will be from the same author or not. I used the .index method in order to access the features in the dataframe and reset the indices each time I took out an entire row. This prevented the program to try and access indices that were larger than the dataframe's shape (and would return an error even though that index row existed). Both the features and label are turned into tensors and make a tuple together.  

__Optimizer__  
I chose Adam since it is adaptive compared to Stochastic Gradient Descent (SGD). From the articles I've read, it seems as if Adam uses techniques used in SGD that are seen to be advantageous and is able to compute individual adaptive learning rates for different parameters.  

__Loss Function__   
I chose the Binary Cross Entropy (BCE). Through reading a multitude of articles and documentation, I learned that BCE is known to be good for classification tasks like the one given in this assignment. It creates a greater penalty when incorrect predictions are predicted with a high confidence. It also penalizes wrong but confident predictions and correct but less confident predictions.    

## Part 3 - augment the model   
__How To Run:__  
```bash
python3 a3_model.py datatable.csv --hiddensize 10 --nonlinearity relu
```  
__Arguments (default in parentheses):__  
* hidden size (0)  
* non-linearity (None) choices = [relu, tanh]  

__Non-linearities:__  
* ReLU (rectified linear unit function element-wise)  
* Tanh (hyperbolic tangent function)  

__Tests:__  
_**Base commands:**_   
```bash
python3 a3_features.py enron_sample datatable 2000  
python3 a3_model.py datatable.csv --trainsize 210 --batchsize 20 --testsize 90 --epochs 3
```  

_**No Non-linearity**_  
| Hidden Layers | Accuracy | Precision | Recall | F-measure |
|:-------------:|:--------:|:---------:|:------:|:---------:|
|       -       |   0.51   |    0.52   |  0.51  |    0.50   |
|       10      |   0.51   |    0.51   |  0.51  |    0.51   |
|       20      |   0.57   |    0.57   |  0.57  |    0.56   |
|       50      |   0.52   |    0.52   |  0.52  |    0.52   |  

_**ReLU**_  
| Hidden Layers | Accuracy | Precision | Recall | F-measure |
|:-------------:|:--------:|:---------:|:------:|:---------:|
|       -       |   0.49   |    0.49   |  0.49  |    0.49   |
|       10      |   0.61   |    0.60   |  0.61  |    0.55   |
|       20      |   0.49   |    0.49   |  0.49  |    0.40   |
|       50      |   0.52   |    0.54   |  0.52  |    0.49   |  

_**Tanh**_  
| Hidden Layers | Accuracy | Precision | Recall | F-measure |
|:-------------:|:--------:|:---------:|:------:|:---------:|
|       -       |   0.48   |    0.48   |  0.48  |    0.47   |
|       10      |   0.51   |    0.51   |  0.51  |    0.51   |
|       20      |   0.56   |    0.53   |  0.56  |    0.53   |
|       50      |   0.48   |    0.43   |  0.48  |    0.44   |  

For the tests with no non-linearity and Tanh, all four evaluation measurements were highest and improved at 20 hidden layers and then decreased at 50 hidden layers. For ReLU, the highest was with 10 hidden layers. Looking at all the tests conducted, the range for all four measurements is quite small--ranging from 0.43 to 0.61. From this small sample of testing that I've done, the best results were achieved using ReLU with 10 hidden layers.  
