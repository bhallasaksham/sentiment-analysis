# Sentiment Analyis on iMDB Movie Reviews 

## About

This python project performs sentiment analysis on movie reviews by classifying the reviews as positive or negative. Sentiment analysis is used to extract people's opinion on a variety of topics at many levels of granularity. My goal, is to took at an entire movie review and classify it as positive or negative.

## Implementation

Using Naive Bayes, I will use words as features, add the logarithmic probability scores for each token and make binary decisions between positive and negative sentiment. I will then explore the impact of stop-word filtering. Stop-word filtering helps to improve performance by removing common words like "this", "the", "a", "of", "it" from your train and test sets. A list of stop words is present in data/english.stop

## What is Naive Bayes?

Naive Bayes is a classification technique that uses Bayes' Theorem with an underlying assumption of independence among predictors. Naive Bayes classifiers assume a features presence is unrelated to another feature. For example, if an orange is considered to be an orange if it is orange, round, and 4 inches in diamenter; regardless of if two of these features are related or even existent upon another feature. Bayes' Theorem is the following 

################ Formula 1 here ###################

where A is the class and B is the predictor. There are many examples online, I will leave that for outside of this document.   

There are several pros and cons of Naive Bayes.  

####Pros

* Easy and fast to predict classes of test data. Performs well with multi class prediction  
* If the assumption of independence holds, a Naive Bayes classifier per- forms better than most of models like logistic regression and with less training data.  

####Cons

* If categorical variables have a category that is not observed well in training, then the model will assign a 0 or near zero probability and will be unable to make predictions. To solve this Zero Frequency problem, we can use a smoothing technique like Laplace estimation.  

* Bayes is known to be a bad estimator
* Assumption of independence is usually rare within a data set
* Performs poorly on a set where more false positives are identified than true positives.

## Binary Version of Naive Bayes

To implement a binary version of Naive Bayes classifier, we use the presence or absence of a feature rather than the feature counts. Formal Equation of Binary Naive Bayes  

############### FORMULA 2 HERE ########################
where P(wk,cj) is equal to the Binary event model,  

############### FORMULA 3 HERE ########################

where wk represents each word in the vocabulary  

## tf-idf weighting

Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.  

TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:  

```
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
```

IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:  

```
IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
```

## Running the classifier

Use the command line to run this code.  

The code will run with the command  
```
$ python NaiveBayes.py data/imdb
```

Adding flags should be done as such. Flag -f adds stop word filtering, flag -b uses your binarized model, and flag -m invokes the best model. These flags should be used one at a time.  

## Scope for further improvements 

There are also some common techniques to improve the performance of Na ̈ıve Bayes. Here are some examples of where improvements should be made.  

* "Many people thought this movie was very good, but I found it bad". This sentence has two strong and opposing words in this sentence (good, bad), but it can be inferred that the sentiment of the sentence can be determined by the word bad and not good. How can you weigh your features for ’good’ and ’bad’ differently to reflect this sentence’s sentiment?
* "Paige’s acting was neither realistic or inspired". Currently, the feature set comprises individual words in a bag of words approach. Because of this, the words inspired and realistic are considered separately despite the surrounding words like neither. How can you model take into consideration this word order?
* "The sites are chosen carefully; it is no wonder this comes across as a very beautiful movie". How can you change the weight for a feature like beautiful, since it is expressed more strongly than saying "it is a beautiful film". How does very impact the next word?

## References 

http://www.tfidf.com/ 
https://machinelearningmastery.com/better-naive-bayes/ 
https://en.wikipedia.org/wiki/Naive_Bayes_classifier 
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/ 
https://www.aclweb.org/anthology/W02-1011/ 


