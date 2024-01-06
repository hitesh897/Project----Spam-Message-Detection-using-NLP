#!/usr/bin/env python
# coding: utf-8

# # Text Classification with SpaCy
# 
# A common task in NLP is **text classification**. This is "classification" in the conventional machine learning sense, and it is applied to text. Examples include spam detection, sentiment analysis, and tagging customer queries. 
# 
# In this tutorial, you'll learn text classification with spaCy. The classifier will detect spam messages, a common functionality in most email clients. Here is an overview of the data you'll use:

# import libraries and read the data from csv file

# In[2]:


import pandas as pd

# Loading the spam data
# ham is the label for non-spam messages
spam = pd.read_csv('data.csv')
spam.head(10)


# # Bag of Words
# Machine learning models don't learn from raw text data. Instead, you need to convert the text to something numeric.
# 
# The simplest common representation is a variation of one-hot encoding. You represent each document as a vector of term frequencies for each term in the vocabulary. The vocabulary is built from all the tokens (terms) in the corpus (the collection of documents). 
# 
# As an example, take the sentences "Tea is life. Tea is love." and "Tea is healthy, calming, and delicious." as our corpus. The vocabulary then is `{"tea", "is", "life", "love", "healthy", "calming", "and", "delicious"}` (ignoring punctuation).
# 
# For each document, count up how many times a term occurs, and place that count in the appropriate element of a vector. The first sentence has "tea" twice and that is the first position in our vocabulary, so we put the number 2 in the first element of the vector. Our sentences as vectors then look like 
# 
# $$
# \begin{align}
# v_1 &= \left[\begin{matrix} 2 & 2 & 1 & 1 & 0 & 0 & 0 & 0 \end{matrix}\right] \\
# v_2 &= \left[\begin{matrix} 1 & 1 & 0 & 0 & 1 & 1 & 1 & 1 \end{matrix}\right]
# \end{align}
# $$
# 
# This is called the **bag of words** representation. You can see that documents with similar terms will have similar vectors. Vocabularies frequently have tens of thousands of terms, so these vectors can be very large.
# 
# Another common representation is **TF-IDF (Term Frequency - Inverse Document Frequency)**. TF-IDF is similar to bag of words except that each term count is scaled by the term's frequency in the corpus. Using TF-IDF can potentially improve your models. You won't need it here. Feel free to look it up though!

# # Building a Bag of Words model
# 
# Once you have your documents in a bag of words representation, you can use those vectors as input to any machine learning model. spaCy handles the bag of words conversion and building a simple linear model for you with the `TextCategorizer` class.
# 
# The TextCategorizer is a spaCy **pipe**. Pipes are classes for processing and transforming tokens. When you create a spaCy model with `nlp = spacy.load('en_core_web_sm')`, there are default pipes that perform part of speech tagging, entity recognition, and other transformations. When you run text through a model `doc = nlp("Some text here")`, the output of the pipes are attached to the tokens in the `doc` object. The lemmas for `token.lemma_` come from one of these pipes.
# 
# You can remove or add pipes to models. What we'll do here is create an empty model without any pipes (other than a tokenizer, since all models always have a tokenizer). Then, we'll create a TextCategorizer pipe and add it to the empty model.

# This code block initializes an empty spaCy model for the English language and adds a TextCategorizer component to it. The TextCategorizer component is used for text classification tasks.
# 
# 
# 
# 

# In[3]:


import spacy

# Create an empty model
nlp = spacy.blank("en")

# Add the TextCategorizer to the empty model
textcat = nlp.add_pipe("textcat")


# Next we'll add the labels to the model. Here "ham" are for the real messages, "spam" are spam messages.

# This code block is adding two labels "ham" and "spam" to the text classifier in order to train the model to distinguish between these two categories of text.
# 
# 
# 
# 

# In[4]:


# Add labels to text classifier
textcat.add_label("ham")
textcat.add_label("spam")


# # Training a Text Categorizer Model
# 
# Next, you'll convert the labels in the data to the form TextCategorizer requires. For each document, you'll create a dictionary of boolean values for each class. 
# 
# For example, if a text is "ham", we need a dictionary `{'ham': True, 'spam': False}`. The model is looking for these labels inside another dictionary with the key `'cats'`.

# This code block prepares the training data for a text classifier. The variable train_texts is a NumPy array that contains the text messages to be classified. The variable train_labels is a list of dictionaries where each dictionary represents the label for a text message. The label is specified as a binary value for each of the two categories (ham and spam).
# 
# 
# 
# 

# In[5]:


train_texts = spam['text'].values
train_labels = [{'cats': {'ham': label == 'ham',
                          'spam': label == 'spam'}} 
                for label in spam['label']]


# Then we combine the texts and labels into a single list.

# The code block creates a list of tuples train_data, where each tuple contains a text message and its corresponding label. The labels are in the form of a dictionary that has two boolean values indicating whether the message is spam or not (True if it is spam, False if it is ham). The first three tuples of the list are printed.
# 
# 
# 
# 

# In[6]:


train_data = list(zip(train_texts, train_labels))
train_data[:3]


# Now you are ready to train the model. First, create an `optimizer` using `nlp.begin_training()`. spaCy uses this optimizer to update the model. In general it's more efficient to train models in small batches. spaCy provides the `minibatch` function that returns a generator yielding minibatches for training. Finally, the minibatches are split into texts and labels, then used with `nlp.update` to update the model's parameters.

# This code block is performing text classification training using the Spacy library. It is creating batches of data, iterating through each batch, creating a document object from each text, creating a training example from each document and its corresponding label, and updating the model using the optimizer. The ultimate goal is to train the text classifier to predict whether a given text is "ham" or "spam".
# 
# 
# 
# 

# In[7]:


from spacy.util import minibatch
from spacy.training.example import Example

spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

# Create the batch generator with batch size = 8
batches = minibatch(train_data, size=8)
# Iterate through minibatches
for batch in batches:
    # Each batch is a list of (text, label) 
    for text, labels in batch:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, labels)
        nlp.update([example], sgd=optimizer)


# This is just one training loop (or epoch) through the data. The model will typically need multiple epochs. Use another loop for more epochs, and optionally re-shuffle the training data at the begining of each loop. 

# This code block trains a text classification model using spaCy's text categorizer. It initializes the model and adds labels, prepares training data and trains the model for 10 epochs using mini-batch gradient descent. It then prints the loss for each epoch.
# 
# 
# 
# 

# In[8]:


import random

random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(losses)


# # Making Predictions

# Now that you have a trained model, you can make predictions with the `predict()` method. The input text needs to be tokenized with `nlp.tokenizer`. Then you pass the tokens to the predict method which returns scores. The scores are the probability the input text belongs to the classes.

# This code block is using a trained spaCy model to classify two example text documents using a text categorizer. It first tokenizes the texts and stores them as spaCy documents. It then retrieves the 'textcat' component from the model and uses it to predict the category scores for each document. The predicted scores are printed to the console.
# 
# 
# 
# 

# In[9]:


texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA" ]
docs = [nlp.tokenizer(text) for text in texts]
    
# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores = textcat.predict(docs)

print(scores)


# The scores are used to predict a single class or label by choosing the label with the highest probability. You get the index of the highest probability with `scores.argmax`, then use the index to get the label string from `textcat.labels`.

# This code block takes the scores predicted by a text classification model using textcat.predict() on a list of documents, and then uses argmax() to find the label with the highest score/probability for each document. It then prints the predicted labels for each document.
# 
# 
# 
# 

# In[10]:


# From the scores, find the label with the highest score/probability
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])


# Evaluating the model is straightforward once you have the predictions. To measure the accuracy, calculate how many correct predictions are made on some test data, divided by the total number of predictions.

# In[ ]:




