# Import the libraries


from matplotlib.pyplot import plot
from textblob import TextBlob # It is a library for processing textual data which use NLP (natural language processing)
import pandas as pd
import numpy as np
import re 

# load the  data 

import csv

import textblob

df = pd.read_csv("test.csv")
#print(df)

# clean the text 
# create a function to clean the tweets  

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text) # Removed @mentions 
    text = re.sub(r'#','',text) # Removing the '#' symbol
    text = re.sub(r'RT[\s]+','',text) # Removing RT which is nothing but retweets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Removing links 
    #text = re.sub(r'[^\w\s]','',text) # Removes everything except words and space 

    return text 

df['tweet'] = df['tweet'].apply(cleanTxt) 

# show the text 
#print(df)

# Create a function to get the subjectivity.
# subjectivity tells us how subjective or opinionated the text is.

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity 
# polarity tells us how positive or negative the text is.

def getPolarity(text):
    return TextBlob(text).sentiment.polarity
# we will create the new columns for subjectivity and polarity 

df['Subjectivity'] = df['tweet'].apply(getSubjectivity)
df['Polarity'] = df['tweet'].apply(getPolarity)
#print(df)

# Create a function to compute the positive, negative, and neutral analysis

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['sentiment'] = df['Polarity'].apply(getAnalysis)
 
#print(df.head(10))

# Create a plot for sentiment 
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
sns.countplot(x='sentiment',data=df)

plt.show()
