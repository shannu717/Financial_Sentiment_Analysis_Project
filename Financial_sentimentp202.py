import pandas as pd
import numpy as np
import pickle
import streamlit as st
import nltk
nltk.download ("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import string
from textblob import Word
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


loaded_model = pickle.load(open('filename', 'rb'))

with open('tfidf1.pkl', 'rb') as f:
	df = pickle.load(f)

def text_cleaning(line_from_column):
    text = line_from_column.lower()
    # Replacing the digits/numbers
    text = text.replace('d', '')
    # remove stopwords
    words = [w for w in text if w not in stopwords.words("english")]
    # apply stemming
    words = [Word(w).lemmatize() for w in words]
    # merge words 
    words = ' '.join(words)
    return text 


if __name__ == '__main__':
    st.title('Financial Sentiment Analysis :bar_chart:')
    st.write('A simple sentiment analysis classification app')
    st.subheader('Give your Input below:')
    sentence = st.text_area('Enter your text here',height=200)
    predict_btt = st.button('predict')
    loaded_model = pickle.load(open('filename', 'rb')) 
   
    if predict_btt:
        clean_text = []
        i = text_cleaning(sentence)
        clean_text.append(i)
        data = df.fit_transform([clean_text])
				
        # st.info(vec)
        prediction = loaded_model.predict(data)

        prediction_prob_negative = prediction[0][-1]
        prediction_prob_neutral = prediction[0][0]
        prediction_prob_positive= prediction[0][1]

        prediction_class = prediction(axis=-1)[0]
        print(prediction)
        if prediction_class == -1:
          st.warning('The sentiment of the given text is:negative sentiment')
        if prediction_class == 0:
          st.success('The sentiment of the given text is:neutral sentiment')
        if prediction_class==1:
          st.success('The sentiment of the given text is:positive sentiment')