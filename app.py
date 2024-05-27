import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from scipy.sparse import csr_matrix

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and the model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.error("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        try:
            vector_input = tfidf.transform([transformed_sms])
        except ValueError as e:
            st.error(f"Error in vectorizing the input: {e}")
            # Create a zero vector with the same dimensionality as the TF-IDF vectorizer
            vector_input = csr_matrix((1, tfidf.vocabulary_.__len__()))
        
        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
