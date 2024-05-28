import streamlit as st
import pickle
import numpy as np
import gensim

# Load the models and preprocessing function
word2vec_model = gensim.models.Word2Vec.load("word2vec_model.model")
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('preprocess.pkl', 'rb') as f:
    preprocess = pickle.load(f)

def vectorize_text(text):
    vector_size = word2vec_model.vector_size
    vec = np.zeros(vector_size)
    count = 0
    for word in text:
        if word in word2vec_model.wv:
            vec += word2vec_model.wv[word]
            count += 1
    if count > 0:
        vec /= count
    return vec

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.error("Please enter a message to classify.")
    else:
        # Preprocess and vectorize
        transformed_sms = preprocess(input_sms)
        vector_input = vectorize_text(transformed_sms).reshape(1, -1)
        
        # Predict
        result = classifier.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
