import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import os
from nltk.stem.porter import PorterStemmer

# Fix NLTK for streamlit Cloud
nltk_data_path = "/tmp/nltk_data"

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)
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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('predict'):

    # 1.preprocess
    transformed_sms = transform_text(input_sms)
    # 2.vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

