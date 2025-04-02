import re
import nltk
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nlp = spacy.load('en_core_web_sm')

class TextSummarizer:
    def __init__(self, method='abstractive'):
        self.method = method
        self.summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9.,;?!\s]', '', text)
        return text.strip()

    def extractive_summary(self, text, num_sentences=5):
        sentences = nltk.sent_tokenize(text)
        vectorizer = TfidfVectorizer(stop_words='english')
        sentence_vectors = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(sentence_vectors)
        sentence_scores = similarity_matrix.sum(axis=1)
        ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-num_sentences:][::-1]]
        return ' '.join(ranked_sentences)

    def abstractive_summary(self, text, max_length=150, min_length=30):
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

    def summarize(self, text):
        cleaned_text = self.clean_text(text)
        if self.method == 'extractive':
            return self.extractive_summary(cleaned_text)
        elif self.method == 'abstractive':
            return self.abstractive_summary(cleaned_text)
        else:
            raise ValueError("Invalid method. Choose 'extractive' or 'abstractive'.")

# Streamlit GUI
st.title('LLM-Based Text Summarization Tool')
st.write("This tool generates concise summaries using both extractive and abstractive techniques.")

text = st.text_area("Paste your long text here:", height=300)
method = st.radio("Choose summarization method:", ('extractive', 'abstractive'))

if st.button('Summarize'):
    if text.strip():
        summarizer = TextSummarizer(method=method)
        summary = summarizer.summarize(text)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
