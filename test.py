import streamlit as st
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("AyoubChLin/DistilBERT_ZeroShot")
model = AutoModelForSequenceClassification.from_pretrained("AyoubChLin/DistilBERT_ZeroShot")

classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
topics = ["Health", "Environment", "Technology", "Economy", "Entertainment", "Sports", "Politics", "Education", "Travel", "Food"]

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import re

# Download NLTK stopwords dataset
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_query(user_query):
    user_query = re.sub(r'[^a-zA-Z0-9\s]', '', user_query)
    tokens = word_tokenize(user_query)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    spell = SpellChecker()
    corrected_tokens = [spell.correction(word) for word in filtered_tokens]
    corrected_tokens = [word.replace('gud', 'good') for word in corrected_tokens]
    preprocessed_query = ' '.join(corrected_tokens)

    return preprocessed_query


def search_solr(query, base_url, return_fields, num_results=10):
    
    q = f'title:(*{query}*)'  # Modified to include wildcard search
    search_url = f'{base_url}/select?q={q}&fl={return_fields}&wt=json&sort=score desc&rows={num_results}'
    response = requests.get(search_url)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': f'Failed to fetch data: Status Code {response.status_code}'}
    
def classify_topic(user_query):
    result = classifier(user_query, topics)
    return result["labels"][0]

def classify_end_continue(prompt):
    url = "http://35.233.185.244:5000/classify_end_continue"
    data = {"prompt": prompt}
    response = requests.post(url, json=data)
    return response.json()

def classify_wiki_chat(prompt):
    url = "http://34.82.173.246:5000/classify_wiki_chat"
    data = {"prompt": prompt}
    response = requests.post(url, json=data)
    return response.json()

def chatterbot_response(prompt):
    url = "http://34.168.104.3:5000/chat"
    data = {"prompt": prompt}
    response = requests.post(url, json=data)
    return response.json()

# def on_send():
#     user_input = st.session_state.user_input.strip()
#     if user_input:
#         end_continue_response = classify_end_continue(user_input)
#         if end_continue_response['output'] == 'continue chat':
#             wiki_chat_response = classify_wiki_chat(user_input)
#             if wiki_chat_response['output'] == 'wiki':
#                 preprocessed_query = preprocess_query(user_input)
#                 results = search_solr(preprocessed_query, 'http://35.245.97.133:8983/solr/IRF23P1', 'topic,title,revision_id,summary')
#                 if 'response' in results and 'docs' in results['response'] and results['response']['docs']:
#                     latest_doc = results['response']['docs'][-1]
#                     response = latest_doc.get('summary', 'Summary not available.')
#                 else:
#                     response = "I couldn't find any information on that topic."
#             elif wiki_chat_response['output'] == 'chat':
#                 response = chatterbot_response(user_input)
#         elif end_continue_response['output'] == 'bye':
#             response = "Goodbye! Chat with you later!"
#         else:
#             response = "I'm not sure how to respond to that."

#         update_chat_history(user_input, response)
        
def on_send():
    user_input = st.session_state.user_input.strip()
    if user_input:
        
        end_continue_response = classify_end_continue(user_input)
        if end_continue_response['output'] == 'continue chat':
            wiki_chat_response = classify_wiki_chat(user_input)
            if wiki_chat_response['output'] == 'wiki':
                #topic = classify_topic(user_input)
                #response = f"Fetching information on this {topic} related topic."
                preprocessed_query = preprocess_query(user_input)
                results = search_solr(preprocessed_query, 'http://35.245.97.133:8983/solr/IRF23P1', 'topic,title,revision_id,summary')
                if 'response' in results and 'docs' in results['response'] and len(results['response']['docs']) > 0:
                    latest_doc = results['response']['docs'][0]
                    response = latest_doc.get('summary', 'Summary not available.')
                else:
                    response = "I couldn't find any information on that topic."

            elif wiki_chat_response['output'] == 'chat':
                response = chatterbot_response(user_input)
        elif end_continue_response['output'] == 'bye':
            response = "Goodbye! Chat with you later!"
        else:
            response = "I'm not sure how to respond to that."

        update_chat_history(user_input, response)



def update_chat_history(user_input, response):
    new_chat_history = st.session_state.chat_history + [('USER', user_input), ('WW', response)]
    st.session_state.chat_history = new_chat_history

def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [('WW', 'Hello! How may I help you today?')]
    if 'last_selected_topic' not in st.session_state:
        st.session_state.last_selected_topic = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''

    st.title('Wisdom-Whisperer')

    # Custom styles for chat bubbles, sidebar, and title
    st.markdown("""
        <style>
            /* General styles */
            h1 { color: black; }

            /* Styles for chat bubbles */
            .message { border-radius: 25px; padding: 10px; margin: 10px 0; border: 1px solid #e6e9ef; position: relative; }
            .user { background-color: #dbf0d4; color: black; }
            .WW { background-color: #f1f0f0; color: black; }

            /* Styles for sidebar */
            .sidebar .sidebar-content { background-color: #f0f0f0; color: black; font-weight: bold; padding-top: 10px; font-size: 25px; }
            .sidebar-heading { color: black; font-weight: bold; font-size: 25px; }
            .sidebar-content a { color: black; font-weight: bold; font-size: 25px; }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        topics_with_placeholder = ["Select a topic..."] + topics
        selected_topic = st.radio("Choose a topic:", topics_with_placeholder, index=0)
        if selected_topic != "Select a topic..." and selected_topic != st.session_state['last_selected_topic']:
            st.session_state['chat_history'].append(('WW', f"Here's the summary on the topic: {selected_topic}."))
            st.session_state['last_selected_topic'] = selected_topic

    # Display chat history
    for role, message in st.session_state['chat_history']:
        bubble_class = "user" if role == "USER" else "WW"
        st.markdown(f"<div class='message {bubble_class}'>{message}</div>", unsafe_allow_html=True)

    # Text input for user query
    st.text_input("Send a message...", value=st.session_state.user_input, on_change=on_send, key="user_input")

    # Button to send the message
    st.button('➤', on_click=on_send)

if __name__ == '__main__':
    main()
