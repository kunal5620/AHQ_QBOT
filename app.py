from flask import Flask, request, jsonify, render_template, redirect, url_for
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from datetime import datetime
import os
import speech_recognition as sr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import re
import random
import torch
from torch.quantization import quantize_dynamic
import re

access_token = "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"
model_name="meta-llama/Llama-3.2-3B-Instruct"


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
 
model= model.to(torch.bfloat16)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=700, device="cuda", temperature=0.7, do_sample=True)

# Define global variables 
global_context = None 
global_generated_answer = None

app = Flask(__name__)


# Initialize Pinecone
api_key = "d32b39c0-2fdb-4652-b6a3-0e1084a88325"
pc = Pinecone(api_key=api_key)
index_name = "anand-chatbot"
index = pc.Index(index_name)

# Load SentenceTransformer model
model_encode = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code="True", device="cpu") 

# Function to query the Pinecone index
def query_pinecone(query):
    query_embedding = model_encode.encode(query)
    results = index.query(vector=query_embedding.tolist(), top_k=8, include_metadata=True)
    return results

# Function to store data in an Excel file
def store_to_excel(username, query, unique_file_names):
    date_time = datetime.now()
    date_str = date_time.strftime('%Y-%m-%d')
    time_str = date_time.strftime('%H:%M:%S')
    
    # Define the file path
    file_path = 'query_log.xlsx'

    # Check if the file exists
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=['Username', 'Date', 'Time', 'Query', 'Files Explored'])
    
    # Append new data
    new_row = {
        'Username': username,
        'Date': date_str,
        'Time': time_str,
        'Query': query,
        'Files Explored': ', '.join(unique_file_names)
    }
    df = df._append(new_row, ignore_index=True)
    
    # Save the DataFrame to an Excel file
    df.to_excel(file_path, index=False)

@app.route('/query', methods=['POST'])
def handle_query():
    global global_context, global_generated_answer
    username='user1'
    user_query = request.json['query']
    #username = request.json['username']

    greet_query =  user_query.strip().lower()
    greet_query = re.sub(r'[^\w\s]', '', greet_query)

    #Predefined greeting reponses
    greeting_reponses = {
        "hi": [ 
            "Hello! I am AHQ Chatbot, you ask the question regarding the AHQ. How can I help you today?",
            "Hi there, What can I do for you?",
            "Hey, I am AHQ Chatbot. Need any assistance regarding AHQ?"
        ],
         "hello": [ 
            "Hello! I am AHQ Chatbot, you ask the question regarding the AHQ. How can I help you today?",
            "Hi there, I am AHQ Chatbot. What can I do for you?",
            "Hey, I am AHQ Chatbot. Need any assistance regarding AHQ?"
        ],
         "hey": [ 
            "Hello! I am AHQ Chatbot, you ask the question regarding the AHQ. How can I help you today?",
            "Hi there, I am AHQ Chatbot. What can I do for you?",
            "Hey, I am AHQ Chatbot. Need any assistance regarding AHQ?"
        ],
        "how are you": [ 
            "I am fine! I am AHQ Chatbot, you ask the question regarding the AHQ. How can I help you today?",
            "Hello, I am fine, What can I do for you?",
            "Fine, Do you Need any assistance regarding AHQ?"
        ],
        "who are you": [ 
            "I am AHQ Chatbot, you ask the question regarding the AHQ. How can I help you today?",
            "Hello, I am AHQ Chatbot, you ask the question regarding the AHQ."
        ],
        "what is your name": [ 
            "Hi, my name is AHQ Chatbot, you ask the question regarding the AHQ.",
            "Hello, I am AHQ Chatbot, how can I help you today."
        ]
    } 

    #matching of query
    for greeting, respones in greeting_reponses.items():
        if greet_query.startswith(greeting):
            return jsonify({"answer":random.choices(respones)})

    #If greet_query does not match    
    results = query_pinecone(user_query)

    matches = []
    unique_file_names = set()  # To store unique file names
    for match in results['matches']:
        file_name = match['metadata']['file_name']
        unique_file_names.add(file_name)
        
        match_info = {
            "chunk_text": match['metadata']['chunk_text']
        }
        matches.append(match_info)
        
    # Store the query details in an Excel file
    store_to_excel(username, user_query, unique_file_names)

    context = matches
    
    # Create the prompt using the retrieved context
    prompt = f"""You are an AHQ chatbot. Generate the detailed answer for the question from the given context. Try to answer pointwise.
    Context: {context}
    Question: {user_query}
    Answer:"""    

    # Generate the response using the language model and print the answer
    answer = pipe(prompt)
    generated_answer  = (answer[0]['generated_text'].split('Answer:')[-1].strip())
    global_generated_answer = generated_answer
    return jsonify({"answer": generated_answer})
    '''
    # Extract numbered points from the generated answer
    numbered_points = {}
    current_point_number = None
    current_point_text = ""
    for line in generated_answer.split('\n'):
        match = re.match(r'(\d+)\. (.*)', line)
        if match:
            if current_point_number is not None:
                numbered_points[current_point_number] = current_point_text.strip()
            current_point_number = int(match.group(1))
            current_point_text = match.group(2)
        else:
            if current_point_number is not None:
                current_point_text += " " + line.strip()
    
    if current_point_number is not None:
        numbered_points[current_point_number] = current_point_text.strip()
    
    # Format the answer with numbered points
    formatted_answer = ""
    for point_number, point_text in numbered_points.items():
        formatted_answer += f"{point_number}. {point_text}\n\n"
    
    return jsonify({"answer": formatted_answer})'''

@app.route('/speak', methods=['POST']) 
def speak(): 
    global global_generated_answer 
    text = global_generated_answer if global_generated_answer else "No answer generated." 
    
    # Initialize the pyttsx3 engine 
    engine = pyttsx3.init() 
    
    # Set properties before adding things to speak 
    engine.setProperty('rate', 155) # Speed rate for speak out 
    engine.setProperty('volume', 0.9) # Volume 0-1 
    
    # Convert the text to speech 
    engine.say(text) 
    
    # Wait for the speech to finish 
    engine.runAndWait() 
    
    return jsonify({'status': 'success', 'spoken_text': text})

# Load user credentials from CSV
credentials_df = pd.read_csv('C:/Users/admin/Desktop/QBOT/LLM Model Build/credentials.csv')

# User validation and login
def validate_user(username, password):
    user = credentials_df[(credentials_df['username'] == username) & (credentials_df['password'] == password)]
    return not user.empty

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if validate_user(username, password):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})

@app.route('/chat')
def chat():
    return render_template('index_trial.html')  # Your chatbot UI

@app.route('/')
def index1():
    return render_template('login.html')  # Login page

from flask import Flask, send_from_directory, render_template_string
import os


@app.route('/files/')
@app.route('/files/<path:subpath>')
def serve_directory(subpath=''):
    directory = 'C:/QBOT/LLM Model Build/LLM From Desktop/'
    full_path = os.path.join(directory, subpath)

    if os.path.isdir(full_path):
        # Return a list of files and directories in the folder
        files = os.listdir(full_path)
        files_links = [f'<a href="{subpath}/{file}">{file}</a>' for file in files]
        return render_template_string('<br>'.join(files_links))
    else:
        # Return the specific file
        return send_from_directory(directory, subpath)
    

@app.route('/files/<path:filename>')
def serve_file(filename):
    # Replace 'E:/Anand_Project/Trial/knowledge_base' with the path to your folder
    directory = 'C:/QBOT/LLM Model Build/LLM From Desktop/'
    return send_from_directory(directory, filename) 


#Speech to text convertor
from flask import Flask, jsonify
import speech_recognition as sr

@app.route('/convert_speech', methods=['POST'])
def convert_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source1:
        print("Silence Please")
        recognizer.adjust_for_ambient_noise(source1, duration=2)
        print("Speak Now Please")
        audio2 = recognizer.listen(source1, timeout=5)

    try:
        text = recognizer.recognize_google(audio2, language='en-IN')
        text = text.lower()
        print("You said: " + text)
        return jsonify({'transcription': text})
    except sr.UnknownValueError:
        return jsonify({'transcription': ''}), 400
    except sr.RequestError as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)
























'''You are a helpful AI assistant. Understand the question and answer it from the given context in informative manner. If the question cannot be answered using the information provided, answer with "I don't know'''
#torch_dtype=torch.float16,