#Code is working Properly 

'''
import os
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
import pinecone
from pptx import Presentation
from PIL import Image
import pytesseract
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
api_key = "d32b39c0-2fdb-4652-b6a3-0e1084a88325"
pc = Pinecone(api_key=api_key)
index_name = "build-bot"

index = pc.Index(index_name)

# Load SentenceTransformer model
model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

# Function to query the Pinecone index
def query_pinecone(query):
    query_embedding = model.encode(query)
    results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
    return results

# Example user query
user_query = "Information on Indirect Tax?"
results = query_pinecone(user_query)

# Print the results
for match in results['matches']:
    print(f"Match Found: {match['metadata']['chunk_text']}")
    print(f"Best Fit Match: {match['metadata']['file_name']}")
    print(f"Matching Score: {match['score'] * 100:.2f}%")
    print(f"Vector Index: {match['metadata']['chunk_index']}")
    print()
'''

#Seeting up with the Frontend

from flask import Flask, request, jsonify
import os
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
import pinecone
from pptx import Presentation
from PIL import Image
import pytesseract
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

# Initialize Pinecone
api_key = "d32b39c0-2fdb-4652-b6a3-0e1084a88325"
pc = Pinecone(api_key=api_key)
index_name = "build-bot"

index = pc.Index(index_name)

# Load SentenceTransformer model
model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

# Function to query the Pinecone index
def query_pinecone(query):
    query_embedding = model.encode(query)
    results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
    return results

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data['query']
    results = query_pinecone(user_query)
    
    response_data = []
    for match in results['matches']:
        response_data.append({
            'chunk_text': match['metadata']['chunk_text'],
            'file_name': match['metadata']['file_name'],
            'score': f"{match['score'] * 100:.2f}%"
        })
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

