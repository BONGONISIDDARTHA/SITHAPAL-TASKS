pip install pypdf2 sentence-transformers faiss-cpu openai transformers PyMuPDF
import fitz  # PyMuPDF for PDF extraction
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize OpenAI API key (ensure it's set up correctly)
openai.api_key = 'your-openai-api-key'

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF using PyMuPDF
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to chunk the text into smaller pieces
def chunk_text(text, chunk_size=500):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to embed text chunks using SentenceTransformer
def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

# Function to initialize and create a FAISS index
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]  # Embedding dimensionality
    index = faiss.IndexFlatL2(dim)  # Use L2 distance (Euclidean)
    index.add(embeddings)  # Add embeddings to the index
    return index

# Function to perform similarity search in FAISS index
def search_faiss_index(query, index, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return indices, distances

# Function to get the most relevant chunks based on a query
def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    indices, _ = search_faiss_index(query, index, top_k)
    return [chunks[i] for i in indices[0]]

# Function to generate response using GPT-3 or similar LLM
def generate_response(query, context):
    prompt = f"Question: {query}\nAnswer: {context}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Process PDF and set up pipeline
def process_pdf(pdf_path):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    return index, chunks

# Query Handling
def handle_query(query, pdf_path, index, chunks):
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    context = " ".join(relevant_chunks)
    response = generate_response(query, context)
    return response

# Example usage
if _name_ == "_main_":
    pdf_path = "path_to_your_pdf.pdf"
    
    # Step 1: Process PDF (extract text, chunk, embed, and create index)
    index, chunks = process_pdf(pdf_path)
    
    # Step 2: Handle a user query
    user_query = "What is the unemployment rate for people with a Bachelor's degree?"
    response = handle_query(user_query, pdf_path, index, chunks)
    print("Response:", response)
[15/12, 9:09 pm] Siddu🕺: def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text
[15/12, 9:09 pm] Siddu🕺: def chunk_text(text, chunk_size=500):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
[15/12, 9:10 pm] Siddu🕺: def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings
[15/12, 9:10 pm] Siddu🕺: def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
[15/12, 9:10 pm] Siddu🕺: def generate_response(query, context):
    prompt = f"Question: {query}\nAnswer: {context}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()