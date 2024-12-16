pip install requests beautifulsoup4 transformers faiss-cpu langchain
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch

# Step 1: Data Ingestion

def scrape_website(url):
    """Scrape content from a website."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract all text from paragraphs and headings
    text = ''
    for para in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):
        text += para.get_text(separator=' ') + '\n'
    
    return text

def chunk_text(text, chunk_size=500):
    """Split text into chunks of specified size."""
    words = text.split()
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [' '.join(chunk) for chunk in chunks]

def generate_embeddings(texts, model, tokenizer):
    """Generate embeddings for text chunks."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def store_embeddings(embeddings, database):
    """Store embeddings in a FAISS vector database."""
    if database is None:
        # Initialize FAISS index
        dim = embeddings.shape[1]
        database = faiss.IndexFlatL2(dim)  # L2 distance
    database.add(embeddings)
    return database

# Step 2: Query Handling

def query_to_embedding(query, model, tokenizer):
    """Convert a query to an embedding."""
    inputs = tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return query_embedding.numpy()

def search_in_database(query_embedding, database, k=5):
    """Search the FAISS database for similar embeddings."""
    distances, indices = database.search(query_embedding, k)
    return indices, distances

# Step 3: Response Generation

def generate_response(query, database, model, tokenizer, k=5):
    """Generate a response to the user's query based on database retrieval."""
    query_embedding = query_to_embedding(query, model, tokenizer)
    
    # Perform similarity search in the FAISS index
    indices, distances = search_in_database(query_embedding, database, k)
    
    # Retrieve the corresponding text chunks from the database
    retrieved_texts = [database.reconstruct(i) for i in indices[0]]
    
    # Use an LLM to generate a detailed response
    context = ' '.join(retrieved_texts)
    full_prompt = f"Answer the following question using the 
import textwrap

# Function to chunk the text into smaller segments
def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    # Use textwrap to split the text into chunks that fit the model's context window
    return textwrap.wrap(text, chunk_size)

# Example usage
text = "Your scraped content goes here..."
chunks = chunk_text(text)
from transformers import AutoTokenizer, AutoModel
import torch

# Load a pre-trained embedding model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-nli-stsb-mean-tokens')
model = AutoModel.from_pretrained('distilbert-base-nli-stsb-mean-tokens')

# Function to encode text chunks into embeddings
def encode_text(chunks: List[str]) -> List[torch.Tensor]:
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings.append(model(**inputs).last_hidden_state.mean(dim=1))
    return embeddings
import faiss
import numpy as np

# Function to create and store embeddings in a FAISS index
def create_faiss_index(embeddings: List[torch.Tensor], metadata: List[str]) -> faiss.IndexFlatL2:
    # Convert embeddings to numpy arrays for FAISS
    embeddings_np = np.vstack([e.numpy() for e in embeddings]).astype('float32')
    
    # Create a FAISS index for similarity search
    index = faiss.IndexFlatL2(embeddings_np.shape[1])  # L2 distance metric
    index.add(embeddings_np)
    
    return index, embeddings_np, metadata

# Example usage
index, embeddings_np, metadata = create_faiss_index(embeddings, ['Metadata about the chunk'])
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Initialize the FAISS vector store and LLM
vector_store = FAISS.from_embeddings(HuggingFaceEmbeddings(model_name="distilbert-base-nli-stsb-mean-tokens"), embeddings_np)
llm = OpenAI(temperature=0, api_key='your-openai-api-key')

# Function to handle user queries
def handle_query(query: str, vector_store: FAISS, index: faiss.IndexFlatL2, metadata: List[str], top_k: int = 3) -> str:
    # Convert query to embedding
    query_embedding = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        query_embedding = model(**query_embedding).last_hidden_state.mean(dim=1).numpy().astype('float32')

    # Perform a similarity search in the FAISS index
    D, I = index.search(query_embedding, top_k)
    
    # Retrieve top-k relevant chunks and their metadata
    relevant_chunks = [metadata[i] for i in I[0]]
    
    # Generate a response using the LLM and the retrieved chunks
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the following question based on the context:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    response = llm(prompt)
    
    return response

# Example usage
query = "What is the latest research at Stanford University?"
response = handle_query(query, vector_store, index, metadata)
print(response)
urls = [
    "https://www.uchicago.edu/",
    "https://www.washington.edu/",
    "https://www.stanford.edu/",
    "https://und.edu/"
]

scraped_data = crawl_websites(urls)