import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import MistralAIEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# Récupère le token depuis l'environnement (définie par le workflow)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable is not set")
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY


# Variables globales
collection_name = "rag_mistral"

# Initialisation du modèle Mistral via HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_length=1024,
)
# Création du modèle de chat avec LangChain
model = ChatHuggingFace(llm=llm)

# Initialise les embeddings
embeddings = MistralAIEmbeddings(model="mistral-embed")

# Initialise le client Qdrant en mémoire
client = QdrantClient(":memory:")

# Taille des vecteurs d'embedding Mistral (par défaut 1024)
vector_size = 1024
# Crée une collection si elle n'existe pas déjà
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

# Extraire les références des articles PDF

def extraire_urls_pdf(file, type=["pdf", "txt"]) -> List[str]:

    """Extrait toutes les URLs d'un PDF, y compris dans les références."""
    
    doc = fitz.open(stream=file.read(), filetype=type)
    urls = set()  # Utiliser un set pour éviter les doublons

    for page in doc:
    # Extraire les liens de la page (y compris les références)
        for link in page.get_links():
            if "uri" in link:  # Vérifie si le lien est une URL
                urls.add(link["uri"])

    doc.close()
    return list(urls)

# Charger documents PDF, texte ou pages web

def load_document(uploaded_file) -> List[Document]:
    """Load a document based on its file extension"""
    
    for file in uploaded_file:
        file_ext = file.split(".")[1]
        try:
            if file.endswith(".pdf"):
                loader_pdf = PyPDFLoader(file)
                documents = loader_pdf.load()
                for url in extraire_urls_pdf(file):
                    try:
                        web_loader = WebBaseLoader(url)
                        web_ref = web_loader.load()
                        documents.extend(web_ref)
                    except Exception as e:
                        print(f"Error loading URL {url}: {e}")

            elif file.endswith(".txt"):
                loader = TextLoader(file, encoding="utf-8")
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            return documents
        except Exception as e:
            raise Exception(f"Error loading document {file}: {str(e)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
    separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for natural breaks
)

# Rag chain : chaine à 2 étapes avec injection de contexte
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = ([("System",
        """You are a helpful assistant for scientific articles. Use the following context from the multiple documents in your response.
        Important:
        1. Cite the sources you used in your answer as a list of references with the name of the documents or the links at the end of your response.
        2. If the context does not contain the answer, say so explicitly.
        3. Synthesize information from ALL relevant documents in the context.
        4. Provide concise and clear answers suitable for academic research."""),
        ("Context below:"

        f"\n\n{docs_content}"
    )])

    return system_message

# Ajout d'une mémoire et d'un middleware de résumé pour gérer la longueur des conversations
checkpointer = InMemorySaver()

agent = create_agent(model, tools=[], middleware=[prompt_with_context, SummarizationMiddleware(
            model=model,
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )], checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}