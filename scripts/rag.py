import os
import tempfile
from typing import List, Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

# Load environment variables from .env file for local execution
load_dotenv()

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


def load_document(uploaded_file) -> List[Document]:
    """Load a document based on its file extension"""
    print(f"Debug: Entering load_document with uploaded_file: {uploaded_file}")  # Debug print
    if not uploaded_file:
        print("Debug: No files to process")  # Debug print
        return []

    documents = []  # Initialize the documents list

    for file in uploaded_file:
        print(f"Debug: Processing file: {file.name}")  # Debug print
        file_ext = file.name.split(".")[-1]  # file.name pour avoir l'extension

        try:
            if file_ext == "pdf":
                print(f"Debug: Processing PDF file: {file.name}")  # Debug print
                try:
                    # Crer un fichier temporaire pour le PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Charger le document PDF
                    loader_pdf = PyPDFLoader(tmp_file_path)
                    file_documents = loader_pdf.load()

                    # Add the documents to the main list
                    documents.extend(file_documents)

                except Exception as e:
                    print(f"Error processing PDF file {file.name}: {str(e)}")
                    raise
                finally:
                    # Nettoyer le fichier temporaire s'il existe
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            elif file_ext == "txt":
                print(f"Debug: Processing TXT file: {file.name}")  # Debug print
                try:
                    loader = TextLoader(file, encoding="utf-8")
                    file_documents = loader.load()
                    documents.extend(file_documents)
                except Exception as e:
                    raise Exception(f"Error processing TXT file {file.name}: {str(e)}")
    

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

        except Exception as e:
            raise Exception(f"Error loading document {file.name}: {str(e)}")

    print(f"Debug: Returning documents: {len(documents)} documents")  # Debug print
    return documents


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
    separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for natural breaks
)



class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Definir le node rag RAG
def rag_node(state: State) -> State:
    """RAG node that answers questions using retrieved documents."""
    last_message = state["messages"][-1].content
    
    # Extraire les documents
    retrieved_docs = vector_store.similarity_search(last_message, k=2)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Créer le prompt avec le contexte des documents
    system_prompt = f"""You are a helpful assistant for scientific articles. Use the following context from the multiple documents in your response.
Important:
1. Cite the sources you used in your answer as a list of references with the name of the documents or the links at the end of your response.
2. Add the link related to each reference if available.
3. If you use information from multiple documents, make sure to cite all of them.
4. If the context does not contain the answer, say so explicitly.
5. Synthesize information from ALL relevant documents in the context.
6. Provide concise and clear answers suitable for academic research.
7. Format the references clearly as follows:
   Références :
   - Ref1: [Description or title of the reference only if available] (URL only if available)
   - Ref2: [Description or title of the reference only if available] (URL only if available)
   ...

Context from documents:
{docs_content}"""
    
    # Generer la réponse avec le modèle
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message)
    ])
    
    return {"messages": [response]}

# Creer le graph LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("rag", rag_node)
graph_builder.add_edge(START, "rag")

# Ajouter une memoire pour le graph 
checkpointer = InMemorySaver()
agent = graph_builder.compile(checkpointer=checkpointer)

# Configuration pour l'agent
config = RunnableConfig(configurable={"thread_id": "1"})

# Fonction pour nettoyer l'index
def nettoyer_index() -> str:
    """Nettoyer le vectorstore Qdrant en mémoire."""
    
    try:
        # Delete collection and recreate
        client = QdrantClient(location=":memory:")
        try:
            client.delete_collection(collection_name)
        except:
            pass
        # Taille des vecteurs d'embedding Mistral (par défaut 1024)
        vector_size = 1024
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
           embedding=embeddings,
        )
        
        try:
            retrieved = vector_store.similarity_search("test", k=2)
        
        except Exception as e:
            #retrieved = vector_store.similarity_search()
            print(f"Erreur lors du test du vector store: {str(e)}")
            raise
    
        
        return "✅ Index nettoyé avec succès!"
    except Exception as e:
        return f"❌ Erreur lors du nettoyage de l'index: {str(e)}"
    