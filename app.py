import streamlit as st
import os

from scripts.rag import load_document, vector_store, text_splitter, agent, config

st.title("Outil d'aide à la recherche documentaire pour des articles scientifiques")

# Upload de fichier
uploaded_file = st.file_uploader("Télécharger un fichier PDF ou TXT", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_file is not None:
    

    # Chargement du document
    documents = load_document(uploaded_file)

    # Division du document en chunks
    all_splits = text_splitter.split_documents(documents)

    # Ajout des chunks au vector store
    vector_store.add_documents(all_splits)

    st.success("Document chargé et traité avec succès !")

    # Champ de saisie pour la question
    question = st.text_input("Posez votre question :")

    if question:
        # Recherche des documents pertinents
        retriever = vector_store.as_retriever()
        docs = retriever.get_relevant_documents(question)

        # Génération de la réponse
        response = agent.invoke({"input": question, "config": config})

        # Affichage de la réponse
        st.write("Réponse :")
        st.write(response.content)