import streamlit as st
from langchain_core.messages import HumanMessage
from scripts.rag import load_document, vector_store, text_splitter, agent, config, nettoyer_index

st.title("Outil d'aide à la recherche documentaire pour des articles scientifiques")


# Upload du fichier
uploaded_file = st.file_uploader("Ajoutez un fichier", type=["pdf", "txt"], accept_multiple_files=True, key="file_uploader")


if uploaded_file is not None and len(uploaded_file) > 0:
    print(f"Debug: Uploaded files: {[file.name for file in uploaded_file]}")


    # Chargement du document
    documents = load_document(uploaded_file)

    print(f"Debug: Loaded documents: {documents}")

    # Division du document en chunks
    all_splits = text_splitter.split_documents(documents)

    # Ajout des chunks au vector store
    vector_store.add_documents(all_splits)

    print(f"Debug: Vector store: {vector_store}")

    st.success("Document chargé et traité avec succès !")

    # Champ de saisie pour la question
    question = st.text_input("Posez votre question :")

    repondre = st.button("Répondre à la question")
    # Ajouter un bouton pour réinitialiser l'index
    reset_index = st.button("Réinitialiser l'index", type="primary")

    if question and repondre:
        with st.spinner("Génération de la réponse..."):
            # Invoquer l'agent avec la question
            response = agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config
            )

            # Affichage de la réponse
            st.write("Réponse :")
            # Extraire le dernier message de la réponse
            last_message = response["messages"][-1].content
            st.write(last_message)

    if reset_index:
        with st.spinner("Réinitialisation de l'index..."):
            result = nettoyer_index()
            st.write(result)
            print(f"Debug: Index clearing result: {result}")
else:
    st.warning("Téléchargez un document")