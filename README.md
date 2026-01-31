# RAG_Mistral pour la fouille documentaire scientifique
Projet étudiant de RAG qui permet de d'interroger une base documentaire fournie à travers une interface de chat. Dans notre exemple, il s'agit d'articles scientifiques. Les réponses générées prennent en compte ce contexte.

### Stack technique
Ce projet utilise une stack technique composée de Mistral 7B pour le modèle de langage, Mistral Embeddings pour les embeddings, QDrant pour le stockage et la recherche vectorielle, et Huggingface pour l'hébergement des modèles. L'interface utilisateur est développée avec Streamlit, offrant une expérience interactive pour interroger la base documentaire.

**Framework et orchestration** : LangChain et LangGraph pour orchestrer le flux de données, gérer la mémoire, et construire des pipelines RAG robustes.

### Avertissement
Vous devez créer un compte sur HuggingFace et générer un token en lecture : `HF_TOKEN`
Vous devez créer un compte sur Mistral et générer un API token :
`MISTRAL_TOKEN`

** Token gratuits mais soumis à un quota d'utilisation **

## Structure du projet

Le projet est structuré de la manière suivante :

- `app.py` : Interface utilisateur Streamlit qui gère l'interaction avec l'utilisateur (upload de fichiers, affichage des résultats).
- `scripts/rag.py` : Script principal contenant la logique du pipeline RAG, l'orchestration LangGraph, et la gestion du vecteur store.
- `.github/workflows/rag.yml` : Fichier de configuration pour les tests automatisés.
- `requirements.txt` : Liste des dépendances nécessaires pour le projet.

## Architecture et flux de données

### Comment les programmes travaillent ensemble

Le projet suit une architecture **client-serveur-logique** où chaque composant a un rôle distinct :

```
┌─────────────────┐         ┌─────────────────┐         ┌──────────────┐
│   app.py        │────────▶│   scripts/      │────────▶│  LLM & VectorStore
│ (Streamlit UI)  │         │   rag.py        │         │  (Mistral, Qdrant)
│                 │◀────────│ (Business Logic)│◀────────│
└─────────────────┘         └─────────────────┘         └──────────────┘
```

#### **app.py - Interface Utilisateur (Streamlit)**
- **Responsabilités** :
  - Affichage d'une interface web intuitive pour l'utilisateur
  - Gestion de l'upload de fichiers PDF/TXT
  - Affichage des réponses générées
  - Gestion des boutons d'action (Répondre, Réinitialiser l'index)

- **Flux** :
  1. L'utilisateur charge des documents via le `file_uploader`
  2. Les fichiers sont traités via `load_document()` de `rag.py`
  3. Les documents sont fragmentés et indexés dans le vector store
  4. L'utilisateur pose une question
  5. La question est envoyée à `agent.invoke()` qui retourne une réponse
  6. La réponse est affichée dans l'interface

#### **scripts/rag.py - Logique métier et Orchestration**
- **Responsabilités** :
  - Initialisation des modèles (LLM, embeddings)
  - Gestion du vector store Qdrant
  - Construction du pipeline RAG avec LangGraph
  - Gestion de la mémoire des conversations
  - Traitement des documents

- **Composants clés** :
  - `load_document()` : Charge et valide les fichiers PDF/TXT
  - `rag_node()` : Nœud RAG qui récupère le contexte et génère une réponse
  - `nettoyer_index()` : Réinitialise le vector store

### Spécificités de l'architecture

#### **1. LangGraph - Orchestration stateful**

LangGraph est utilisé pour construire un **graph d'états** qui gère le flux de la conversation :

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

- **StateGraph** : Définit le graphe d'états avec un nœud `rag` unique
- **add_messages** : Agrège automatiquement les messages dans l'état
- **Avantages** :
  - Gestion élégante des états complexes
  - Traçabilité du flux d'exécution
  - Extensibilité pour ajouter des nœuds supplémentaires

#### **2. Mémoire à court terme (Short-term Memory)**

La mémoire des conversations est gérée via le **checkpoint** LangGraph :

```python
checkpointer = InMemorySaver()
agent = graph_builder.compile(checkpointer=checkpointer)
```

- **InMemorySaver** : Stocke l'historique des messages en mémoire
- **Thread ID** : Identifie chaque conversation unique
- **Avantages** :
  - Les réponses considèrent le contexte des questions précédentes
  - Permet les conversations multi-turns
  - L'historique est conservé dans la session Streamlit

#### **3. Mémoire vectorielle avec Qdrant**

Le **vector store** Qdrant stocke les embeddings des documents :

- **Collection Qdrant** : `rag_mistral` avec distance COSINE
- **Embeddings** : Générés via `MistralAIEmbeddings` (modèle `mistral-embed`)
- **Recherche** : `similarity_search(query, k=2)` récupère les 2 documents les plus pertinents
- **Avantages** :
  - Recherche sémantique efficace
  - En mémoire (pas de base de données persistante)
  - Récupération rapide du contexte pertinent

#### **4. Pipeline RAG complet**

Le pipeline RAG (Retrieval-Augmented Generation) suit ce flux :

```
Question de l'utilisateur
    ↓
Récupération des documents pertinents (similarity_search)
    ↓
Construction du prompt avec contexte (system prompt + question)
    ↓
Invocation du LLM Mistral 7B
    ↓
Génération de la réponse avec citations
    ↓
Affichage à l'utilisateur
```

#### **5. LangChain - Intégration des composants**

LangChain fournit les abstractions pour lier tous les composants :

- **ChatHuggingFace** : Wrapper pour le modèle Mistral via HuggingFace
- **MistralAIEmbeddings** : Génération des embeddings
- **QdrantVectorStore** : Intégration avec Qdrant
- **BaseMessage** : Type de message standardisé (SystemMessage, HumanMessage, AIMessage)
- **RunnableConfig** : Configuration de l'exécution avec état

### Flux d'une question utilisateur

1. **Input** : `"Quelle est la principale contribution du papier ?"`
2. **State Creation** : `{"messages": [HumanMessage(content="...")]}`
3. **RAG Node Execution** :
   - Récupère les documents pertinents
   - Construit le system prompt avec contexte
   - Invoque Mistral 7B
4. **Output** : `{"messages": [..., AIMessage(content="La principale contribution est...")]}`
5. **Checkpoint** : L'état est sauvegardé pour les prochaines questions
6. **Display** : Le message est affiché dans Streamlit

## Prérequis

Avant de pouvoir exécuter le projet, vous devez installer les dépendances listées dans `requirements.txt`. Vous pouvez le faire en exécutant la commande suivante :

```bash
pip install -r requirements.txt
```

## Configuration

### Exécution locale

Pour exécuter le projet localement, vous devez configurer les variables d'environnement suivantes :

- `HF_TOKEN` : Token d'accès à Huggingface.
- `MISTRAL_API_KEY` : Token d'accès à Mistral AI.

Ces variables peuvent être configurées dans le fichier `.env` ou directement dans votre environnement.

Exécutez l'application Streamlit avec la commande suivante :

```bash
streamlit run app.py
```

Cette commande lancera l'application Streamlit sur `http://localhost:8501` (ou le prochain port disponible si celui-ci est utilisé).

### Exécution sur GitHub

Pour exécuter le projet sur GitHub, vous devez configurer les secrets suivants dans les paramètres de votre dépôt sous actions :

- `HF_TOKEN` : Token d'accès à Huggingface.
- `MISTRAL_API_KEY` : Token d'accès à Mistral AI.

Ces secrets sont utilisés dans le workflow GitHub Actions défini dans `.github/workflows/rag.yml`.

Le projet est configuré pour exécuter des tests automatisés à chaque commit. Les tests sont définis dans le fichier `.github/workflows/rag.yml`. Pour exécuter le workflow, poussez vos changements sur votre dépôt GitHub.

## Déploiement

Pour déployer votre application, vous pouvez utiliser des services comme Streamlit Sharing, Heroku, ou Render. Voici un exemple pour Streamlit Sharing :

1. **Créer un compte sur Streamlit Sharing** :
   - Allez sur [Streamlit Sharing](https://share.streamlit.io/) et connectez-vous avec votre compte GitHub.

2. **Déployer l'application** :
   - Cliquez sur "New App" et sélectionnez votre dépôt GitHub.
   - Configurez les variables d'environnement `HF_TOKEN` et `MISTRAL_TOKEN` dans les paramètres de l'application.
   - Déployez l'application.

Votre application sera alors accessible via un lien fourni par Streamlit Sharing.

## Contribution

Les contributions sont les bienvenues. Pour contribuer, veuillez créer une branche à partir de la branche principale et soumettre une pull request.

## Licence

Ce projet est sous licence MIT.