# RAG_Mistral pour la fouille documentaire scientifique
Projet étudiant de RAG qui permet de d'interroger une base documentaire fournie à travers une interface de chat. Dans notre exemple, il s'agit d'articles scientifiques. Les réponses générées prennent en compte ce contexte.

### Stack technique
Ce projet utilise une stack technique composée de Mistral 7B pour le modèle de langage, econre Mistral pour l'embeddings, QDrant pour le stockage et la recherche vectorielle, et Huggingface pour l'hébergement des modèles. L'interface utilisateur est développée avec Streamlit, offrant une expérience interactive pour interroger la base documentaire.

### Avertissement
Vous devez créer un compte sur HuggingFace et générer un token en lecture : `HF_TOKEN`
Vous devez créer un compte sur Mistral et générer un API token :
`MISTRAL_TOKEN`

** Token gratuits mais soumis à un quota d'utilisation **

## Structure du projet

Le projet est structuré de la manière suivante :

- `scripts/rag.py` : Script principal contenant la logique du pipeline RAG.
- `.github/workflows/rag.yml` : Fichier de configuration pour les tests automatisés.
- `requirements.txt` : Liste des dépendances nécessaires pour le projet.

## Prérequis

Avant de pouvoir exécuter le projet, vous devez installer les dépendances listées dans `requirements.txt`. Vous pouvez le faire en exécutant la commande suivante :

```bash
pip install -r requirements.txt
```

## Configuration

### Exécution locale


Pour exécuter le projet localement, vous devez configurer les variables d'environnement suivantes :

- `HF_TOKEN` : Token d'accès à Huggingface.
- `MISTRAL_TOKEN` : Token d'accès à Mistral AI.

Ces variables peuvent être configurées dans le fichier `.env` ou directement dans votre environnement.



Pour exécuter le projet localement, vous devez d'abord installer Streamlit si ce n'est pas déjà fait :

```bash
pip install streamlit
```

Ensuite, exécutez le script `scripts/rag.py` avec la commande suivante :

```bash
streamlit run scripts/rag.py
```

Cette commande lancera l'application Streamlit et ouvrira une interface web dans votre navigateur par défaut.

### Exécution sur GitHub

Pour exécuter le projet sur GitHub, vous devez configurer les secrets suivants dans les paramètres de votre dépôt sous actions :

- `HF_TOKEN` : Token d'accès à Huggingface.
- `MISTRAL_TOKEN` : Token d'accès à Mistral AI.

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