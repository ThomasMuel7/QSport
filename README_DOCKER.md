# QSport — Lancer avec Docker

## Prérequis
- Docker Desktop installé et lancé
- Fichier `.env` à la racine (copier `.env.example` et remplir les valeurs)

## Démarrage

### 1. Variables d'environnement
cp .env.example .env
# Remplir SUPABASE_URL, SUPABASE_KEY, BSD_API_TOKEN dans .env

### 2. Lancer tous les services
docker compose up --build

### 3. Télécharger le modèle Ollama (première fois uniquement)
docker exec -it qsport-ollama-1 ollama pull mistral

### 4. Accéder à l'application
→ Interface web : http://localhost:3000
→ API backend :  http://localhost:8000
→ Ollama :       http://localhost:11434

## Arrêter
docker compose down

## Réinitialiser complètement (supprime les modèles Ollama)
docker compose down -v
