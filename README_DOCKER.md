# Q-Sport — Déploiement avec Docker

## 📋 Table des matières

1. [Prérequis](#prérequis)
2. [Configuration initiale](#configuration-initiale)
3. [Démarrage de l'application](#démarrage-de-lapplication)
4. [Architecture Docker](#architecture-docker)
5. [Services en détail](#services-en-détail)
6. [Architecture du backend](#architecture-du-backend)
7. [Ollama sur Google Colab](#ollama-sur-google-colab)
8. [Accès à l'application](#accès-à-lapplication)
9. [Gestion des services](#gestion-des-services)
10. [Étendre le projet](#étendre-le-projet)
11. [Troubleshooting](#troubleshooting)

---

## Prérequis

### Logiciels requis

- **Docker Desktop** (v4.0+) — [Windows](https://www.docker.com/products/docker-desktop) / [Mac](https://www.docker.com/products/docker-desktop) / [Linux](https://docs.docker.com/engine/install/)
- **Git**

```bash
git clone https://github.com/ThomasMuel7/QSport.git
cd QSport
git checkout smartpld
```

### Ressources recommandées

| Composant | Minimum     | Recommandé   |
| --------- | ----------- | ------------ |
| CPU       | 4 cores     | 8+ cores     |
| RAM       | 8GB         | 16GB         |
| Disque    | 50GB libres | 100GB libres |

---

## Configuration initiale

```bash
cp .env.example .env
```

Remplir `.env` :

```env
# LLM via Google Colab (voir section dédiée)
OLLAMA_HOST=https://abc123-def456.trycloudflare.com
OLLAMA_MODEL=llama3.1:8b

# Football Data API
API_KEY_FOOTBALL_DATA=ton_token  # football-data.org/client/register
```

---

## Démarrage de l'application

```bash
# Premier lancement (build complet ~20 min)
docker compose up --build

# Lancements suivants
docker compose up
```

Le backend est prêt quand cette ligne apparaît :

```
backend-1  | INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## Architecture Docker

```
┌────────────────────────────────────────────┐
│              DOCKER COMPOSE                │
│                                            │
│  ┌─────────────────┐  ┌─────────────────┐  │
│  │    FRONTEND     │  │    BACKEND      │  │
│  │   (Port 3000)   ◄─►  (Port 8000)     │  │
│  │  React + Vite   │  │ FastAPI +       │  │
│  │                 │  │ LangChain + ML  │  │
│  └─────────────────┘  └────────┬────────┘  │
│                                │           │
└────────────────────────────────┼───────────┘
                                 │ HTTPS
                    ┌────────────▼────────────┐
                    │    OLLAMA (Colab T4)    │
                    │  via Cloudflare Tunnel  │
                    │     llama3.1:8b         │
                    └─────────────────────────┘
```

### Services

| Service  | Port | Image       | Rôle             |
| -------- | ---- | ----------- | ---------------- |
| frontend | 3000 | Node.js 18  | Interface web    |
| backend  | 8000 | Python 3.10 | API + modèles ML |

> ℹ️ Ollama tourne en dehors de Docker, sur Google Colab (voir section dédiée).

---

## Services en détail

### Frontend

**URL :** http://localhost:3000

Interface React/Vite — formulaires de prédiction et chat avec l'agent LLM.

### Backend

**URL :** http://localhost:8000  
**Swagger :** http://localhost:8000/docs

Contient les modèles ML chargés en mémoire au démarrage :

| Sport    | Modèles                |
| -------- | ---------------------- |
| Football | XGBoost, CatBoost      |
| NBA      | XGBoost, CatBoost, QNN |
| Tennis   | LightGBM               |

**Endpoints principaux :**

```
POST /chat
GET  /health
```

---

## Architecture du backend

```
backend/
├── main.py              # Point d'entrée FastAPI
├── agent.py             # Agent LLM LangChain
├── tools.py             # Tools appelés par l'agent
├── foot_prediction.py   # Pipeline prédiction football
├── nba_prediction.py    # Pipeline prédiction NBA
├── tennis_prediction.py # Pipeline prédiction tennis
├── odds.py              # Récupération des cotes (The Odds API)
├── paths.py             # Chemins vers data/ et models/
└── teams_mapping.csv    # Alias → noms officiels des équipes
```

### main.py

Point d'entrée FastAPI. Expose deux endpoints :

- `GET /health` — vérifie que le backend est opérationnel
- `POST /chat` — reçoit un message utilisateur, le transmet à l'agent et retourne la réponse

Gère également le CORS pour autoriser les requêtes du frontend.

### agent.py

Initialise l'agent LangChain avec le modèle Ollama configuré via `.env`. L'agent suit un prompt système strict en 3 étapes : identification du sport, appel obligatoire au bon tool, retour brut du résultat sans modification.

L'agent est instancié une seule fois (singleton `_agent`) pour éviter de recharger le modèle à chaque requête.

### tools.py

Définit les trois tools LangChain appelables par l'agent :

- `predict_foot(teams)` — prédiction football, résout les alias d'équipes via `teams_mapping.csv`
- `predict_nba(teams)` — prédiction NBA, attend des trigrammes (ex: `LAL vs GSW`)
- `predict_tennis(players)` — prédiction tennis ATP, détecte la surface dans la chaîne en entrée

Chaque predictor est chargé une fois au démarrage. Si un modèle est indisponible, le tool renvoie un message d'erreur explicite sans faire crasher le backend.

### foot_prediction.py / nba_prediction.py / tennis_prediction.py

Contiennent la logique métier de prédiction pour chaque sport. Chargent les modèles ML depuis `models/` au démarrage et exposent une fonction unique (`predire`, `NBAPredictor.predict_game`, `predict_match`) appelée par les tools.

### paths.py

Centralise tous les chemins du projet (`data/`, `models/`, etc.) en chemins absolus résolus depuis la position du fichier. Évite les problèmes de chemins relatifs selon le répertoire de lancement.

---

## Ollama sur Google Colab

L'agent LLM tourne sur **Google Colab (GPU T4)** et est exposé via un tunnel Cloudflare.

> ⚠️ Le runtime Colab est limité à **~5 heures**. Le tunnel et l'URL changent à chaque session — penser à mettre à jour `.env`.

### Étape 1 — Configurer le runtime GPU

Dans Colab : **Runtime → Change runtime type → GPU (T4)**

### Étape 2 — Installer et lancer Ollama

```python
# Cellule 1
!apt-get install -y zstd pciutils
!curl -fsSL https://ollama.com/install.sh | sh
!nvidia-smi  # Vérifier le GPU

# Cellule 2
import os, subprocess, time
os.environ["OLLAMA_ORIGINS"] = "*"
subprocess.Popen(["ollama", "serve"], stdout=open("ollama.log","w"), stderr=subprocess.STDOUT)
time.sleep(5)
print("✅ Ollama lancé")
```

### Étape 3 — Télécharger le modèle

```python
!ollama pull llama3.1:8b
!ollama list  # Vérifier
```

### Étape 4 — Créer le tunnel Cloudflare

```python
!curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:11434
```

Copier l'URL affichée (ex: `https://abc123.trycloudflare.com`) et la mettre dans `.env` :

```env
OLLAMA_HOST=https://abc123.trycloudflare.com
```

Puis redémarrer le backend :

```bash
docker compose restart backend
```

### Modèles disponibles

| Modèle       | VRAM | Vitesse   | Qualité     |
| ------------ | ---- | --------- | ----------- |
| orca-mini:3b | 2GB  | ⚡ Rapide | moyenne     |
| mistral      | 5GB  | Moyen     | bonne       |
| llama3.1:8b  | 8GB  | Moyen     | ✅ Meilleur |

> Pour changer de modèle, modifier `OLLAMA_MODEL` dans `.env` et relancer `docker compose restart backend`.

---

## Accès à l'application

| Interface   | URL                        |
| ----------- | -------------------------- |
| Frontend    | http://localhost:3000      |
| Backend API | http://localhost:8000      |
| Swagger UI  | http://localhost:8000/docs |

---

## Gestion des services

```bash
# Démarrer / arrêter
docker compose up -d
docker compose down

# Logs
docker compose logs -f backend
docker compose logs -f frontend

# Statut
docker compose ps

# Redémarrer un service
docker compose restart backend

# Rebuild un service
docker compose up -d --build backend
```

---

## Troubleshooting

**Backend crash au démarrage**  
→ Insuffisance mémoire. Augmenter la RAM dans Docker Desktop (Settings → Resources → Memory : 16GB minimum).

**"Connection timeout" vers Ollama**  
→ Vérifier que le tunnel Colab est actif. Tester :

```bash
curl -I $OLLAMA_HOST
```

**Port déjà utilisé**

```bash
lsof -i :8000   # Trouver le processus
kill -9 <PID>
```

**Le tunnel Cloudflare se ferme**  
→ Normal. Relancer la cellule Colab et mettre à jour `.env`.

**L'agent n'appelle pas le bon tool**  
→ Vérifier la docstring du tool concerné dans `tools.py` — c'est elle que le LLM utilise pour router. La rendre plus explicite sur les cas d'usage.

**L'agent répond sans appeler de tool**  
→ Le modèle LLM est peut-être trop petit ou trop "créatif". Réduire `temperature` dans `agent.py` (ex: `0.0`) ou passer à `llama3.1:8b`.

---

> Pour faire tourner Ollama en local (sans Colab), il faut un GPU avec suffisamment de VRAM, le toolkit NVIDIA, et adapter le `docker-compose.yml` pour exposer le GPU au conteneur. Le setup Colab reste recommandé pour du développement et des tests.

---

**Dernière mise à jour :** 06 Mai 2026 — Docker Compose v2.17+
