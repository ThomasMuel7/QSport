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

## Étendre le projet

### Modifier le comportement de l'agent

Le comportement de l'agent est entièrement contrôlé par le `SYSTEM_PROMPT` dans `agent.py`. C'est le premier levier à modifier pour changer la façon dont l'agent interprète les requêtes ou formate ses réponses.

```python
# agent.py
SYSTEM_PROMPT = """You are QSport, an expert sports prediction assistant.
...
"""
```

**Exemples de modifications utiles :**

Changer la langue de réponse par défaut :

```python
# Remplacer dans SYSTEM_PROMPT
"Be concise and structured."
# par
"Always respond in French, be concise and structured."
```

Ajouter un nouveau sport reconnu :

```python
# Dans la section STEP 1: IDENTIFICATION, ajouter
"- For rugby: use full team names (ex: 'Stade Toulousain', 'Racing 92')."

# Dans STEP 2: TOOL CALL, ajouter
"- predict_rugby → for ANY rugby match, no matter what."

# Dans EXAMPLES, ajouter
"- 'Toulouse vs Racing' → call predict_rugby('Stade Toulousain vs Racing 92')"
```

Modifier le niveau de verbosité des réponses :

```python
# Remplacer
"Be concise and structured."
# par
"Always include a short explanation of the key factors driving the prediction."
```

### Changer le modèle LLM

Le modèle est configuré exclusivement via `.env` — aucune modification de code nécessaire :

```env
OLLAMA_MODEL=mistral        # Plus rapide, moins précis
OLLAMA_MODEL=llama3.1:8b    # Meilleur équilibre (défaut)
OLLAMA_MODEL=llama2:13b     # Plus précis, trop lourd pour T4
```

Pour les paramètres fins du modèle (température, etc.), modifier `agent.py` :

```python
# agent.py
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "mistral-nemo"),
    base_url=os.getenv("OLLAMA_HOST"),
    temperature=0.1,   # 0.0 = déterministe, 1.0 = créatif
)
```

### Ajouter un nouveau tool

Un tool est une fonction Python décorée avec `@tool` que l'agent peut appeler. Voici comment en ajouter un de A à Z.

#### Étape 1 — Créer la logique métier

Créer un nouveau fichier (ex: `rugby_prediction.py`) avec une fonction exposant le résultat :

```python
# rugby_prediction.py
def predict_rugby(team1: str, team2: str) -> dict:
    # Charger le modèle, construire les features, inférer...
    return {
        "winner": team1,
        "p_home_win": 0.62,
        "p_away_win": 0.38,
    }
```

#### Étape 2 — Déclarer le tool dans tools.py

```python
# tools.py
from langchain.tools import tool

try:
    from rugby_prediction import predict_rugby as predire_rugby
except Exception as e:
    predire_rugby = None
    print(f"[tools] Rugby predictor non disponible: {e}")

@tool
def predict_rugby(teams: str) -> str:
    """Predit le resultat d'un match de rugby Top 14 ou Champions Cup.
    Format: 'EQUIPE1 vs EQUIPE2' (ex: 'Stade Toulousain vs Racing 92')."""
    code
```

> **Important :** La docstring du `@tool` est ce que le LLM lit pour décider quand appeler ce tool. Elle doit être claire, courte, et mentionner le format d'entrée attendu.

#### Étape 3 — Enregistrer le tool dans l'agent

```python
# agent.py
from tools import predict_nba, predict_foot, predict_tennis, predict_rugby  # Ajouter ici

tools = [predict_nba, predict_foot, predict_tennis, predict_rugby]  # Ajouter ici
```

#### Étape 4 — Mettre à jour le prompt système

```python
# agent.py — dans SYSTEM_PROMPT, section STEP 2
"- predict_rugby → for ANY rugby match, no matter what."

# Section EXAMPLES
"- 'Toulouse vs Racing' → call predict_rugby('Stade Toulousain vs Racing 92')"
```

#### Étape 5 — Rebuild le backend

```bash
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
