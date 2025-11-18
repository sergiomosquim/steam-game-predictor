# Steam Game Popularity Predictor

Predict the potential popularity of Steam games using an XGBoost model.

This project includes a full ML workflow: exploratory data analysis, model training, hyperparameter tuning, and deployment via FastAPI and Fly.io.

---

## Features

- Explore data and train multiple models in a Jupyter notebook  
- Select the best model via hyperparameter tuning  
- Train final model with `train.py` and save final model and score quantiles for use in prediction  
- Predict game popularity via FastAPI API (`predict_docker.py`)  
- See interpretable feature contributions for each prediction  
- JSON input templates for easy testing (`game_example.json`)  
- Fully containerized with Docker for deployment

---

## Project Structure

```
steam-game-predictor/
│
├── Dockerfile                   # Container setup
├── LICENSE
├── README.md
├── best_xgboost_params.json     # Set of best parameters from RandomizedSearchCV for XGBoost model
├── fly.toml                     # Fly.io deployment config
├── game_example.json            # Example game input for requests
├── log_score_quantiles.json     # Quantiles based on training data for generating prediction categories
├── notebook.ipynb               # EDA, model comparison, hyperparameter tuning
├── predict_docker.py            # FastAPI app for serving predictions via Docker
├── predict_local.py             # FastAPI app for serving predictions locally
├── pyproject.toml               # Project dependencies
├── request.py                   # Sends example game JSON to API and prints tables
├── runapp.sh                    # Entrypoint for Docker container (runs uvicorn)
├── train.py                     # Train final model with best parameters, save model & log_scores
├── transformers.py              # Column transformers & pipeline for train.py
└── uv.lock                      # Project dependencies
```

---

## Workflow

1. **EDA & Model Selection** (`notebook.ipynb`)  
   - Explore Steam game dataset  
   - Train multiple models (XGBoost, etc.)  
   - Tune hyperparameters and select best model  

2. **Train Final Model** (`train.py`)  
   - Uses cleaned data and best hyperparameters  
   - Saves trained XGBoost model (joblib)  
   - Saves log_scores from training data  

3. **Transformers** (`transformers.py`)  
   - Defines preprocessing steps  
   - Used by both `train.py` and FastAPI for consistent feature processing  

4. **FastAPI App** (`predict_docker.py` or `predict_local.py`)  
   - Serves `/predict` endpoint  
   - Returns predictions + top feature contributions  

5. **Request Script** (`request.py`)  
   - Reads `game_example.json`  
   - Sends request to deployed API  
   - Prints **Prediction Summary** & **Top Feature Contributions** as readable tables  

6. **Deployment**  
   - Dockerized app with `Dockerfile` and `entrypoint.sh`  
   - Fly.io deployment via `fly.toml`  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sergiomosquim/steam-game-predictor.git
cd steam-game-predictor
```

Set up Python environment:

```bash
uv sync --locked
```

---

## Usage

### 1. Train Model Locally

```bash
uv run train.py
```

- Saves final  
- Saves training log_scores for analysis

### 2. Run FastAPI Locally

```bash
uv run predict_local.py
```

- Endpoint: `POST /predict`  
- Input: JSON matching `game_example.json`  
- Output: JSON prediction + top contributors

### 3. Test Predictions

```bash
uv run request.py
```

- Reads `game_example.json`  
- Prints **Prediction Summary** & **Top Feature Contributions** in tables

---

## Example `game_example.json`

```json
{
    "price": 19.99,
    "release_year": 2021,
    "days_since_release": 300,
    "dlc_count": 0,
    "categories": ["Action", "Singleplayer"]
}
```

---

## Docker

- Pull Docker image:

```bash
docker pull ghcr.io/sergiomosquim/steam-game-predictor:latest
```

- Run container locally:

```bash
docker run -p 9696:9696 steam-game-predictor
```

---

## 📖 References

- [Kaggle Dataset Link](https://doi.org/10.34740/KAGGLE/DSV/11017460)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)  
- [FastAPI Documentation](https://fastapi.tiangolo.com/)  
- [Fly.io Docs](https://fly.io/docs/)

---

## License

MIT License – see LICENSE file for details.
