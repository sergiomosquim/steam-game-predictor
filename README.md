![Python](https://img.shields.io/badge/python-3.13-blue)
![Docker](https://img.shields.io/badge/docker-ready-green)
![GitHub License](https://img.shields.io/badge/license-MIT-blue)


# Steam Game Popularity Predictor

With thousands of games on Steam, it can be difficult to guess how well a new game might perform just by looking at its basic information. I created this project to explore whether publicly available game metadata, e.g.,  price, release date, and categories—can help predict how popular a game will become. The result is a simple tool that turns game metadata into an estimated popularity score, helping highlight what aspects of a game may contribute most to its success.

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
├── data
│   ├── data_preprocessed.pkl    # Pickle file for preprocessed data
└── uv.lock                      # Project dependencies
```

---

## Workflow

1. **EDA & Model Selection** (`notebook.ipynb`)  
   - Explore Steam game dataset  
   - Feature engineering
   - Train multiple models (DecisionTree, RandomForest, XGBoost)  
   - Tune hyperparameters with `RandomizedSearchCV`and select best model
   - Plot metrics and feature importances (built into the models, permutation-based and SHapley Additive eXplanations (SHAP) based)

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

To download the raw data and extract the csv file:
```bash
curl -L -O https://www.kaggle.com/api/v1/datasets/download/artermiloff/steam-games-dataset
mv steam-games-dataset steam-games-dataset.zip

unzip -l steam-games-dataset.zip

unzip steam-games-dataset.zip games_march2025_cleaned.csv -d data
rm steam-games-dataset.zip
```

The pickle file `data_preprocessed.pkl` can be used to load the preprocessed data

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

### 3. Test Predictions using fly.io deployment

- Edit `game_example.json`

```bash
uv run request.py
```

- Reads `game_example.json`  
- Prints **Prediction Summary** & **Top Feature Contributions** in tables

### 4. Test Predictions locally

- Edit `game_example.json`
- Change url in `requenst.py` to:
```python
url = "http://0.0.0.0:9696/predict"
```

- Run `request.py`:
```bash
uv run request.py
```

- Reads `game_example.json`  
- Prints **Prediction Summary** & **Top Feature Contributions** in tables

---

## Example `game_example.json`

```json
{
    "price": 29.99,
    "release_date": "2025-01-15",
    "genres": [
        "Action",
        "Adventure"
    ],
    "categories": [
        "Single-player",
        "Co-op"
    ],
    "developers": [
        "IndieDevStudio"
    ],
    "publishers": [
        "IndiePublisher"
    ],
    "discount": 0.2,
    "required_age": 12,
    "dlc_count": 1,
    "windows": true,
    "mac": false,
    "linux": false
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

- Artemiy Ermilov, Arina Nevolina, Artem Pospelov, and Assol Kubaeva. (2025). Steam Games Dataset 2025 [Data set](https://doi.org/10.34740/KAGGLE/DSV/11017460). Kaggle.
- [XGBoost Documentation](https://xgboost.readthedocs.io/)  
- [FastAPI Documentation](https://fastapi.tiangolo.com/)  
- [Fly.io Docs](https://fly.io/docs/)

---

## License

MIT License – see LICENSE file for details.

## Considerations

The model uses `release_date` as a numeric feature based on the dates available in the training data (up to March 2025).  
Tree-based models like XGBoost cannot extrapolate beyond the date range they have seen, so predictions for games with **future release dates** or dates **outside the training range** may be less reliable or produce flattened feature contributions.
The current implementation caps the data to latest `2025-03-31`, which reduces utility of the model in its current implementation.

This does not affect API functionality, but it is important to consider when interpreting prediction results.
