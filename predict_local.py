from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import numpy as np
import shap
from pydantic import BaseModel, Field
from typing import List
import json
from datetime import date

class GameInput(BaseModel):
    price: float = Field(default=0.0, ge=0)
    release_date: date = Field(..., le=date(2025,3,31))
    genres: List[str] = Field(..., min_length=1)
    categories: List[str] = Field(..., min_length=1)
    developers: List[str] = Field(..., min_length=1)
    publishers: List[str] = Field(..., min_length=1)
    discount: float = Field(default=0.0, ge=0)
    required_age: int = Field(default=0)
    dlc_count: int = Field(default=0)
    windows: bool = Field(default=True)
    mac: bool = Field(default=False)
    linux: bool = Field(default=False)

class PredictSuccess(BaseModel):
    log_score: float
    estimated_popular_players: int
    popularity_category: str
    top_contributors: dict

with open('final_xgb_model.pkl', 'rb') as f:
    pipeline = joblib.load(f)

def predict_single(game:GameInput):
    # convert input to dataframe
    input_df = pd.DataFrame([game.model_dump()])
    log_score = pipeline.predict(input_df)[0]
    estimated_popular_players = np.expm1(log_score)
    result = dict({
        'log_score': float(log_score),
        'estimated_popular_players': int(estimated_popular_players)
    })
    return result

with open('log_score_quantiles.json', 'r') as f:
    quantiles = json.load(f)

def popularity_category(log_score: float):
    if log_score <= quantiles['20%']:
        return 'Very Low'
    elif log_score <= quantiles['40%']:
        return 'Low'
    elif log_score <= quantiles['60%']:
        return 'Medium'
    elif log_score <= quantiles['80%']:
        return 'High'
    else:
        return 'Very High'

# Return top contributors based on SHAP values
model = pipeline.named_steps['model']
preprocessor = pipeline.named_steps['preprocess']
explainer = shap.TreeExplainer(model)

def get_feature_contributions(game:GameInput):
    input_df = pd.DataFrame([game.model_dump()])
    feature_names = preprocessor.get_feature_names_out()
    input_df = pd.DataFrame(preprocessor.transform(input_df), columns = feature_names).astype(float)
    shap_val = explainer(input_df).values[0,:]
    contributions = dict(zip(
        input_df.columns,
        map(float, shap_val)
    ))
    return contributions

app = FastAPI(title='Steam Game Success Predictor')

@app.post("/predict", response_model=PredictSuccess)
def predict(game:GameInput):
    pred_value = predict_single(game)
    category = popularity_category(pred_value['log_score'])
    contributions = get_feature_contributions(game)
    sorted_contributions = sorted(
        contributions.items(), 
        key = lambda x: np.abs(x[1]),
        reverse=True)
    top_contributions = dict(sorted_contributions[:5])


    return PredictSuccess(
        log_score=pred_value['log_score'],
        estimated_popular_players=pred_value['estimated_popular_players'],
        popularity_category=category,
        top_contributors=top_contributions
    )

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port = 9696)