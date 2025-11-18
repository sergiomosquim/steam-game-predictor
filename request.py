import json
import requests
import pandas as pd

url = "https://steam-game-predictor.fly.dev/predict"

with open('game_example.json', 'r') as f_in:
    game = json.load(f_in)

response = requests.post(url, json=game)

if response.status_code==200:
    response_json = response.json()
    main_values = {
        "log_score": response_json["log_score"],
        "estimated_popular_players": response_json["estimated_popular_players"],
        "popularity_category": response_json["popularity_category"]
    }

    df_main = pd.DataFrame(
        [{"variable": k, "value": v} for k, v in main_values.items()]
    )

    print("\n=== Prediction Summary ===")
    print(df_main.to_string(index=False))

    tc = response_json["top_contributors"]

    df_contrib = pd.DataFrame(
        [{"feature": k, "contribution": v} for k, v in tc.items()]
    )

    print("\n=== Top Feature Contributions ( Ranked in absolute values ) ===")
    print(df_contrib.to_string(index=False))
else:
    print(f"Error {response.status_code}: {response.text}")