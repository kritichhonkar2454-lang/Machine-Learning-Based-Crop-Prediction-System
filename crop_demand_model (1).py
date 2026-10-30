from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "crop_demand_model.pkl"
DATA_FILE = BASE_DIR / "crop_demand_data.csv"

CROPS = ["Maize", "Onion", "Potato", "Rice", "Tomato", "Wheat"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
FEATURES = [
    "price",
    "inventory",
    "active_buyers",
    "days_to_harvest",
    "market_distance",
    *[f"crop_{crop}" for crop in CROPS],
    *[f"season_{season}" for season in SEASONS],
]


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("crop_demand_model.pkl is missing from this folder.")
    return joblib.load(MODEL_FILE)


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError("crop_demand_data.csv is missing from this folder.")

    data = pd.read_csv(DATA_FILE)
    required_columns = FEATURES + ["demand"]
    missing = [column for column in required_columns if column not in data.columns]

    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    return data


def predict_demand(
    price,
    inventory,
    active_buyers,
    days_to_harvest,
    market_distance,
    crop,
    season,
):
    crop = str(crop).strip().title()
    season = str(season).strip().title()

    if crop not in CROPS:
        raise ValueError(f"Choose a crop from: {', '.join(CROPS)}")

    if season not in SEASONS:
        raise ValueError(f"Choose a season from: {', '.join(SEASONS)}")

    values = {
        "price": price,
        "inventory": inventory,
        "active_buyers": active_buyers,
        "days_to_harvest": days_to_harvest,
        "market_distance": market_distance,
    }

    values.update({f"crop_{name}": int(name == crop) for name in CROPS})
    values.update({f"season_{name}": int(name == season) for name in SEASONS})

    input_data = pd.DataFrame([values], columns=FEATURES)
    return float(load_model().predict(input_data)[0])


def evaluate_model(data):
    model = load_model()
    predictions = model.predict(data[FEATURES])
    actual = data["demand"]

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5

    return mae, rmse


if __name__ == "__main__":
    dataset = load_dataset()
    mae, rmse = evaluate_model(dataset)

    print(f"Dataset loaded: {len(dataset)} records")
    print(f"Model MAE: {mae:.2f}")
    print(f"Model RMSE: {rmse:.2f}")

    result = predict_demand(
        price=25,
        inventory=500,
        active_buyers=20,
        days_to_harvest=7,
        market_distance=12,
        crop="Tomato",
        season="Kharif",
    )

    print(f"Predicted crop demand: {result:.2f}")
