import sys
import json
import webbrowser
import pickle
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
    MEIPASS  = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MEIPASS  = None

CONFIG_PATH = BASE_DIR / "config.json"

if not CONFIG_PATH.exists():
    print(f"[WARN] config.json was not fount {CONFIG_PATH}, using default value")
    config = {}
else:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

HOST = config.get("host", "0.0.0.0")
PORT = config.get("port", 5000)
OPEN_BROWSER = config.get("open_browser", True)

if MEIPASS is not None:
    FRONTEND_DIR = MEIPASS / config.get("frontend_dir", "frontend")
    MODEL_PATH = MEIPASS / config.get("model_path",   "model/model.pkl")
else:
    FRONTEND_DIR = (BASE_DIR / config.get("frontend_dir", "frontend")).resolve()
    MODEL_PATH = (BASE_DIR / config.get("model_path",   "model/model.pkl")).resolve()

print(f"[INFO] frontend : {FRONTEND_DIR}")
print(f"[INFO] model    : {MODEL_PATH}")
print(f"[INFO] server   : http://{HOST}:{PORT}")

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
HERO_LIST = saved["heroes"]   
feature_cols = saved["features"]
scaler = saved["scaler"]    
needs_scale = saved["needs_scale"]


def build_features(candidate_hero: str, my: list, enemy: list) -> pd.DataFrame:
    team = list(my) + [candidate_hero]

    row = {}
    for h in HERO_LIST:
        row[f"adv_{h}"] = (1.0 if h in team else 0.0) - (1.0 if h in enemy else 0.0)
    row["size_diff"] = float(len(team) - len(enemy))

    return pd.DataFrame([row])[feature_cols]


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/counterpick", methods=["POST"])
def counterpick():
    data   = request.json
    my     = data.get("my_heroes", [])
    enemy  = data.get("enemy_heroes", [])
    banned = set(data.get("banned", []))
    used   = set(my) | set(enemy) | banned

    results = []
    for hero in HERO_LIST:
        if hero in used:
            continue

        X_row = build_features(hero, my, enemy)

        if needs_scale and scaler is not None:
            X_row = scaler.transform(X_row)

        prob = float(model.predict_proba(X_row)[0][1])
        results.append({"hero": hero, "win_prob": round(prob, 4)})

    results.sort(key=lambda x: x["win_prob"], reverse=True)
    return jsonify({"recommendations": results})


@app.route("/predict", methods=["POST"])
def predict():
    data  = request.json
    my    = data.get("my_heroes", [])
    enemy = data.get("enemy_heroes", [])

    row = {}
    for h in HERO_LIST:
        row[f"adv_{h}"] = (1.0 if h in my else 0.0) - (1.0 if h in enemy else 0.0)
    row["size_diff"] = float(len(my) - len(enemy))

    X_row = pd.DataFrame([row])[feature_cols]

    if needs_scale and scaler is not None:
        X_row = scaler.transform(X_row)

    prob = float(model.predict_proba(X_row)[0][1])
    pred = "Win" if prob >= 0.5 else "Lost"

    return jsonify({
        "prediction": pred,
        "win_prob":   round(prob, 4),
        "confidence": round(abs(prob - 0.5) * 2, 4),
    })


if __name__ == "__main__":
    if OPEN_BROWSER:
        webbrowser.open(f"http://localhost:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)