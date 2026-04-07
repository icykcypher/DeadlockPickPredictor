from pathlib import Path
from flask import Flask, request, jsonify
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path=""
)

MODEL_PATH = BASE_DIR / "model" / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model        = saved["model"]
HERO_LIST    = saved["heroes"]    # порядок героев как при обучении
feature_cols = saved["features"]  # ['adv_Abrams', 'adv_Bebop', ..., 'size_diff']
scaler       = saved["scaler"]    # None если не LogisticRegression
needs_scale  = saved["needs_scale"]


def build_features(candidate_hero: str, my: list, enemy: list) -> pd.DataFrame:
    """
    Строит adv_-фичи для кандидата.
    Модель обучалась на: adv_{h} = team_{h} - enemy_{h}, size_diff.
    Кандидат считается уже добавленным в my.
    """
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
    """
    Прямой предикт по готовым командам.
    Тело: { "my_heroes": [...], "enemy_heroes": [...] }
    """
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
    app.run(debug=True)