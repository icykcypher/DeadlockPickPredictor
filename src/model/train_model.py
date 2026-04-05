import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
STATS_PATH = os.path.join(BASE_DIR, "model_stats.json")
TEST_SIZE = 0.2
RANDOM_SEED = 21

df = pd.read_csv(DATA_PATH, encoding="utf-8")
print(f"Lenght: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Wins: {df['win'].sum()}  ({df['win'].mean()*100:.1f}%)")
print(f"Lost: {(df['win']==0).sum()}  ({(1-df['win'].mean())*100:.1f}%)")

team_cols = [c for c in df.columns if c.startswith("team_")  and not c.endswith("size")]
enemy_cols = [c for c in df.columns if c.startswith("enemy_") and not c.endswith("size")]
heroes = [c.replace("team_", "") for c in team_cols]

print(f"Heroes count: {len(heroes)}")
print(f"Heroes: {', '.join(heroes)}")

for h in heroes:
    df[f"adv_{h}"] = df[f"team_{h}"].astype(float) - df[f"enemy_{h}"].astype(float)

df["size_diff"] = df["team_size"].astype(float) - df["enemy_size"].astype(float)

feature_cols = [f"adv_{h}" for h in heroes] + ["size_diff"]

X = df[feature_cols].astype(float)
y = df["win"].astype(int)

print(f"Totals features: {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"Train: {len(X_train)} rows")
print(f"Test:  {len(X_test)} rows")

print("\nComparing different models")

candidates = {
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=RANDOM_SEED
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=RANDOM_SEED
    ),
}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
for name, clf in candidates.items():
    Xtr = X_train_scaled if "Logistic" in name else X_train
    Xte = X_test_scaled  if "Logistic" in name else X_test

    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    results[name] = {"acc": acc, "auc": auc, "clf": clf, "Xte": Xte}
    print(f"    {name:25s}  accuracy={acc:.4f}  AUC={auc:.4f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print(f"\nBest Model: {best_name} (AUC={best['auc']:.4f})")

print(f"\nDetailed best model: '{best_name}'...")

best_clf = best["clf"]
Xte = best["Xte"]
pred = best_clf.predict(Xte)
proba = best_clf.predict_proba(Xte)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Lost", "Win"], digits=4))

print("Cross report (5 fold):")
Xtr_cv = X_train_scaled if "Logistic" in best_name else X_train
cv_scores = cross_val_score(
    best_clf, Xtr_cv, y_train, cv=5, scoring="accuracy", n_jobs=-1
)
print(f"Accuracy by folds: {[round(s,4) for s in cv_scores]}")
print(f"Avg: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")

print("\nImportance of features (top 15):")

if hasattr(best_clf, "feature_importances_"):
    importances = best_clf.feature_importances_
elif hasattr(best_clf, "coef_"):
    importances = np.abs(best_clf.coef_[0])
else:
    importances = np.zeros(len(feature_cols))

feat_imp = sorted(
    zip(feature_cols, importances),
    key=lambda x: -x[1]
)

print(f"{'Feature':<20} {'Importance':>10}  Hero")
print("    " + "-" * 45)
for feat, imp in feat_imp[:15]:
    hero = feat.replace("adv_", "") if feat != "size_diff" else "size_diff"
    bar  = "█" * int(imp * 200)
    print(f"    {feat:<20} {imp:>10.4f}  {bar}")

print(f"\nSaving model in '{MODEL_PATH}'")

save_data = {
    "model": best_clf,
    "model_name": best_name,
    "features": feature_cols,
    "heroes": heroes,
    "scaler": scaler if "Logistic" in best_name else None,
    "needs_scale": "Logistic" in best_name,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(save_data, f)

stats = {
    "model": best_name,
    "accuracy": round(best["acc"], 4),
    "auc": round(best["auc"], 4),
    "cv_mean": round(float(cv_scores.mean()), 4),
    "cv_std": round(float(cv_scores.std()),  4),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "n_features": len(feature_cols),
    "heroes": heroes,
    "top_heroes": [f.replace("adv_","") for f, _ in feat_imp[:5] if f != "size_diff"],
}

with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"Model saved: {MODEL_PATH}")
print(f"Statistic saved: {STATS_PATH}")

print("\nDemo trying:")

def predict_win(team_heroes: list, enemy_heroes: list) -> dict:
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    m = data["model"]
    features = data["features"]
    h_list = data["heroes"]
    scaler_ = data["scaler"]
    scale = data["needs_scale"]

    row = {}
    for h in h_list:
        t = 1.0 if h in team_heroes  else 0.0
        e = 1.0 if h in enemy_heroes else 0.0
        row[f"adv_{h}"] = t - e

    t_size = len(team_heroes)
    e_size = len(enemy_heroes)
    row["size_diff"] = float(t_size - e_size)

    X_row = pd.DataFrame([row])[features]

    if scale:
        X_row = scaler_.transform(X_row)

    proba = m.predict_proba(X_row)[0][1]
    pred  = int(proba >= 0.5)

    return {
        "team": team_heroes,
        "enemy": enemy_heroes,
        "win_prob": round(proba, 4),
        "prediction": "Win" if pred else "Lost",
        "confidence": round(abs(proba - 0.5) * 2, 4),
    }


demo1 = predict_win(
    team_heroes = ["Lash", "Dynamo", "Haze", "Kelvin", "Bebop"],
    enemy_heroes = ["Seven", "Vindicta", "Yamato", "McGinnis", "Pocket"],
)

print(f"\nExample 1:")
print(f"My team:  {', '.join(demo1['team'])}")
print(f"Enemies: {', '.join(demo1['enemy'])}")
print(f"Prediction:  {demo1['prediction']}")
print(f"Win probability: {demo1['win_prob']*100:.1f}%")
print(f"Confidence:   {demo1['confidence']*100:.1f}%")

demo2 = predict_win(
    team_heroes = ["Shiv", "Paradox"],
    enemy_heroes = ["Haze", "Warden", "Wraith"],
)

print(f"\nExample 2 (not complete draft):")
print(f"My team:  {', '.join(demo2['team'])}")
print(f"Enemies: {', '.join(demo2['enemy'])}")
print(f"Prediction:  {demo2['prediction']}")
print(f"Win probability: {demo2['win_prob']*100:.1f}%")

print("\n" + "=" * 60)
print(f"Accuracy: {best['acc']*100:.1f}%")
print(f"AUC-ROC: {best['auc']:.4f}")
print("=" * 60)