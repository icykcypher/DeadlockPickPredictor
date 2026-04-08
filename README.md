# Deadlock Pick Predictor

Tool for real-time hero pick recommendations during Deadlock drafts. After the first enemy pick, the app suggests which heroes to pick based on ML predictions trained on real match data.

## How it works

A Gradient Boosting model is trained on match history. For each available hero, the server predicts win probability given the current draft state and returns a ranked list. The frontend displays these probabilities as badges and bars on each hero card.

## Stack

- **Backend** — Python, Flask, scikit-learn
- **Frontend** — Vanilla JS/HTML/CSS
- **Build** — PyInstaller (single binary), GitHub Actions (CI for Linux + Windows)

## Project structure

```
build/
├── server.py        # Flask server + ML inference
├── config.json      # Runtime config (port, paths)
├── frontend/
│   ├── index.html   # Draft UI
│   └── public/      # Hero images
└── model/
    └── model.pkl    # Trained model
```

## Running from source

```bash
pip install flask pandas scikit-learn numpy
python build/server.py
```

Opens at `http://localhost:5000`.

## Config

Edit `config.json` next to the binary:

```json
{
  "port": 5000,
  "host": "0.0.0.0",
  "open_browser": true,
  "frontend_dir": "frontend",
  "model_path": "model/model.pkl"
}
```

## Building

```bash
cd build
pyinstaller --onefile \
  --add-data "frontend:frontend" \
  --add-data "model/model.pkl:model" \
  --add-data "config.json:." \
  --hidden-import sklearn \
  --hidden-import sklearn.ensemble \
  --hidden-import sklearn.ensemble._gb \
  --hidden-import sklearn.ensemble._forest \
  --hidden-import sklearn.linear_model \
  --hidden-import sklearn.preprocessing \
  --hidden-import sklearn.tree \
  --hidden-import sklearn.tree._classes \
  server.py
```

Or trigger the GitHub Actions workflow — it builds for both Linux and Windows automatically.

## Training your own model

Place your `data.csv` in the `model/` directory and run:

```bash
python train.py
```