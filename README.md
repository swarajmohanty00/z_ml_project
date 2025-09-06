
# 🏏 IPL Win Predictor

A Machine Learning project to predict the outcome of IPL matches based on historical match and ball-by-ball data.
This project preprocesses raw IPL datasets, trains a model, and provides predictions through an interactive app.

---

## 📂 Project Structure

```
ipl_win_predictor/
│── __pycache__/              # Auto-generated Python cache files
│── data/                     # Raw IPL datasets
│   ├── deliveries.csv        # Ball-by-ball data
│   └── matches.csv           # Match-level data
│
│── app.py                    # Main app (Streamlit)
│── data_preprocessing.py     # Data cleaning & preprocessing
│── model_training.py         # Training script for ML model
│── predictor.py              # Loads model & predicts outcomes
│
│── city.pkl                  # Encoded city/venue mappings
│── team.pkl                  # Encoded team mappings
│── model.pkl                 # Trained ML model
│
│── requirements.txt          # Project dependencies
```

---

## ⚙️ Features

* Preprocess raw IPL data (`matches.csv` and `deliveries.csv`)
* Train a machine learning model to predict match winners
* Save and load trained model with `pickle`
* Interactive prediction using `Streamlit` app
* Encodes team and city data for model compatibility

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ashishota/z_ml_project.git
cd z_ml_project/ipl_win_predictor
```

### 2️⃣ Install Dependencies

Make sure you have **Python 3.9+** installed, then run:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
pandas
scikit-learn
streamlit
```

### 3️⃣ Run Preprocessing

```bash
python data_preprocessing.py
```

### 4️⃣ Train the Model

```bash
python model_training.py
```

### 5️⃣ Make Predictions

```bash
python predictor.py
```

### 6️⃣ Launch the App

```bash
streamlit run app.py
```

---

## 📊 Dataset

The project uses IPL data from Kaggle:

* **matches.csv** → Match-level info (teams, venue, winner, toss, etc.)
* **deliveries.csv** → Ball-by-ball deliveries data

---

## 🧠 Model

* Trained using scikit-learn
* Encoded features: teams, venue, toss decision, etc.
* Saved as `model.pkl` for reusability

---

## 📌 Future Improvements

* Add more features (player stats, weather, toss impact, etc.)
* Use advanced models (XGBoost, Random Forest, Deep Learning)
* Deploy the app on **Streamlit Cloud / Heroku / AWS**

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.

---

## 📜 License

This project is licensed under the MIT License.


