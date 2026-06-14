# Projekat: Churn Predikcija


## Opis projekta

Kompanija gubi korisnike ali ne zna ko će otići sljedeći. Ovaj projekat predstavlja sistem koji:
- Predviđa koji korisnici su u riziku od odlaska (XGBoost)
- Objašnjava zašto (SHAP)
- Računa očekivani ROI retention kampanje

Koristio sam Telco Customer Churn Dataset sa Kaggle-a.

Takodje, koriscena je biblioteka Streamlit za web aplikaciju, kako bih sebi olaksao rad na frontendu a dobio nesto sto izgleda kao standardna web aplikacija ovog tipa.

## Pokretanje projekta

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 3. Treniranje modela
```bash
python model.py
```
Ovo će kreirati: `churn_model.pkl`, `shap_explainer.pkl`, `X_test_sample.csv`, `y_test_sample.csv`, `y_proba.npy`

### 4. Pokretanje web aplikacije
```bash
streamlit run app.py
```
