import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')


# ucitava csv i vraca dataframe
def ucitaj_podatke(putanja="Telco_customer_churn.csv"):
    print("Ucitavanje podataka...")
    df = pd.read_csv(putanja)
    print(f"Ucitano {len(df)} redova i {len(df.columns)} kolona.")
    return df



# cisti podatke
def pripremi_podatke(df):
    print("\nPriprema podataka za model...")

    df = df.copy()

    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(0)

    # CustomerID nije koristan za predikciju
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)

    df['Churn'] = df['Churn Value']

    kolone_za_izbaciti = [
        'Churn Label',
        'Churn Reason',
        'Churn Score',
        'CLTV',
        'Churn Value'
    ]

    for kolona in kolone_za_izbaciti:
        if kolona in df.columns:
            df = df.drop(kolona, axis=1)

    # Sve tekstualne kolone pretvaramo u brojeve
    le = LabelEncoder()
    kategoricke = df.select_dtypes(include=['object']).columns

    for kolona in kategoricke:
        df[kolona] = le.fit_transform(df[kolona].astype(str))

    print("Priprema podataka gotova.")
    return df
# dodaje nove kolone na osnovu duzine koristenja
def dodaj_kolone(df):
    df = df.copy()

    if 'Tenure Months' in df.columns:
        df['rani_rizik'] = (df['Tenure Months'] <= 3).astype(int)
        df['srednji_rizik'] = (
            (df['Tenure Months'] > 3) &
            (df['Tenure Months'] <= 12)
        ).astype(int)

        df['kasni_rizik'] = (
            df['Tenure Months'] > 24
        ).astype(int)

        if 'Monthly Charges' in df.columns:
            # mjesecna cijena podijeljeno sa brojem mjeseci
            df['charges_per_tenure'] = (
                df['Monthly Charges'] /
                (df['Tenure Months'] + 1)
            )

    return df

# dijeli dataset na training (80%) i test (20%)
def podijeli_podatke(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=67,
        stratify=y        
    )

    print(f"Training set: {len(X_train)} redova")
    print(f"Test set: {len(X_test)} redova")
    return X_train, X_test, y_train, y_test



# trenira XGBoost klasifikator
def treniraj_model(X_train, y_train):
    print("\nTreniram XGBoost model...")

    # dataset ima vise 'No' nego 'Yes' pa racunamo tezinu da model ne ignorise manjinu
    broj_no = (y_train == 0).sum()
    broj_yes = (y_train == 1).sum()
    omjer = broj_no / broj_yes
    print(f"  Odnos No/Yes: {omjer:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,        
        max_depth=4,             
        learning_rate=0.05,      
        scale_pos_weight=omjer,  
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)
    print("Treniranje modela gotovo.")
    return model



# stampa classification report i confusion matrix
def evaluiraj_model(model, X_test, y_test):
    print("\n- REZULTATI MODELA:")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ostaje', 'Odlazi']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return y_proba


# kreira SHAP explainer koji objasnjava zasto model donosi odredjenu odluku
def izracunaj_shap(model, X_train, X_test, feature_names):
    print("\nRacunanje SHAP vrijednosti...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return explainer, shap_values


# racuna ocekivani ROI ako se pokrene retention kampanja nad rizicnim korisnicima
def izracunaj_roi(y_test, y_proba, cijena_kampanje=20, vrijednost_korisnika=150):
    print("\n- ROI ANALIZA RETENTION KAMPANJE:")

    # zanima nas top 30% korisnika
    threshold = np.percentile(y_proba, 70)
    targetovani = (y_proba >= threshold)

    stvarni_churni_targetovani = ((y_test == 1) & targetovani).sum()
    ukupno_targetovanih = targetovani.sum()

    trosak = ukupno_targetovanih * cijena_kampanje
    zadrzani = int(stvarni_churni_targetovani * 0.3)
    prihod = zadrzani * vrijednost_korisnika
    roi = ((prihod - trosak) / trosak) * 100 if trosak > 0 else 0

    print(f"Targetovani korisnici: {ukupno_targetovanih}")
    print(f"Od kojih su stvarni churni: {stvarni_churni_targetovani}")
    print(f"Procjena zadrzanih: {zadrzani}")
    print(f"Trosak kampanje: ${trosak:,}")
    print(f"Ocekivani prihod: ${prihod:,}")
    print(f"ROI: {roi:.1f}%")

    return {
        'targetovani': int(ukupno_targetovanih),
        'stvarni_churni': int(stvarni_churni_targetovani),
        'zadrzani': zadrzani,
        'trosak': trosak,
        'prihod': prihod,
        'roi': roi
    }


if __name__ == "__main__":
    df = ucitaj_podatke()
    df = dodaj_kolone(df)
    df_clean = pripremi_podatke(df)

    X_train, X_test, y_train, y_test = podijeli_podatke(df_clean)
    model = treniraj_model(X_train, y_train)
    y_proba = evaluiraj_model(model, X_test, y_test)

    explainer, shap_values = izracunaj_shap(
        model, X_train, X_test,
        feature_names=X_train.columns.tolist()
    )
    roi_rezultati = izracunaj_roi(y_test, y_proba)

    import joblib
    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(explainer, 'shap_explainer.pkl')
    X_test.to_csv('X_test_sample.csv', index=False)
    y_test.to_csv('y_test_sample.csv', index=False)
    np.save('y_proba.npy', y_proba)

    print("\nModel sacuvan.")
