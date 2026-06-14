import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(
    page_title="Churn Predikcija - Telco",
    page_icon="📡",
    layout="wide"
)

@st.cache_resource
def ucitaj_model():
    if not os.path.exists('churn_model.pkl'):
        st.error("Model nije pronadjen! Pokrenite prvo model.py")
        st.stop()

    model = joblib.load('churn_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    return model, explainer



# ucitava test podatke i vjerovatnoce koje je model izracunao
@st.cache_data
def ucitaj_test_podatke():
    X_test = pd.read_csv('X_test_sample.csv')
    y_test = pd.read_csv('y_test_sample.csv').squeeze()
    y_proba = np.load('y_proba.npy')
    return X_test, y_test, y_proba

# vraca oznaku rizika na osnovu vjerovatnoce
def nivo_rizika(proba):
    if proba >= 0.7:
        return "Visok rizik"
    elif proba >= 0.4:
        return "Srednji rizik"
    else:
        return "Nizak rizik"


# vraca preporucenu akciju zavisno od nivoa rizika
def predlozi_akciju(proba, row):
    if proba >= 0.7:
        if row.get('Contract', 0) == 0:  # month-to-month ugovor
            return "Ponuditi godisnji ugovor sa 20% popusta"
        else:
            return "Hitno kontaktirati - personalizovana ponuda"
    elif proba >= 0.4:
        return "Poslati loyalty email sa benefitima"
    else:
        return "Nije potrebna akcija"


st.title("Churn Predikcija")
st.markdown("*Alat za identifikaciju korisnika koji bi mogli napustiti kompaniju*")
st.markdown("---")

try:
    model, explainer = ucitaj_model()
    X_test, y_test, y_proba = ucitaj_test_podatke()
except Exception as e:
    st.error(f"Greska pri ucitavanju: {e}")
    st.stop()

# sidebar kontrole za filtriranje tabele
st.sidebar.header("Filteri")
min_rizik = st.sidebar.slider(
    "Minimalni rizik churna (%)",
    min_value=0,
    max_value=100,
    value=30,
    step=5
)
prikazi_n = st.sidebar.selectbox(
    "Broj korisnika za prikaz",
    options=[10, 25, 50, 100, "Svi"],
    index=1
)

# kartice na vrhu stranice
col1, col2, col3, col4 = st.columns(4)
ukupno = len(y_proba)
visok_rizik = (y_proba >= 0.7).sum()
srednji_rizik = ((y_proba >= 0.4) & (y_proba < 0.7)).sum()
prosjecni_churn = y_proba.mean() * 100

with col1:
    st.metric("Ukupno korisnika", ukupno)
with col2:
    st.metric("Visok rizik", visok_rizik, delta=f"{visok_rizik/ukupno*100:.1f}%")
with col3:
    st.metric("Srednji rizik", srednji_rizik, delta=f"{srednji_rizik/ukupno*100:.1f}%")
with col4:
    st.metric("Prosjecni rizik", f"{prosjecni_churn:.1f}%")



st.markdown("---")

st.subheader("Rizicni korisnici - rangirani po riziku")

# tabela
df_prikaz = X_test.copy()
df_prikaz['Vjerovatnoca churna'] = y_proba
df_prikaz['Rizik'] = [nivo_rizika(p) for p in y_proba]
df_prikaz['Preporucena akcija'] = [predlozi_akciju(p, X_test.iloc[i]) for i, p in enumerate(y_proba)]
df_prikaz['Stvarni churn'] = y_test.values

filter_proba = min_rizik / 100
df_filtrirano = df_prikaz[df_prikaz['Vjerovatnoca churna'] >= filter_proba]
df_filtrirano = df_filtrirano.sort_values('Vjerovatnoca churna', ascending=False)

# skracuje listu na zadati broj redova
if prikazi_n != "Svi":
    df_filtrirano = df_filtrirano.head(prikazi_n)

st.write(f"Prikazujem {len(df_filtrirano)} korisnika sa rizikom >= {min_rizik}%")

prikaz_kolone = ['Vjerovatnoca churna', 'Rizik', 'Preporucena akcija', 'Stvarni churn']
for kol in ['tenure', 'MonthlyCharges', 'Contract']:
    if kol in df_filtrirano.columns:
        prikaz_kolone.insert(0, kol)

st.dataframe(
    df_filtrirano[prikaz_kolone].style.format({'Vjerovatnoca churna': '{:.1%}'}),
    use_container_width=True,
    height=400
)

st.markdown("---")

st.subheader("Objasnjenje predikcije - SHAP analiza")

indeksi_rizicnih = df_prikaz[df_prikaz['Vjerovatnoca churna'] >= filter_proba].index.tolist()



if len(indeksi_rizicnih) == 0:
    st.warning("Nema korisnika sa ovim nivoom rizika. Smanjite filter.")
else:
    odabrani_idx = st.selectbox(
        "Odaberi korisnika za analizu:",
        options=range(len(indeksi_rizicnih)),
        format_func=lambda i: f"Korisnik #{indeksi_rizicnih[i]} - Rizik: {y_proba[indeksi_rizicnih[i]]:.1%}"
    )

    pravi_idx = indeksi_rizicnih[odabrani_idx]
    korisnik = X_test.iloc[pravi_idx]
    rizik_korisnika = y_proba[pravi_idx]

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown(f"### Korisnik #{pravi_idx}")
        st.metric("Vjerovatnoca churna", f"{rizik_korisnika:.1%}")
        st.write(f"**Status:** {nivo_rizika(rizik_korisnika)}")
        st.write(f"**Akcija:** {predlozi_akciju(rizik_korisnika, korisnik)}")
        st.markdown("**Detalji korisnika:**")
        st.json(korisnik.to_dict())

    with col_b:
        # waterfall pokazuje koji feature-i povecavaju (crveno) a koji smanjuju (plavo) rizik
        st.markdown("**SHAP Waterfall - Zasto ovaj korisnik?**")
        shap_vals = explainer.shap_values(X_test.iloc[[pravi_idx]])

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=korisnik.values,
                feature_names=X_test.columns.tolist()
            ),
            max_display=10,
            show=False
        )
        st.pyplot(fig)
        plt.close()
st.markdown("---")
st.subheader("ROI Retention Kampanje")

col_roi1, col_roi2 = st.columns(2)

with col_roi1:
    cijena = st.number_input("Cijena kontakta po korisniku ($)", value=20, min_value=1, max_value=500)
    vrijednost = st.number_input("Prosjecna vrijednost korisnika ($)", value=150, min_value=10, max_value=5000)
    stopa_zadrzavanja = st.slider("Pretpostavljena stopa zadrzavanja (%)", 10, 60, 30) / 100



with col_roi2:
    # racuna trosak, prihod i ROI na osnovu unesenih parametara i filtera
    targetovani = (y_proba >= filter_proba).sum()
    stvarni_churni_target = ((y_proba >= filter_proba) & (y_test == 1)).sum()
    zadrzani = int(stvarni_churni_target * stopa_zadrzavanja)

    trosak = targetovani * cijena
    prihod = zadrzani * vrijednost
    roi = ((prihod - trosak) / trosak * 100) if trosak > 0 else 0

    st.metric("Targetovani korisnici", targetovani)
    st.metric("Pravi churni u targetu", int(stvarni_churni_target))
    st.metric("Procijenjeno zadrzanih", zadrzani)
    st.metric("Trosak kampanje", f"${trosak:,}")
    st.metric("Ocekivani prihod", f"${prihod:,}")
    st.metric("ROI", f"{roi:.1f}%", delta="pozitivan" if roi > 0 else "negativan")

st.markdown("---")
