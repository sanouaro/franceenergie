import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import statsmodels.api as sm

# Définir une palette de couleurs
couleurs = {
    "fond": "#f5f5f5",
    "accent": "#1abc9c",
    "texte": "#333333",
    "texte_secondaire": "#777777"
}

st.title("Projet Energie")

@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df_monthly_mean = pd.read_csv(url)
    return df_monthly_mean

df_monthly_mean = load_data("https://raw.githubusercontent.com/miraculix95/franceenergie/main/df_monthly_mean.csv?token=GHSAT0AAAAAAB75KNMFZNRSFAZ6VR4IZ5JWZAN2Q4A")
st.dataframe(df_monthly_mean)
df_monthly_mean = np.log(df_monthly_mean)

st.button("Rerun")




# Créer le modèle SARIMA
modele = sm.tsa.SARIMAX(df_monthly_mean, order=(1,2,3), seasonal_order=(1,1,1,12))
sarima = modele.fit()

# Sélection des paramètres du modèle
st.write("## Sélection des paramètres du modèle")
p = st.slider("Ordre de l'autorégression (p)", min_value=0, max_value=10, value=1)
d = st.slider("Ordre de la différenciation (d)", min_value=0, max_value=10, value=1)
q = st.slider("Ordre de la moyenne mobile (q)", min_value=0, max_value=10, value=1)
P = st.slider("Ordre de l'autorégression saisonnière (P)", min_value=0, max_value=10, value=0)
D = st.slider("Ordre de la différenciation saisonnière (D)", min_value=0, max_value=10, value=1)
Q = st.slider("Ordre de la moyenne mobile saisonnière (Q)", min_value=0, max_value=10, value=0)
s = st.slider("Période saisonnière (s)", min_value=1, max_value=36, value=12)

# Entraînement du modèle SARIMA
model =sm.tsa.SARIMAX(df_monthly_mean, order=(p,d,q), seasonal_order=(P,D,Q,s), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit()

# Affichage des paramètres du modèle
st.write("## Paramètres du modèle")
st.write(results.summary())

# Prévisions avec curseur
st.write("## Prévisions")
start_index = st.slider("Index de départ", min_value=80, max_value=len(df_monthly_mean)-1, value=len(df_monthly_mean)-12)
end_index = st.slider("Index de fin", min_value=start_index, max_value=len(df_monthly_mean)+260, value=len(df_monthly_mean))
forecast = results.predict(start=start_index, end=end_index, dynamic=True)
data_with_forecast = df_monthly_mean.copy()
data_with_forecast['forecast'] = forecast

# Affichage du graphique de prévision
fig, ax = plt.subplots(figsize=(10,5))
df_monthly_mean.plot(ax=ax, label='Données observées')
forecast.plot(ax=ax, label='Prévisions', alpha=0.7)
ax.set_xlabel('Date')
ax.set_ylabel('Valeur')
ax.set_title('Prévisions SARIMA')
ax.legend()
st.pyplot(fig)