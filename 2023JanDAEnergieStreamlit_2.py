# -*- coding: utf-8 -*-
"""
Created on Thu March 09 11:33:29 2023

@author: France Energie Team 
"""
#####################################################################################################################
######################################## Inport des modules et path  ################################################
#####################################################################################################################

# paths

path = r"C:\Users\Brand\Downloads\_DataScientest\Scripts\Project\eco2mix-regional-cons-def.csv"
path_orig = r"C:\Users\Anwender\Downloads\eco2mix-regional-cons-def.csv"
path_url =r"https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-regional-cons-def/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B"

path_agg_local = r"C:\Users\Brand\Downloads\_DataScientest\Scripts\Project\fe_agg_day.csv"
meteo_path_local = r"C:\Users\Brand\Downloads\_DataScientest\Project\France weather data 2013-01-01 to 2020-12-31.csv"

path_agg = r"fe_agg_day.csv"
meteo_path = r"France weather data 2013-01-01 to 2020-12-31.csv"


# importation 

import pandas as pd 
import numpy as np 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px
import io

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures


import warnings
warnings.filterwarnings('ignore')


#####################################################################################################################
######################################## Import + Traitement Données ################################################
#####################################################################################################################

@st.cache_data
def load_prepare_data():
    ### Importation des données
    dfx = pd.read_csv(path, sep=';', index_col=0)
    # dfx = pd.read_csv(path_url, sep=';') # takes way too long
    
    # données supplémentaires
    dfx['Date'] = pd.to_datetime(dfx['Date'])
    dfx['Année']=dfx['Date'].dt.year
    dfx['Mois']=dfx['Date'].dt.month
    # suppression des données
    dfx = dfx[(dfx['Année'] != 2021) &(dfx['Année'] != 2022)]
    dfx.drop(columns=['Nature','Column 30','Date - Heure', 'Stockage batterie','Déstockage batterie','Eolien terrestre', 'Eolien offshore', 'TCO Thermique (%)','TCO Nucléaire (%)','TCO Eolien (%)','TCO Solaire (%)','TCO Hydraulique (%)','TCO Bioénergies (%)'], inplace=True)
    # Correction MW  MWh
    columnnum = len(dfx.columns)-2
    dfx.iloc[:, 4:columnnum] = dfx.iloc[:, 4:columnnum] / 2
    # traitement données manquantes
    medians = dfx.median()
    dfx = dfx.fillna(medians)
    return dfx

@st.cache_data
def load_meteo_data():
    france_city_meteo = pd.read_csv(meteo_path, sep = ",")
    return france_city_meteo

@st.cache_data
def aggregate_by_day(df):
    return df.groupby(['Date'], as_index=False).sum()

@st.cache_data
def scaler(X_train, X_test):
    sc = StandardScaler()
    X_trainS = sc.fit_transform(X_train)
    X_testS = sc.transform(X_test)
    return X_trainS, X_testS 




#####################################################################################################################
######################################## Création des testsets ######################################################
#####################################################################################################################


# function to load data
# df = load_prepare_data() #commented out because now using preprocessed data
# aggregation actually by day not by month (all of france gets summed up)
# df_jour = aggregate_by_day(df)  #commented out because now using preprocessed data

# Loading of preprocessed data
df_jour = pd.read_csv(path_agg, sep=';', index_col=0)

#Definition des variable categorielle et numerique
df_cat = df_jour.select_dtypes(include=['object'])
df_num = df_jour.select_dtypes(include=[ 'float'])
X, y = df_num.drop('Consommation (MW)', axis=1), df_jour['Consommation (MW)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle = False)

# Transormation des données par Standardisation

X_trainS, X_testS = scaler(X_train, X_test)

#####################################################################################################################
######################################## Streamlit  #################################################################
#####################################################################################################################

### variables disponibles
### df = base de données pour tous les régions et chaque 30 minutes
### df_mois = based de données aggregé par jour et sur toutes les régions

st.title('Projet Souverainité Énergétique')
st.sidebar.title("Sommaire")
pages = ["Intro", "Dataviz", "Modélisation - JDD", "Modélisation - Temperature", "Modélisation - Series Temporelle"]
page = st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.title("Introduction")
    st.write("Energie")
    st.image("energie.jpg", use_column_width=True)
    st.write("Non-aggregated dataframe")
    #st.dataframe(df.head())
    
    st.write("Aggregated dataframe")
    st.dataframe(df_jour.head())
          
    # if st.checkbox('Afficher les valeurs manquantes'):
    #    st.dataframe(df.isna().sum())

elif page == pages[1]:
    st.title("Data Visualisation")

    fig = plt.figure()
    #sns.boxplot(x = "Mois", y="", data = df_jour)
    df_jour["yMonth"] = df_jour["Année"] + df_jour["Mois"]/12
    df_mois = df_jour.groupby(["yMonth"]).sum()
    plt.plot(df_mois.index, df_mois["Solaire (MW)"])
    plt.title("Dévelopement de l'énergie solaire")
    plt.xlabel("Temps")
    plt.ylabel("Production MWh Mensuelle")
    st.pyplot(fig)


elif page == pages[2] : 
    
    st.write("Modélisation")
    
    model_choisi = st.selectbox(label = "Choix de mon modèle", options = ['Linear Regression', 'RidgeCV', 'Lasso', 'Decision Tree', 'Random Forest'])

    if model_choisi == 'Linear Regression' :
        regressor = LinearRegression() #pas de hyperparamètres disponibles
    elif model_choisi == 'Decision Tree' :
        prof = st.slider("Profondeur de l'arbre", min_value=0, max_value=20, value=10)
        regressor = DecisionTreeRegressor(random_state=42, max_depth=prof) 
    elif model_choisi == 'Random Forest' :
        num = st.slider("Nombres d'arbres", min_value=0, max_value=100, value=10,step = 10)
        prof = st.slider("Profondeur de l'arbre", min_value=0, max_value=20, value=10)
        regressor = RandomForestRegressor(random_state=42, n_estimators=num, max_depth=prof, n_jobs=-1)  
    elif model_choisi == 'RidgeCV' :
        regressor = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100)) # alpha = 0 égale régression linéaire
    elif model_choisi == 'Lasso' :
        regressor = Lasso(alpha=1) # alpha = 0 égale régression linéaire

    def regression_output():
        # Entraîner le modèle de régression linéaire
        regressor.fit(X_trainS, y_train)
    
        # Faire des prédictions sur l'ensemble de test
        y_pred_train = regressor.predict(X_trainS)
        y_pred_test = regressor.predict(X_testS)

        # Évaluer les performances du modèle
        st.markdown('**Train**')
        st.write('Coefficient de détermination du modèle:', round(regressor.score(X_trainS, y_train),2))
        st.write('Mean_squared_error:', round(mean_squared_error(y_pred_train, y_train),2))
        st.write("Mean absolute error: ", round(mean_absolute_error(y_pred_train, y_train),2))
        st.write("R-squared: ", round(r2_score(y_train, y_pred_train),2))

        st.markdown('**Test**')
        st.write('Coefficient de détermination du modèle:', round(regressor.score(X_testS, y_test),2))
        st.write('Mean_squared_error:', round(mean_squared_error(y_pred_test, y_test),2))
        st.write("Mean absolute error: ", round(mean_absolute_error(y_pred_test, y_test),2))
        st.write("R-squared: ", round(r2_score(y_test, y_pred_test),2))

    
        # Coefficients si régression linéaire
        if model_choisi in ['Linear Regression','RidgeCV', 'Lasso']:
            fig = plt.figure(figsize = (7,7))
            coeffs = list(regressor.coef_)
            coeffs.insert(0, regressor.intercept_)
            feats2 = list(X.columns)
            feats2.insert(0, 'intercept')
            df_val = pd.DataFrame({'valeur_estimee': coeffs}, index=feats2)
            df_val.sort_values(by = 'valeur_estimee', ascending= True, inplace=True)
            plt.barh(df_val.index, df_val.valeur_estimee)
            plt.title('Coefficients de la régression linéaire')
            st.pyplot(fig)

        
        # Feature importance si pas régression linéaire
        if model_choisi in ['Decision Tree', 'Random Forest']:
            fig = plt.figure(figsize = (7,7))
            feat_importances = pd.DataFrame(regressor.feature_importances_, index=X_train.columns, columns=["Importance"])
            feat_importances.sort_values(by='Importance', ascending=True, inplace=True)
            plt.barh(feat_importances.index, feat_importances.Importance)
            plt.title('Facteur d importance des variables')
            st.pyplot(fig)
            
            if model_choisi == 'Decision Tree':
                st.write("Decision Tree")
                st.write("Depth used: ", regressor.tree_.max_depth)
                orig_depth = regressor.tree_.max_depth
                # Estimation de la courbe de l'accuracy
                score_train = []
                score_test = []
                for depth in range(1,orig_depth+1):
                    tregressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
                    tregressor.fit(X_trainS, y_train)
                    score_train.append(tregressor.score(X_trainS, y_train))
                    score_test.append(tregressor.score(X_testS, y_test))

                fig = plt.figure(figsize = (15,15))
                score_df = pd.DataFrame({'number_splits': range(1,orig_depth+1), 
                                         'score_train': score_train, 
                                         'score_test': score_test})
                plt.plot(score_df.number_splits, score_df.score_train, label='Train', color='red') 
                plt.plot(score_df.number_splits, score_df.score_test, label='Test', color='blue')
                plt.legend()
                plt.title("Accuracy en fonction de la profondeur de l'arbre")
                plt.xlabel("Number of splits")
                plt.xlabel("Accuracy")
                st.pyplot(fig)
                
                fig = plt.figure(figsize = (15,15))
                plot_tree(regressor, feature_names=X_train.columns, max_depth=2, filled=True)
                plt.title('Decision Tree')
                st.pyplot(fig)


            if model_choisi == 'Random Forest':
                st.write("Random Forest")
                st.write("Depth used: ", len(regressor.estimators_))
                orig_depth = len(regressor.estimators_)
                # Estimation de la courbe de l'accuracy
                score_train = []
                score_test = []
                for num in range(1, 100, 10):
                    tregressor = RandomForestRegressor(n_jobs=-1, n_estimators=num, random_state=42)
                    tregressor.fit(X_trainS, y_train)
                    score_train.append(tregressor.score(X_trainS, y_train))
                    score_test.append(tregressor.score(X_testS, y_test))

                fig = plt.figure(figsize = (15,15))
                score_df = pd.DataFrame({'number_estimators': range(1,100, 10), 
                                         'score_train': score_train, 
                                         'score_test': score_test})
                plt.plot(score_df.number_estimators, score_df.score_train, label='Train', color='red') 
                plt.plot(score_df.number_estimators, score_df.score_test, label='Test', color='blue')
                plt.legend()
                plt.title("Accuracy en fonction du nombre d'arbres")
                plt.xlabel("Number of Trees")
                plt.xlabel("Accuracy")
                st.pyplot(fig)
                

        # Plot a graph
        fig = plt.figure(figsize = (7,7))
        plt.scatter(y_pred_test, y_test, c='green')
        plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
        
        plt.xlabel("Prediction")
        plt.ylabel("Vrai valeur")
        plt.title('Comparaison prediction vrai valeur')
        plt.legend()
        st.pyplot(fig)
        return "Success"

    regression_output()   


elif page == pages[3] :
    #st.write("Temperature")
    order_options = ["Lineaire", "Polynomiale"]
    order_selection = st.radio("Ordre de la regréssion", order_options)

    ### loading the meteorological data
    france_city_meteo = load_meteo_data()
    france_city_meteo.datetime = pd.to_datetime(france_city_meteo.datetime)
    france_city_meteo = france_city_meteo.set_index("datetime")

    ### Info
    if st.checkbox('Information sur les données'):
        buffer = io.StringIO()
        france_city_meteo.info(buf=buffer)
        st.text(buffer.getvalue())
        st.dataframe(france_city_meteo.head(), use_container_width = True)

    ### dfs separés par ville
    france_city_meteo["month"] = france_city_meteo.index.month

    df_paris = france_city_meteo[france_city_meteo["name"]=="Paris"]
    df_marseille = france_city_meteo[france_city_meteo["name"]=="Marseille"]
    df_bordeaux = france_city_meteo[france_city_meteo["name"]=="Bordeaux"]
    df_strasbourg = france_city_meteo[france_city_meteo["name"]=="Strassbourg"]
    df_nantes = france_city_meteo[france_city_meteo["name"]=="Nantes"]
    
    # nouveau df avec les temperatures moyennes par jour
    avg_temp = [[time, *temp_tup,round(np.array(temp_tup).mean(),2)] for time, temp_tup in 
                zip(df_paris.index, zip(df_paris.temp, df_marseille.temp, df_nantes.temp, df_strasbourg.temp, df_bordeaux.temp))] 
    colnams = ["Time","Paris", "Marseile","Nantes","Strasbourg", "Bordeaux", "Average"]
    df_x = pd.DataFrame(avg_temp, columns=colnams)
    df_x = df_x.set_index("Time")  
    
    # nouveau df avec les temperatures moyennes par mois
    df_x["yMonth"] = np.round(df_x.index.year + (df_x.index.month-1)/12,2)
    df_x_ymonth = df_x.groupby(by = "yMonth").mean()
    
    
    df_energy = df_jour.copy()
    # maybe not needed
    df_energy["Date"] = pd.to_datetime(df_energy["Date"])
    df_energy = df_energy.set_index("Date")
    # new time index and calculation of monthly energy need
    df_energy["yMonth"] = np.round(df_energy.index.year + (df_energy.index.month-1)/12,2)
 
    df_mconsumption = df_energy.groupby(by = "yMonth")[["Consommation (MW)"]].sum()
    # creation on new df    
    df_x_ymonth = pd.concat([df_x_ymonth, df_mconsumption], axis=1)
 
    if order_selection == "Lineaire":
        # graph
        plt.figure(figsize=(6,6));
        fig = sns.lmplot(x="Average", y="Consommation (MW)", data=df_x_ymonth, fit_reg = True, height=10, aspect=1);
        plt.xlabel("Température moyenne journalière");
        plt.ylabel("Consommation MWh");
        plt.title("Temperature Consommation Régression Linéaire");
        st.pyplot(fig)

        # statistiques
        model = sm.OLS(df_x_ymonth["Consommation (MW)"], df_x_ymonth.Average).fit()
        ypred = model.predict(df_x_ymonth.Average) 
        st.write("Récapitulatif régression linéaire: \n")
        st.write(model.summary())

    if order_selection == "Polynomiale":
        # graph
        order_degree = st.slider("Polynomal-Degree", min_value=2, max_value=20, value=2)
        plt.figure(figsize=(6,6));
        fig = sns.lmplot(x="Average", y="Consommation (MW)", data=df_x_ymonth, fit_reg = True, order = order_degree, height=10, aspect=1);
        plt.xlabel("Température moyenne journalière");
        plt.ylabel("Consommation MWh");
        plt.title("Temperature Consommation Régression Quadratique");
        st.pyplot(fig)

        # quadratic regression

        # reshaping necessary see link below  https://stackoverflow.com/questions/51150153/valueerror-expected-2d-array-got-1d-array-instead
        X = np.array(df_x_ymonth.Average).reshape(-1, 1)

        from sklearn.preprocessing import PolynomialFeatures
        poly2 = PolynomialFeatures(degree= order_degree, include_bias=False)
        xp = poly2.fit_transform(X)

        # information  https://towardsdatascience.com/polynomial-regression-in-python-dd655a7d9f2b

        model = sm.OLS(df_x_ymonth["Consommation (MW)"], xp).fit()
        ypred = model.predict(xp) 

        st.write("Récapitulatif de la régression pronominale quadratique:\n")
        st.write(model.summary())


### Next steps : Show table with data to see it has been correctly loaded - done
### make one of the regression functions work - done
### create a mockup of the windows - done
### put the different regression functions into the different panels - done
### incorporate hyperparameters - done
### improve execution speed - done
### incorporate interactivity - done

### incorporate the modelisation of the temperature - done
### check all the graphs


### deploy in the Cloud / Github

# creation données supplémentaires
# df['Energie verte'] = df['Eolien (MW)']+df['Solaire (MW)']+df['Hydraulique (MW)']+df['Bioénergies (MW)']
# df['Production totale'] = df['Thermique (MW)']+df['Nucléaire (MW)']-df['Pompage (MW)']+df['Eolien (MW)']+df['Solaire (MW)']+df['Hydraulique (MW)']+df['Bioénergies (MW)']
# df["TCH Total"] = (df["TCH Thermique (%)"] + df["TCH Nucléaire (%)"] +
#    df["TCH Eolien (%)"] + df["TCH Solaire (%)"] + df["TCH Hydraulique (%)"] +
#    df["TCH Bioénergies (%)"])

# données supplémentaires pour Anova
# df['ConsommationMW'] = df['Consommation (MW)']
# df['ProductionTotale'] = df['Production totale']
# df['NucleaireMW'] = df['Nucléaire (MW)']
# df['HydrauliqueMW'] = df['Hydraulique (MW)']
# df['EnergieVerte'] = df['Energie verte']

#suppression des colonnes créées pour Anova
# df.drop(columns=['ConsommationMW','ProductionTotale','NucleaireMW', 'HydrauliqueMW',
#                 'EnergieVerte'], inplace = True)

# conversion Date en datetime - déjà fait
# df["Date"] = pd.to_datetime(df["Date"])

#suppression des colonnes autocorrélées
# df =df.drop('Production totale', axis = 1)
# df = df.drop('Energie verte', axis = 1)