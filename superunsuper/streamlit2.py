import streamlit as st
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load('financial_model.pkl')

# Charger les noms des colonnes sauvegardées pendant l'entraînement
feature_columns = joblib.load('feature_columns.pkl')  # Assurez-vous d'avoir sauvegardé ces colonnes au préalable

st.title("Financial Inclusion Prediction App")

# Définir les champs d'entrée pour l'utilisateur
country = st.selectbox("Country", ["Kenya", "Uganda", "Tanzania", "Rwanda"])
location_type = st.selectbox("Location Type", ["Rural", "Urban"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1, step=1)
age_of_respondent = st.number_input("Age", min_value=10, step=1)
gender_of_respondent = st.selectbox("Gender", ["Male", "Female"])
relationship_with_head = st.selectbox("Relationship with Head", ["Spouse", "Head of Household", "Child", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widowed"])
education_level = st.selectbox("Education Level", ["No formal education", "Primary education", "Secondary education", "Tertiary education"])
job_type = st.selectbox("Job Type", ["Self employed", "Government Dependent", "Informally employed", "Formally employed Private"])

# Bouton pour prédire
if st.button("Predict"):
    # Préparer les données pour la prédiction
    input_data = pd.DataFrame({
        'country': [country], 'location_type': [location_type],
        'cellphone_access': [cellphone_access], 'household_size': [household_size],
        'age_of_respondent': [age_of_respondent], 'gender_of_respondent': [gender_of_respondent],
        'relationship_with_head': [relationship_with_head], 'marital_status': [marital_status],
        'education_level': [education_level], 'job_type': [job_type]
    })
    
    # Encoder les données en utilisant pd.get_dummies pour correspondre aux colonnes d'entraînement
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)

    # Réaligner les colonnes avec celles utilisées pendant l'entraînement
    input_data_encoded = input_data_encoded.reindex(columns=feature_columns, fill_value=0)

    # Effectuer la prédiction
    prediction = model.predict(input_data_encoded)
    result = "Has Bank Account" if prediction[0] == 1 else "No Bank Account"
    
    # Afficher le résultat dans la barre latérale
    st.sidebar.write(f"Prediction: {result}")
