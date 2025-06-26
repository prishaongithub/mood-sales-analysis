import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="MOODSpend Insights", layout="wide")
st.title("🎯 Customer Behaviour Predictor")
st.markdown("Predict customer spending behavior based on mood, season, weather, and more.")

# Load models
spend_model = joblib.load("spend_amount_model.pkl")
category_model = joblib.load("category_model.pkl")
birthday_model = joblib.load("birthday_model.pkl")
category_encoder = joblib.load("label_encoder_category.pkl")

# Helper: Check if near birthday
def is_near_birthday(transaction_date, birthdate):
    bday_this_year = birthdate.replace(year=transaction_date.year)
    delta = (bday_this_year - transaction_date).days
    return 0 <= delta <= 15

# --- Upload Data Preview Section ---
st.sidebar.header("📂 Upload CSV (Optional)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.subheader("📈 Data Preview")
    st.dataframe(df_uploaded.head())
    st.markdown("---")

    st.subheader("📊 Explore Data")
    num_cols = df_uploaded.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_uploaded.select_dtypes(include='object').columns.tolist()

    if num_cols:
        col = st.selectbox("Numerical Column for Histogram", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df_uploaded[col], kde=True, ax=ax)
        st.pyplot(fig)

    if cat_cols:
        cat = st.selectbox("Categorical Column for Count Plot", cat_cols)
        fig, ax = plt.subplots()
        sns.countplot(data=df_uploaded, x=cat, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    st.markdown("---")

# --- Prediction Selection ---
st.subheader("🧠 Select Prediction Type")
task = st.radio("What would you like to predict?", ["💸 Spend Amount", "📦 Category", "🎂 Birthday Spend Probability"])

# --- Dynamic Input Section ---
with st.form("prediction_form"):
    age = st.slider("Customer Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy", "Low Energy"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rain", "Foggy", "Snow", "Clear", "Humid", "Windy", "Overcast"])

    # Common input for tasks 1 & 3
    if task in ["💸 Spend Amount", "🎂 Birthday Spend Probability"]:
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])

    if task in ["💸 Spend Amount", "📦 Category"]:
        city = st.selectbox("City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Seattle', 'Miami', 'Denver', 'Boston', 'Atlanta'])

    # Birthday + Spend Amount always require these
    transaction_date = st.date_input("Transaction Date", datetime.today())
    birthdate = st.date_input("Customer Birthdate", datetime(2000, 1, 1))

    if task == "🎂 Birthday Spend Probability":
        category_input = st.selectbox("Product Category", ["Food", "Electronics", "Entertainment", "Personal Care", "Groceries"])

    if task == "📦 Category":
        transaction_amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=100.0)

    submit = st.form_submit_button("🎯 Predict")

# --- Perform Prediction ---
if submit:
    st.subheader("📌 Results")

    near_bday = 1 if is_near_birthday(transaction_date, birthdate) else 0

    if task == "💸 Spend Amount":
        df_spend = pd.DataFrame({
            "Customer_Age": [age],
            "Gender": [gender],
            "Mood": [mood],
            "Season": [season],
            "Weather": [weather],
            "Is_Near_Birthday": [near_bday],
            "City": [city]
        })
        df_spend = pd.get_dummies(df_spend)
        df_spend = df_spend.reindex(columns=spend_model.feature_names_in_, fill_value=0)
        predicted_amount = spend_model.predict(df_spend)[0]
        st.metric("💸 Predicted Spend Amount", f"₹{round(predicted_amount, 2)}")

    elif task == "📦 Category":
        df_cat = pd.DataFrame({
            "Customer_Age": [age],
            "Gender": [gender],
            "Mood": [mood],
            "Weather": [weather],
            "Is_Near_Birthday": [near_bday],
            "Transaction Amount": [transaction_amount],
            "City": [city],
            "Birthdate": [birthdate]  # even if unused in model, kept for boost logic
        })
        df_cat = pd.get_dummies(df_cat)
        df_cat = df_cat.reindex(columns=category_model.feature_names_in_, fill_value=0)
        predicted_category_code = category_model.predict(df_cat)[0]
        predicted_category = category_encoder.inverse_transform([predicted_category_code])[0]
        st.metric("📦 Likely Spending Category", predicted_category)

    elif task == "🎂 Birthday Spend Probability":
        df_bday = pd.DataFrame({
            "Customer_Age": [age],
            "Gender": [gender],
            "Mood": [mood],
            "Season": [season],
            "Weather": [weather],
            "Category": [category_input]
        })
        df_bday = pd.get_dummies(df_bday)
        df_bday = df_bday.reindex(columns=birthday_model.feature_names_in_, fill_value=0)
        base_prob = birthday_model.predict_proba(df_bday)[0][1]

        # --- Boost logic (refined) ---
        boost = 0
        if gender == "Female" and mood in ["Happy", "Relaxed"] and near_bday:
            boost += 0.25
        if weather in ["Sunny", "Clear"] and near_bday:
            boost += 0.15
        if category_input in ["Electronics", "Personal Care", "Entertainment"] and near_bday:
            boost += 0.15
        if transaction_date.weekday() >= 5 and near_bday:
            boost += 0.10
        if mood in ["Happy", "Relaxed"] and season == "Summer":
            boost += 0.05

        boosted_prob = min(base_prob + boost, 1.0)
        st.metric("🎂 Birthday Spend Probability", f"{round(boosted_prob * 100, 2)}%")

st.markdown("---")
st.caption("🚀 Built with ❤️ by Prisha Arora | Powered by Streamlit")
