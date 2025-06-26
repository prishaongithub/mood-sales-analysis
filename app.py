import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="MOODSpend Insights", layout="wide")
st.title("🧠 MOODSpend: Customer Behavior Analysis")
st.caption("🔍 Upload your dataset and explore spending predictions.")

# --- Load Models ---
spend_model = joblib.load("spend_amount_model.pkl")
category_model = joblib.load("category_model.pkl")
birthday_model = joblib.load("birthday_model.pkl")
category_encoder = joblib.load("label_encoder_category.pkl")

# --- Helper ---
def is_near_birthday(transaction_date, birthdate):
    bday_this_year = birthdate.replace(year=transaction_date.year)
    delta = (bday_this_year - transaction_date).days
    return 0 <= delta <= 15

# --- Section 1: Upload + EDA ---
st.header("📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV to explore", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.subheader("📈 Data Preview")
    st.dataframe(df_uploaded.head())

    with st.expander("📊 Explore Dataset"):
        num_cols = df_uploaded.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df_uploaded.select_dtypes(include='object').columns.tolist()

        if num_cols:
            col = st.selectbox("📉 Numerical Column", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df_uploaded[col], kde=True, ax=ax)
            st.pyplot(fig)

        if cat_cols:
            cat = st.selectbox("📊 Categorical Column", cat_cols)
            fig, ax = plt.subplots()
            sns.countplot(data=df_uploaded, x=cat, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.markdown("---")

# --- Predictor 1: Spend Amount ---
with st.expander("💸 Predict Spend Amount", expanded=False):
    st.subheader("Enter Details:")
    with st.form("spend_form"):
        age1 = st.slider("Age", 18, 70, 30)
        gender1 = st.selectbox("Gender", ["Male", "Female"], key="gender1")
        mood1 = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy", "Low Energy"], key="mood1")
        season1 = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"], key="season1")
        weather1 = st.selectbox("Weather", ["Sunny", "Cloudy", "Rain", "Foggy", "Snow", "Clear", "Humid", "Windy", "Overcast"], key="weather1")
        birthdate1 = st.date_input("Birthdate", datetime(2000, 1, 1), key="birthdate1")
        transaction_date1 = st.date_input("Transaction Date", datetime.today(), key="transdate1")
        city1 = st.selectbox("City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Seattle', 'Miami', 'Denver', 'Boston', 'Atlanta'], key="city1")
        submit1 = st.form_submit_button("Predict 💸")

    if submit1:
        near_birthday1 = 1 if is_near_birthday(transaction_date1, birthdate1) else 0
        df_spend = pd.DataFrame({
            "Customer_Age": [age1],
            "Gender": [gender1],
            "Mood": [mood1],
            "Season": [season1],
            "Weather": [weather1],
            "Is_Near_Birthday": [near_birthday1],
            "City": [city1]
        })
        df_spend = pd.get_dummies(df_spend)
        df_spend = df_spend.reindex(columns=spend_model.feature_names_in_, fill_value=0)
        amount = spend_model.predict(df_spend)[0]
        st.success(f"💸 Estimated Spend Amount: ₹{round(amount, 2)}")

# --- Predictor 2: Category ---
with st.expander("📦 Predict Category", expanded=False):
    st.subheader("Enter Details:")
    with st.form("cat_form"):
        age2 = st.slider("Age", 18, 70, 30, key="age2")
        gender2 = st.selectbox("Gender", ["Male", "Female"], key="gender2")
        mood2 = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy", "Low Energy"], key="mood2")
        weather2 = st.selectbox("Weather", ["Sunny", "Cloudy", "Rain", "Foggy", "Snow", "Clear", "Humid", "Windy", "Overcast"], key="weather2")
        city2 = st.selectbox("City", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Seattle', 'Miami', 'Denver', 'Boston', 'Atlanta'], key="city2")
        txn_amount = st.number_input("Transaction Amount", min_value=1.0, value=500.0, key="txn_amount")
        submit2 = st.form_submit_button("Predict Category 📦")

    if submit2:
        df_cat = pd.DataFrame({
            "Customer_Age": [age2],
            "Gender": [gender2],
            "Mood": [mood2],
            "Weather": [weather2],
            "City": [city2],
            "Transaction Amount": [txn_amount],
            "Is_Near_Birthday": [0]
        })
        df_cat = pd.get_dummies(df_cat)
        df_cat = df_cat.reindex(columns=category_model.feature_names_in_, fill_value=0)
        category_code = category_model.predict(df_cat)[0]
        category = category_encoder.inverse_transform([category_code])[0]
        st.success(f"📦 Most Likely Category: **{category}**")

# --- Predictor 3: Birthday Probability ---
with st.expander("🎂 Predict Birthday Spend Probability", expanded=False):
    st.subheader("Enter Details:")
    with st.form("bday_form"):
        age3 = st.slider("Age", 18, 70, 30, key="age3")
        gender3 = st.selectbox("Gender", ["Male", "Female"], key="gender3")
        mood3 = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy", "Low Energy"], key="mood3")
        season3 = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"], key="season3")
        weather3 = st.selectbox("Weather", ["Sunny", "Cloudy", "Rain", "Foggy", "Snow", "Clear", "Humid", "Windy", "Overcast"], key="weather3")
        category3 = st.selectbox("Category Interested", ["Food", "Electronics", "Entertainment", "Personal Care", "Groceries"], key="category3")
        birthdate3 = st.date_input("Customer Birthdate", datetime(2000, 1, 1), key="birthdate3")
        transaction_date3 = st.date_input("Transaction Date", datetime.today(), key="transdate3")
        submit3 = st.form_submit_button("Predict Birthday Probability 🎂")

    if submit3:
        near_birthday3 = is_near_birthday(transaction_date3, birthdate3)

        df_bday = pd.DataFrame({
            "Customer_Age": [age3],
            "Gender": [gender3],
            "Mood": [mood3],
            "Season": [season3],
            "Weather": [weather3],
            "Category": [category3]
        })

        df_bday = pd.get_dummies(df_bday)
        df_bday = df_bday.reindex(columns=birthday_model.feature_names_in_, fill_value=0)
        prob = birthday_model.predict_proba(df_bday)[0][1]

        # 🌟 Boost Logic
        boost = 0
        if gender3 == "Female" and mood3 == "Happy" and near_birthday3:
            boost += 0.25
        if mood3 in ["Relaxed", "Happy"] and transaction_date3.weekday() >= 5 and near_birthday3:
            boost += 0.20
        if weather3 in ["Sunny", "Clear"] and near_birthday3:
            boost += 0.15
        boost = min(boost, 0.45)
        final_prob = min(prob + boost, 1.0)

        st.metric("🎂 Birthday Spend Probability", f"{round(final_prob * 100, 2)}%")

# --- Footer ---
st.markdown("---")
st.caption("🚀 Built with ❤️ by Prisha Arora | Powered by Streamlit")
