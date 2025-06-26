import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 🔧 Page Setup
st.set_page_config(page_title="Customer Spend Intelligence", layout="wide")
st.title("🛍️ Customer Spending Intelligence Dashboard")
st.write("Upload your dataset or enter customer details to generate insights and predictions.")

# 🔹 Load Models
spend_model = joblib.load("spend_amount_model.pkl")      # RandomForestRegressor
category_model = joblib.load("category_model.pkl")       # XGBoost Classifier
birthday_model = joblib.load("birthday_model.pkl")       # Logistic Regression

# 🔸 Upload Section + EDA
st.subheader("📂 Upload Your Own Dataset (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file to explore and visualize your data", type="csv")

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.write("### 🧾 Data Preview", df_uploaded.head())

    st.write("Rows:", df_uploaded.shape[0], " | Columns:", df_uploaded.shape[1])
    st.write("### 📊 Statistical Summary")
    st.write(df_uploaded.describe())

    # Auto Visualizations
    num_cols = df_uploaded.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_uploaded.select_dtypes(include='object').columns.tolist()

    if num_cols:
        selected_num = st.selectbox("Numerical column for distribution plot", num_cols)
        fig1, ax1 = plt.subplots()
        sns.histplot(df_uploaded[selected_num], kde=True, ax=ax1)
        st.pyplot(fig1)

    if cat_cols:
        selected_cat = st.selectbox("Categorical column for count plot", cat_cols)
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df_uploaded, x=selected_cat, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    st.markdown("---")

# 🔮 Prediction Section
st.subheader("🤖 Predict Customer Spending Behavior")

with st.form("input_form"):
    st.markdown("### 🧍 Enter Customer Information")
    age = st.slider("Customer Age", 15, 75, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Relaxed", "Lazy"])
    season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Spring"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Foggy"])
    category_input = st.selectbox("Likely Category (for birthday model)", ["Apparel", "Food", "Electronics", "Entertainment"])
    city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Other"])
    is_near_birthday = st.selectbox("Is it near their birthday?", ["Yes", "No"])
    submit = st.form_submit_button("Predict")

if submit:
    is_birthday = 1 if is_near_birthday == "Yes" else 0

    # -------- Model 1: Spend Amount Prediction --------
    df_spend = pd.DataFrame({
        "Customer_Age": [age],
        "Gender": [gender],
        "Mood": [mood],
        "Season": [season],
        "Weather": [weather],
        "Is_Near_Birthday": [is_birthday]
    })
    df_spend = pd.get_dummies(df_spend)
    df_spend = df_spend.reindex(columns=spend_model.feature_names_in_, fill_value=0)
    predicted_amount = spend_model.predict(df_spend)[0]

    # -------- Model 2: Category Prediction --------
    df_cat = pd.DataFrame({
        "Customer_Age": [age],
        "Gender": [gender],
        "Mood": [mood],
        "Weather": [weather],
        "Is_Near_Birthday": [is_birthday],
        "Transaction Amount": [predicted_amount],
        "City": [city]
    })
    df_cat = pd.get_dummies(df_cat)
    df_cat = df_cat.reindex(columns=category_model.feature_names_in_, fill_value=0)
    predicted_category = category_model.predict(df_cat)[0]

    # -------- Model 3: Birthday Spending Probability --------
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
    birthday_prob = birthday_model.predict_proba(df_bday)[0][1]

    # -------- Display Results --------
    st.subheader("🧠 Prediction Results")
    st.success(f"💸 Estimated Spend Amount: ₹{round(predicted_amount, 2)}")
    st.info(f"📦 Likely Purchase Category: {predicted_category}")
    st.warning(f"🎂 Probability of Spending Near Birthday: {round(birthday_prob * 100, 2)}%")
