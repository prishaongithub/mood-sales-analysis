# 🎯 MOODSpend Insights

**MOODSpend** is an intelligent customer behavior analysis and prediction tool built using Python, Streamlit, and machine learning. It analyzes a customer’s **mood, weather, season, city, age**, and more to **predict spending behavior**, **category preference**, and **birthday-based shopping probability**.

---

## 📂 Dataset Overview

The core dataset (`credit_card_transaction.csv`) includes:
- Customer details (Name, Gender, Birthdate)
- Transaction info (Amount, Date, Category)
- Geographic location

### 🔄 Data Cleaning & Transformation:
- 🧹 Cleaned missing gender using `gender-guesser` on first names  
- 📅 Converted `Date` and `Birthdate` to datetime  
- 🎂 Computed `Customer_Age` and `Is_Near_Birthday` (if within 15 days)  
- 📍 Randomly assigned realistic `City` values for simulation  
- 🌤️ Retrieved or simulated `Weather` using API / logic per `Season`  
- 😄 Derived `Mood` based on `Weather`, `Weekday`, and `Season`  
- 🧼 Removed outliers and ensured clean categorical groupings (e.g., `Other` for rare `Category`)

---

## 🤖 Machine Learning Models

### 1. 🔢 **Spend Amount Prediction (Regression)**
- **Model:** Random Forest Regressor
- **Target:** `Transaction Amount`
- **Features:** Age, Gender, Mood, Season, Weather, Near Birthday

### 2. 🧠 **Category Prediction (Classification)**
- **Model:** Random Forest Classifier / XGBoost
- **Target:** `Category`
- **Features:** Age, Gender, Mood, Weather, Transaction Amount, City, Near Birthday

### 3. 🎂 **Birthday Spend Probability (Classification)**
- **Model:** Logistic Regression
- **Target:** `Is_Near_Birthday`
- **Features:** Age, Gender, Mood, Season, Weather, Category

Each model is trained, evaluated, and exported using `joblib` for real-time inference in Streamlit.

---

## 📺 Streamlit App Functionality

The `app.py` file powers a full-stack interactive dashboard.

### 🔮 Smart Prediction Engine
Users choose **what they want to predict**:
1. **💸 Predict Spend Amount**  
   ➤ Inputs: Age, Gender, Mood, Season, Weather, Transaction Date, Birthdate, City  
   ➤ Output: Estimated spend (₹)

2. **📦 Predict Spend Category**  
   ➤ Inputs: Age, Gender, Mood, Weather, Transaction Amount, City, Birthdate  
   ➤ Output: Likely product category

3. **🎂 Predict Birthday Probability**  
   ➤ Inputs: Age, Gender, Mood, Season, Weather, Category, Transaction Date, Birthdate  
   ➤ Output: Probability of birthday-based purchase

### 🎯 Smart Boosting Logic
To enhance birthday prediction accuracy, custom boosts were applied:
- +25%: Female + Happy + Near Birthday  
- +15%: Relaxed/Happy on Weekend near birthday  
- +10%: Sunny/Clear Weather near birthday  
- +10%: Category = Electronics or Personal Care near birthday

### 📊 Upload & EDA
- Upload your own CSV file  
- Preview dataset  
- Plot **numerical histograms** or **categorical bar plots**

### 🧠 Model Integration
- All models are pre-trained and loaded with `joblib`  
- Feature alignment ensured using `model.feature_names_in_`  
- Category decoded using `LabelEncoder` used during training

---

## ⚙️ How to Run

1. **Clone this repo**  
   ```bash
   git clone https://github.com/prishaongithub/mood-sales-analysis.git
   cd mood-sales-analysis
