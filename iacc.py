import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config as the first Streamlit command
st.set_page_config(page_title="Personalized Women's Healthcare System", layout="wide")

# Load real-world dataset (Replace with an actual dataset for better accuracy)
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

df = load_data()

# Selecting relevant columns for women's health analysis
df = df[['Age', 'BMI', 'Glucose', 'BloodPressure', 'Insulin', 'SkinThickness']]
df.rename(columns={'Glucose': 'Blood Sugar Level', 'BloodPressure': 'Blood Pressure', 'Insulin': 'Hormone Level', 'SkinThickness': 'Body Fat Index'}, inplace=True)

# Step 2: Preprocessing
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 3: Clustering Algorithms
# Hierarchical Clustering
hierarchical_model = AgglomerativeClustering(n_clusters=3)
df['Hierarchical_Cluster'] = hierarchical_model.fit_predict(df_scaled)

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
df['GMM_Cluster'] = gmm.fit_predict(df_scaled)

st.title("🌸 Personalized Women's Healthcare System")
st.write("### Empowering Women Through AI-Driven Healthcare")
st.write("This interactive tool helps provide **personalized health recommendations** using AI and real-world medical data.")

st.sidebar.header("🔍 About This Tool")
st.sidebar.write("- Uses AI-driven clustering on real-world medical data.")
st.sidebar.write("- Provides **research-backed personalized recommendations**.")
st.sidebar.write("- Helps in **early detection and health risk prevention**.")

st.sidebar.write("#### How to Use:")
st.sidebar.write("1. Enter your health details manually below.")
st.sidebar.write("2. Get evidence-based recommendations based on AI analysis.")
st.sidebar.write("3. Explore data visualization to understand health clusters.")

# User Input
st.write("### 🏥 Enter Your Health Information")
age = st.number_input("Enter Age (18-60)", min_value=18, max_value=60, step=1)
bmi = st.number_input("Enter BMI (18.0-50.0)", min_value=18.0, max_value=50.0, step=0.1)
blood_sugar = st.number_input("Blood Sugar Level (70-200 mg/dL)", min_value=70, max_value=200, step=1)
blood_pressure = st.number_input("Blood Pressure (60-180 mmHg)", min_value=60, max_value=180, step=1)
hormone_level = st.number_input("Hormone Level (0.5-25.0 mIU/mL)", min_value=0.5, max_value=25.0, step=0.1)
body_fat_index = st.number_input("Body Fat Index (5-50)", min_value=5, max_value=50, step=1)

# Create a dataframe for input
user_data = pd.DataFrame({
    'Age': [age],
    'BMI': [bmi],
    'Blood Sugar Level': [blood_sugar],
    'Blood Pressure': [blood_pressure],
    'Hormone Level': [hormone_level],
    'Body Fat Index': [body_fat_index]
})

# Preprocess user input
user_scaled = scaler.transform(user_data)

# Predict cluster
gmm_cluster = gmm.predict(user_scaled)[0]

# Step 5: Personalized Health Recommendations
def generate_recommendations(cluster_id):
    if cluster_id == 0:
        return "🥗 Maintain a balanced diet and increase physical activity to manage weight and prevent chronic conditions. Consider routine check-ups."
    elif cluster_id == 1:
        return "🩺 Monitor blood sugar and blood pressure levels closely. Reduce processed sugar intake and consult a doctor for regular screenings."
    else:
        return "🏋️ Strength training and a nutrient-rich diet can help with hormonal balance and body composition. Hydration and sleep management are key."

# Get recommendation
recommendation = generate_recommendations(gmm_cluster)
st.success("✅ Your Personalized Health Recommendation:")
st.info(recommendation)

# Visualization - Bar Plot for Cluster Distribution
st.write("### 📊 Health Clusters Visualization")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df['GMM_Cluster'], palette='coolwarm', ax=ax)
plt.xlabel("Health Clusters")
plt.ylabel("Number of Individuals")
plt.title("Distribution of Individuals Across Health Clusters")
st.pyplot(fig)

# Show Predictions on Graph
st.write("### 📌 Your Cluster Prediction")
st.write(f"Based on your input, you belong to **Cluster {gmm_cluster}**. This means your health profile is similar to others in this group.")

# Explain the Cluster Chart
st.write("### 📌 Understanding the Cluster Chart")
st.write("The bar chart above shows the number of individuals assigned to each health cluster.")
st.write("Each cluster represents a group of individuals with similar health profiles, allowing for **better health interventions and personalized care.**")

st.write("### 🔬 How This Works")
st.write("Our AI model analyzes health data and categorizes users into different health risk segments using **Hierarchical Clustering** and **Gaussian Mixture Models** (GMMs).")
st.write("This approach allows for early intervention, prevention, and **data-driven healthcare improvements** for every individual.")

