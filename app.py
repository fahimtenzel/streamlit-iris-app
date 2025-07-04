# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load model
model = joblib.load('iris_model.pkl')

# Load Iris dataset for labels and reference
iris = load_iris()
target_names = iris.target_names
feature_names = iris.feature_names

st.title("🌼 Iris Flower Classifier")
st.write("Enter flower measurements to predict its species:")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"🌸 Predicted Species: **{target_names[prediction]}**")

    # Visualize input against the dataset
    df = pd.DataFrame(iris.data, columns=feature_names)
    df['species'] = pd.Series([target_names[i] for i in iris.target])
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', ax=ax)
    ax.scatter(petal_length, petal_width, color='black', marker='X', s=100, label='Your Input')
    ax.legend()
    st.pyplot(fig)
