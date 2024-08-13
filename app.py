import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def generate_data(a, c, n):
    np.random.seed(42)  # for reproducibility
    x = np.linspace(-100, 100, n)
    noise = c * np.random.normal(0, 1, n)
    y = a * x + 50 + noise
    return x, y

def plot_regression(a, c, n):
    x, y = generate_data(a, c, n)
    x = x.reshape(-1, 1)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(x, y)
    
    # Make predictions
    y_pred = model.predict(x)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Plot data points and regression line
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data points')
    ax.plot(x, y_pred, color='red', label='Regression line')
    ax.set_title(f'Linear Regression\na={a}, c={c}, n={n}\nMSE={mse:.2f}, R2={r2:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    st.pyplot(fig)

# Streamlit sliders
st.title("Linear Regression with Interactive Sliders")
a = st.slider('a', -10.0, 10.0, 1.0, 0.1)
c = st.slider('c', 0.0, 100.0, 10.0, 1.0)
n = st.slider('n', 100, 1000, 100, 10)

plot_regression(a, c, n)

