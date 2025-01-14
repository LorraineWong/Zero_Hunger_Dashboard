import streamlit as st

st.set_page_config(page_title="Zero Hunger Dashboard", layout="wide")

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Mock Data Loading (expand data for better coverage and model training)
@st.cache_data
def load_data():
    countries = [
        "USA", "China", "India", "Brazil", "Nigeria", "Russia", "Australia", "Canada", "Germany", "France",
        "Japan", "Mexico", "Indonesia", "South Africa", "UK", "Argentina", "Egypt", "Thailand", "Vietnam", "Pakistan",
        "Bangladesh", "Turkey", "Italy", "Spain", "Philippines", "Colombia", "South Korea", "Iran", "Poland", "Malaysia"
    ]
    crops = ["Wheat", "Rice", "Maize", "Barley", "Soybean", "Cotton"]
    data = []
    years = range(2000, 2024)  # Generate data for 2000 to 2022
    for country in countries:
        for crop in crops:
            for year in years:
                rainfall = 50 + (hash(country + crop + str(year)) % 2950)  # Random rainfall 50-3000 mm
                temperature = 5 + (hash(crop + country + str(year)) % 25)  # Random temperature 5-30 °C
                yield_tons = 0.5 + (hash(crop + country + str(year)) % 10)  # Random yield 0.5-10 tons/ha
                data.append({
                    'Country': country,
                    'Crop': crop,
                    'Year': year,
                    'Rainfall (mm)': rainfall,
                    'Temperature (°C)': temperature,
                    'Yield (tons/ha)': yield_tons
                })
    return pd.DataFrame(data)

yield_data = load_data()

# Train a Random Forest Model on Mock Data
@st.cache_data
def train_model(data):
    # Use a subset of the data for training
    train_data = data[['Rainfall (mm)', 'Temperature (°C)', 'Yield (tons/ha)']].dropna()
    X = train_data[['Rainfall (mm)', 'Temperature (°C)']]
    y = train_data['Yield (tons/ha)']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Train the model with the entire dataset
rf_model = train_model(yield_data)

# Streamlit App
st.title("🌍 Zero Hunger Dashboard")
st.markdown("<style>.title { font-size: 2.5em; color: #2c3e50; text-align: center; }</style>", unsafe_allow_html=True)
st.markdown("Using data science to predict crop yields and support food security efforts.")

# Sidebar Filters
st.sidebar.header("🔍 Filters")
selected_country = st.sidebar.selectbox("Select Country:", options=yield_data['Country'].unique())
selected_crop = st.sidebar.selectbox("Select Crop:", options=yield_data['Crop'].unique())

# Filter Data Based on Selection
filtered_data = yield_data[(yield_data['Country'] == selected_country) & (yield_data['Crop'] == selected_crop)]

# Main Dashboard Sections
# 1. Summary Section
st.markdown("### 📊 Key Metrics", unsafe_allow_html=True)
metrics_section = st.container()
col1, col2, col3 = metrics_section.columns(3)
if not filtered_data.empty:
    avg_rainfall = filtered_data['Rainfall (mm)'].mean()
    avg_temperature = filtered_data['Temperature (°C)'].mean()
    avg_yield = filtered_data['Yield (tons/ha)'].mean()
    col1.metric("Avg Rainfall (mm)", f"{avg_rainfall:.2f}")
    col2.metric("Avg Temperature (°C)", f"{avg_temperature:.2f}")
    col3.metric("Avg Yield (tons/ha)", f"{avg_yield:.2f}")
else:
    st.warning("No data available for the selected filters.")

# 2. Prediction Section
st.markdown("### 📈 Crop Yield Prediction", unsafe_allow_html=True)

# User Inputs with Manual Entry Fields
st.write("#### Rainfall (mm):")
rainfall_input = st.number_input("Enter Rainfall (mm):", min_value=50, max_value=3000, value=800, step=1, format="%d")
st.write("#### Temperature (°C):")
temperature_input = st.number_input("Enter Temperature (°C):", min_value=5, max_value=30, value=22, step=1, format="%d")

# Predict yield using the trained Random Forest model
input_features = pd.DataFrame({
    'Rainfall (mm)': [rainfall_input],
    'Temperature (°C)': [temperature_input]
})

predicted_yield = rf_model.predict(input_features)[0]

# Display Predicted Yield
st.markdown(f"""
<div style='
    background-color: rgba(50, 168, 82, 0.8); 
    padding: 15px; 
    border-radius: 10px; 
    text-align: center; 
    color: white; 
    font-size: 24px; 
    font-weight: bold;'>
    Predicted Yield: {predicted_yield:.2f} tons/ha
</div>
""", unsafe_allow_html=True)

# Display Input Ranges for Reference
st.write(f"""
    * Valid range for Rainfall: 50 to 3000 mm
    * Valid range for Temperature: 5 to 30 °C
""")

# 3. Annual Summary Section
st.markdown("### 🌾 Annual Yield Summary", unsafe_allow_html=True)
if not filtered_data.empty:
    annual_summary = filtered_data.groupby('Year').agg({
        'Rainfall (mm)': 'mean',
        'Temperature (°C)': 'mean',
        'Yield (tons/ha)': 'sum'
    }).reset_index()
    fig_summary = px.bar(
        annual_summary,
        x="Year",
        y="Yield (tons/ha)",
        title=f"Annual Yield Summary for {selected_crop} in {selected_country}",
        labels={"Yield (tons/ha)": "Total Yield (tons/ha)", "Year": "Year"},
        text_auto=True,
        color_discrete_sequence=["#27ae60"]
    )
    st.plotly_chart(fig_summary, use_container_width=True)

    # Separate Line Plots for Rainfall and Temperature
    st.markdown("### 🌧️ Annual Average Rainfall", unsafe_allow_html=True)
    fig_rainfall = px.bar(
        annual_summary,
        x="Year",
        y="Rainfall (mm)",
        title=f"Annual Average Rainfall for {selected_crop} in {selected_country}",
        labels={"Rainfall (mm)": "Rainfall (mm)", "Year": "Year"},
        color_discrete_sequence=["#3498db"]
    )
    st.plotly_chart(fig_rainfall, use_container_width=True)

    st.markdown("### 🌡️ Annual Average Temperature", unsafe_allow_html=True)
    fig_temperature = px.bar(
        annual_summary,
        x="Year",
        y="Temperature (°C)",
        title=f"Annual Average Temperature for {selected_crop} in {selected_country}",
        labels={"Temperature (°C)": "Temperature (°C)", "Year": "Year"},
        color_discrete_sequence=["#e74c3c"]
    )
    st.plotly_chart(fig_temperature, use_container_width=True)
else:
    st.warning("No annual data available for the selected filters.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Built with ❤️ using Streamlit.")