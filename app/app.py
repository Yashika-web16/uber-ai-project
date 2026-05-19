import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import networkx as nx

# PAGE CONFIG
st.set_page_config(
    page_title="Uber AI System",
    page_icon="🚖",
    layout="wide"
)

# LOAD DATA
df = pd.read_csv("data/uber.csv")

# LOAD MODEL
model = joblib.load("models/fare_prediction_model.pkl")

# SIDEBAR
with st.sidebar:
    selected = option_menu(
        menu_title="Uber AI Dashboard",
        options=[
            "Dashboard",
            "Fare Prediction",
            "Route Optimization",
            "Driver Allocation"
        ],
        icons=[
            "speedometer2",
            "cash-stack",
            "geo-alt",
            "person-badge"
        ],
        menu_icon="robot",
        default_index=0,
    )

# CUSTOM CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }

    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# DASHBOARD PAGE
if selected == "Dashboard":

    st.title("🚖 Uber AI Ride Optimization Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rides", len(df))

    with col2:
        st.metric("Average Fare", round(df['fare_amount'].mean(), 2))

    with col3:
        st.metric("Max Fare", round(df['fare_amount'].max(), 2))

    st.markdown("---")

    # FARE DISTRIBUTION
    fig = px.histogram(
        df,
        x='fare_amount',
        nbins=50,
        title="Fare Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # RIDES BY PASSENGER COUNT
    fig2 = px.histogram(
        df,
        x='passenger_count',
        title="Passenger Count Analysis"
    )

    st.plotly_chart(fig2, use_container_width=True)

# FARE PREDICTION PAGE
elif selected == "Fare Prediction":

    st.title("💰 Fare Prediction System")

    col1, col2 = st.columns(2)

    with col1:
        pickup_longitude = st.number_input(
            "Pickup Longitude",
            value=-73.985428
        )

        pickup_latitude = st.number_input(
            "Pickup Latitude",
            value=40.748817
        )

        dropoff_longitude = st.number_input(
            "Dropoff Longitude",
            value=-73.985428
        )

        dropoff_latitude = st.number_input(
            "Dropoff Latitude",
            value=40.748817
        )

    with col2:
        passenger_count = st.slider(
            "Passenger Count",
            1,
            6,
            1
        )

        hour = st.slider("Hour", 0, 23, 12)

        day = st.slider("Day", 1, 31, 15)

        month = st.slider("Month", 1, 12, 6)

        weekday = st.slider("Weekday", 0, 6, 3)

    if st.button("Predict Fare"):

        input_data = np.array([[
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            passenger_count,
            hour,
            day,
            month,
            weekday
        ]])

        prediction = model.predict(input_data)

        st.success(
            f"Predicted Fare: ${prediction[0]:.2f}"
        )

# ROUTE OPTIMIZATION PAGE
elif selected == "Route Optimization":

    st.title("🗺️ Route Optimization System")

    G = nx.Graph()

    G.add_edge('A', 'B', weight=4)
    G.add_edge('A', 'C', weight=2)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('B', 'D', weight=5)
    G.add_edge('C', 'D', weight=8)
    G.add_edge('D', 'E', weight=2)

    source = st.selectbox(
        "Select Pickup Location",
        ['A', 'B', 'C', 'D']
    )

    target = st.selectbox(
        "Select Destination",
        ['B', 'C', 'D', 'E']
    )

    if st.button("Find Best Route"):

        path = nx.shortest_path(
            G,
            source=source,
            target=target,
            weight='weight'
        )

        distance = nx.shortest_path_length(
            G,
            source=source,
            target=target,
            weight='weight'
        )

        st.success(f"Optimal Route: {' → '.join(path)}")

        st.info(f"Total Distance: {distance}")

# DRIVER ALLOCATION PAGE
elif selected == "Driver Allocation":

    st.title("🚗 Driver Allocation AI")

    drivers = {
        'Driver 1': 'A',
        'Driver 2': 'C',
        'Driver 3': 'D'
    }

    passenger = st.selectbox(
        "Passenger Location",
        ['A', 'B', 'C', 'D', 'E']
    )

    traffic_graph = nx.Graph()

    traffic_graph.add_edge('A', 'B', weight=4)
    traffic_graph.add_edge('A', 'C', weight=2)
    traffic_graph.add_edge('B', 'C', weight=1)
    traffic_graph.add_edge('B', 'D', weight=5)
    traffic_graph.add_edge('C', 'D', weight=8)
    traffic_graph.add_edge('D', 'E', weight=2)

    driver_distances = {}

    for driver, location in drivers.items():

        distance = nx.shortest_path_length(
            traffic_graph,
            source=location,
            target=passenger,
            weight='weight'
        )

        driver_distances[driver] = distance

    best_driver = min(
        driver_distances,
        key=driver_distances.get
    )

    st.success(f"Assigned Driver: {best_driver}")

    st.info(
        f"Estimated Arrival Cost: {driver_distances[best_driver]}"
    )