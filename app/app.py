import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import networkx as nx
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import st_folium

# PAGE CONFIG
st.set_page_config(
    page_title="Uber AI System",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD DATA
df = pd.read_csv("data/uber.csv")

# LOAD MODEL
model = joblib.load("models/fare_prediction_model.pkl")

# CUSTOM CSS
st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background: linear-gradient(to right, #0B0F19, #111827);
    color: white;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0B0F19;
    border-right: 1px solid #1F2937;
}

/* METRIC CARDS */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #111827, #1F2937);
    border: 1px solid #374151;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

/* TITLES */
h1, h2, h3, h4 {
    color: white !important;
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(to right, #00C6FF, #0072FF);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(to right, #0072FF, #00C6FF);
}

/* INPUT BOXES */
.stNumberInput,
.stSelectbox,
.stSlider {
    background-color: #111827;
    border-radius: 10px;
}

/* HIDE MENU */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:

    st.markdown("""
        <h1 style='text-align:center; color:#00C6FF;'>
            🚖 Uber AI
        </h1>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Fare Prediction",
            "Live Maps",
            "Route Optimization",
            "Driver Allocation"
        ],
        icons=[
            "speedometer2",
            "cash-stack",
            "map",
            "geo-alt-fill",
            "person-workspace"
        ],
        default_index=0,
    )

# DASHBOARD PAGE
if selected == "Dashboard":

    st.title("🚖 AI Ride Optimization Dashboard")

    st.markdown("### Real-Time Transportation Intelligence System")

    # METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Rides",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Average Fare",
            f"${df['fare_amount'].mean():.2f}"
        )

    with col3:
        st.metric(
            "Maximum Fare",
            f"${df['fare_amount'].max():.2f}"
        )

    with col4:
        st.metric(
            "Total Passengers",
            f"{df['passenger_count'].sum():,.0f}"
        )

    st.markdown("---")

    # CHARTS
    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="fare_amount",
            nbins=50,
            title="Fare Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig2 = px.histogram(
            df,
            x="passenger_count",
            title="Passenger Count Analysis"
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # RIDES BY HOUR
    if 'hour' in df.columns:

        st.markdown("## 📈 Peak Ride Hours")

        fig3 = px.line(
            df.groupby('hour').size().reset_index(name='rides'),
            x='hour',
            y='rides',
            markers=True
        )

        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827"
        )

        st.plotly_chart(fig3, use_container_width=True)

# FARE PREDICTION
elif selected == "Fare Prediction":

    st.title("💰 AI Fare Prediction")

    st.markdown("### Predict ride fares using Machine Learning")

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
            value=-73.985130
        )

        dropoff_latitude = st.number_input(
            "Dropoff Latitude",
            value=40.758896
        )

    with col2:

        passenger_count = st.slider(
            "Passenger Count",
            1,
            6,
            1
        )

        hour = st.slider(
            "Hour",
            0,
            23,
            12
        )

        day = st.slider(
            "Day",
            1,
            31,
            15
        )

        month = st.slider(
            "Month",
            1,
            12,
            6
        )

        weekday = st.slider(
            "Weekday",
            0,
            6,
            3
        )

    if st.button("🚖 Predict Fare"):

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
            f"Estimated Fare: ${prediction[0]:.2f}"
        )

# LIVE MAPS
elif selected == "Live Maps":

    st.title("🗺️ Live GPS Ride Map")

    st.markdown("### Real-Time Pickup & Drop Visualization")

    # MAP CENTER
    map_center = [40.748817, -73.985428]

    # CREATE MAP
    m = folium.Map(
        location=map_center,
        zoom_start=13,
        tiles='CartoDB dark_matter'
    )

    # SAMPLE PICKUP
    pickup = [40.748817, -73.985428]

    # SAMPLE DROPOFF
    dropoff = [40.758896, -73.985130]

    # PICKUP MARKER
    folium.Marker(
        pickup,
        popup="Pickup Location",
        icon=folium.Icon(
            color='green',
            icon='play'
        )
    ).add_to(m)

    # DROPOFF MARKER
    folium.Marker(
        dropoff,
        popup="Dropoff Location",
        icon=folium.Icon(
            color='red',
            icon='stop'
        )
    ).add_to(m)

    # ROUTE LINE
    folium.PolyLine(
        [pickup, dropoff],
        color='cyan',
        weight=5
    ).add_to(m)

    # DISPLAY MAP
    st_folium(
        m,
        width=1200,
        height=600
    )

# ROUTE OPTIMIZATION
elif selected == "Route Optimization":

    st.title("🗺️ Smart Route Optimization")

    st.markdown("### AI-Powered Shortest Path System")

    G = nx.Graph()

    G.add_edge('A', 'B', weight=4)
    G.add_edge('A', 'C', weight=2)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('B', 'D', weight=5)
    G.add_edge('C', 'D', weight=8)
    G.add_edge('D', 'E', weight=2)

    col1, col2 = st.columns(2)

    with col1:

        source = st.selectbox(
            "Pickup Location",
            ['A', 'B', 'C', 'D']
        )

    with col2:

        target = st.selectbox(
            "Destination",
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

        st.success(
            f"Optimal Route: {' → '.join(path)}"
        )

        st.info(
            f"Total Distance: {distance} km"
        )

# DRIVER ALLOCATION
elif selected == "Driver Allocation":

    st.title("🚗 AI Driver Allocation")

    st.markdown("### Intelligent Driver Dispatch System")

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

    st.success(
        f"Assigned Driver: {best_driver}"
    )

    st.info(
        f"Estimated Arrival Cost: {driver_distances[best_driver]}"
    )

    st.markdown("## 🚘 Driver Distances")

    driver_df = pd.DataFrame({
        "Driver": list(driver_distances.keys()),
        "Cost": list(driver_distances.values())
    })

    fig4 = px.bar(
        driver_df,
        x="Driver",
        y="Cost",
        color="Driver",
        title="Driver ETA Comparison"
    )

    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827"
    )

    st.plotly_chart(fig4, use_container_width=True)