import joblib
import streamlit as st
from streamlit_folium import st_folium
import folium
import networkx as nx
import osmnx as ox
import math
import random
import requests
import time
import gpxpy
import pandas as pd
import numpy as np

st.set_page_config(page_title="Générateur d'itinéraire cycliste", layout="wide")

# BACKEND

# Point de départ (Castanet)

Point_depart = (43.52448896013804, 1.4964669559174226)

# Ressources

@st.cache_resource
def load_model():
    model = joblib.load("performance_cyclisme.pkl")
    return model

@st.cache_resource
def load_graph():
    graph = ox.load_graphml("Data/graph_bike.graphml")
    return graph

@st.cache_resource
def load_node_counter():
    node_counter = joblib.load("Data/node_counter.pkl")
    return node_counter

@st.cache_resource
def load_node_score():
    node_score = joblib.load("Data/node_score.pkl")
    return node_score

def predict_speed(route, is_solo, model):
    
    features = extract_features(route, is_solo)
    
    speed = model.predict([features])
    
    return speed

def extract_features(route, is_solo):
    return [1, 2, is_solo]

def load_data(df):
    df = df.rename(columns={'Distance.1': 'Distance_m'})

    colonnes_a_garder = [
        'Activity ID', 'Activity Date', 'Activity Name', 'Activity Type',
        'Elapsed Time', 'Distance', 'Max Heart Rate',
        'Relative Effort', 'Filename', 'Moving Time', 'Max Speed', 'Average Speed',
        'Elevation Gain', 'Elevation Loss', 'Elevation Low', 'Elevation High',
        'Average Heart Rate', 'Calories', 'Start Time', 'Distance_m'
    ]

    df = df[colonnes_a_garder]
    df['Vitesse_moyenne'] = round((df.Distance_m / (df['Moving Time'] / 3600))/1000,1)

    patterns = {
        'pere': r'père|pere',
        'nausicaa': r'nausicaa|naunau|copine',
        'avec': r'père|pere|loïc|loic|nausicaa|mère|yann|naunau|hugo|balade|promenade|arnaud|mere|avec|ACE'
    }

    for col, regex in patterns.items():
        df[col] = (
            df['Activity Name']
            .str.contains(regex, case=False, na=False)
            .astype('category')
        )

    df = df.assign(
        **{
            'annee': df["Activity Date"].str.split(',', expand=True)[1].str.strip().astype(int),
            'mois': df["Activity Date"].str.split(' ', expand=True)[0],
            'elevation_km': (df["Elevation Gain"] / df["Distance_m"]) * 1000,
            'date': pd.to_datetime(df["Activity Date"], format='%b %d, %Y, %I:%M:%S %p')
        }
    )

    return df

# Fonctions pour générer le parcours

def nearest_node(G, lat, lon):
    return ox.nearest_nodes(G, lon, lat)

def route_distance(G, route):

    dist = 0

    for u,v in zip(route[:-1], route[1:]):
        edge = G.get_edge_data(u,v)[0]
        dist += edge["length"]

    return round(dist / 1000, 2)

def waypoint_from_angle(lat, lon, distance_m, angle):

    R = 6371000

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1)*math.cos(distance_m/R) +
        math.cos(lat1)*math.sin(distance_m/R)*math.cos(angle)
    )

    lon2 = lon1 + math.atan2(
        math.sin(angle)*math.sin(distance_m/R)*math.cos(lat1),
        math.cos(distance_m/R)-math.sin(lat1)*math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)

def generation_parcours(Point_depart, Distance_souhaitee, nb_noeuds=5, premier_noeud=None):

    start_node = ox.nearest_nodes(G, Point_depart[1], Point_depart[0])

    base_angle = random.uniform(0, 2 * math.pi)

    radius = Distance_souhaitee * 0.55 * 1000 / nb_noeuds

    angles = [
        base_angle,
        base_angle + random.uniform(0, math.pi / 3),
        base_angle + random.uniform(math.pi / 3, 2 * math.pi / 3),
        base_angle + random.uniform(2 * math.pi / 3, 3 * math.pi / 3),
        base_angle + random.uniform(3 * math.pi / 3, 4 * math.pi / 3),
        base_angle + random.uniform(4 * math.pi / 3, 5 * math.pi / 3)
    ]

    waypoints = []

    current_lat, current_lon = Point_depart

    for a in angles:
        new_point = waypoint_from_angle(
            current_lat,
            current_lon,
            random.uniform(2 / 3 * radius, 4 / 3 * radius),
            a
        )
        waypoints.append(new_point)

        current_lat, current_lon = new_point

    nodes = [
        ox.nearest_nodes(G, lon, lat)
        for lat, lon in waypoints
    ]

    route = []

    segments = [
        (start_node, nodes[0]),
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[3]),
        (nodes[3], nodes[4]),
        (nodes[4], nodes[5]),
        (nodes[5], start_node)
    ]

    for u, v in segments:
        path = nx.shortest_path(G, u, v, weight="length")

        if route:
            route.extend(path[1:])
        else:
            route.extend(path)

    return route

# Fonctions pour calculer les scores des parcours

def get_elevations_opentopo(coords, chunk_size=100, delay=1.0, retries=3):
    """
    coords: [(lat, lon), ...]
    return: [elevation, ...]
    """

    URL = "https://api.opentopodata.org/v1/eudem25m"
    results = []

    for i in range(0, len(coords), chunk_size):
        chunk = coords[i:i + chunk_size]

        locations = "|".join(f"{lat},{lon}" for lat, lon in chunk)

        for attempt in range(retries):
            try:
                response = requests.get(
                    URL,
                    params={"locations": locations},
                    timeout=10
                )

                if response.status_code == 429:
                    print("Rate limit atteint → pause...")
                    time.sleep(delay * (attempt + 2))
                    continue

                response.raise_for_status()

                data = response.json()

                if "results" not in data:
                    raise ValueError(f"Réponse invalide: {data}")

                elevations = [r["elevation"] for r in data["results"]]

                if len(elevations) != len(chunk):
                    raise ValueError("Mismatch entre coords et résultats")

                results.extend(elevations)
                break  

            except Exception as e:
                print(f"[chunk {i}] tentative {attempt+1} échouée:", e)

                if attempt == retries - 1:
                    results.extend([None] * len(chunk))

                time.sleep(delay * (attempt + 1))

        time.sleep(delay)

    return results


def compute_denivele_from_nodes(route, elevations):
    denivele = 0

    for i in range(1, len(route)):
        e1 = elevations.get(route[i - 1])
        e2 = elevations.get(route[i])

        if e1 is not None and e2 is not None:
            diff = e2 - e1
            if diff > 0:
                denivele += diff

    return denivele

def score_itineraire(
    routes,
    node_score,
    distance_souhaitee,
    denivele_souhaite,
    poids_noeud=0.5,
    poids_distance=1,
    poids_denivele=0.05,
    nb_noeuds=5
):

    all_nodes = set(node for route in routes for node in route)

    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in all_nodes]

    elevations_list = get_elevations_opentopo(coords)
    elevations = dict(zip(all_nodes, elevations_list))

    liste_score = []
    liste_denivele = []

    for route in routes:
        score = 0

        # score noeuds
        for node in route:
            score += (node_score.get(node, 0) * poids_noeud) / nb_noeuds

        # distance
        dist = route_distance(G, route)

        # dénivelé 
        denivele_total = compute_denivele_from_nodes(route, elevations)

        score -= poids_distance * abs(distance_souhaitee - dist)
        score -= poids_denivele * abs(denivele_souhaite - denivele_total)

        liste_score.append(score)
        liste_denivele.append(denivele_total)

    return liste_score, liste_denivele

def choix_meilleur_parcours(nb_parcours = 10, Distance_souhaitee = 80, Denivele_souhaite = 1000): 

    routes = []

    for i in range(nb_parcours):
        parcours = generation_parcours(Point_depart, Distance_souhaitee)
        routes.append(parcours)

    for i, route in enumerate(routes):
        print(i, route_distance(G, route))
        
    scores, deniveles = score_itineraire(routes, node_score, distance_souhaitee = Distance_souhaitee, denivele_souhaite = Denivele_souhaite)

    best_index = scores.index(max(scores))
    best_route = routes[best_index]
    denivele = deniveles[best_index]

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    return best_route, denivele

def export_route_to_gpx(G, route, filename="route.gpx"):

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for node in route:

        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]

        point = gpxpy.gpx.GPXTrackPoint(lat, lon)
        gpx_segment.points.append(point)

    with open(filename, "w") as f:
        f.write(gpx.to_xml())
    return gpx


# Paramètres carte

lat, lon = 43.4557049347904, 1.610179849942512

m = folium.Map(location=[lat, lon], zoom_start=12)

# Chargement des ressources

G = load_graph()
model = load_model()
node_score = load_node_score()
node_counter = load_node_counter()

# Etat session

if "route" not in st.session_state:
    st.session_state.route = None
    st.session_state.denivele = None

# Ressources dashboard

df = pd.read_csv('D:/Documents/R/Vélo/Data_Strava/activities.csv')
strava = load_data(df)
strava = strava.sort_values(['annee','mois'], ascending=False)

#FRONTEND

# Interface utilisateur

onglets = ["Générateur d'itinéraire cycliste", 'Statistiques', 'Détail parcours']
onglet1, onglet2, onglet3 = st.tabs(onglets)

with onglet1:
    st.header("Paramètres de génération du parcours")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        solo = st.selectbox(
            "Conditions de sortie",
            ["Seul", "Groupe"]
        )

    avec = 0 if solo == "Seul" else 1

    with col2:
        distance = st.slider("Distance (km)", 10, 150, 5)

    with col3:
        elevation = st.slider("Dénivelé (m)", 0, 3000, 50)

    with col4:
        nb_parcours = st.slider("Nombre de parcours", 1, 50, 1)

    generate = st.button("🚀 Générer l'itinéraire")

    # Generation intinéraire

    if generate:
        with st.spinner("Génération en cours..."):
            
            route, denivele = choix_meilleur_parcours(
                nb_parcours=nb_parcours,
                Distance_souhaitee=distance,
                Denivele_souhaite=elevation
            )

        # stockage pour persistance
        st.session_state.route = route
        st.session_state.denivele = denivele

    # Carte

    st.subheader("Carte")

    col1, col2 = st.columns([2,1])

    m = folium.Map(location=Point_depart, zoom_start=12)

    if st.session_state.route is not None:
        coords_route = [
            (G.nodes[n]["y"], G.nodes[n]["x"])
            for n in st.session_state.route
        ]

        if coords_route:
            folium.PolyLine(coords_route, color="red", weight=4).add_to(m)

    with col1:
        st_folium(m, width=600, height=600)

    with col2:
      
        if st.session_state.route is not None:
            st.write("Informations sur l'itinéraire généré")
            st.metric("Distance (km)", f"{route_distance(G, st.session_state.route):.2f}")
            st.metric("Dénivelé (m)", f"{st.session_state.denivele:.0f}")   

        # Vitesse prédite
        if st.session_state.route is not None:

            route = st.session_state.route
            denivele = st.session_state.denivele

            Distance_m = route_distance(G, route)*1000

            elevation_km = denivele / (Distance_m / 1000)
            elevation_km_2 = elevation_km**2

            Predicted_speed = model.predict([
                [Distance_m, avec, elevation_km, elevation_km_2]
            ])

            st.metric("Vitesse prédite (km/h)", f"{Predicted_speed[0]:.1f}")

with onglet2:
    st.header("Statistiques générales")


with onglet3:
    st.header("Détail du parcours")

parcours_a_afficher = st.selectbox(
    "Parcours à afficher",
    strava["Activity ID"],
    format_func=lambda x: (
        f"{strava.loc[strava['Activity ID'] == x, 'Activity Name'].values[0]} | "
        f"{strava.loc[strava['Activity ID'] == x, 'Activity Date'].values[0]} | "
        f"{strava.loc[strava['Activity ID'] == x, 'Distance'].values[0]} km | "
        f"{strava.loc[strava['Activity ID'] == x, 'Elevation Gain'].values[0]} m"
    )
)