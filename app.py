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
import pickle
import plotly.express as px
import plotly.graph_objects as go

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

    df["Distance"] = df["Distance"].str.replace(",", "").astype(float)
    df.loc[df['Activity Type'] == 'Swim', 'Distance'] = (df.loc[df['Activity Type'] == 'Swim', 'Distance'] / 1000)

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
            'jour': df["Activity Date"].str.split(' ', expand=True)[1].str.replace(',', '').astype(int),
            'elevation_km': (df["Elevation Gain"] / df["Distance_m"]) * 1000,
            'date': pd.to_datetime(df["Activity Date"], format='%b %d, %Y, %I:%M:%S %p')
        }
    )

    df["Filename"] = (
        df["Filename"]
        .astype(str)
        .str.replace("activities/", "", regex=False)
        .str.split("/")
        .str[-1]
    )

    return df

def get_dist_cum_elev(df, id_parcours):

    R = 6371

    df_dist = pd.DataFrame(df.iloc[0][id_parcours], columns=["lat", "lon"])

    lat = np.radians(df_dist["lat"])
    lon = np.radians(df_dist["lon"])

    dlat = lat.diff()
    dlon = lon.diff()

    a = np.sin(dlat/2)**2 + np.cos(lat)*np.cos(lat.shift())*np.sin(dlon/2)**2
    df_dist["dist"] = 2 * R * np.arcsin(np.sqrt(a))

    df_dist["dist"] = df_dist["dist"].fillna(0)
    df_dist["dist_cum"] = round(df_dist["dist"].cumsum(),4)

    df_dist["elev"] = df.iloc[1][id_parcours]

    return df_dist[["dist_cum", "elev", "lat", "lon"]]

def carte_folium_parcours(df):

    center_lat = df["lat"].iloc[0]
    center_lon = df["lon"].iloc[0]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    coords = list(zip(df["lat"], df["lon"]))

    folium.PolyLine(coords, color="red", weight=3, opacity=0.8).add_to(m)

    folium.Marker(coords[0], tooltip="Départ", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], tooltip="Arrivée", icon=folium.Icon(color="red")).add_to(m)

    return m



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

def generation_parcours(Point_depart, Distance_souhaitee, nb_noeuds=5, max_attempts=3):
    for attempt in range(max_attempts):
        try:
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
            nodes = [ox.nearest_nodes(G, lon, lat) for lat, lon in waypoints]
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
        except nx.exception.NetworkXNoPath:
            continue
    raise RuntimeError("Impossible de générer un parcours après plusieurs tentatives.")

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
    poids_distance=3,
    poids_denivele=1,
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

def choix_meilleur_parcours(nb_parcours = 10, Distance_souhaitee = 80, Denivele_souhaite = 1000, Point_depart = None): 

    if Point_depart is None:
        Point_depart = st.session_state.get("Point_depart", Point_depart)

    routes = []

    for i in range(nb_parcours):
        parcours = generation_parcours(Point_depart, Distance_souhaitee)
        routes.append(parcours)

    for i, route in enumerate(routes):
        print(i, route_distance(G, route))
        
    scores, deniveles = score_itineraire(routes, node_score, distance_souhaitee = Distance_souhaitee, denivele_souhaite = Denivele_souhaite)

    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_n = min(3, len(routes))

    top_routes = [routes[i] for i in sorted_idx[:top_n]]
    top_deniveles = [deniveles[i] for i in sorted_idx[:top_n]]
    top_scores = [scores[i] for i in sorted_idx[:top_n]]

    return top_routes, top_deniveles, top_scores

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


def route_to_gpx_bytes(G, route):
    """Convertit un itinéraire en bytes GPX pour téléchargement sans fichier temporaire."""
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for node in route:
        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]
        segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

    return gpx.to_xml().encode("utf-8")


def calcul_pente(df, step = 300):

    df['dist_cum_m'] = df["dist_cum"] * 1000
    df["Tranche"] = (df["dist_cum_m"] // step) * step
    bins = (df["dist_cum_m"] // step) * step
    g = df.groupby(bins)

    result = g.agg(
        dist_start=("dist_cum_m", "first"),
        dist_end=("dist_cum_m", "last"),
        elev_start=("elev", "first"),
        elev_end=("elev", "last"),
    )

    result["delta_dist"] = result["dist_end"] - result["dist_start"]
    result["delta_elev"] = result["elev_end"] - result["elev_start"]

    result = result[result["delta_dist"] > 0]

    result["pente_%"] = round((result["delta_elev"] / result["delta_dist"]) * 100, 2)

    result.reset_index(drop=False, inplace=True)
    result = result.rename(columns={"dist_cum_m": "Tranche"})

    df = pd.merge(df, result[["Tranche", "pente_%"]], on="Tranche", how="left")
    df.drop(columns=["dist_cum_m", "Tranche"], inplace=True)

    df["couleur_pente"] = df["pente_%"].apply(lambda x: 'black' if x > 10 else (
    'red' if (x > 8 and x <= 10) else (
        'orange' if (x > 6 and x <= 8) else (
            'yellow' if (x > 4 and x <= 6) else (
                'green' if (x > 2 and x <= 4) else (
                    'blue' if (x > 0 and x <= 2) else 'gray'
                )
            )
        ))))

    return df

def get_profile_from_route(G, route):
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

    R = 6371

    df = pd.DataFrame(coords, columns=["lat", "lon"])

    lat = np.radians(df["lat"])
    lon = np.radians(df["lon"])

    dlat = lat.diff()
    dlon = lon.diff()

    a = np.sin(dlat/2)**2 + np.cos(lat)*np.cos(lat.shift())*np.sin(dlon/2)**2
    df["dist"] = 2 * R * np.arcsin(np.sqrt(a))

    df["dist"] = df["dist"].fillna(0)
    df["dist_cum"] = df["dist"].cumsum()

    elevations = get_elevations_opentopo(coords)
    df["elev"] = elevations

    return df

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

if "top_routes" not in st.session_state:
    st.session_state.top_routes = []
    st.session_state.top_deniveles = []
    st.session_state.top_scores = []
    st.session_state.selected_top_index = 0

if "activities_path" not in st.session_state:
    st.session_state.activities_path = 'Data/activities.csv'

if "Point_depart" not in st.session_state:
    st.session_state.Point_depart = Point_depart

if "strava" not in st.session_state:
    try:
        df = pd.read_csv(st.session_state.activities_path)
        st.session_state.strava = load_data(df)
        st.session_state.load_error = ""
    except Exception as e:
        st.session_state.strava = pd.DataFrame()
        st.session_state.load_error = str(e)

# Ressources dashboard

strava = st.session_state.strava
if not strava.empty:
    strava = strava.sort_values(['annee','mois','jour'], ascending=False)
dataset_gpx = pickle.load(open("Data/dataset_points.pkl", "rb"))
dataset_points = pd.DataFrame(dataset_gpx)

#FRONTEND

# Interface utilisateur

onglets = ["Générateur d'itinéraire cycliste", 'Statistiques', 'Détail parcours', 'Importer activities.csv']
onglet1, onglet2, onglet3, onglet4 = st.tabs(onglets)

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
        nb_parcours = st.slider("Nombre de parcours", 3, 50, 1)

    generate = st.button("🚀 Générer l'itinéraire")

    # Generation intinéraire

    Point_depart = st.session_state.Point_depart

    if generate:
        with st.spinner("Génération en cours..."):
            
            top_routes, top_deniveles, top_scores = choix_meilleur_parcours(
                nb_parcours=nb_parcours,
                Distance_souhaitee=distance,
                Denivele_souhaite=elevation,
                Point_depart=Point_depart
            )

        # stockage pour persistance
        st.session_state.top_routes = top_routes
        st.session_state.top_deniveles = top_deniveles
        st.session_state.top_scores = top_scores
        st.session_state.selected_top_index = 0

    # Carte

    st.subheader("Carte")

    if st.session_state.top_routes:
        nb_top = len(st.session_state.top_routes)
        options = [f"Top {i+1}" for i in range(nb_top)]
        selected_label = st.radio(
            "Sélectionnez un itinéraire (top 3)",
            options,
            index=st.session_state.selected_top_index,
            horizontal=True
        )
        selected_idx = options.index(selected_label)
        st.session_state.selected_top_index = selected_idx

        route_selected = st.session_state.top_routes[selected_idx]
        denivele_selected = st.session_state.top_deniveles[selected_idx]
        score_selected = st.session_state.top_scores[selected_idx]
    else:
        route_selected = None
        denivele_selected = None
        score_selected = None

    col1, col2 = st.columns([2,1])

    m = folium.Map(location=Point_depart, zoom_start=12)

    if route_selected is not None:
        coords_route = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_selected]
        if coords_route:
            folium.PolyLine(coords_route, color="red", weight=4).add_to(m)

    with col1:
        st_folium(m, width=800, height=800, returned_objects=[], key="m")

    with col2:
        if route_selected is not None:
            st.write("Informations sur l'itinéraire sélectionné")
            st.metric("Distance (km)", f"{route_distance(G, route_selected):.2f}")
            st.metric("Dénivelé (m)", f"{denivele_selected:.0f}")

            gpx_bytes = route_to_gpx_bytes(G, route_selected)
            st.download_button(
                label="📥 Télécharger le GPX (itinéraire sélectionné)",
                data=gpx_bytes,
                file_name=f"parcours_cycliste_top{selected_idx+1}.gpx",
                mime="application/gpx+xml"
            )

            Distance_m = route_distance(G, route_selected) * 1000
            elevation_km = denivele_selected / (Distance_m / 1000) if Distance_m > 0 else 0
            elevation_km_2 = elevation_km**2

            Predicted_speed = model.predict([
                [Distance_m, avec, elevation_km, elevation_km_2]
            ])
            st.metric("Vitesse prédite (km/h)", f"{Predicted_speed[0]:.1f}")

    if route_selected is not None:
        st.subheader("Profil d'élévation")
        df_profile = get_profile_from_route(G, route_selected)
        df_profile = calcul_pente(df_profile)
        fig_profile = go.Figure()

        for _, group in df_profile.groupby(
            (df_profile["couleur_pente"] != df_profile["couleur_pente"].shift()).cumsum()
        ):
            color = group["couleur_pente"].iloc[0]
            fillcolor = color.replace(")", ",0.2)").replace("rgb", "rgba") if "rgb" in color else color
            fig_profile.add_trace(go.Scatter(
                x=group["dist_cum"],
                y=group["elev"],
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=fillcolor,
                showlegend=False
            ))

        fig_profile.update_layout(
            title="Profil d'élévation du parcours généré",
            xaxis_title="Distance (km)",
            yaxis_title="Altitude (m)"
        )

        st.plotly_chart(fig_profile, use_container_width=True)

    if st.session_state.top_routes:
        st.markdown("---")
        st.subheader("Top 3 itinéraires retenus")
        for idx, route in enumerate(st.session_state.top_routes):
            with st.expander(f"Itinéraire n°{idx+1}"):
                dist = route_distance(G, route)
                deniv = st.session_state.top_deniveles[idx]
                st.metric("Distance (km)", f"{dist:.2f}")
                st.metric("Dénivelé (m)", f"{deniv:.0f}")

                gpx_bytes_i = route_to_gpx_bytes(G, route)
                st.download_button(
                    label=f"📥 Télécharger GPX n°{idx+1}",
                    data=gpx_bytes_i,
                    file_name=f"parcours_cycliste_top{idx+1}.gpx",
                    mime="application/gpx+xml",
                    key=f"download_top_{idx}"
                )

with onglet2:
    st.header("Statistiques")
    ongleta, ongletb, ongletc = st.tabs(["Statistiques générales", "", ""])
    with ongleta:

        strava["annee"] = strava["annee"].astype(str)

        st.subheader("Statistiques générales")
        annee_filtre = st.selectbox(
            "Filtrer par année",np.append(["Tous les temps"], pd.unique(strava["annee"])), index=0)
        if annee_filtre != "Tous les temps":
            strava_filtre = strava[strava["annee"] == annee_filtre]
        else:
            strava_filtre = strava
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre d'activités", strava_filtre.shape[0])
        with col2:
            st.metric("Distance totale (km)", strava_filtre["Distance"].sum().astype(int))
        with col3:
            st.metric("Dénivelé total (m)", strava_filtre["Elevation Gain"].sum().astype(int))
        
        st.subheader("Distance parcourue par mois")
        fig_distance_mois = px.bar(
            strava_filtre.groupby("mois")[["Distance","Elevation Gain"]].sum().reindex(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
            labels={"value": "Total", "mois": "Mois"},
            title="Distance et dénivelé totaux parcourus par mois",
            barmode="group",
        )
        st.plotly_chart(fig_distance_mois, use_container_width=True)



with onglet3:
    st.header("Détail du parcours")

    tri_par = st.radio(
        "Trier les parcours par",
        ["📆 Date (décroissant)", "🛣️ Distance (décroissant)", "⛰️ Dénivelé (décroissant)"],
        index=0
    )

    if tri_par == "📆 Date (décroissant)":
        strava_triee = strava.sort_values(['annee','mois','jour'], ascending=False)
    elif tri_par == "🛣️ Distance (décroissant)":
        strava_triee = strava.sort_values('Distance_m', ascending=False)
    elif tri_par == "⛰️ Dénivelé (décroissant)":
        strava_triee = strava.sort_values('Elevation Gain', ascending=False)

    parcours_a_afficher = st.selectbox(
        "Parcours à afficher",
        strava_triee["Activity ID"],
        format_func=lambda x: (
            f"{strava.loc[strava['Activity ID'] == x, 'Activity Name'].values[0]} | "
            f"{strava.loc[strava['Activity ID'] == x, 'Activity Date'].values[0]} | "
            f"{strava.loc[strava['Activity ID'] == x, 'Distance'].values[0]} km | "
            f"{strava.loc[strava['Activity ID'] == x, 'Elevation Gain'].values[0]} m"
        )
    )

    m_parcours = folium.Map(location=[lat, lon], zoom_start=200)

    if parcours_a_afficher is not None:
        filename = strava.loc[strava["Activity ID"] == parcours_a_afficher, 'Filename'].iloc[0]
        df_parcours = get_dist_cum_elev(dataset_points, filename)
        df_parcours = calcul_pente(df_parcours)
        m_parcours = carte_folium_parcours(df_parcours)


    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:
        st_folium(m_parcours, width=1280, height=720, returned_objects=[], key="m_parcours")

    fig_profile = go.Figure()

    for _, group in df_parcours.groupby(
        (df_parcours["couleur_pente"] != df_parcours["couleur_pente"].shift()).cumsum()
    ):
        color = group["couleur_pente"].iloc[0]
        fillcolor = color.replace(")", ",0.2)").replace("rgb", "rgba") if "rgb" in color else color
        fig_profile.add_trace(go.Scatter(
            x=group["dist_cum"],
            y=group["elev"],
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=fillcolor,
            showlegend=False
        ))

    fig_profile.update_layout(
        title=f"Profil du parcours: {strava[strava['Activity ID'] == parcours_a_afficher]['Activity Name'].iloc[-1]}",
        xaxis_title="Distance (km)",
        yaxis_title="Altitude (m)"
    )

    with col3:
        st.write("Informations sur le parcours sélectionné")
        st.metric("Distance (km)", f"{strava[strava['Activity ID'] == parcours_a_afficher]['Distance'].iloc[-1]}")
        st.metric("Dénivelé (m)", f"{strava[strava['Activity ID'] == parcours_a_afficher]['Elevation Gain'].iloc[-1]}")
        st.metric("Vitesse moyenne (km/h)", f"{strava[strava['Activity ID'] == parcours_a_afficher]['Vitesse_moyenne'].iloc[-1]}")

    col1, col2, col3 = st.columns([1, 5, 1])

    with col2:
        st.plotly_chart(fig_profile, use_container_width=True)

with onglet4:
    st.header("Importer un fichier activities.csv")
    st.write("Entrez le chemin complet du fichier `activities.csv` pour recharger le jeu de données Strava.")

    with st.form("load_activities_csv"):
        activities_path = st.text_input(
            "Chemin du fichier activities.csv",
            value=st.session_state.activities_path,
            placeholder="C:/Users/.../activities.csv"
        )
        load_file = st.form_submit_button("Charger le fichier")

    if load_file:
        try:
            df = pd.read_csv(activities_path)
            st.session_state.strava = load_data(df)
            st.session_state.activities_path = activities_path
            st.session_state.load_error = ""
            st.success(f"Fichier chargé avec succès : {activities_path}")
        except Exception as e:
            st.session_state.load_error = str(e)
            st.error(f"Impossible de charger le fichier : {e}")

    st.markdown("---")
    st.write(f"Chemin actuel : `{st.session_state.activities_path}`")

    with st.form("load_start_point"):
        start_point_input = st.text_input(
            "Nouveau point de départ (lat, lon)",
            value=f"{st.session_state.Point_depart[0]}, {st.session_state.Point_depart[1]}",
            placeholder="43.52448896013804, 1.4964669559174226"
        )
        set_start_point = st.form_submit_button("Mettre à jour le point de départ")

    if set_start_point:
        try:
            lat_str, lon_str = [s.strip() for s in start_point_input.split(",", 1)]
            lat_val = float(lat_str)
            lon_val = float(lon_str)
            st.session_state.Point_depart = (lat_val, lon_val)
            st.success(f"Point de départ mis à jour : {lat_val}, {lon_val}")
        except Exception as e:
            st.error("Format invalide. Utilisez `lat, lon` avec des valeurs numériques.")

    st.write(f"Point de départ actuel : `{st.session_state.Point_depart[0]}, {st.session_state.Point_depart[1]}`")
    if st.session_state.load_error:
        st.error(f"Dernière erreur de chargement : {st.session_state.load_error}")
    elif not st.session_state.strava.empty:
        st.success(f"Données chargées : {st.session_state.strava.shape[0]} activités")
    else:
        st.info("Aucune donnée chargée pour le moment.")