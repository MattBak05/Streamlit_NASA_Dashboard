import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import base64
import os
from datetime import datetime


# -------------------------- Pagina configuratie --------------------------
st.set_page_config(
    page_title="NASA Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- CSS voor layout --------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1600px;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        margin-left: auto;
        margin-right: auto;
    }
    .stColumns {
        gap: 3rem;
        justify-content: flex-start !important;
    }
    .stImage {
        max-width: 800px;
        height: auto;
    }
    /* Oplossing voor plotly grafieken */
    .js-plotly-plot {
        width: 100% !important;
        height: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------- API Sleutels --------------------------
NASA_API_KEY = "R7LKmJX5zeV6PLSpm2ggaXFWvATo0aggnhCmpiQh"
ASTRONOMY_APP_ID = "033dd952-e68f-4c47-850b-e11878699b9a"
ASTRONOMY_APP_SECRET = "79bfa1c64c3a45639c841c19df589b1599d4694ed486225b9b69387a5c5b3de9d8165b397892a110c3da39d4a3bc1f9efb875457acdd5531e908d6b13cd1e28be6ce036a76c2751bbca12288338bec97373f176853aa1933c0ebe2c7915f8325d6b276221c74e40a18664ff31a790ed9"

# Codeer inloggegevens voor Astronomy API
userpass = f"{ASTRONOMY_APP_ID}:{ASTRONOMY_APP_SECRET}"
auth_string = base64.b64encode(userpass.encode()).decode()
AUTH_HEADER = {"Authorization": f"Basic {auth_string}"}

# ================= Functies =================

@st.cache_data(ttl=3600)  # Cache voor 1 uur
def get_moon_phase(date):
    """
    Haal maanfase afbeelding op van AstronomyAPI.
    Zie voor documentatie, https://astronomyapi.com/.
    """
    moon_url = "https://api.astronomyapi.com/api/v2/studio/moon-phase"
    payload = {
        "style": {
            "moonStyle": "default",
            "backgroundStyle": "stars",
            "backgroundColor": "#000000",
            "headingColor": "#ffffff",
            "textColor": "#ffffff"
        },
        "observer": {
            "latitude": 52.3558182,
            "longitude": 4.9557263,
            "date": date
        },
        "view": {
            "type": "landscape-simple",
            "parameters": {}
        }
    }
    try:
        response = requests.post(moon_url, headers=AUTH_HEADER, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data["data"]["imageUrl"]
    except Exception as e:
        st.error(f"Fout bij ophalen maanfase: {e}")
    return None


@st.cache_data(ttl=3600)
def load_neo_for_date(date):
    """
    Laad NEOs voor een specifieke datum via feed endpoint.
    Zie voor documentatie, https://api.nasa.gov/.
    """
    url = "https://api.nasa.gov/neo/rest/v1/feed"
    params = {"start_date": date, "end_date": date, "api_key": NASA_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        neos_for_date = data.get("near_earth_objects", {}).get(date, [])

        neo_list = []
        for neo in neos_for_date:
            diameter_info = neo.get("estimated_diameter", {}).get("kilometers", {})
            est_min = diameter_info.get("estimated_diameter_min", np.nan)
            est_max = diameter_info.get("estimated_diameter_max", np.nan)
            avg_diameter = (est_min + est_max) / 2 if not np.isnan(est_min) and not np.isnan(est_max) else np.nan
            close_approach = neo.get("close_approach_data", [])
            if close_approach:
                approach = close_approach[0]
                approach_date = approach.get("close_approach_date", "Onbekend")
                miss_distance_km = approach.get("miss_distance", {}).get("kilometers", np.nan)
                relative_velocity_kmh = approach.get("relative_velocity", {}).get("kilometers_per_hour", np.nan)
            else:
                approach_date = "Onbekend"
                miss_distance_km = np.nan
                relative_velocity_kmh = np.nan
            neo_list.append({
                "Naam": neo.get("name", "Onbekend"),
                "Gemiddelde Diameter (km)": round(avg_diameter, 3) if not np.isnan(avg_diameter) else np.nan,
                "Datum Nadering": approach_date,
                "Mis Afstand (km)": miss_distance_km,
                "Snelheid (km/u)": relative_velocity_kmh,
                "Potentieel Gevaarlijk": neo.get("is_potentially_hazardous_asteroid", False)
            })
        return pd.DataFrame(neo_list)
    except Exception as e:
        st.error(f"Fout bij ophalen NEO data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_single_mars_rover_photo(date_str):
    """
    Haal 1 roverfoto op van de gekozen dag.
    Controleert of er foto's zijn en geeft None als er geen zijn.
    Zie voor documentatie, https://api.nasa.gov/.
    """
    rovers = ["curiosity", "opportunity", "spirit"]
    accepted_cameras = ["FHAZ", "RHAZ", "NAVCAM", "MAST", "CHEMCAM", "PANCAM", "MASTCAM_Z"]

    for rover in rovers:
        url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos"
        params = {"earth_date": date_str, "api_key": NASA_API_KEY}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                continue
            data = response.json()

            photos = data.get("photos", [])
            if not photos:
                continue

            # Zoek eerst foto's van geaccepteerde camera's
            for p in photos:
                if p["camera"]["name"] in accepted_cameras:
                    img_url = p["img_src"].replace("http://", "https://")
                    return {
                        "url": img_url,
                        "camera": p["camera"]["full_name"],
                        "rover": rover.capitalize()
                    }
            # Als geen geaccepteerde camera, pak eerste foto
            p = photos[0]
            img_url = p["img_src"].replace("http://", "https://")
            return {
                "url": img_url,
                "camera": p["camera"]["full_name"],
                "rover": rover.capitalize()
            }
        except Exception as e:
            continue

    # Geen foto gevonden
    return None


def col_exists_any(df, candidates):
    """Hulpfunctie om flexibel kolomnamen te vinden."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data
def load_and_prepare_neo_data():
    """
    Laad en bereid NEO dataset voor - gecombineerde functie.
    Zie voor documentatie, https://api.nasa.gov/.
    """
    # Bouw een pad relatief aan dit script
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "nearest-earth-objects(1910-2024).csv")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Kon CSV niet laden: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Kon CSV niet laden: {e}")
        return pd.DataFrame()

    # Zoek relevante kolommen (flexibel met verschillende naamconventies)
    col_name = col_exists_any(df, ["name", "naam", "Name"])
    col_min = col_exists_any(df, ["estimated_diameter_min", "est_diam_min", "diameter_min"])
    col_max = col_exists_any(df, ["estimated_diameter_max", "est_diam_max", "diameter_max"])
    col_miss = col_exists_any(df, ["miss_distance", "miss_distance_km", "missDistanceKilometers", "miss_distance_km.1"])
    col_vel = col_exists_any(df, ["relative_velocity", "relative_velocity_kmh", "relative_velocity_km_h", "relative_velocity.km_per_hour"])
    col_hazard = col_exists_any(df, ["is_potentially_hazardous_asteroid", "is_hazardous", "is_hazardous_asteroid"])

    # Bereken gemiddelde diameter
    if col_min and col_max:
        df["Gemiddelde Diameter (km)"] = (pd.to_numeric(df[col_min], errors="coerce") + pd.to_numeric(df[col_max], errors="coerce")) / 2
    else:
        if "diameter" in df.columns:
            df["Gemiddelde Diameter (km)"] = pd.to_numeric(df["diameter"], errors="coerce")
        else:
            df["Gemiddelde Diameter (km)"] = np.nan

    # Zet mis afstand om naar numeriek
    if col_miss:
        df["miss_distance"] = pd.to_numeric(df[col_miss], errors="coerce")
    else:
        df["miss_distance"] = pd.to_numeric(df.get("miss_distance_km", df.get("missDistance", np.nan)), errors="coerce")

    # Zet relatieve snelheid om naar numeriek
    if col_vel:
        df["relative_velocity"] = pd.to_numeric(df[col_vel], errors="coerce")
    else:
        df["relative_velocity"] = pd.to_numeric(df.get("relative_velocity_kmh", np.nan), errors="coerce")

    # Zet gevaar boolean om
    if col_hazard:
        df["is_hazardous"] = df[col_hazard].astype(bool)
    else:
        df["is_hazardous"] = df.get("is_hazardous", False).astype(bool)

    # Zorg voor naam kolom
    if col_name:
        df["name"] = df[col_name].astype(str)
    else:
        if "designation" in df.columns:
            df["name"] = df["designation"].astype(str)
        else:
            df["name"] = df.index.astype(str)

    # Bereken risico score: diameter gedeeld door mis afstand
    df["risk_score"] = df["Gemiddelde Diameter (km)"] / (df["miss_distance"].replace(0, np.nan))
    df["risk_score"] = df["risk_score"].replace([np.inf, -np.inf], np.nan)

    return df


# -------------------------- Maak tabs --------------------------
tab1, tab2 = st.tabs(["📷 Dagelijkse Beelden", "📊 NEO Analyse"])

# ================= Tab 1 =================
with tab1:
    st.header("Dagelijkse Astronomie")

    # Datum selector
    selected_date = st.date_input(
        "Kies een datum",
        value=datetime.today(),
        min_value=datetime(1995, 6, 16),
        max_value=datetime.today()
    )

    # ---------------- APOD ----------------
    with st.spinner("Laden van APOD..."):
        APOD_URL = "https://api.nasa.gov/planetary/apod"
        params_apod = {"api_key": NASA_API_KEY, "date": selected_date.strftime("%Y-%m-%d")}

        try:
            response_apod = requests.get(APOD_URL, params=params_apod, timeout=10)
            response_apod.raise_for_status()
            apod_data = response_apod.json()
        except Exception as e:
            st.error(f"Kon APOD niet ophalen: {e}")
            apod_data = {}

    col1, col2 = st.columns([5, 5])

with col1:
    st.subheader(apod_data.get("title", "Geen titel gevonden"))
    st.caption(apod_data.get("date", ""))

    if apod_data.get("media_type") == "image":
        img_url = apod_data["url"]
        st.markdown(
            f'<a href="{img_url}" target="_blank">'
            f'<img src="{img_url}" style="max-width:100%; height:auto; border-radius:10px;" />'
            f'</a>',
            unsafe_allow_html=True
        )
    elif apod_data.get("media_type") == "video":
        st.video(apod_data["url"])
    else:
        st.warning(f"Onbekend mediatype: {apod_data.get('media_type')}")

    if apod_data.get("explanation"):
        with st.expander("Toon uitleg"):
            st.markdown(f"<div style='max-width: 800px;'>{apod_data['explanation']}</div>", unsafe_allow_html=True)

    # ---------------- Mars Rover Foto ----------------
with col2:
    st.subheader("Mars Rover Photo")
    st.caption(apod_data.get("date", ""))
    with st.spinner("Laden van Mars Rover foto..."):
        rover_photo = get_single_mars_rover_photo(selected_date.strftime("%Y-%m-%d"))
        if rover_photo:
            img_url = rover_photo["url"]
            st.markdown(
                f'<a href="{img_url}" target="_blank">'
                f'<img src="{img_url}" style="max-width:100%; height:auto; border-radius:10px;" />'
                f'</a>',
                unsafe_allow_html=True
            )
            st.caption(f"{rover_photo['rover']} Rover ({rover_photo['camera']})")
        else:
            st.info("Geen roverfoto beschikbaar voor deze datum.")

    # ---------------- Maanfase ----------------
    col3, col4 = st.columns([4, 6])

    with col3:
        st.subheader("Maanfase")
        st.caption(apod_data.get("date", ""))
        with st.spinner("Laden van maanfase..."):
            moon_image_url = get_moon_phase(selected_date.strftime("%Y-%m-%d"))
            if moon_image_url:
                st.image(moon_image_url, use_container_width=True)
            else:
                st.warning("Kon maanfase niet ophalen.")

    # ---------------- Neo ----------------
    with col4:
        st.subheader("Aardscheerders (NEO)")
        st.caption(apod_data.get("date", ""))
        with st.spinner("Laden van NEO data..."):
            df_neos = load_neo_for_date(selected_date.strftime("%Y-%m-%d"))
            if not df_neos.empty:
                st.dataframe(df_neos, height=300, use_container_width=True)
            else:
                st.info("Geen nabije objecten gevonden voor deze datum.")

# ================= Tab 2 =================
with tab2:
    st.header("Interactief NEO onderzoeksdashboard — Wat maakt een object gevaarlijk?")

    # Laad data
    df = load_and_prepare_neo_data()
    if df.empty:
        st.stop()

    # --- Sidebar filters voor tab 2 ---
    with st.sidebar:
        st.header("Dashboard Filters")

        # Bereken standaardwaarden (ongefilterde bereiken)
        diam_min = float(np.nanmin(df["Gemiddelde Diameter (km)"].dropna())) if not df["Gemiddelde Diameter (km)"].dropna().empty else 0.0
        diam_max = float(np.nanmax(df["Gemiddelde Diameter (km)"].dropna())) if not df["Gemiddelde Diameter (km)"].dropna().empty else 10.0
        miss_min = int(np.nanmin(df["miss_distance"].dropna())) if not df["miss_distance"].dropna().empty else 0
        miss_max = int(np.nanmax(df["miss_distance"].dropna())) if not df["miss_distance"].dropna().empty else 1_000_000

        # Gevaar checkbox - start met False (geen filtering)
        hazard_only = st.checkbox("Toon alleen potentieel gevaarlijke objecten", value=False)

        # Diameter slider met stapgrootte
        diam_range = st.slider(
            "Diameter range (km)",
            min_value=diam_min,
            max_value=diam_max,
            value=(diam_min, diam_max),  # Start ongefilterd
            step=0.1,
            format="%.1f"
        )

        # Mis afstand slider met stapgrootte
        miss_step = max(int((miss_max - miss_min) / 1000), 1000)
        miss_range = st.slider(
            "Mis Afstand range (km)",
            min_value=miss_min,
            max_value=miss_max,
            value=(miss_min, miss_max),  # Start ongefilterd
            step=miss_step,
            format="%d"
        )

        # Kies variabele om tegen diameter te plotten
        y_option = st.selectbox(
            "Y-as variabele",
            options=["miss_distance", "relative_velocity", "risk_score"],
            index=0,
            format_func=lambda x: {
                "miss_distance": "Mis Afstand (km)",
                "relative_velocity": "Snelheid (km/u)",
                "risk_score": "Risico Score"
            }[x]
        )

        labels_map = {
            "miss_distance": "Mis Afstand",
            "relative_velocity": "Snelheid)",
            "risk_score": "Risico Score"
        }

        # Toon logaritmische schaal?
        log_scale = st.checkbox("Logaritmische schaal (y-as)", value=False)

    # --- Pas filters toe ---
    df_filtered = df.copy()

    # Pas filters toe
    df_filtered = df_filtered[
        (df_filtered["Gemiddelde Diameter (km)"].between(diam_range[0], diam_range[1], inclusive="both")) &
        (df_filtered["miss_distance"].between(miss_range[0], miss_range[1], inclusive="both"))
    ]

    if hazard_only:
        df_filtered = df_filtered[df_filtered["is_hazardous"] == True]

    # --- Data kwaliteit & samenvattingssectie ---
    st.subheader("Data Samenvatting")

    # Belangrijkste statistieken, KPI's
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Totaal NEO's", f"{len(df):,}", help="Totaal aantal objecten in dataset")
    with k2:
        st.metric("Gefilterd", f"{len(df_filtered):,}", help="Aantal na filtering")
    with k3:
        avg_diam = df_filtered["Gemiddelde Diameter (km)"].mean()
        st.metric("Gem. Diameter", f"{avg_diam:.3f} km" if not np.isnan(avg_diam) else "N/A")
    with k4:
        top_risk = df_filtered["risk_score"].max()
        st.metric("Max Risico", f"{top_risk:.6f}" if not np.isnan(top_risk) else "N/A")

    # --- Layout: twee-kolom hoofdgebied ---
    st.markdown("---")
    colA, colB = st.columns([7, 5])

    # Links: Hoofd scatter plot
    with colA:
        st.subheader(f"Scatter: Diameter vs {labels_map.get(y_option, y_option)}")

        if len(df_filtered) > 0:
            # Sample data voor prestaties als er te veel punten zijn
            sample_size = min(10000, len(df_filtered))
            df_plot = df_filtered.sample(n=sample_size, random_state=42) if len(df_filtered) > sample_size else df_filtered

            fig_sc = px.scatter(
                df_plot,
                x="Gemiddelde Diameter (km)",
                y=y_option,
                color="is_hazardous",
                color_discrete_map={True: "#FF4444", False: "#44AA44"},
                hover_data=["name", "miss_distance", "relative_velocity", "risk_score"],
                title=f"Diameter vs {labels_map.get(y_option, y_option)} ({len(df_plot):,} objecten)",
                labels={
                    "is_hazardous": "Potentieel Gevaarlijk",
                    "Gemiddelde Diameter (km)": "Diameter (km)",
                    "miss_distance": "Mis Afstand (km)",
                    "relative_velocity": "Snelheid (km/u)",
                    "risk_score": "Risico Score"
                },
                template="plotly_white"
            )

            if log_scale and y_option in df_plot.columns:
                fig_sc.update_yaxes(type="log")

            fig_sc.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig_sc, use_container_width=True)

            if len(df_filtered) > sample_size:
                st.info(f"ℹ️ Voor prestaties worden {sample_size:,} van de {len(df_filtered):,} punten getoond")
        else:
            st.warning("⚠️ Geen data beschikbaar met huidige filters. Probeer de filters aan te passen.")

    # Rechts: Top lijsten
    with colB:
        st.subheader("Top Lijsten")

        if len(df_filtered) > 0:
            # Top 5 op risico score
            st.markdown("**Hoogste Risico**")
            top_by_risk = df_filtered.sort_values("risk_score", ascending=False).head(5)[
                ["name", "Gemiddelde Diameter (km)", "miss_distance", "risk_score", "is_hazardous"]
            ]
            st.dataframe(
                top_by_risk.reset_index(drop=True),
                use_container_width=True,
                height=200
            )

            st.markdown("**Grootste Diameter**")
            top_by_diam = df_filtered.sort_values("Gemiddelde Diameter (km)", ascending=False).head(5)[
                ["name", "Gemiddelde Diameter (km)", "miss_distance", "risk_score", "is_hazardous"]
            ]
            st.dataframe(
                top_by_diam.reset_index(drop=True),
                use_container_width=True,
                height=200
            )
            with st.expander("ℹ️ Uitleg: Hoe wordt de Risico Score berekend?"):
                st.markdown("""
                De **Risico Score** geeft een snelle indicatie van het relatieve risico van een object.

                **Formule:**

                ```
                Risico Score = Gemiddelde Diameter (km) ÷ Mis Afstand (km)
                ```

                **Belangrijk:**
                - **Groter object** → hoger risico
                - **Kleinere nadering** → hoger risico

                > ⚠️ Let op: Dit is een vereenvoudigde indicator en geen officiële NASA-methode.
                >
                > Zie voor de daadwerkelijke methode van NASA https://cneos.jpl.nasa.gov/sentry/.
                """)
        else:
            st.info("Geen data voor top lijsten")

    st.markdown("---")

    # --- Extra visualisaties rij ---
    viz_col1, viz_col2 = st.columns(2)

    # Boxplot
    with viz_col1:
        st.subheader("Risico Verdeling")
        if len(df_filtered) > 0:
            # Sample voor prestaties
            subset_box = df_filtered.sample(min(5000, len(df_filtered)), random_state=42)
            fig_box = px.box(
                subset_box,
                x="is_hazardous",
                y="risk_score",
                color="is_hazardous",
                color_discrete_map={True: "#FF4444", False: "#44AA44"},
                labels={
                    "is_hazardous": "Potentieel Gevaarlijk",
                    "risk_score": "Risico Score"
                },
                template="plotly_white"
            )

            if log_scale and "risk_score" in subset_box.columns:
                fig_box.update_yaxes(type="log")
            
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Geen data voor boxplot")

    # Histogram
    with viz_col2:
        st.subheader("Diameter Verdeling")
        if len(df_filtered) > 0:
            fig_hist = px.histogram(
                df_filtered,
                x="Gemiddelde Diameter (km)",
                color="is_hazardous",
                color_discrete_map={True: "#FF4444", False: "#44AA44"},
                nbins=50,
                labels={"is_hazardous": "Potentieel Gevaarlijk"},
                template="plotly_white"
            )
            fig_hist.update_yaxes(type="log")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Geen data voor histogram")

    # --- Correlatie heatmap (groter) ---
    st.subheader("Correlatie Matrix")
    corr_cols = [c for c in ["Gemiddelde Diameter (km)", "miss_distance", "relative_velocity", "risk_score"] if c in df.columns]

    if len(corr_cols) >= 2 and len(df_filtered) > 0:
        corr = df_filtered[corr_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlatie tussen variabelen",
            template="plotly_white",
            aspect="auto"
        )
        fig_corr.update_layout(
            height=600,  # Veel groter voor leesbaarheid
            width=800,
            font=dict(size=14)  # Grotere tekst
        )
        fig_corr.update_xaxes(side="bottom")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Niet genoeg numerieke kolommen voor correlatie matrix")

    # --- Data kwaliteit sectie ---
    with st.expander("Data Kwaliteit & Statistieken", expanded=False):
        col_stats1, col_stats2 = st.columns(2)

        with col_stats1:
            st.markdown("**Ontbrekende Waarden**")
            missings = df.isna().sum().rename("Ontbrekend").to_frame()
            st.dataframe(
                missings[missings["Ontbrekend"] > 0].sort_values("Ontbrekend", ascending=False),
                use_container_width=True
            )

        with col_stats2:
            st.markdown("**Basis Statistieken**")
            stat_cols = [c for c in ["Gemiddelde Diameter (km)", "miss_distance", "relative_velocity", "risk_score"] if c in df.columns]
            if stat_cols:
                st.dataframe(
                    df[stat_cols].describe().T.round(4),
                    use_container_width=True
                )

    # --- Gedetailleerde tabel + download ---
    st.markdown("---")
    st.subheader("Gedetailleerde Gegevens")

    display_cols = [c for c in ["name", "Gemiddelde Diameter (km)", "miss_distance", "relative_velocity", "risk_score", "is_hazardous"] if c in df.columns]

    if not df_filtered.empty:
        # Toon top resultaten gesorteerd op risico
        st.dataframe(
            df_filtered[display_cols].sort_values("risk_score", ascending=False).reset_index(drop=True),
            height=350,
            use_container_width=True
        )

        # Download knop
        csv_bytes = df_filtered[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Gefilterde Data (CSV)",
            data=csv_bytes,
            file_name=f"neos_filtered_{len(df_filtered)}_records.csv",
            mime="text/csv"
        )
    else:
        st.info("Geen records met de huidige filters. Probeer de filters aan te passen.")

    # --- Help sectie ---
    with st.expander("❓ Help & Tips", expanded=False):
        st.markdown("""
        ### Hoe gebruik je dit dashboard?

        **Filters:**
        - **Gevaarlijke objecten**: Bekijk alleen potentieel gevaarlijke asteroïden
        - **Diameter**: Filter op grootte van het object
        - **Afstand**: Filter op hoe dichtbij het object komt

        **Grafieken:**
        - **Scatter plot**: Ontdek relaties tussen variabelen
        - **Boxplot**: Vergelijk risico tussen gevaarlijke en veilige objecten  
        - **Histogram**: Bekijk de verdeling van object groottes
        - **Correlatie matrix**: Zie hoe sterk variabelen samenhangen

        **Tips:**
        - Combineer meerdere filters voor specifieke analyses
        - Gebruik de logaritmische schaal voor grote waarden
        - Hover over punten voor meer details
        - Download gefilterde data voor eigen analyses

        """)




