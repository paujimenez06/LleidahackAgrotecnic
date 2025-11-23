import streamlit as st
import pandas as pd
import pydeck as pdk
import sqlite3
import numpy as np

# configuració de la pàgina
st.set_page_config(
    page_title="Agrotècnic",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Mapa i taula per visualitzar")


# funció de càrrega de dades
@st.cache_data
def load_data_from_db():
    db_path = "logistics.db"
    try:
        conn = sqlite3.connect(db_path)

        df_res = pd.read_sql("SELECT * FROM results_simulation", conn)
        df_rutas = pd.read_sql("SELECT * FROM results_routes", conn)
        df_farms = pd.read_sql("SELECT * FROM farms", conn)
        df_slaughter = pd.read_sql("SELECT * FROM slaughterhouses", conn)

        conn.close()

        # correcció de columnes inexistents
        if "pigs_waiting" not in df_farms.columns:
            df_farms["pigs_waiting"] = 0

        return df_res, df_rutas, df_farms, df_slaughter

    except Exception:
        return None, None, None, None


# carregar dades
df_res, df_rutas, df_farms, df_slaughter = load_data_from_db()

# gestió d'errors de connexió
if df_res is None:
    st.error("Error de Connexió: No es troba la base de dades 'logistics.db'.")
    st.stop()

# barra lateral de controls
with st.sidebar:
    st.header("Panell de Control")
    max_dia = df_res["dia"].max() if not df_res.empty else 1
    dia_select = st.slider("Dia d'Operació", 1, int(max_dia), int(max_dia))

    st.divider()

    rutas_dia_raw = df_rutas[df_rutas["dia"] == dia_select]

    if not rutas_dia_raw.empty and "truck_id" in rutas_dia_raw.columns:
        lista_camiones = ["Tots"] + sorted(rutas_dia_raw["truck_id"].unique().tolist())
    else:
        lista_camiones = ["Tots"]

    camion_select = st.selectbox("Filtrar per Camió:", lista_camiones)

    st.info(f"Dades del Dia {dia_select}")


# filtratge de dades
if camion_select != "Tots":
    # Reset index para poder iterar con idx_ruta limpiamente
    rutas_dia = rutas_dia_raw[rutas_dia_raw["truck_id"] == camion_select].reset_index(
        drop=True
    )
    st.markdown(f"### Vista Detallada: Ruta del Camió {camion_select}")
else:
    rutas_dia = rutas_dia_raw.copy()


# secció de kpis globals
st.subheader("Rendiment Financer Acumulat")

total_ingresos = df_res["beneficio_bruto"].sum()
total_cerdos = df_res["cerdos_entregados"].sum()
total_pienso = df_res["coste_pienso"].sum()
total_alquiler = df_res["coste_alquiler"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Benefici Net Total", f"{total_ingresos:,.0f} €")
col2.metric("Porcs Entregats", f"{total_cerdos:,.0f}")
col3.metric("Despesa Pinso", f"-{total_pienso:,.0f} €")
col4.metric("Despesa Lloguer", f"-{total_alquiler:,.0f} €")

st.divider()


# secció de mapa i rutes
col_map, col_info = st.columns([3, 1])

with col_info:
    st.markdown(f"**Detalls del Dia {dia_select}:**")
    st.write(f"Viatges: {len(rutas_dia_raw)}")
    st.write(f"Porcs moguts: {rutas_dia_raw['cerdos'].sum()}")

    camiones_unicos = rutas_dia_raw["truck_id"].unique()
    st.write(f"Camions Actius: {len(camiones_unicos)}")
    st.caption(f"{', '.join(camiones_unicos)}")

with col_map:
    # preparació de dades de rutes
    def get_route_color_and_status(load_factor):
        if load_factor >= 0.90:
            return [255, 50, 50, 200], "Ple rendiment"
        else:
            return [0, 255, 100, 180], "Amb espai lliure"

    if not rutas_dia.empty:
        result_df = rutas_dia["load_factor"].apply(
            lambda x: pd.Series(get_route_color_and_status(x))
        )
        rutas_dia["color"] = result_df[0]
        rutas_dia["status_text"] = result_df[1]

        rutas_dia["pes_viu_mitja"] = (
            (rutas_dia["pes_total"] / rutas_dia["cerdos"]).fillna(0).round(1)
        )

        rutas_dia["html_tooltip"] = rutas_dia.apply(
            lambda row: f"""
            <div style='font-family: sans-serif; font-size: 12px; min-width: 160px;'>
                <b>Ruta/Viatge:</b> {row.name}<br/>
                <hr style='margin: 4px 0; border-color: #555;'/>
                <b>Estat:</b> {row["status_text"]}<br/>
                <b>Càrrega:</b> {row["pes_total"]:,.0f} kg / {row.get("truck_capacity", 0):,.0f} kg<br/>
                <b>Factor Càrrega:</b> {row["load_factor"] * 100:.1f}%<br/>
                <b>Nº Porcs:</b> {row["cerdos"]}<br/>
                <b>Pes Viu Mitjà:</b> {row["pes_viu_mitja"]} kg<br/>
                <b>Cost Variable:</b> {row["coste_ruta"]:,.2f} €<br/>
                <b>Granges visitades:</b> {row["n_granges_visitades"]}
            </div>
        """,
            axis=1,
        )
    else:
        for col in ["color", "status_text", "html_tooltip"]:
            rutas_dia[col] = pd.Series(dtype="object")

    # lògica de reconstrucció de ruta
    map_data_arcs = rutas_dia
    map_data_path = pd.DataFrame()
    map_data_text = pd.DataFrame()

    if camion_select != "Tots" and not rutas_dia.empty:
        all_paths = []
        all_texts = []

        for idx_ruta, row_ruta in rutas_dia.iterrows():
            n_paradas = row_ruta.get("n_granges_visitades", 1)
            path_coords = []

            # 1. Origen
            path_coords.append([row_ruta["lon_origen"], row_ruta["lat_origen"]])

            ordered_farm_ids = []

            if (
                "stops_str" in row_ruta
                and isinstance(row_ruta["stops_str"], str)
                and row_ruta["stops_str"]
            ):
                ordered_farm_ids = row_ruta["stops_str"].split(",")
            else:
                if n_paradas > 0:
                    posibles_granjas = df_farms.sample(n=min(n_paradas, len(df_farms)))
                    ordered_farm_ids = posibles_granjas["farm_id"].tolist()

            # 2. (Granja -> Granja)
            for i_stop, fid in enumerate(ordered_farm_ids):
                farm_row = df_farms[df_farms["farm_id"] == fid]
                if not farm_row.empty:
                    f_lon = farm_row.iloc[0]["lon"]
                    f_lat = farm_row.iloc[0]["lat"]
                    path_coords.append([f_lon, f_lat])

                    # Etiqueta: "Viatge.Parada" (ej. 1.1, 1.2, 2.1)
                    all_texts.append(
                        {
                            "position": [f_lon, f_lat],
                            "text": f"{idx_ruta + 1}.{i_stop + 1}",
                            "color": [255, 255, 255, 255],
                            "size": 16,
                        }
                    )

            # 3. Desti
            path_coords.append([row_ruta["lon_dest"], row_ruta["lat_dest"]])

            color_viaje = [255, 255, 0, 200]

            all_paths.append(
                {
                    "truck_id": camion_select,
                    "path": path_coords,
                    "color": color_viaje,
                    "html_tooltip": row_ruta["html_tooltip"],
                }
            )

        map_data_path = pd.DataFrame(all_paths)
        map_data_text = pd.DataFrame(all_texts)

        map_data_arcs = pd.DataFrame()

        with col_info:
            st.success(f"Full de Ruta: {camion_select}")
            st.markdown("---")

            for i, row in rutas_dia.iterrows():
                st.markdown(f"**Viatge {i + 1}**")
                st.write(f"Càrrega: {row['pes_total']:,.0f} kg")
                st.caption(
                    f"{row['slaughterhouse_id']} -> {row.get('stops_str', '...')} -> {row['slaughterhouse_id']}"
                )
                st.markdown("---")

    # preparació de dades de granges
    df_farms["pigs_waiting"] = df_farms["pigs_waiting"].fillna(0).astype(int)
    df_farms["inventory"] = df_farms["inventory"].fillna(0).astype(int)

    def get_farm_color(val):
        if val > 100:
            return [200, 0, 0, 200]
        elif val <= 50:
            return [0, 200, 0, 200]
        else:
            return [255, 165, 0, 200]

    df_farms["color"] = df_farms["pigs_waiting"].apply(get_farm_color)
    df_farms["html_tooltip"] = df_farms.apply(
        lambda row: f"""
        <div style='font-family: sans-serif; font-size: 12px;'>
            <b>Granja:</b> {row["farm_id"]}<br/>
            <b>Inventari:</b> {row["inventory"]}<br/>
            <b>A recollir:</b> {row["pigs_waiting"]} porcs
        </div>
    """,
        axis=1,
    )

    # preparació de dades d'escorxadors
    daily_slaughter_usage = (
        rutas_dia_raw.groupby("slaughterhouse_id")["cerdos"].sum().reset_index()
    )
    daily_slaughter_usage.rename(columns={"cerdos": "n_sacrificats_dia"}, inplace=True)

    df_slaughter_status = pd.merge(
        df_slaughter, daily_slaughter_usage, on="slaughterhouse_id", how="left"
    ).fillna(0)

    df_slaughter_status["n_sacrificats_dia"] = df_slaughter_status[
        "n_sacrificats_dia"
    ].astype(int)
    df_slaughter_status["ocupacio_pct"] = (
        df_slaughter_status["n_sacrificats_dia"] / df_slaughter_status["daily_limit"]
    )

    def get_slaughter_status(pct):
        if pct < 0.50:
            return [0, 255, 0, 200], "<50% ple (Verd)"
        elif pct < 0.90:
            return [255, 165, 0, 200], "50-90% ple (Taronja)"
        else:
            return [255, 0, 0, 200], "Ple (Vermell)"

    df_slaughter_status[["color", "status_label"]] = df_slaughter_status[
        "ocupacio_pct"
    ].apply(lambda x: pd.Series(get_slaughter_status(x)))

    df_slaughter_status["html_tooltip"] = df_slaughter_status.apply(
        lambda row: f"""
        <div style='font-family: sans-serif; font-size: 12px; min-width: 150px;'>
            <b>Escorxador:</b> {row["slaughterhouse_id"]}<br/>
            <hr style='margin: 4px 0; border-color: #555;'/>
            <b>Sacrificats AVUI:</b> {row["n_sacrificats_dia"]:,.0f} ({row["daily_limit"]:,.0f})<br/>
            <b>% Ocupació:</b> {row["ocupacio_pct"] * 100:.1f}%<br/>
            <b>Capacitat:</b> {row["status_label"]}
        </div>
    """,
        axis=1,
    )

    df_slaughter = df_slaughter_status

    # renderitzat final
    layer_farms = pdk.Layer(
        "ScatterplotLayer",
        data=df_farms,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=1000,
        pickable=True,
        auto_highlight=True,
    )

    layer_slaughter = pdk.Layer(
        "ScatterplotLayer",
        data=df_slaughter,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=3500,
        pickable=True,
        auto_highlight=True,
    )

    layer_arcs = pdk.Layer(
        "ArcLayer",
        data=map_data_arcs,
        get_source_position=["lon_origen", "lat_origen"],
        get_target_position=["lon_dest", "lat_dest"],
        get_source_color="color",
        get_target_color="color",
        get_width=4,
        get_tilt=15,
        pickable=True,
        auto_highlight=True,
    )

    layers_list = [layer_farms, layer_slaughter, layer_arcs]

    if camion_select != "Tots" and not map_data_path.empty:
        layer_path = pdk.Layer(
            "PathLayer",
            data=map_data_path,
            get_path="path",
            get_color="color",
            width_scale=1,
            width_min_pixels=3,
            get_width=5,
            pickable=True,
            auto_highlight=True,
        )

        layer_text = pdk.Layer(
            "TextLayer",
            data=map_data_text,
            get_position="position",
            get_text="text",
            get_color="color",
            get_size=20,
            get_angle=0,
            get_text_anchor="middle",
            get_alignment_baseline="center",
            get_background_color=[0, 0, 0, 200],
            get_border_color=[255, 255, 255],
            get_border_width=1,
            pickable=False,
        )

        # Reemplazamos los arcos con el path detallado
        layers_list = [layer_farms, layer_slaughter, layer_path, layer_text]

    view_state = pdk.ViewState(latitude=41.8, longitude=1.5, zoom=7.5, pitch=45)

    tooltip_config = {
        "html": "{html_tooltip}",
        "style": {
            "backgroundColor": "#111",
            "color": "white",
            "zIndex": "1000",
            "borderRadius": "8px",
        },
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=layers_list,
            tooltip=tooltip_config,
        )
    )

# gràfics finals
st.subheader("Evolució Econòmica")
st.line_chart(df_res, x="dia", y="beneficio_bruto")
with st.expander("Veure Taula de Resultats Detallada"):
    st.dataframe(df_res)
