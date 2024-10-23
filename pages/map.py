import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import time

geojson_path = 'datasets/CSV/map/mesaugat-geoJSON-Nepal-37f73c5/mesaugat-geoJSON-Nepal-37f73c5/hotosm_npl_waterways_polygons_geojson.geojson'

st.title("Waterways Map")

# Loading spinner
with st.spinner("Loading map..."):
    time.sleep(1)  
    try:
        with open(geojson_path, encoding='utf-8') as f:
            geojson_data = json.load(f)

        nepal_map = folium.Map(location=[28.3949, 84.1240], zoom_start=6)

        folium.GeoJson(
            geojson_data,
            name='Waterways',
            tooltip=folium.GeoJsonTooltip(
                fields=['natural', 'water', 'source', 'waterway', 'covered', 'depth', 'width', 'blockage'],
                aliases=['Natural:', 'Water:', 'Source:', 'Waterway:', 'Covered:', 'Depth:', 'Width:', 'Blockage:'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    font-family: Arial, 
                    color: black;
                    font-size: 12px;
                    background-color: white;
                    border: 1px solid grey;
                    border-radius: 3px;
                    padding: 5px;
                """,
            ),
            style_function=lambda feature: {
                'fillColor': 'blue',
                'color': 'blue',
                'weight': 1,  
                'fillOpacity': 0.5,
            }
        ).add_to(nepal_map)

        st_folium(nepal_map, returned_objects=[])

    except FileNotFoundError:
        st.error("GeoJSON file not found. Please check the path.")
    except json.JSONDecodeError:
        st.error("Error decoding the GeoJSON file. Please check the file format.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
