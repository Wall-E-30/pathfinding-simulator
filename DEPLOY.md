# Deployment Guide: Streamlit Community Cloud

This guide provides instructions for deploying the **Dynamic Emergency Vehicle Rerouting** web simulator to Streamlit Community Cloud.

---

## Deployment Settings

1. **Repository**: `Wall-E-30/pathfinding-simulator`
2. **Branch**: `main`
3. **Main File Path**:
   ```text
   app/app.py
   ```
4. **App Requirements File**:
   Streamlit Cloud will automatically detect and install [`app/requirements.txt`](app/requirements.txt).
   - Contains only 5 lightweight pure dependencies: `streamlit`, `folium`, `streamlit-folium`, `networkx`, `numpy`.
   - Explicitly avoids heavy C-extensions and GIS binaries (`osmnx`, `geopandas`, `gdal`, `pyproj`).
5. **Advanced Settings**:
   - Under **Python Version**, select **3.11** (or **3.12**).

---

## Known Load Times & Performance

- **Cold Boot / Initial Deployment**: ~15–30 seconds for Streamlit container startup and wheel installation.
- **Graph Ingestion & Heuristic Cache (`data/app_graph.pkl`)**: **~25 ms**. The pre-cropped 3,015-node graph unpickles and computes the pre-cached $v_{\max}$ heuristic almost instantaneously on first load.
- **Interactive Map Rendering**: **< 1.0 second**. The Leaflet/Folium component renders immediately on initial load and stays responsive during interactions (`returned_objects=[]` prevents map panning/zooming from triggering Python reruns).
- **Scenario Simulation**: Dynamic scenario execution (6 events across Baseline A* and D* Lite) executes in **~50–80 ms**.

---

## Live Application

- **URL**: [https://pathfinding-simulator.streamlit.app/](https://pathfinding-simulator.streamlit.app/)

