# Deployment Guide: Streamlit Community Cloud

This guide covers deployment settings and operational metrics for the **Dynamic Emergency Vehicle Rerouting** web app.

---

## 1. Cloud Configuration Settings

On [Streamlit Community Cloud](https://share.streamlit.io):

| Setting | Value |
| :--- | :--- |
| **Repository** | `Wall-E-30/pathfinding-simulator` |
| **Branch** | `main` |
| **Main file path** | `app/app.py` |
| **Python Version** (Advanced Settings) | **`3.11`** or **`3.12`** |

> **Viva Note on Dependencies:** The deployment relies exclusively on `app/requirements.txt` containing only lightweight dependencies (`streamlit`, `folium`, `streamlit-folium`, `networkx`, `numpy`). Heavy C-extension GIS libraries (`osmnx`, `geopandas`, `gdal`, `pyproj`) are strictly excluded, ensuring quick builds and avoiding cloud memory limits.

---

## 2. Graph Data & Performance Metrics

- **Graph Storage:** `data/app_graph.pkl` (pre-processed and cropped road network).
- **File Size:** **0.39 MB (410 KB)** (well within Streamlit Cloud's 25 MB limit).
- **First-Load Time:**
  - Graph unpickling and heuristic initialization: **~20 ms**.
  - Total cold start to interactive map: **~1.5 seconds**.
- **Subsequent Reruns:** Instantaneous via `@st.cache_resource` and `st.session_state`.

---

## 3. Redeployment & Reboot

- **Automatic Redeploy:** Pushing commits to `main` automatically triggers a zero-downtime hot reload.
- **Manual Reboot:** If the app encounters any cloud timeout or state lock, click the three-dot menu (`⋮`) in the lower-right corner of the web page and select **Reboot app**.
- **Memory Contingency:** If the container encounters resource constraints, `scripts/export_app_graph.py` can be re-run with a smaller `RADIUS_M` parameter to produce a more compact pickle.
