"""Streamlit deployment skeleton for Dynamic Emergency Vehicle Rerouting.

Visualizes live street network and dynamic pathfinding simulations
using an interactive Folium map rendered via streamlit-folium.
"""

from typing import Sequence
import folium
import streamlit as st
from streamlit_folium import st_folium

# Default coordinate center: Ahmedabad, Gujarat, India (Urban Road Network)
# Hard-coded for deployment skeleton to ensure quick rendering without GIS overhead.
DEFAULT_CENTER: list[float] = [23.0225, 72.5714]
DEFAULT_ZOOM: int = 12


def create_map(
    center: Sequence[float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
) -> folium.Map:
    """Create a Folium map centered at the specified coordinates.

    Args:
        center: Latitude and longitude pair [lat, lon].
        zoom: Initial zoom level for the interactive Leaflet map.

    Returns:
        Configured folium.Map instance.
    """
    # OpenStreetMap tiles provide clear street layouts suitable for emergency routing
    m = folium.Map(
        location=[center[0], center[1]],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Initial dispatch center marker (Ahmedabad City Center)
    folium.Marker(
        location=[center[0], center[1]],
        popup="Emergency Dispatch HQ (Ahmedabad)",
        tooltip="Dispatch Center",
        icon=folium.Icon(color="red", icon="plus", prefix="fa"),
    ).add_to(m)

    return m


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Emergency Vehicle Rerouting",
        page_icon="🚑",
        layout="wide",
    )

    st.title("🚑 Dynamic Emergency Vehicle Rerouting on Live Street Networks")
    st.markdown(
        "Interactive pathfinding simulation comparing **Baseline Dijkstra / A\\*** "
        "and **D\\* Lite** under dynamic urban traffic events (closures, congestion, and incident response)."
    )

    # Sidebar for simulation controls and network info
    with st.sidebar:
        st.header("Simulation Settings")
        st.info("Deployment Skeleton: Live street network visualization ready.")
        st.write(f"**Map Center:** Lat {DEFAULT_CENTER[0]}, Lon {DEFAULT_CENTER[1]}")
        st.write(f"**Default Zoom:** {DEFAULT_ZOOM}")

    # Main layout columns: Map visualization and status panel
    col_map, col_info = st.columns([3, 1])

    with col_map:
        st.subheader("Interactive Road Network")
        folium_map = create_map(DEFAULT_CENTER, DEFAULT_ZOOM)
        # Render map using streamlit-folium component
        st_folium(folium_map, width=800, height=520)

    with col_info:
        st.subheader("System Status")
        st.success("Deployment Skeleton Active")
        st.metric(label="Network Nodes", value="Ready")
        st.metric(label="Active Algorithm", value="D* Lite / A*")
        st.caption(
            "Note: Full routing algorithms and live event injection will be loaded "
            "interactively in subsequent phases."
        )


if __name__ == "__main__":
    main()
