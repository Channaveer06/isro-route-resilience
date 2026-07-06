import streamlit as st
import torch
import numpy as np
from PIL import Image
import networkx as nx
import sys
import os
import folium
from streamlit_folium import st_folium
import matplotlib as mpl
import matplotlib.colors as mcolors
import uuid
from scipy import ndimage
from folium.features import DivIcon
import time

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from models.segmentation import UNet
from graph.skeletonize import extract_skeleton
from graph.topology import build_graph_from_skeleton, heal_topology
from graph.analysis import calculate_betweenness_centrality, find_critical_nodes
from graph.ablation import simulate_node_ablation, calculate_resilience_index

# --- Premium UI Configuration ---
st.set_page_config(page_title="ISRO Route Resilience", layout="wide", page_icon="🛰️")

# Deep Glassmorphism & Animated Custom CSS
st.markdown("""
    <style>
    /* 🚀 BULLETPROOF THEME ENFORCEMENT 🚀 */
    
    .stApp {
        background: radial-gradient(circle at top left, #ffffff, #eef2f5) !important;
    }
    
    /* FORCE THE SIDEBAR TO BE LIGHT REGARDLESS OF USER THEME */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e2e8f0 100%) !important;
        border-right: 1px solid #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1a365d !important;
    }
    
    /* FORCE MAIN BODY TEXT TO BE DARK ON THE LIGHT BACKGROUND */
    .stApp p, .stApp span, div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] h2, div[data-testid="stMarkdownContainer"] h3 {
        font-size: 1.25rem;
        color: #1a365d !important;
    }

    .main-header {
        color: #1a365d !important;
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        margin-bottom: 0px !important;
        text-align: left !important;
        letter-spacing: -1px !important;
    }
    .sub-header {
        color: #e28743 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 5px !important;
        margin-bottom: 40px !important;
        border-bottom: 3px solid #e28743 !important;
        padding-bottom: 15px !important;
    }
    
    /* Stunning Glass Cards with Hover Lifting */
    .glass-card {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 16px;
        padding: 30px;
        margin-top: 20px;
        box-shadow: 0 10px 30px -10px rgba(26, 54, 93, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px -10px rgba(226, 135, 67, 0.2);
        border: 1px solid rgba(226, 135, 67, 0.4);
    }
    
    /* Huge Pulse Metrics for Impact - TEXT MUST BE WHITE HERE */
    .metric-box {
        text-align: center;
        padding: 25px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1a365d 0%, #2a4a7f 100%) !important;
        box-shadow: 0 10px 20px rgba(26, 54, 93, 0.2);
    }
    .metric-box h2, .metric-box p, .metric-box span, .metric-box div {
        color: #ffffff !important;
    }
    .metric-box h2 {
        font-size: 4rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        text-shadow: 0px 4px 10px rgba(0,0,0,0.3) !important;
    }
    .metric-box p {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        margin: 0 !important;
    }
    
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(226, 135, 67, 0.6); }
        70% { box-shadow: 0 0 0 20px rgba(226, 135, 67, 0); }
        100% { box-shadow: 0 0 0 0 rgba(226, 135, 67, 0); }
    }
    .pulse-alert {
        animation: pulse-ring 2s infinite;
        border: 2px solid #e28743 !important;
        background: #1a365d !important;
    }
    
    /* Enlarge Sidebar Typography */
    .css-1544g2n {
        padding-top: 3rem;
    }
    .sidebar-title {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #1a365d !important;
        margin-top: 20px;
        border-bottom: 2px solid #1a365d;
        padding-bottom: 5px;
    }
    
    /* Make Tabs Huge */
    button[data-baseweb="tab"] p {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #1a365d !important;
    }
    
    /* Sliders, Uploader, and Dropdown labels */
    .stSlider label, .stFileUploader label, .stSelectbox label, .stSelectbox div {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #1a365d !important;
    }
    
    /* Make the native Metric text huge and visible */
    [data-testid="stMetricLabel"] p {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1a365d !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 3.5rem !important;
        color: #1a365d !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Titles EXACTLY as requested
st.markdown('<h1 class="main-header">Route Resilience</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility</p>', unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=1).to(device)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "models", "checkpoint.pth.tar")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device

model, device = load_model()

# --- Leaflet Folium Renderer ---
def draw_graph_folium(G, image, critical_nodes=[], disabled_node=None, original_G=None):
    bounds = [[0, 0], [512, 512]]
    m = folium.Map(location=[256, 256], zoom_start=1, crs='Simple', tiles=None, control_scale=False)
    
    tmp_path = f"temp_overlay_{uuid.uuid4().hex}.jpg"
    image.save(tmp_path)
    
    folium.raster_layers.ImageOverlay(
        image=tmp_path,
        bounds=bounds,
    ).add_to(m)
    
    centralities = nx.get_node_attributes(G, 'centrality').values()
    max_cent = max(centralities) if centralities else 1.0
    cmap = mpl.colormaps['YlOrRd'] 
    
    # Draw edges with Heatmap coloring
    for u, v, data in G.edges(data=True):
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        
        # Invert Y for Leaflet CRS.Simple
        lat_u, lon_u = 512 - pos_u[1], pos_u[0]
        lat_v, lon_v = 512 - pos_v[1], pos_v[0]
        
        cu = G.nodes[u].get('centrality', 0)
        cv = G.nodes[v].get('centrality', 0)
        avg_cent = (cu + cv) / 2
        
        is_healed = data.get('healed', False)
        
        if is_healed:
            folium.PolyLine(
                locations=[[lat_u, lon_u], [lat_v, lon_v]],
                color='#00d2ff', # Bright Cyan for visibility on light/dark backgrounds
                weight=5,
                dash_array='8, 8',
                opacity=0.9,
                tooltip="Auto-Healed by MST Algorithm"
            ).add_to(m)
        else:
            color = mcolors.to_hex(cmap(avg_cent / max_cent if max_cent > 0 else 0))
            weight = 5 if avg_cent > max_cent * 0.5 else 2.5
            
            folium.PolyLine(
                locations=[[lat_u, lon_u], [lat_v, lon_v]],
                color=color,
                weight=weight,
                opacity=0.8
            ).add_to(m)
        
    # Draw Gatekeeper Nodes with Permanent Labels
    for n, c in critical_nodes:
        if n == disabled_node:
            continue
        pos = G.nodes[n]['pos']
        lat, lon = 512 - pos[1], pos[0]
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color='#d32f2f',
            fill=True,
            fill_color='#d32f2f',
        ).add_to(m)
        
        # Permanent Text Label next to the dot
        folium.map.Marker(
            [lat, lon],
            icon=DivIcon(
                icon_size=(120,30),
                icon_anchor=(-12, 12),
                html=f'<div style="font-size: 13pt; color: #1a365d; font-weight: 900; background-color: rgba(255,255,255,0.95); padding: 4px 8px; border-radius: 6px; border: 2px solid #1a365d; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); white-space: nowrap;">NODE {n}</div>',
            )
        ).add_to(m)
        
    if disabled_node is not None and original_G is not None:
        pos = original_G.nodes[disabled_node]['pos']
        lat, lon = 512 - pos[1], pos[0]
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='black', icon='remove-sign'),
            tooltip=f"DISABLED NODE {disabled_node}"
        ).add_to(m)
        
    return m

# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=180)

st.sidebar.markdown('<p class="sidebar-title">📁 File Input</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload Satellite Imagery (JPG/PNG)", type=["jpg", "png"])

st.sidebar.markdown('<p class="sidebar-title">🧠 Core Settings</p>', unsafe_allow_html=True)
confidence_threshold = st.sidebar.slider("AI Neural Confidence", 0.05, 0.95, 0.30, help="Lower value detects fainter roads but increases noise.")
morph_kernel = st.sidebar.slider("Pixel Matrix Healing", 0, 40, 5, help="Fills small cuts in the raw pixel mask before graph extraction.")
heal_dist = st.sidebar.slider("MST Edge Connect Distance", 5.0, 300.0, 50.0, help="Max distance to connect broken road endpoints.")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((512, 512))
    img_array = np.array(image)
    
    # Interactive TABS for cleaner layout
    tab1, tab2, tab3 = st.tabs(["🛰️ Neural Extraction", "🕸️ Graph Topology", "🚨 Disaster Simulation"])
    
    with st.spinner("Processing deep neural networks & topology..."):
        tensor_img = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            preds = (torch.sigmoid(model(tensor_img)) > confidence_threshold).float()
            mask = preds[0][0].cpu().numpy()
            
            if morph_kernel > 0:
                mask = ndimage.binary_closing(mask, structure=np.ones((morph_kernel, morph_kernel))).astype(np.float32)
                
        skeleton = extract_skeleton(mask)
        G = build_graph_from_skeleton(skeleton)
        G = heal_topology(G, max_distance=heal_dist)
        if len(G.nodes) > 0:
            G = calculate_betweenness_centrality(G)
            critical_nodes = find_critical_nodes(G, top_k=5)
        else:
            critical_nodes = []

    # TAB 1: AI Vision
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card"><h3>Live Satellite Feed</h3></div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with col2:
            st.markdown('<div class="glass-card"><h3>AI Segmented Mask</h3></div>', unsafe_allow_html=True)
            st.image(mask, use_container_width=True, clamp=True)
            
    # TAB 2: Math Topology
    with tab2:
        st.markdown('<div class="glass-card"><h3>Mathematical Graph Engine Statistics</h3><p style="font-size: 1.1rem; color: #7f8c8d;">Showing real-time structural analysis of the road network.</p></div>', unsafe_allow_html=True)
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Total Intersections (Nodes)", f"{len(G.nodes):,}")
        stat_col2.metric("Total Connections (Edges)", f"{len(G.edges):,}")
        healed_count = sum([1 for u, v, d in G.edges(data=True) if d.get('healed')])
        stat_col3.metric("MST Healed Cuts", f"{healed_count}")

    # TAB 3: Core Analytics & Simulation
    with tab3:
        if len(G.nodes) > 0:
            col_map, col_controls = st.columns([2.5, 1.2])
            
            with col_controls:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("<h2>⚡ Simulation Engine</h2>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 1.1rem;'>Trigger a localized disaster (e.g., flooding, roadblock) at a critical Gatekeeper node to analyze cascade failure.</p>", unsafe_allow_html=True)
                
                node_options = [("🟢 System Normal (None)", None)] + [(f"🔴 Disable NODE {n}", n) for n, c in critical_nodes]
                selected_option = st.selectbox("Select Target:", options=node_options, format_func=lambda x: x[0])
                
                disabled_node = selected_option[1]
                
                report_text = "ISRO ROUTE RESILIENCE ENGINE - CRITICAL REPORT\n"
                report_text += "="*50 + "\n\n"
                
                if disabled_node is None:
                    ri = 1.0
                    st.markdown('<br><div class="metric-box"><p>Network Efficiency</p><h2>100%</h2></div>', unsafe_allow_html=True)
                    report_text += "STATUS: ALL SYSTEMS NOMINAL\nEfficiency: 100%\n"
                else:
                    G_ablated = simulate_node_ablation(G, disabled_node)
                    ri = calculate_resilience_index(G, G_ablated)
                    
                    travel_time_increase = (1.0 / ri - 1.0) * 100 if ri > 0 else float('inf')
                    
                    alert_class = "pulse-alert" if ri < 0.8 else ""
                    st.markdown(f'<br><div class="metric-box {alert_class}"><p>Network Survival</p><h2>{ri*100:.1f}%</h2></div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("Estimated Travel Time Delay", f"{travel_time_increase:.1f} mins", delta=f"+{travel_time_increase:.1f} mins", delta_color="inverse")
                    
                    report_text += f"DISASTER SIMULATED: NODE {disabled_node} OFFLINE\n"
                    report_text += f"Network Survival Rate: {ri*100:.1f}%\n"
                    report_text += f"Estimated Delay: +{travel_time_increase:.1f} mins\n"
                    
                    if ri < 0.8:
                        st.error("🚨 CRITICAL WARNING: Network fragmentation detected. Dispatch emergency routing immediately.")
                        report_text += "\nCRITICAL WARNING: Network fragmentation detected.\n"
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 GENERATE OFFICIAL REPORT",
                    data=report_text,
                    file_name="isro_official_report.txt",
                    mime="text/plain",
                    type="primary",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with col_map:
                st.markdown('<div class="glass-card" style="padding: 10px;">', unsafe_allow_html=True)
                plot_G = G if disabled_node is None else G_ablated
                m = draw_graph_folium(plot_G, image, critical_nodes, disabled_node, original_G=G)
                st_folium(m, width=900, height=700, returned_objects=[])
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👈 Please upload a satellite image from your dataset to launch the dashboard.")
