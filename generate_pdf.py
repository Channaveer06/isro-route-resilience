import base64
import requests
from fpdf import FPDF
import sys
import os

mermaid_code = """
graph TD
    classDef frontend fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;
    classDef backend fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:white;
    classDef ml fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:white;
    classDef graphclass fill:#e67e22,stroke:#d35400,stroke-width:2px,color:white;
    classDef data fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:black;

    A[Google Earth Engine Data]:::data -->|Automated Script| B[src/data/dataset.py]:::backend
    A2[OSM Vector Graph]:::data -->|Ground Truth| B
    B -->|Augmented Tensors| C[src/models/segmentation.py]:::ml
    C -->|U-Net Architecture| D[src/training/train.py]:::ml
    D -->|DiceBCELoss Optimization| E[(checkpoint.pth.tar)]:::ml
    E -->|Loads Weights| F[app/main.py]:::frontend
    F -->|Raw Prediction Mask| G[src/graph/skeletonize.py]:::graphclass
    G -->|1-pixel skeleton| H[src/graph/topology.py]:::graphclass
    H -->|Kruskal MST Healing| I[src/graph/analysis.py]:::graphclass
    I -->|Betweenness Centrality| J[src/graph/ablation.py]:::graphclass
    J -->|Simulate Disaster| K[Streamlit Folium UI]:::frontend
"""

# Mermaid API uses urlsafe base64 of the plain code
encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
img_url = f"https://mermaid.ink/img/{encoded}"

print("Downloading Mermaid Diagram...")
res = requests.get(img_url)
if res.status_code != 200:
    print(f"Error downloading image: {res.status_code} - {res.text}")
    sys.exit(1)

with open("pipeline.png", "wb") as f:
    f.write(res.content)

print("Diagram downloaded. Generating PDF with FPDF...")

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 18)
        self.set_text_color(26, 54, 93)
        self.cell(0, 10, 'ISRO Route Resilience - Complete Project Report', border=0, new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, new_x='LMARGIN', new_y='NEXT', align='C')

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

html_content = """
<h2 align="center">1. Project Overview & Objective</h2>
<p>
This project is engineered for the <b>ISRO NNRMS Hackathon</b>. The primary goal is to extract road networks from noisy satellite imagery (where roads are blocked by trees or clouds) and mathematically analyze the network's resilience to disasters.
<br><br>
Standard AI fails when a tree covers a road (it predicts a broken road). Our project uses a hybrid architecture: a <b>Deep Learning model</b> (U-Net) to find the roads, and a <b>Graph Theory Topological Engine</b> (Minimum Spanning Tree) to automatically "heal" the broken roads and calculate the impact if a critical node fails (e.g., flooding).
</p>

<h2 align="center">2. Complete Pipeline Diagram</h2>
<p align="center">
<img src="pipeline.png" width="450">
</p>

<h2>3. Technologies Used & Their Purpose</h2>
<ul>
<li><b>PyTorch & U-Net (Machine Learning):</b> Used to train the neural network to perform pixel-level semantic segmentation (identifying exactly which pixels are roads).</li>
<li><b>NetworkX & Scipy (Graph Theory):</b> Used to convert the AI's pixel output into a mathematical graph (nodes and edges) to run topology algorithms like Betweenness Centrality.</li>
<li><b>Streamlit & Folium (Frontend):</b> Used to create the beautiful glassmorphism web dashboard and interactive maps.</li>
<li><b>Albumentations (Data Augmentation):</b> Used to inject artificial noise (shadows, rotations) into the training data so the AI learns to see through bad weather.</li>
<li><b>Google Earth Engine / geemap (Data Pipeline):</b> Used to automatically pull cloud-free Sentinel-2 satellite imagery.</li>
</ul>

<h2>4. Detailed File-By-File Breakdown</h2>

<h3>Frontend & User Interface</h3>
<b>app/main.py</b>
<br>
<b>What it is:</b> The core User Interface file powered by Streamlit.
<br>
<b>Why we used it:</b> It acts as the bridge between the user and the complex math models. It contains custom CSS to make the app look stunning (glassmorphism cards). It loads the AI weights, runs inference on a user's uploaded satellite image, and renders the healed graph interactively on a Leaflet map. Crucially, it handles the "Disaster Simulation" logic where a user disables a node to see the Travel Time Delay.

<h3>Machine Learning (Model & Training)</h3>
<b>src/models/segmentation.py</b>
<br>
<b>What it is:</b> Defines the exact mathematical architecture of the AI.
<br>
<b>Why we used it:</b> It implements a standard U-Net with skip connections. This is the industry standard for biomedical and satellite segmentation because it preserves high-resolution details (essential for thin roads).
<br><br>
<b>src/training/train.py & src/training/losses.py</b>
<br>
<b>What it is:</b> The engine that teaches the AI how to find roads.
<br>
<b>Why we used it:</b> Because roads make up only 5% of a satellite image, standard accuracy metrics fail. The <b>losses.py</b> file implements <b>DiceBCELoss</b> to force the AI to care about thin roads. <b>train.py</b> uses the Adam optimizer and PyTorch Automatic Mixed Precision to drastically speed up training on Kaggle GPUs. It outputs the <b>checkpoint.pth.tar</b> weights.

<h3>Data Processing Pipeline</h3>
<b>data_pipeline.py & src/data/dataset.py</b>
<br>
<b>What it is:</b> The scripts that feed information to the AI.
<br>
<b>Why we used it:</b> <b>data_pipeline.py</b> connects to Google Earth Engine to automatically pull raw Sentinel-2 imagery and matches it with OpenStreetMap vectors for zero-manual-effort dataset creation. <b>dataset.py</b> is the PyTorch DataLoader that applies Albumentations (fake shadows, dropouts) to mathematically torture the AI so it becomes robust against real-world noise.

<h3>Graph Theory & Topology (The Healing Engine)</h3>
<b>src/graph/skeletonize.py</b>
<br>
<b>What it is:</b> Thins the thick AI prediction down to a 1-pixel wide line using the Medial Axis Transform.
<br>
<b>Why we used it:</b> You cannot run graph math on thick blobs of pixels. It must be perfectly thin.
<br><br>
<b>src/graph/topology.py</b>
<br>
<b>What it is:</b> The core "Healing" algorithm.
<br>
<b>Why we used it:</b> If a tree covers a road, the AI draws a gap. This script converts the pixels to a NetworkX graph, uses a Disjoint Set (Union-Find) and <b>Kruskal's Minimum Spanning Tree (MST)</b> to mathematically hunt down broken dead-ends and stitch them together using the shortest possible path, completely healing the occlusion.
<br><br>
<b>src/graph/analysis.py & src/graph/ablation.py</b>
<br>
<b>What it is:</b> The disaster stress-testing modules.
<br>
<b>Why we used it:</b> <b>analysis.py</b> calculates "Betweenness Centrality" to find bottlenecks (major bridges). <b>ablation.py</b> simulates a catastrophic failure by deleting a bottleneck node and calculates the "Resilience Index" (ratio of average shortest paths before vs. after failure) to tell ISRO how vulnerable the network is.
"""

pdf.write_html(html_content)
pdf.output("ISRO_Project_Complete_Report.pdf")
print("PDF successfully generated!")
