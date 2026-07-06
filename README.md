# Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis

This project is built for the ISRO NNRMS hackathon. It provides an end-to-end pipeline for:
1. **Occlusion-Aware Extraction**: Deep learning models (U-Net/Transformer) to extract road continuity under tree cover, shadows, etc.
2. **Topological Reconstruction**: Converting pixel masks to vector graphs using MST healing.
3. **Structural Intelligence**: Identifying bottlenecks using Betweenness Centrality.
4. **Simulated Stress Testing**: Simulating network impacts (ablations) when critical nodes fail.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Place raw datasets (e.g. SpaceNet/DeepGlobe/Sentinel-2) in `data/raw/`.

3. Run the interactive dashboard:
   ```bash
   streamlit run app/main.py
   ```
