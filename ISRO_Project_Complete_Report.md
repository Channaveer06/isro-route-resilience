# ISRO Route Resilience: Complete Project Documentation

## 1. Project Overview & Objective
This project is built for the **ISRO NNRMS Hackathon**. Its primary objective is to solve the critical problem of extracting continuous road networks from satellite imagery—even when occluded by trees, shadows, or clouds—and then analyzing the resilience of that network against disasters (e.g., flooding, roadblocks) using graph theory.

## 2. Complete Architecture & Pipeline Diagram
Below is the end-to-end pipeline of the project:

```mermaid
graph TD
    %% Styling
    classDef frontend fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;
    classDef backend fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:white;
    classDef ml fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:white;
    classDef graph fill:#e67e22,stroke:#d35400,stroke-width:2px,color:white;
    classDef data fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:black;

    subgraph "1. Data Acquisition & Pipeline (data_pipeline.py)"
        A[Google Earth Engine<br>Sentinel-2 Imagery]:::data --> B[Automated Alignment & Extraction]:::backend
        A2[OpenStreetMap<br>Vector Road Graphs]:::data --> B
    end

    subgraph "2. Model Training & Augmentation (src/training & src/models)"
        B --> C[src/data/dataset.py<br>Albumentations Augmentation]:::ml
        C --> D[src/models/segmentation.py<br>U-Net Deep Learning Model]:::ml
        D --> E[src/training/train.py<br>Dice & BCE Loss Optimization]:::ml
        E --> F[(checkpoint.pth.tar<br>Trained Weights)]:::ml
    end

    subgraph "3. Graph Reconstruction & Topology Healing (src/graph)"
        F --> G[src/graph/skeletonize.py<br>Medial Axis Transform]:::graph
        G --> H[src/graph/topology.py<br>MST Auto-Healing of Broken Roads]:::graph
        H --> I[src/graph/analysis.py<br>Betweenness Centrality & Bottlenecks]:::graph
        I --> J[src/graph/ablation.py<br>Disaster Simulation & Resilience Index]:::graph
    end

    subgraph "4. User Interface & Dashboard (app/main.py)"
        J --> K[Streamlit Glassmorphism UI]:::frontend
        F -.-> K
        K --> L[Interactive Folium Map<br>Real-time Disaster Impact]:::frontend
    end
```

## 3. Technologies & Models Used
- **Deep Learning Model Trained**: **U-Net** (A Fully Convolutional Network architecture perfect for precise, pixel-level semantic segmentation of roads).
- **Core Frameworks**:
  - **PyTorch**: Used for building and training the U-Net model and defining custom loss functions (`DiceBCELoss`). Uses Automatic Mixed Precision (AMP) for speed.
  - **Streamlit**: Used for the frontend to create a reactive, high-end glassmorphism web dashboard.
  - **NetworkX**: Used heavily in the backend to convert road pixels into mathematical graphs and run MST and Centrality algorithms.
  - **Folium / Leaflet**: Used for interactive geographical mapping and rendering graph overlaps on the frontend.
  - **Albumentations**: Used for aggressive data augmentation (adding fake shadows, rotations) during training to simulate bad weather.
  - **Google Earth Engine (geemap)** & **osmnx**: Used for automated extraction of Sentinel-2 satellite data and OSM ground truth.

---

## 4. Pin-to-Pin Detail: File & Folder Breakdown

### 📂 Root Directory
- **`data_pipeline.py`**
  - **Purpose**: Zero-manual-effort automated script to acquire training data.
  - **How it works**: It connects to Google Earth Engine (`geemap`) to download cloud-free Sentinel-2 satellite imagery for a specific city. It simultaneously pulls the mathematical driveable road graph from OpenStreetMap (`osmnx`) to serve as the ground truth.
- **`requirements.txt`**
  - **Purpose**: Lists every Python library required to run the project (e.g., torch, streamlit, networkx).
- **`README.md` & `PROJECT_DOCUMENTATION.md`**
  - **Purpose**: High-level overviews, setup instructions, and summaries of the topological healing engine.

### 📂 Frontend Interface
- **`app/main.py`**
  - **Section**: Frontend & UI Integration
  - **Purpose**: The core user-facing Streamlit application. 
  - **How it works**: 
    - Contains deep custom CSS injected via `st.markdown` to create a premium glassmorphism UI.
    - Loads the trained U-Net model weights (`checkpoint.pth.tar`) into GPU/CPU memory.
    - Allows the user to upload a satellite image, passes it through the AI, and extracts the road mask.
    - Coordinates the backend graph scripts to convert the mask into a healed NetworkX graph.
    - Renders the final result interactively using Folium maps.
    - Handles the **"Disaster Simulation"** tab, where a user can manually disable a "Gatekeeper Node" and instantly see the calculated Travel Time Delay and Network Survival Rate.

### 📂 Data Processing & Augmentation (Backend)
- **`src/data/dataset.py`**
  - **Section**: Backend Data Loader
  - **Purpose**: Feeds data into the AI during training.
  - **How it works**: Contains the `SatelliteRoadDataset` PyTorch class. It loads satellite images and their corresponding black-and-white road masks. Crucially, it uses `albumentations` to apply random transformations (rotations, shadows, pixel dropouts) so the model learns to detect roads even in terrible conditions, rather than just memorizing the exact dataset images.

### 📂 Model Architecture (Model Training)
- **`src/models/segmentation.py`**
  - **Section**: Deep Learning Model
  - **Purpose**: Defines the exact architecture of the AI.
  - **How it works**: Implements the **U-Net** architecture. It consists of a "down" path (encoders to capture context) and an "up" path (decoders for precise localization) connected via skip connections. This is the exact mathematical structure that is trained to look at an RGB image and output a road mask.

### 📂 Training Engine (Model Training)
- **`src/training/train.py`**
  - **Section**: Training Loop
  - **Purpose**: The main engine that teaches the AI how to find roads.
  - **How it works**: Uses an Adam optimizer and the U-Net model. It loops over the dataset (epochs), makes a prediction, compares it to the ground truth, calculates the error (loss), and adjusts the model's weights using Backpropagation. It utilizes `torch.cuda.amp.GradScaler` to train faster on modern GPUs. It saves the best results to `checkpoint.pth.tar`.
- **`src/training/losses.py`**
  - **Section**: Loss Functions
  - **Purpose**: Calculates how "wrong" the AI is during training.
  - **How it works**: Roads are very thin, meaning 90% of a satellite image is "Not Road". Standard accuracy metrics completely fail here. This file implements `DiceLoss` and `DiceBCELoss` (Binary Cross Entropy) which mathematically forces the AI to care about the thin road pixels rather than just guessing "background" all the time. It also calculates IoU (Intersection over Union).

### 📂 Graph Theory & Topology (Backend)
- **`src/graph/skeletonize.py`**
  - **Section**: Backend Graph Processing
  - **Purpose**: Thins down the thick AI prediction.
  - **How it works**: Uses `skimage.morphology.skeletonize` to reduce a blobby, multi-pixel wide road mask down to a single-pixel wide line. This is a mandatory step before converting pixels into a mathematical graph.
- **`src/graph/topology.py`**
  - **Section**: Backend Graph Processing
  - **Purpose**: The core "Healing" engine.
  - **How it works**: Converts the 1-pixel skeleton into a NetworkX graph (Nodes and Edges). If a tree occluded the road, the graph will have a cut. It uses a Disjoint Set (Union-Find) and Kruskal's Minimum Spanning Tree (MST) algorithm to mathematically search for disconnected endpoints and stitch them together using the shortest distance, bridging the gap left by the tree automatically.
- **`src/graph/analysis.py`**
  - **Section**: Backend Graph Processing
  - **Purpose**: Identifies the most critical points in the city network.
  - **How it works**: Calculates **Betweenness Centrality** for every node. A node with high centrality is a bottleneck (like a major bridge or main highway intersection). These are designated as "Gatekeeper Nodes".
- **`src/graph/ablation.py`**
  - **Section**: Backend Graph Processing
  - **Purpose**: Simulates disasters on the network.
  - **How it works**: Takes the full road graph, deletes one of the Gatekeeper nodes (simulating a flood or blockade), and re-calculates the shortest paths. It calculates a "Resilience Index" by comparing the average travel distance before and after the node failed. If the network shatters into pieces, it applies a catastrophic fragmentation penalty.
