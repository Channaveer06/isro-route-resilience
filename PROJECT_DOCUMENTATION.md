# ISRO Route Resilience & Topological Healing Engine
## Comprehensive Project Documentation

### 1. What is this Project?
This project was engineered for an ISRO (Indian Space Research Organisation) hackathon. The core challenge was to extract a topological road network from satellite imagery and mathematically analyze its resilience to disaster events (like severe flooding or roadblocks). 

Satellite imagery often suffers from occlusions (trees, building shadows, or clouds blocking the view of the road). Standard AI models fail here, creating "broken" roads. This project solves that using a hybrid approach: a Deep Learning model to extract the initial roads, and a rigorous Mathematical Graph Theory engine to automatically "heal" the broken roads and test network survival rates.

---

### 2. Step-by-Step Procedure: How We Built It
1. **Phase 1: Data Preparation:** We utilized a satellite road extraction dataset. We built a PyTorch `DataLoader` with the `Albumentations` library to augment the images (adding artificial shadows, rotations, and pixel dropouts). This forced the AI to learn how to handle noisy, imperfect satellite conditions.
2. **Phase 2: Deep Learning (Semantic Segmentation):** We trained a PyTorch U-Net convolutional neural network to classify every single pixel in the image as "Road" or "Not Road", producing a raw black-and-white pixel mask.
3. **Phase 3: Mathematical Morphology:** We applied a computer vision technique called "Pixel Closing" to physically melt nearby pixels together, bridging tiny micro-cuts in the raw AI output.
4. **Phase 4: Skeletonization:** We used the Medial Axis Transform to thin the road pixels down to a 1-pixel wide line, perfectly preserving the mathematical connectivity (Euler characteristic) of the roads.
5. **Phase 5: Graph Theory & Topological Healing:** We converted the 1-pixel skeleton into a NetworkX mathematical graph. Because trees caused massive gaps in the roads, we implemented a custom Disjoint Set (Union-Find) and Kruskal's Minimum Spanning Tree (MST) algorithm. This mathematically searched for broken dead-ends and stitched them back together using the shortest possible distance without creating infinite loops.
6. **Phase 6: Criticality Analysis:** We calculated the "Betweenness Centrality" for every intersection to find the most vulnerable "Gatekeeper Nodes" (bottlenecks).
7. **Phase 7: Disaster Simulation UI:** We built a beautiful glass-morphism dashboard with Streamlit and Leaflet.js. We allowed the user to manually disable a critical node, which triggers our catastrophic fragmentation math to calculate the new Network Survival Rate and Travel Time Delays in real-time.

---

### 3. Technology Stack: What We Used and WHY
*   **PyTorch (Deep Learning Framework)**
    *   *Why we used it:* It allows for custom loss functions (DiceBCELoss) and supports Automatic Mixed Precision (`torch.cuda.amp`) which allowed us to train the model on Kaggle GPUs at lightning speed.
    *   *Why not TensorFlow/Keras?:* PyTorch provides deeper access to tensor manipulation, which is critical for custom computer vision tasks, and is the current industry standard for AI researchers.
*   **U-Net (Neural Architecture)**
    *   *Why we used it:* U-Net is a Fully Convolutional Network designed originally for biomedical imaging. It uses "skip connections" to preserve high-resolution spatial details, making it absolutely perfect for extracting thin, winding roads from 512x512 images.
    *   *Why not YOLO or ResNet?:* YOLO is for drawing bounding boxes (detecting cars), not pixels. ResNet is for classifying whole images (e.g., "Is this a dog?"), not drawing a pixel-perfect map.
*   **NetworkX & Kruskal's MST (Graph Theory)**
    *   *Why we used it:* NetworkX allows us to treat intersections as "Nodes" and roads as "Edges". Kruskal's Minimum Spanning Tree mathematically guarantees that we bridge the shortest distance between broken dead-ends without accidentally creating redundant circular loops.
*   **Streamlit & Leaflet.js (Dashboard)**
    *   *Why we used it:* Streamlit allows for rapid deployment of Python AI models into highly interactive web apps without writing complex React/Node.js backend servers. Leaflet handles the complex geospatial rendering.

---

### 4. How to Run the Project
**Step 1: Setup the Environment**
Open your command prompt and ensure all libraries are installed:
```cmd
pip install torch torchvision numpy Pillow albumentations networkx scipy streamlit folium streamlit-folium matplotlib
```

**Step 2: Run the Web Dashboard (Presentation Mode)**
Open a command prompt in your `isro_hack` folder and run:
```cmd
streamlit run app/main.py
```
*Note: This will automatically load the highly-trained 26-epoch model weights from `src/models/checkpoint.pth.tar` and launch the browser interface.*

**Step 3: Training the Model from Scratch (Optional)**
If you ever want to re-train the model, run:
```cmd
python src/training/train.py
```
*(CRITICAL NOTE: It is highly recommended to upload the dataset to Kaggle.com and run the training script on a free Kaggle T4 GPU. Training 50 epochs on a standard Windows CPU will take days).*
