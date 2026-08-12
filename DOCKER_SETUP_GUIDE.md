# 🚀 ISRO Route Resilience: Quick Start Guide (Docker)

Welcome to the ISRO Route Resilience project! We have fully **Dockerized** this repository. This means you do **not** need to manually install PyTorch, setup Python environments, or debug missing system libraries (like OpenCV bindings). Docker handles everything in an isolated, guaranteed-to-work container.

If you don't have Docker installed yet, don't worry. Just follow the steps below!

---

## 🛠️ Step 1: Install Docker Desktop
Since you don't have Docker on your laptop, you need to install it first.

1. **Download Docker Desktop:**
   - Go to the official Docker website: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
   - Click **Download for Windows** (or Mac/Linux depending on your system).
2. **Install and Run:**
   - Run the installer you just downloaded. Leave the default settings (it will likely recommend using the WSL 2 backend on Windows, which is great).
   - Once installed, open the **Docker Desktop** application from your start menu. 
   - *Note: Docker Desktop must be running in the background for you to use Docker commands!*

---

## 📥 Step 2: Download the Project from GitHub
Next, you need to get the code onto your computer.

1. Open your terminal (Command Prompt or PowerShell on Windows).
2. Clone the repository by running:
   ```bash
   git clone https://github.com/YOUR_USERNAME/isro-route-resilience.git
   ```
   *(Make sure to replace the URL with the actual GitHub link to this repository!)*
3. Navigate into the downloaded folder:
   ```bash
   cd isro-route-resilience
   ```

---

## 🐳 Step 3: Build and Run the Docker Image
Now that you have the code and Docker is running, it's incredibly simple to start the project. We just need to build the "Image" (which packages all the code and libraries together) and then run it.

**1. Build the Docker Image**
Make sure you are in the `isro-route-resilience` folder, then run:
```bash
docker build -t isro-project .
```
*(This command looks at the `Dockerfile` we created, downloads Python, installs PyTorch and OpenCV, and prepares the model. It might take a few minutes the first time!)*

**2. Run the Docker Container**
Once it finishes building, start the UI server by running:
```bash
docker run -p 8501:8501 isro-project
```
- The `-p 8501:8501` flag is very important! It forwards the internal Docker port to your laptop's port so you can access the website.

---

## 🌍 Step 4: Access the Dashboard!
The server is now running! 

Open your favorite web browser (Chrome, Edge, Safari) and go to:
### **[http://localhost:8501](http://localhost:8501)**

You should instantly see the beautifully designed Streamlit Glassmorphism Interface for the ISRO Route Resilience project. You can upload satellite images and simulate disasters directly from your browser!

---

### 💡 Extra Tips for Developers
- **Stopping the server:** If you want to stop the server, just go back to your terminal where it is running and press `Ctrl + C`.
- **Editing Code:** If you edit any Python files in the folder (like `app/main.py`), you will need to stop the container and run `docker build -t isro-project .` again so Docker can package your new changes into the image!
- **Data ignored:** We intentionally added the massive 4GB `training_files.zip` to a `.dockerignore` file. This ensures your Docker image remains extremely fast and lightweight!
