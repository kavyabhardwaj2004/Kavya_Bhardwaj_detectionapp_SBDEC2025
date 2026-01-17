# 🏛️ HeritageLens AI — Preserving History Through Technology

HeritageLens AI is a comprehensive computer vision platform designed to detect and analyze heritage sites, archaeological structures, and cultural landmarks. Powered by YOLOv11 and Random Forest, it offers real-time site mapping, environmental risk assessment (erosion prediction), and professional reporting tools for researchers and historians.

## 🌟 Core Modules & Features
### 📸 1. Image Detection Module

Multiple Uploads: Support for batch processing of heritage site photos.

Bounding Boxes: High-precision identification of architectural elements.

Detections Statistics: Real-time summary of objects found, average confidence, and most common classes.

Automated Reporting: Instant generation of professional PDF reports with detection highlights.

### 🎥 2. Video Analysis Module

Dynamic Inputs: Support for local .mp4, .avi, .mov files or direct YouTube links.

Real-time Processing: Live frame-by-frame analysis with on-screen bounding boxes.

Stats Tracker: Monitor processed frames, elapsed time, and detection progress.

Export Capabilities: Download processed videos and dedicated video analysis reports.

### 📊 3. Interactive Summary Dashboard

Global Overview: Merged analytics from both your image and video sessions.

Advanced Visuals:

Detection distribution by heritage class.

Confidence score frequency histograms.

Trend analysis and class-wise performance.

Download Hub: Export a combined master report for official documentation.

### ⏳ 4. Soil Erosion Susceptibility Prediction

Risk Assessment: Specifically analyzes the site terrain to predict the impact of erosion on ancient structures.

Erosion Heatmaps: Generates specialized visual overlays to highlight vulnerable areas.

ML Integration: Uses a Random Forest model to output risk levels (Low/Medium/High) and exact probability scores.

### 📚 5. Educational Portal & Documentation

Heritage Knowledge Base: Detailed information on characteristics of Temples, Forts, and Megaliths.

Conservation Guidelines: Learn about global best practices for archaeological preservation and AI’s role in modern history.

## 🏺 Heritage Classes Identified

The system is custom-trained to recognize four specific categories:

Heritage Sites: 🏛️ Temples, palaces, forts, museums, and historical monuments.

Stones/Structures: 🗿 Stone pillars, walls, megaliths, and architectural remnants.

Crops/Farmland: 🌾 Traditional farming areas and irrigation landscapes.

Non-Archaeological: 🏔️ Natural geography like deserts, mountains, and water bodies.

## 🛠️ Technical Stack

Frontend: Streamlit with custom CSS (Earthy tones: Bronze, Sandstone, Olive).

Computer Vision: YOLOv11 (Ultralytics) & OpenCV.

Predictive Analytics: Scikit-Learn (Random Forest Model).

Visualization: Plotly, Matplotlib, and Seaborn.

Language: Python 3.10+.

Data Export: ReportLab for professional PDF formatting.

## 📦 Installation & Setup
### Step 1: Repository Clone

```bash
git clone https://github.com/your-username/HeritageLens-AI.git
cd HeritageLens-AI
```

### Step 2: Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verification

Ensure the trained weights file (best.pt) is in the project root folder.

## 🚀 Execution

To launch the application:
```bash
streamlit run app.py
```

Access: Open http://localhost:8501 in your browser.

## 🎨 Design & Accessibility

UI/UX: Earthy, professional color palette designed for high contrast and readability.

Responsive: Sidebar navigation allows seamless transitions between Analysis and Analytics.

Hardware Support: Optimized for both CPU and CUDA-compatible NVIDIA GPUs for faster video processing.

## 📄 License & Acknowledgments

License: MIT License.

Credits: Special thanks to Ultralytics (YOLOv11), Streamlit, and the archaeology community for providing site documentation datasets.

### 🏛️ HeritageLens AI — Preserving our shared past for a digital future.
