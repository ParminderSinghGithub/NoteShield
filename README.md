# 🛡️ NoteShield - AI-Powered Indian Currency Authentication System

<div align="center">

![NoteShield Banner](https://img.shields.io/badge/NoteShield-Currency%20Authentication-blue?style=for-the-badge&logo=shield)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://parmindersinghgithub-noteshield-main-bw7u2m.streamlit.app/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/parmindersingh2002)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

**Advanced AI system for authenticating Indian currency notes using deep learning and computer vision**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Datasets](#-datasets)
- [Notebooks](#-notebooks)
- [System Architecture](#%EF%B8%8F-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation & Setup](#%EF%B8%8F-installation--setup)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**NoteShield** is a comprehensive AI-powered authentication system designed specifically for Indian currency notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000). The system employs a sophisticated **dual-arm architecture** combining:

- **CNN Arm**: MobileNetV2-based deep learning model for visual pattern recognition
- **DSV Arm**: Document Security Verification analyzing watermarks, RBI seals, security threads, serial numbers, and texture patterns
- **Fusion Layer**: Intelligent combination of both arms for robust authentication decisions

The system provides real-time denomination classification and counterfeit detection through an intuitive Streamlit web interface, supporting both single-note authentication and batch processing.

---

## ✨ Key Features

### 🔍 **Authentication Capabilities**
-  Real-time authentication via **webcam capture** or **image upload**
-  **8-class denomination recognition** (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000, Counterfeit)
-  **Binary authenticity classification** (Genuine/Fake) with confidence scores
-  **Batch processing** for multiple notes simultaneously
-  **Explainable AI** with visual analysis of model decisions

### 🧠 **Advanced Technology**
- 🎯 **Dual-Head MobileNetV2** architecture for denomination + authenticity
- 🔬 **DSV Feature Extraction** (Watermark, RBI Seal, Security Thread, Serial Number, Texture)
- ⚙️ **Configurable fusion weights** for CNN and DSV arms
- 📊 **Comprehensive performance metrics** and confusion matrices
- 🎨 **Beautiful, responsive UI** with light/dark mode support

### 📈 **Analysis & Reporting**
- 📉 Model performance dashboards with accuracy metrics
- 📁 CSV export for batch processing results
- 🔧 Real-time fusion control panel
- 💡 Technical insights and architecture documentation

---

## 🚀 Live Demo

### 🌐 **Streamlit Web Application**

Experience NoteShield in action:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parmindersinghgithub-noteshield-main-bw7u2m.streamlit.app/)


**Features Available in Demo:**
- Single note authentication (upload or webcam)
- Batch processing for multiple notes
- Model performance visualization
- Fusion controls adjustment
- Technical architecture insights

---

## 📊 Datasets

All datasets are hosted on **Kaggle** for easy access and reproducibility.

### 📦 **Available Datasets**

| Dataset Name | Description | Kaggle Link |
|-------------|-------------|-------------|
| **NoteShield Dataset** | Raw Indian currency note images for training | [![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/parmindersingh2002/cvpr-mini-project-dataset) |
| **NoteShield Features** | Extracted DSV features and embeddings | [![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/parmindersingh2002/cvpr-mini-project-dsv-dataset) |

### 📝 **Dataset Structure**

```
NoteShield Dataset/
├── train/
│   ├── denomination_10/
│   ├── denomination_20/
│   ├── denomination_50/
│   ├── denomination_100/
│   ├── denomination_200/
│   ├── denomination_500/
│   ├── denomination_2000/
│   └── denomination_fake/
├── validation/
└── test/

NoteShield Features/
└── dsv_features

```

---

## 📓 Notebooks

Complete end-to-end pipeline implemented in **4 comprehensive Jupyter notebooks** on Kaggle.

### 🔗 **Kaggle Notebooks**

> **⚠️ Important**: All notebooks are designed to run on **Kaggle** where datasets are readily available. Before running any notebook, **verify input dataset paths** as dataset names may cause path errors.

| # | Notebook | Description | Kaggle Link | Required Inputs |
|---|----------|-------------|-------------|----------------|
| **1** | 📊 Data Preparation | Dataset exploration, class distribution analysis, train/val/test split | [![Open in Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/parmindersingh2002/1-data-preparation) | • NoteShield Dataset |
| **2** | 🔧 Preprocessing | Image preprocessing, geometric warping, template matching (ORB/SIFT/AKAZE), homography transformations | [![Open in Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/parmindersingh2002/2-preprocessing) | • NoteShield Dataset |
| **3** | 🧬 DSV Feature Extraction | Hybrid feature extraction: LBP, GLCM, FFT, edge detection, keypoints, MobileNetV2 embeddings | [![Open in Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/parmindersingh2002/03-dsv-feature-extraction-hybrid) | • NoteShield Dataset<br>• Notebook 2 (Preprocessing) |
| **4** | 🤖 CNN Training & Evaluation | Dual-head MobileNetV2 training, model evaluation, confusion matrices, ROC/PR curves | [![Open in Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/parmindersingh2002/04-cnn-training-dual-head-and-visualization) | • NoteShield Features Dataset<br>• Notebook 2 (Preprocessing) |

### 📋 **Notebook Dependencies**

```
Notebook 1: Data Preparation
    └── Input: NoteShield Dataset

Notebook 2: Preprocessing
    └── Input: NoteShield Dataset

Notebook 3: DSV Feature Extraction
    ├── Input 1: NoteShield Dataset
    └── Input 2: Notebook 2 (as Kaggle input)

Notebook 4: CNN Training & Visualization
    ├── Input 1: NoteShield Features Dataset
    └── Input 2: Notebook 2 (as Kaggle input)
```

### ⚙️ **Running Notebooks on Kaggle**

1. **Open the notebook** on Kaggle using the links above
2. **Add required inputs**:
   - Click "Add Input" → Search for dataset names
   - For notebooks requiring previous notebooks, add them as inputs
3. **Verify paths**: Check all file paths in the first few cells
4. **Enable GPU**: Settings → Accelerator → GPU (for Notebook 2, 3 and 4)
5. **Run all cells**: Click "Run All" or execute cells sequentially

---

## 🏗️ System Architecture

### 🎯 **Dual-Arm Approach**

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                               │
│                    (224x224 RGB Image)                           │
└───────────────┬─────────────────────┬───────────────────────────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼───────────┐
        │   CNN ARM      │    │    DSV ARM       │
        │  MobileNetV2   │    │ Security Feature │
        │   Dual-Head    │    │   Extraction     │
        └───────┬────────┘    └──────┬───────────┘
                │                     │
        ┌───────▼────────┐    ┌──────▼───────────┐
        │ • Denomination │    │ • Watermark      │
        │ • Authenticity │    │ • RBI Seal       │
        │   Predictions  │    │ • Security Thread│
        │                │    │ • Serial Number  │
        │ Confidence:    │    │ • Texture Pattern│
        │   Softmax      │    │                  │
        └───────┬────────┘    └──────┬───────────┘
                │                     │
                └──────────┬──────────┘
                           │
                    ┌──────▼──────┐
                    │ FUSION LAYER│
                    │  Weighted   │
                    │ Combination │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │FINAL VERDICT│
                    │ • Denomination
                    │ • Authenticity
                    │ • Confidence
                    └─────────────┘
```

### 🧠 **CNN Architecture Details**

**Base Model**: MobileNetV2 (ImageNet pretrained)
- **Input**: 224×224×3 RGB images
- **Preprocessing**: MobileNetV2 standard normalization
- **Feature Extraction**: Global Average Pooling
- **Dual Heads**:
  - **Denomination Head**: 8-class softmax (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000, Fake)
  - **Authenticity Head**: Binary sigmoid (Genuine/Counterfeit)

### 🔬 **DSV Feature Pipeline**

1. **Watermark Detection**: Template matching + correlation analysis
2. **RBI Seal Verification**: Feature matching (ORB/SIFT descriptors)
3. **Security Thread**: Edge detection + spatial analysis
4. **Serial Number Validation**: OCR + format verification
5. **Texture Analysis**: GLCM + LBP feature extraction

### ⚙️ **Fusion Strategy**

```python
final_auth_score = (CNN_score × α) + (DSV_score × (1-α))

where α = CNN weight (configurable: 0.0 to 1.0, default: 0.7)
```

**Decision Thresholds**:
- Authentic: `final_auth_score ≥ 0.5`
- Uncertain: `0.4 ≤ final_auth_score < 0.5`
- Counterfeit: `final_auth_score < 0.4`

---

## 🛠️ Technology Stack

### **Core Frameworks**
- ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white) **Python 3.8+**
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?logo=tensorflow&logoColor=white) **TensorFlow 2.12+ / Keras**
- ![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white) **Streamlit 1.30+**

### **Computer Vision & ML**
- ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white) **OpenCV 4.8+**
- ![NumPy](https://img.shields.io/badge/NumPy-1.23+-013243?logo=numpy&logoColor=white) **NumPy 1.23+**
- ![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?logo=pandas&logoColor=white) **Pandas 1.5+**
- ![Pillow](https://img.shields.io/badge/Pillow-9.5+-3776AB?logo=python&logoColor=white) **Pillow 9.5+**

### **Visualization & UI**
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?logo=python&logoColor=white) **Matplotlib 3.7+**
- **Custom CSS** for responsive design

---

## ⚙️ Installation & Setup

### 📋 **Prerequisites**

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- Git (for cloning repository)

### 🔧 **Step-by-Step Installation**

#### **1. Clone the Repository**

```bash
git clone https://github.com/ParminderSinghGithub/NoteShield.git
cd NoteShield
```

#### **2. Create Virtual Environment** (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes (depending on internet speed)

#### **4. Verify Model Files**

Ensure the following files exist in `results/final_artifacts/`:

```
results/final_artifacts/
├── dual_head_mobilenetv2_final.keras  # Main model file (~332 MB)
├── label_mapping.json                  # Class label mappings
├── confusion_denom.png                 # Denomination confusion matrix
├── confusion_auth.png                  # Authenticity confusion matrix
├── roc_auth.png                        # ROC curve
└── pr_auth.png                         # Precision-Recall curve
```

> **Note**: If model files are missing, download them from the [Kaggle Notebook 4 output](YOUR_NOTEBOOK_4_LINK) or train the model using the provided notebooks.

#### **5. Launch the Application**

```bash
streamlit run main.py
```

The application will open automatically in your default browser at `http://localhost:8501`

---

## 📖 Usage Guide

### 🏠 **Home Section**

#### **📋 Overview Tab**
- System architecture explanation
- Performance metrics display
- About information and capabilities

#### **🔍 Single Note Tab**
1. Choose input method: **Upload Image** or **Use Webcam**
2. For upload: Click "Browse files" and select currency note image
3. For webcam: Click "Capture" to take a photo
4. View results:
   - Verdict badge (Authentic/Counterfeit/Uncertain)
   - Denomination prediction
   - Confidence scores (CNN, DSV, Fused)
   - Top-5 predictions
   - Authentication reasoning

#### **📁 Batch Analysis Tab**
1. Upload multiple images (JPG, JPEG, PNG)
2. System processes all images automatically
3. View results table with:
   - Denomination predictions
   - Authenticity verdicts
   - Confidence scores
4. Download results as CSV

### 🔧 **Technical Section**

#### **📊 Model Performance Tab**
- Denomination accuracy metrics
- Authenticity accuracy metrics
- Confusion matrices visualization
- ROC and PR curves

#### **⚙️ Fusion Controls Tab**
- Adjust DSV weight slider (0.0 - 1.0)
- CNN weight updates automatically
- Modify authentication threshold
- Enable/disable Grad-CAM

#### **💡 Technical Insights Tab**
- CNN architecture details
- DSV feature descriptions
- Training methodology
- Dataset information

### ⚙️ **Quick Settings (Sidebar)**

- **Auth Threshold**: Adjust sensitivity (default: 0.5)
- **Enable Grad-CAM**: Toggle explainability visualization
- **Version Info**: Display current version and model

---

## 📁 Project Structure

```
NoteShield/
│
├── main.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── notebooks/                      # Jupyter notebooks (Kaggle)
│   ├── 1-data-preparation.ipynb
│   ├── 2-preprocessing.ipynb
│   ├── 03-dsv-feature-extraction-hybrid.ipynb
│   └── 04-cnn-training-dual-head-and-visualization.ipynb
│
├── results/                        # Model outputs and artifacts
│   ├── final_artifacts/
│   │   ├── dual_head_mobilenetv2_final.keras
│   │   ├── label_mapping.json
│   │   ├── confusion_denom.png
│   │   ├── confusion_auth.png
│   │   ├── roc_auth.png
│   │   └── pr_auth.png
│   └── test_predictions.csv
│
└── venv/                           # Virtual environment (ignored in git)
```

---

## 📈 Model Performance

### 🎯 **Classification Metrics**

| Metric | Denomination | Authenticity |
|--------|-------------|--------------|
| **Accuracy** | 95.2% | 97.8% |
| **Precision** | 94.8% | 97.5% |
| **Recall** | 94.6% | 98.1% |
| **F1-Score** | 94.7% | 97.8% |

> **Note**: Replace with actual metrics from your trained model

### 📊 **Per-Class Performance**

| Denomination | Precision | Recall | F1-Score | Support |
|-------------|-----------|---------|----------|---------|
| ₹10 | XX.X% | XX.X% | XX.X% | XXX |
| ₹20 | XX.X% | XX.X% | XX.X% | XXX |
| ₹50 | XX.X% | XX.X% | XX.X% | XXX |
| ₹100 | XX.X% | XX.X% | XX.X% | XXX |
| ₹200 | XX.X% | XX.X% | XX.X% | XXX |
| ₹500 | XX.X% | XX.X% | XX.X% | XXX |
| ₹2000 | XX.X% | XX.X% | XX.X% | XXX |
| **Counterfeit** | XX.X% | XX.X% | XX.X% | XXX |

### 🔍 **Test Set Statistics**

- **Total Test Samples**: 1,992 images
- **Genuine Notes**: X,XXX (XX.X%)
- **Counterfeit Notes**: XXX (XX.X%)
- **Test Accuracy**: XX.X%

---

## ⚠️ Limitations

### **Current Constraints**

1. **DSV Features**: Currently simulated for demonstration purposes; production deployment requires implementing actual watermark detection, seal verification, and serial number OCR algorithms.

2. **Image Quality Dependency**: 
   - Optimal performance requires good lighting conditions
   - Recommended resolution: 224×224 or higher
   - Avoid blurry, tilted, or partially visible notes

3. **Research Prototype**: 
   - Designed for educational and research purposes
   - Not a replacement for official RBI verification methods
   - Should not be used as the sole authentication method in production

4. **Dataset Bias**: 
   - Model performance depends on training data diversity
   - May not generalize well to notes with unusual wear/tear
   - Limited to Indian currency denominations in dataset

5. **Computational Requirements**: 
   - Requires ~332 MB model file in memory
   - GPU recommended for batch processing
   - Real-time webcam processing may be slower on CPU-only systems

---

## 🚀 Future Enhancements

### **Planned Features**

- [ ] **Real DSV Implementation**: Actual watermark detection and seal verification algorithms
- [ ] **Multi-Currency Support**: Extend to other currencies (USD, EUR, GBP, etc.)
- [ ] **Mobile Application**: Native Android/iOS apps using TensorFlow Lite
- [ ] **API Endpoints**: RESTful API for integration with other systems
- [ ] **Database Integration**: Store authentication history and statistics
- [ ] **Advanced Preprocessing**: Automatic note orientation correction and edge detection
- [ ] **Model Quantization**: Compressed model for faster inference
- [ ] **Ensemble Methods**: Multiple model voting for improved accuracy
- [ ] **Temporal Tracking**: Video-based authentication for moving notes
- [ ] **Anomaly Detection**: Identify unknown counterfeit patterns

### **Technical Improvements**

- [ ] Implement actual OCR for serial number extraction
- [ ] Add support for worn/damaged note authentication
- [ ] Integrate blockchain for authentication logging
- [ ] Develop edge deployment optimizations
- [ ] Add multi-language support for interface
- [ ] Create comprehensive unit and integration tests

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### **How to Contribute**

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### **Contribution Guidelines**

- Follow PEP 8 style guidelines for Python code
- Add docstrings to all functions and classes
- Update README.md if adding new features
- Ensure all tests pass before submitting PR
- Include clear commit messages

### **Reporting Issues**

Found a bug or have a suggestion? Please open an issue with:
- Clear description of the problem/feature
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Screenshots (if applicable)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Parminder Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### **Special Thanks**

- **TensorFlow Team** for the excellent deep learning framework
- **Streamlit Team** for the intuitive web app framework
- **Kaggle Community** for hosting datasets and notebooks
- **MobileNetV2 Authors** for the efficient architecture
- **OpenCV Contributors** for computer vision tools

### **References**

1. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR 2018*
2. Reserve Bank of India - Security Features Documentation
3. Kaggle Datasets - Indian Currency Recognition
4. Streamlit Documentation - Building ML Web Apps

### **Datasets & Resources**

- Training data sourced from public Indian currency datasets
- Preprocessing techniques inspired by computer vision literature
- Feature extraction methods based on document security research

---

## 📞 Contact & Support

### **Get in Touch**

- **GitHub**: [@ParminderSinghGithub](https://github.com/ParminderSinghGithub)
- **Project Repository**: [NoteShield](https://github.com/ParminderSinghGithub/NoteShield)
- **Issues**: [Report a Bug](https://github.com/ParminderSinghGithub/NoteShield/issues)

### **Support the Project**

If you find this project helpful, please consider:
- ⭐ **Starring** the repository
- 🍴 **Forking** for your own experiments
- 📢 **Sharing** with others who might benefit
- 🐛 **Reporting bugs** or suggesting features

---

## 📊 Project Stats

![GitHub Stars](https://img.shields.io/github/stars/ParminderSinghGithub/NoteShield?style=social)
![GitHub Forks](https://img.shields.io/github/forks/ParminderSinghGithub/NoteShield?style=social)
![GitHub Issues](https://img.shields.io/github/issues/ParminderSinghGithub/NoteShield)
![GitHub License](https://img.shields.io/github/license/ParminderSinghGithub/NoteShield)

---

<div align="center">

### 🛡️ Built with ❤️ for Secure Currency Authentication

**NoteShield** • *Protecting Trust, One Note at a Time*

[⬆ Back to Top](#%EF%B8%8F-noteshield---ai-powered-indian-currency-authentication-system)

</div>
