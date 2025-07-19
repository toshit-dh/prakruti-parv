# 🐾 Prakruti-Parv: A Wildlife Conservation Project

Prakruti-Parv is a cutting-edge wildlife conservation platform that combines **deep learning**, **image/audio processing**, and **community engagement** to protect endangered species. From poaching detection to species identification and educational outreach, Prakruti-Parv brings technology and awareness together to safeguard biodiversity.

---

## 📽️ Demo Video & 📊 PPT

- 🎥 **Demo Video**: [Watch on YouTube](#)

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Screenshots](#screenshots)
- [Performance Metrics](#performance-metrics)
- [Future Scope](#future-scope)
- [Team](#team)

---

## 🌱 Project Overview

> **Goal**: A unified platform for wildlife conservation that leverages AI for species identification, poaching detection, audio recognition, educational resources, and a community-driven data enrichment ecosystem.

---

## 🚀 Core Features

- 🐯 **Species Identification** – Upload animal images and get instant species predictions with ResNet50 + Gemini.
- 🔫 **Poaching Detection** – Video-based detection using ResNet101 to flag illegal activities.
- 🔊 **Audio Species Detection** – Analyze animal calls and identify species via AudioCNN.
- 📚 **Educational Module** – Fetches data from Wikipedia/YouTube with downloadable PDFs.
- 📱 **Wildlife Social** – Aggregates live conservation news from Twitter/X & Instagram.
- 🏆 **Gamification** – Users earn "Prakruti-Mudra" coins and badges for contributions.

---

## 🧰 Tech Stack

| Layer | Tools/Tech |
|------|-------------|
| **Frontend** | HTML, CSS, React.js |
| **Backend** | Node.js, Express.js |
| **AI/ML Models** | PyTorch, TorchVision, Scikit-learn |
| **Database** | MongoDB |
| **Others** | Flask API, Postman, GitHub, Jupyter Notebook |

---

## 🧠 System Architecture

### 🔧 Modular Breakdown

- **Frontend**: React UI with public/private views, education + contribution modules.
- **Backend**: User auth, API routing, data handling via Node.js.
- **Flask ML API**: Processes image/audio for prediction using trained models.
- **Database**: MongoDB to store species, users, and alerts.
- **Integrations**: Twitter/YouTube APIs, Notification Service.

---

## 🛠️ Installation

### 2. Backend Setup

```bash
cd backend
npm install
npm start
```

### 3. Flask ML API Setup

```bash
cd flask-api
pip install -r requirements.txt
python app.py
```

### 4. Frontend Setup

```bash
cd frontend
npm install
npm start
```

---

## 🖼️ Screenshots

| Feature               | Screenshot                                  |
|-----------------------|----------------------------------------------|
| Register Page         | ![Register](assets/screens/register.png)     |
| Home Page             | ![Home](assets/screens/home.png)             |
| Poaching Detection    | ![Poaching](assets/screens/poaching.png)     |
| Species Identification| ![Species](assets/screens/species.png)       |
| Audio Detection       | ![Audio](assets/screens/audio.png)           |
| Education Page        | ![Education](assets/screens/education.png)   |

---

## 📊 Performance Metrics

| Model                  | Training Accuracy | Testing Accuracy |
|------------------------|-------------------|------------------|
| Species Identification | 93%               | 90%              |
| Poaching Detection     | 99%               | 98%              |
| Audio Recognition      | 94.4%             | 78.4%            |

---

## 🔭 Future Scope

- 🌫️ Add support for foggy/night vision poaching detection.
- 📱 Mobile App version for forest officers.
- 🧠 Real-time streaming audio surveillance.
- 🗣️ Voice command & multilingual support.
- 📈 Dashboard analytics for conservationists.


   
