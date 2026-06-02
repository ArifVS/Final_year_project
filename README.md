[README.md](https://github.com/user-attachments/files/26377634/README.md)
# 🚦 Smart Traffic Management System
### Federated Learning + YOLOv8 + LSTM | Final Year Project

A real-time intelligent traffic management system that uses **YOLOv8** for vehicle detection and **Federated Learning with LSTM** for adaptive signal timing prediction.

---

## 🧠 Features

- 🔍 **Vehicle Detection** — YOLOv8m detects cars, motorcycles, buses, trucks, and bicycles
- 📊 **Real-time Metrics** — Precision, Recall, F1-score, and Accuracy per frame
- 🧬 **Federated LSTM** — Distributed edge-node learning for traffic prediction
- 🟢 **Congestion Levels** — LOW / MEDIUM / HIGH / CRITICAL with adaptive signal timing
- 📷 **Image Upload** — Upload a traffic image for instant analysis
- 🎥 **Live Webcam** — Real-time detection via webcam feed

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download YOLOv8 model
```bash
# Automatically downloaded on first run, or manually:
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### 3. Run the Flask app
```bash
python demo.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
ArifVS-clone/
├── demo.py              # Main Flask application
├── app.py               # Alternate app entry
├── templates/
│   ├── index.html       # Upload page
│   ├── result.html      # Detection results
│   └── webcam.html      # Live webcam feed
├── static/
│   ├── uploads/         # Uploaded images
│   └── outputs/         # Processed images
└── requirements.txt
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python) |
| Detection | YOLOv8 (Ultralytics) |
| Prediction | LSTM (TensorFlow/Keras) |
| Learning | Federated Averaging |
| Frontend | HTML, CSS, JavaScript |

---

---

## 📈 Advantages

- Enables real-time traffic monitoring and congestion prediction.
- Reduces traffic congestion through intelligent signal optimization.
- Preserves data privacy by avoiding the transfer of raw traffic data.
- Minimizes network bandwidth usage through federated learning.
- Provides low-latency decision-making using edge computing.
- Offers a scalable solution for smart city traffic management.
  
---

---

## 🔮 Future Enhancements

- Integration with real-world IoT-enabled traffic cameras and sensors.
- Deployment on cloud and edge hybrid infrastructures.
- Emergency vehicle detection and priority-based signal control.
- Multi-intersection traffic coordination and optimization.
- Advanced federated learning algorithms for improved model performance.
- Large-scale implementation across smart city transportation networks.
  
---

---


## 👨‍💻 Author

**Final Year Project** — Alliance University  
*Smart Traffic Management using Federated Learning*
