📹 VisionGuard – CCTV Object Detection System

An AI-powered object detection system designed for CCTV surveillance.
The project uses YOLOv8 for real-time detection of people, vehicles, and other objects from video streams or images.

🚀 Features

🎯 Real-time object detection using YOLOv8

🖥️ Supports CCTV/Live Camera feeds and pre-recorded video files

📊 Detection statistics (counts of persons, vehicles, etc.)

💾 Option to save detections with bounding boxes

⚡ Lightweight Flask API for integration with frontend (React)

🛠️ Tech Stack

Deep Learning: YOLOv8, PyTorch

Backend: Flask (Python)

Frontend: React.js (optional live dashboard)

Other Tools: OpenCV, NumPy

📂 Project Structure
VisionGuard/
│── backend/
│   ├── app.py            # Flask API
│   ├── detect.py         # Detection logic with YOLOv8
│   └── requirements.txt  # Python dependencies
│
│── frontend/
│   ├── src/              # React frontend code
│   └── package.json
│
│── models/
│   └── yolov8n.pt        # Pre-trained YOLOv8 model
│
└── README.md

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/YourUsername/VisionGuard.git
cd VisionGuard

2️⃣ Backend Setup
cd backend
pip install -r requirements.txt


Run Flask server:

python app.py

3️⃣ Frontend Setup (Optional)
cd frontend
npm install
npm start

🎯 Usage
Run on video file:
python detect.py --source data/test_video.mp4

Run on CCTV camera (replace with your IP):
python detect.py --source rtsp://username:password@ip_address:port/stream

Run via Flask API:
POST /predict  
Body: { "image": "base64_encoded_image" }

📊 Sample Output

✅ Detected persons, cars, and bikes with bounding boxes.
✅ Can count and log number of detections.

📈 Future Improvements

🚀 Integrate with attendance system (face recognition)

📡 Push alerts (email/SMS) when intruder detected

🖥️ Web dashboard with live detection analytics

🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.
