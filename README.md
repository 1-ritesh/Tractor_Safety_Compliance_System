# 🚜 Tractor Safety Compliance Portal

The **Tractor Safety Compliance Portal** is an AI-powered web application built to assist the RTO in monitoring agricultural vehicle safety. It automates video analysis to detect red cloth reflectors or number plates on tractors, flagging violations and extracting evidence frames to ensure compliance with road safety regulations.

## 🚀 How to Run the Project

### ✅ Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Tractor-Safety-Compliance-Portal.git
cd Tractor-Safety-Compliance-Portal
```

### ✅ Step 2: Create a Virtual Environment
```bash
python -m venv venv

# Activate the environment:
# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### ✅ Step 3: Install Dependencies
Install requirements for both apps:
```bash
pip install -r combined_tractor/requirements.txt
pip install -r tractor_frame_app/requirements.txt
```

### ✅ Step 4: Run the Apps

🔹 **To launch the Compliance Detection Portal:**
```bash
cd combined_tractor
streamlit run app.py
```

🔹 **To launch the Tractor Frame Extractor:**
```bash
cd tractor_frame_app
streamlit run app.py
```

## 📋 Features

- **AI-Powered Detection**: Automatically detects red cloth reflectors and number plates on tractors
- **Video Analysis**: Processes video files to identify safety compliance violations
- **Frame Extraction**: Extracts evidence frames for violation documentation
- **Real-time Monitoring**: Streamlined interface for RTO officials
- **Compliance Reporting**: Generates detailed reports for safety violations

## 🛠️ Technologies Used

- **Computer Vision**: YOLOv8 for object detection
- **OCR**: Tesseract for number plate recognition
- **Frontend**: Streamlit for web interface
- **Backend**: Python with OpenCV
- **Machine Learning**: Custom CNN models for classification

## 📁 Project Structure

```
Tractor-Safety-Compliance-Portal/
├── combined_tractor/
│   ├── app.py
│   ├── requirements.txt
│   └── model/
├── tractor_frame_app/
│   ├── app.py
│   ├── requirements.txt
│   └── model/
├── output_frames/
├── uploads/
├── README.md
├── .gitignore
└── LICENSE
```

## 📝 Usage Notes

- Place your model files (YOLO weights, CNN models) inside the respective `model/` folders
- Upload video/image files via the app UI
- Output frames will be automatically saved in the `output_frames/` directory
- Ensure proper lighting and video quality for optimal detection accuracy

## 📁 .gitignore

```gitignore
**__pycache__**/
*.pyc
*.pyo
*.pyd
*.DS_Store
*.log
venv/
uploads/
output_frames/
model/
*.mp4
*.avi
*.mov
*.jpg
*.jpeg
*.png
.env
```

## 📸 Sample Output

In the output file

## 🔧 Configuration

1. Ensure you have the required model files in the appropriate directories
2. Adjust detection confidence thresholds in the configuration files
3. Modify output paths as needed in the application settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/) & [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Streamlit](https://streamlit.io/) community
- RTO officials for domain expertise and requirements

## 📞 Support

For support and questions, please open an issue in the GitHub repository or contact the development team.

---

**Made with ❤️ for Agricultural Vehicle Safety**
