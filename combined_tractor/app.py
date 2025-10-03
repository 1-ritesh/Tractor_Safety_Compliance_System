from flask import Flask, render_template, request
import os
from deeplearning import object_detection as plate_detection
from cloth_detection import load_cloth_model, predict_image
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Configure paths
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

# Load red cloth detection model
cloth_model, device = load_cloth_model("static/models/red_cloth_model.pth")

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # Handle file upload
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        
        # Get detection type from form
        detection_type = request.form.get('detection_type')
        
        if detection_type == 'plate':
            # License plate detection
            text_list = plate_detection(path_save, filename)
            return render_template('index.html', 
                                 upload=True,
                                 upload_image=filename,
                                 text=text_list,
                                 no=len(text_list),
                                 detection_type='plate')
        
        elif detection_type == 'cloth':
            # Red cloth detection
            try:
                result = predict_image(path_save, cloth_model, device)
                return render_template('index.html',
                                     upload=True,
                                     upload_image=filename,
                                     cloth_result=result,
                                     detection_type='cloth')
            except Exception as e:
                return render_template('index.html',
                                     error=f"Error in cloth detection: {str(e)}")
    
    return render_template('index.html', upload=False)

if __name__ == "__main__":
    app.run(debug=True)