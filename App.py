import os
import cv2
import torch
from torch import nn
from torchvision import transforms
import face_recognition
from flask import Flask, render_template, redirect, request, url_for, json,jsonify,send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
from time import time as current_time
from torch.utils.data import Dataset
from pytube import YouTube
import re

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'Uploaded_Files'
app.config['PROCESSED_FOLDER'] = 'Processed_Files'
app.config['JSON_FOLDER'] = 'Json_Files'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create directories if they do not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['JSON_FOLDER'], exist_ok=True)


# Constants for image preprocessing
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Model setup
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Image transformations for preprocessing
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])



# Prediction function
def predict(model, img):
    model.eval()
    sm = nn.Softmax(dim=1)
    fmap, logits = model(img.to('cuda')) if torch.cuda.is_available() else model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return [int(prediction.item()), confidence]

def preprocess_video(video_path, max_duration=5):
    face_frames = []
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    max_frames = int(fps * max_duration)
    
    success, image = vidcap.read()
    count = 0
    while success and len(face_frames) < max_frames:
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB conversion
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            frame = image[top * 4:bottom * 4, left * 4:right * 4, :]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            face_frames.append(frame)
        success, image = vidcap.read()
        count += 1
    return face_frames

def create_video(frames, video_filename):
    output_dir = app.config['PROCESSED_FOLDER']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, secure_filename(video_filename))

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    return output_path

def detectFakeVideo(video_path):
    model = Model(num_classes=2)
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_95.22_acc_40_sl.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Limit video duration to 5 seconds (or any suitable duration based on fps)
    face_frames = preprocess_video(video_path, max_duration=5)
    if len(face_frames) == 0:
        return [0, 0]  # No faces found, could not process

    processed_frames = []
    for frame in face_frames:
        processed_frame = train_transforms(frame)
        processed_frames.append(processed_frame)

    frames_tensor = torch.stack(processed_frames).unsqueeze(0)

    # Perform prediction
    prediction = predict(model, frames_tensor)

    # Save processed video
    processed_video_path = create_video(face_frames, os.path.basename(video_path))

    return prediction, processed_video_path

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_valid_youtube_url(url):
    pattern = r"^(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+(&\S*)?$"
    return re.match(pattern, url) is not None

def download_video(url):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        if stream:
            download_path = os.path.join(app.config['UPLOAD_FOLDER'])
            if not os.path.exists(download_path):
                os.makedirs(download_path)
            
            filename = stream.download(output_path=download_path)
            return True, filename
        else:
            return False, "No progressive MP4 streams available for download."
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False, str(e)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected video file'}), 400
    
    if not is_allowed_file(video.filename):  
        return jsonify({'error': 'Invalid file type'}), 400

    video_filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    video.save(video_path)

    # Assuming detectFakeVideo returns output and processed_video_path
    prediction, processed_video_path = detectFakeVideo(video_path)

    if prediction[0] == 1:
        output = "REAL"
    else:
        output = "FAKE"

    confidence = prediction[1]  # Extract confidence value
    # Save processed video in the configured folder
    processed_video_filename = os.path.basename(processed_video_path)
    processed_video_dest = os.path.join(app.config['PROCESSED_FOLDER'], processed_video_filename)
    os.rename(processed_video_path, processed_video_dest)

    # Create a JSON file with video details
    video_details = {
        'output': output,
        'confidence': confidence,  # Add confidence value to JSON
        'uploaded_video': video_filename,
        'processed_video': processed_video_filename
    }
    json_filename = video_filename + '.json'
    json_path = os.path.join(app.config['JSON_FOLDER'], json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(video_details, json_file)

    # Return JSON filename for AJAX response
    return jsonify(json_filename)


@app.route('/upload-url', methods=['POST'])
def upload_url():
    url = request.form['url']
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    if is_valid_youtube_url(url):
        success, filename = download_video(url)
        if success:
            video_path = filename
            # Adjust unpacking to handle two values from detectFakeVideo()
            prediction, processed_video_path = detectFakeVideo(video_path)
            
            if prediction[0] == 1:
                output = "REAL"
            else:
                output = "FAKE"

            confidence = prediction[1]  # Extract confidence value

            processed_video_filename = os.path.basename(processed_video_path)
            processed_video_dest = os.path.join(app.config['PROCESSED_FOLDER'], processed_video_filename)
            os.rename(processed_video_path, processed_video_dest)

            # Save video details in JSON file
            video_details = {
                'output': output,
                'confidence': confidence,  # Add confidence value to JSON
                'uploaded_video': os.path.basename(filename),
                'processed_video': processed_video_filename
            }
            json_filename = os.path.basename(filename) + '.json'
            json_path = os.path.join(app.config['JSON_FOLDER'], json_filename)
            with open(json_path, 'w') as json_file:
                json.dump(video_details, json_file)

            return jsonify(json_filename)
        else:
            return jsonify({'error': filename}), 400
    else:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    

@app.route('/json_files/<filename>')
def download_json(filename):
    return send_from_directory(app.config['JSON_FOLDER'], filename)

@app.route('/result')
def show_result():
    json_filename = request.args.get('json_filename')
    
    if not json_filename:
        return "Missing json_filename parameter", 400  # Handle error if parameter is missing

    json_path = os.path.join(app.config['JSON_FOLDER'], json_filename)
    
    if not os.path.exists(json_path):
        app.logger.error(f"JSON file not found: {json_path}")
        return "JSON file not found", 404  # Handle error if JSON file does not exist

    try:
        with open(json_path, 'r') as json_file:
            video_info = json.load(json_file)
    except Exception as e:
        app.logger.error(f"Error loading JSON file: {json_path}, {e}")
        return "Error loading JSON file", 500  # Handle any other errors during JSON file loading
    
    return render_template('result.html', video_info=video_info)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)