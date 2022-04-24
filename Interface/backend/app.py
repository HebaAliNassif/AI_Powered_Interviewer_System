from flask import Flask, jsonify,request
from flask_cors import CORS,cross_origin
import os
import subprocess
from werkzeug.utils import secure_filename
# importing our modules files
import SpeechToText
import moviepy
from moviepy.editor import VideoFileClip
# configuration
DEBUG = True
UPLOAD_VIDEOS_FOLDER = 'Videos'
UPLOAD_CVS_FOLDER = 'CVs'
UPLOAD_SPEECHTEXT_FOLDER='SpeechText'
# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_VIDEOS_FOLDER'] = UPLOAD_VIDEOS_FOLDER
app.config['UPLOAD_CVS_FOLDER'] = UPLOAD_CVS_FOLDER
app.config['UPLOAD_SPEECHTEXT_FOLDER'] = UPLOAD_SPEECHTEXT_FOLDER
# enable CORS
CORS(app, resources={r'/video_processing': {'origins': 'http://localhost:8080'}})


# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    os.chdir('C:/Users/THINK/Desktop/College/GP/emotion_extraction/Emotion-Recognition-From-Facial-Expressions-master/Trail1')
    #live - emotion extraction
    os.system('python live_ManualHOG.py')
    #input image - emotion extraction 
    #res=subprocess.check_output('python trail2_test.py')
    #res.decode("utf-8")
    return jsonify('hi')

@app.route('/video_processing', methods=['POST'])
@cross_origin(origin='http://localhost:8080',headers=['Content-Type'])
def video_processing():
    #recieving the videos and saving it locally 
    file = request.files['webcam']
    username=request.form['username']
    filename = secure_filename(file.filename)
    new_file = username + '.' + 'mp4'
    file_path=f"{app.config['UPLOAD_VIDEOS_FOLDER']}/{new_file}"
    file.save(file_path)
    
    #passing videos to speech to text module:
    print(file_path)
    SpeechToText.convert_video_to_audio_moviepy(file_path)
    new_audio_file = username + '.' + 'wav'

    text=SpeechToText.get_large_audio_transcription(f"{app.config['UPLOAD_VIDEOS_FOLDER']}/{new_audio_file}")
    f= open(f"{app.config['UPLOAD_SPEECHTEXT_FOLDER']}/{username}.txt","w+")
    f.write(text)
    f.close()
    #passing videos to face detection module:
    

    return jsonify('video received')

@app.route('/add_resume', methods=['POST'])
@cross_origin(origin='http://localhost:8080',headers=['Content-Type'])
def add_resume():
    #response = jsonify({'some': 'data'})
    #response.headers.add('Access-Control-Allow-Origin', '*')
    #video = request.files['webcam'].stream.read()
    file = request.files['cv']
    username=request.form['username']
    filename = secure_filename(file.filename)
    #base_file, ext = os.path.splitext(filename)
    new_file = username + '.' + 'pdf'

    file.save(os.path.join(app.config['UPLOAD_CVS_FOLDER'], new_file))
    return jsonify('file received')


if __name__ == '__main__':
    app.run()