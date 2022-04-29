from flask import Flask, jsonify,request
from flask_cors import CORS,cross_origin
import os
import subprocess
from werkzeug.utils import secure_filename
# importing our modules files
import SpeechToText

import moviepy
from moviepy.editor import VideoFileClip
import sys
#sys.path.insert(1, "../../Video/emotions_analysis")
#import emotions_analysis
sys.path.insert(0, '../../')
###############OUR MODULES IMPORTS#################
from Video.emotions_analysis import analyse_emotions
from FaceEmotionExtraction.FaceEmotionExtraction_ManualImplementedHOG_IntegrationTest import accumaltive_emotion_extraction_probabilities
from PersonalityAssessment.Files.PersonalityAssessment import predictPersonality

# configuration
DEBUG = True
UPLOAD_VIDEOS_FOLDER = 'Videos'
UPLOAD_CVS_FOLDER = 'CVs'
UPLOAD_SPEECHTEXT_FOLDER='SpeechText'
FRAMES_FOLDER='Frames'
FACES_DETECTED_FOLDER='FacesDetectedImages'
# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_VIDEOS_FOLDER'] = UPLOAD_VIDEOS_FOLDER
app.config['UPLOAD_CVS_FOLDER'] = UPLOAD_CVS_FOLDER
app.config['UPLOAD_SPEECHTEXT_FOLDER'] = UPLOAD_SPEECHTEXT_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER
app.config['FACES_DETECTED_FOLDER'] = FACES_DETECTED_FOLDER
# enable CORS
CORS(app, resources={r'/video_processing': {'origins': 'http://localhost:8080'}})

# Temp test route
# sanity check route
@app.route('/result', methods=['GET'])
@cross_origin(origin='http://localhost:8080')
def ping_pong():
    #os.chdir('C:/Users/THINK/Desktop/College/GP/emotion_extraction/Emotion-Recognition-From-Facial-Expressions-master/Trail1')
    #live - emotion extraction
    #os.system('python live_ManualHOG.py')
    #input image - emotion extraction 
    #res=subprocess.check_output('python trail2_test.py')
    #res.decode("utf-8")

    #temp list
    Emotions_Extracted={"Happy":10,"Contempt":5,"Anger":5,"Disgust":5,"Fear":5,"Sad":10,"Surprise":10,"Neutral":50}
    return jsonify(Emotions_Extracted)
    #return jsonify('hi')

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
    #converting .mp4 to .wav
    SpeechToText.convert_video_to_audio_moviepy(file_path)
    new_audio_file = username + '.' + 'wav'
    #extracting the speech and writting in external files
    text=SpeechToText.get_large_audio_transcription(f"{app.config['UPLOAD_VIDEOS_FOLDER']}/{new_audio_file}")
    f= open(f"{app.config['UPLOAD_SPEECHTEXT_FOLDER']}/{username}.txt","w+")
    f.write(text)
    f.close()
    
    faces_detected_path=f"{app.config['FACES_DETECTED_FOLDER']}/{username}"
    #passing videos to face detection module:
    analyse_emotions(file_path, opencv_fd=True, frames_path=f"{app.config['FRAMES_FOLDER']}/{username}", faces_path=faces_detected_path)
    #passing list of cropped images from face detection module to face emotion extraction module:
    Acc_emotion_extraction_probs=accumaltive_emotion_extraction_probabilities(faces_detected_path)
    #return dictionary of classes probabilities
    return jsonify([username,Acc_emotion_extraction_probs])
    #return jsonify('ji')

@app.route('/personality_assessment', methods=['GET'])
@cross_origin(origin='http://localhost:8080')
def personality_assessment():
    file_path=f"{app.config['UPLOAD_SPEECHTEXT_FOLDER']}"
    
    predictPersonalityResults=predictPersonality(file_path)
    return jsonify(predictPersonalityResults)
    
@app.route('/add_resume', methods=['POST'])
@cross_origin(origin='http://localhost:8080',headers=['Content-Type'])
def add_resume():
    file = request.files['cv']
    username=request.form['username']
    filename = secure_filename(file.filename)
    new_file = username + '.' + 'pdf'
    file.save(os.path.join(app.config['UPLOAD_CVS_FOLDER'], new_file))

    #passing the file to resume filtering module

    # return dictionary of user statistics and probabilities
    return jsonify('file received')


if __name__ == '__main__':
    app.run()