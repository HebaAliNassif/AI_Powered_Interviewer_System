from flask import Flask, jsonify,request
from flask_cors import CORS,cross_origin
import os
import subprocess
from werkzeug.utils import secure_filename
import time
# importing our modules files
import SpeechToText
import moviepy
from moviepy.editor import VideoFileClip
import sys
#sys.path.insert(1, "../../Video/emotions_analysis")
#import emotions_analysis
sys.path.insert(0, '../../')
sys.path.insert(1, "../../Video")
sys.path.insert(4, "../../Video/face_alignment")
sys.path.insert(2, "../../Video/face_detection/vj")
sys.path.insert(3, "../../Video/face_detection/hog")
###############OUR MODULES IMPORTS#################
from Video.emotions_analysis_v2 import analyse_emotions
from FaceEmotionExtraction.FaceEmotionExtraction_ManualImplementedHOG_IntegrationTest import accumaltive_emotion_extraction_probabilities
from PersonalityAssessment.Files.PersonalityAssessment import predictPersonality
from Video.face_detection.hog.hog import HogClassifier
from Video.face_detection.face_detector import FaceDetector
from Video.face_alignment.face_aligner import FaceAligner
from Video import config
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

faceDetector=""
hogModel=""
faceAligner=""
@app.before_first_request
def before_first_request():
    start = time.time()
    app.logger.info("before_first_request")
    global faceDetector 
    global hogModel
    global faceAligner
    print("Loading the models")
    faceDetector= FaceDetector("../../Video/"+config.VJ_MODEL_PATH)
    hogModel = HogClassifier.loadModel("../../Video/"+config.HOG_MODEL_PATH)
    faceAligner = FaceAligner(desiredFaceWidth=100) 
    end = time.time()
    print("loading Models takes: "+str(end - start)+" secs")

@app.route('/load_models', methods=['GET'])
@cross_origin(origin='http://localhost:8080')
def load_models():
    return jsonify("Models are loaded")

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
    print("Video processing start")
    #recieving the videos and saving it locally 
    #file = request.files['webcam']
    username=request.form['username']
    print("Saving video "+username)
    #filename = secure_filename(file.filename)
    new_file = username + '.' + 'mp4'
    file_path=f"{app.config['UPLOAD_VIDEOS_FOLDER']}/{new_file}"
    #file.save(file_path)
    
    
    #passing videos to speech to text module:
    #converting .mp4 to .wav
    print("Converting .mp4 to .wav of video "+username)
    #SpeechToText.convert_video_to_audio_moviepy(file_path)
    #new_audio_file = username + '.' + 'wav'

    #extracting the speech and writting in external files
    print("Extracting the speech of video "+username)
    #start = time.time()
    #text=SpeechToText.get_large_audio_transcription(f"{app.config['UPLOAD_VIDEOS_FOLDER']}/{new_audio_file}")
    #f= open(f"{app.config['UPLOAD_SPEECHTEXT_FOLDER']}/{username}.txt","w+")
    #f.write(text)
    #f.close()
    #end = time.time()
    #print("Speech to Text module of video "+username+" takes: "+str(end - start)+" secs")
    
    #passing videos to face detection module:
    while faceDetector=="" and hogModel=="" and faceAligner=="":
        print("waiting for models")
    print("Face Detection start of video "+username)
    start = time.time()
    ListOfFaceDetectedImages=analyse_emotions(file_path,faceDetector,hogModel,faceAligner,opencv_fd=False)
    end = time.time()
    print("Face Detection module of video "+username+" takes: "+str(end - start)+" secs")

    #passing list of cropped images from face detection module to face emotion extraction module:
    print("Emotion Extraction start of video "+username)
    start = time.time()
    Acc_emotion_extraction_probs=accumaltive_emotion_extraction_probabilities(ListOfFaceDetectedImages)
    end = time.time()
    print("Emotion Extraction module of video "+username+" takes: "+str(end - start)+" secs")

    #return dictionary of classes probabilities
    return jsonify([username,Acc_emotion_extraction_probs])
    #return jsonify('ji')

@app.route('/personality_assessment', methods=['GET'])
@cross_origin(origin='http://localhost:8080')
def personality_assessment():
    print("Personality Assessment start")
    start = time.time()
    file_path=f"{app.config['UPLOAD_SPEECHTEXT_FOLDER']}"
    predictPersonalityResults=predictPersonality(file_path)
    end = time.time()
    print("Personality Assessment module takes: "+str(end - start)+" secs")
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
    print("Resume Filtering start")
    start = time.time()
    end = time.time()
    print("Resume Filtering module takes: "+str(end - start)+" secs")
    # return dictionary of user statistics and probabilities
    return jsonify('file received')


if __name__ == '__main__':
    app.run()
    