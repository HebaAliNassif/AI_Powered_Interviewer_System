from flask import Flask, jsonify,request
from flask_cors import CORS,cross_origin
import os
import subprocess
#import connexion
#from werkzeug.datastructures import FileStorage
#from flask_uploads import UploadSet, configure_uploads
from werkzeug.utils import secure_filename
# configuration
DEBUG = True
UPLOAD_FOLDER = ''
# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
    #response = jsonify({'some': 'data'})
    #response.headers.add('Access-Control-Allow-Origin', '*')
    #video = request.files['webcam'].stream.read()
    file = request.files['webcam']
    username=request.form['username']
    filename = secure_filename(file.filename)
    #base_file, ext = os.path.splitext(filename)
    new_file = username + '.' + 'mp4'

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_file))
    return jsonify('video received')


if __name__ == '__main__':
    app.run()