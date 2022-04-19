from flask import Flask, jsonify
from flask_cors import CORS
import os
import subprocess

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    os.chdir('C:/Users/THINK/Desktop/College/GP/emotion_extraction/Emotion-Recognition-From-Facial-Expressions-master/Trail1')
    #live - emotion extraction
    os.system('python live_ManualHOG.py')
    #input image - emotion extraction 
    res=subprocess.check_output('python trail2_test.py')
    res.decode("utf-8")
    return jsonify(str(int(res)))


if __name__ == '__main__':
    app.run()