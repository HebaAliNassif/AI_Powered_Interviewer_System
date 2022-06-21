<template>
  <v-app class="back">
    <br />
    <h1 style="font-family:Times New Roman;color:#443842" class="text-center">
      Now, we would like to ask you a few questions...
    </h1>
    
    <br />
    <div style="margin-left:1rem">
    <h2 style="font-family:Times New Roman;color:#443842">Interview Phase Steps:</h2>
    <ol style="font-size:20px;font-family:Times New Roman;color:#443842">
      <li v-for="step in this.Steps" :key="step">
        {{step}}
      </li>
    
    </ol>
    <br/>
    <h2 style="color:#B22222">Important Notes:</h2>
    <ul style="font-size:20px;font-family:Times New Roman;color:#443842">
      <li v-for="note in this.Notes" :key="note">
        {{note}}
      </li>
    
    </ul>
    <br/>
    <h3 style="color:#443842" class="text-center">
     Are you ready to start?
    </h3>
    </div>
    <br />
    <div class="text-center">
      <v-btn
        v-show="this.firstTime"
        :disabled="skippedVideo"
        depressed
        color="#5b738c" 
      elevation="5"
       x-large 
       dark
        @click="VideoProcessing()"
        >{{ bottun[0] }}</v-btn
      >
      <v-btn
        v-show="this.nextTime"
        :disabled="skippedVideo"
        depressed
        color="#5b738c" 
      elevation="5"
       x-large 
       dark 
       
        @click="VideoProcessing()"
        >{{ bottun[1] }}</v-btn
      >
    </div>
    <br/>
    <v-alert
      prominent
      v-if="skippedVideo"
      type="error"
      color="#B22222"
    >
      <v-row align="center">
        <v-col class="grow">
          You skipped recording your answers. Press on the restart button to start over.
        </v-col>
        <v-col class="shrink">
          <v-btn depressed
        color="#5b738c" 
      elevation="5"
       large
       dark @click="Repeat()">Restart</v-btn>
        </v-col>
      </v-row>
    </v-alert>
    <br />
    <div class="text-center">
      <p style="font-size:20px;font-family:Times New Roman;color:#443842" v-if="this.showQuestion">
        Question {{ this.$store.state.questionNumber }}:
        {{ questions[this.$store.state.questionNumber - 1] }}
      </p>
    </div>

    <div >
      <video id="myVideo"  playsinline class="video-js vjs-default-skin " ></video>
    </div>
    <br />
    <br />
  </v-app>
</template>
<script>
import axios from "axios";

import "video.js/dist/video-js.min.css";
import videojs from "video.js";
import RecordRTC from "recordrtc";
import "videojs-record/dist/css/videojs.record.css";
import Record from "videojs-record/dist/videojs.record.js";
import WaveSurfer from "wavesurfer.js";
import MicrophonePlugin from "wavesurfer.js/dist/plugin/wavesurfer.microphone.js";
WaveSurfer.microphone = MicrophonePlugin;

// register videojs-wavesurfer plugin
import TsEBMLEngine from 'videojs-record/dist/plugins/videojs.record.ts-ebml.js';
import "videojs-wavesurfer/dist/css/videojs.wavesurfer.css";
import Wavesurfer from "videojs-wavesurfer/dist/videojs.wavesurfer.js";
export default {
  components: {},
  data() {
    return {
      skippedVideo:false,
      recorded_videos:{},
      Steps:[
      "When you press the start button, the first question will appear and a black screen will appear.",
      "You should press on the microphone in the middle of the screen.",
      "Press the record button when you are ready to start recording.",
      "Press the stop recording button when you are done. Otherwise, it will stop recording automatically after 1 minute.",
      "Press the next button to go to the next question."
      ],
      Notes:["You will hear each question once by our machine and it'll appear in front of you.",
      "You will be given 1 minute to answer each question with your webcam opened.",
      "Please make your voice clear, loud and speak in English."],
      firstTime: true,
      nextTime: false,
      bottun: ["Start", "Next"],
      showQuestion: false,
      numberOfVideosRecorded:0,
      questions: [
        "Please tell us about yourself in 1 minute.",
        "Why did you decide to apply to this role?",
        "What do you know about our company?",
      ],
    };
  },
  methods: {
    Repeat(){
      location.reload();
    },
    startcamera() {
      
      let options = {
        // video.js options
        controls: true,
        
        bigPlayButton: false,
        loop: false,
        fluid: true,
        width: 300,
        height: 300,
        responsive: true,
        plugins: {
          // videojs-record plugin options
          record: {
            image: false,
            audio: true,
            video: true,
            maxLength: 60,
            displayMilliseconds: true,
            debug: false,
            convertEngine: 'ts-ebml',
          },
        },
      };
      let player = videojs("myVideo", options, function () {
        const msg =
          "Using video.js " +
          videojs.VERSION +
          " with videojs-record " +
          videojs.getPluginVersion("record");
        videojs.log(msg);

        console.log("videojs-record is ready!");
      });
      
      
      player.on("finishConvert", ()=> {
        
        if (player.convertedData.length > 1 ) {
          this.recorded_videos[this.$store.state.questionNumber]=player.convertedData[player.convertedData.length - 1];
          
        } else {
          this.recorded_videos[this.$store.state.questionNumber]=player.convertedData;
          
        }
        console.log(this.recorded_videos)
        this.numberOfVideosRecorded=Object.keys(this.recorded_videos).length
        console.log(this.numberOfVideosRecorded)
        if (Object.keys(this.recorded_videos).length == 3) {
          
          Object.entries(this.recorded_videos).forEach(([index, video]) => {
  
            
            
            
            let formData = new FormData();
            console.log("sending video " + index);
            //console.log(video);
            formData.append("webcam", video);
            formData.append("username", "user_Q_" + index );
            const path = "http://localhost:5000/video_processing";
            axios
              .post(path, formData, {
                headers: {
                  "Content-Type": "multipart/form-data",
                },
              })
              .then( res => {
                
                localStorage.setItem('EmotionExtractionResults_'+res.data[0],JSON.stringify(res.data[1]));
              })
              .catch((error) => {
                console.error(error);
              });
            const path2 = "http://localhost:5000/speech_to_text";
            axios
              .post(path2, formData, {
                headers: {
                  "Content-Type": "multipart/form-data",
                },
              })
              .then( res => {
                console.log(res);
              })
              .catch((error) => {
                console.error(error);
              });  
          });
        }

        player.record().reset();
      });
      
    },
    VideoProcessing() {
      
      console.log(this.$store.state.questionNumber)
      console.log(this.numberOfVideosRecorded)
      if(this.numberOfVideosRecorded!=this.$store.state.questionNumber){
        this.skippedVideo=true
      }
      
      else if (this.$store.state.questionNumber > 2 ) {
        const path = "http://localhost:5000/personality_assessment";
            axios
              .get(path, {
               
              })
              .then( res => {
                localStorage.setItem('PersonalityAssessmentResults',JSON.stringify(res.data));
              })
              .catch((error) => {
                console.error(error);
              });
        this.$router.push("/result");
        return;
      }
      this.showQuestion = true;
      this.firstTime = false;
      this.nextTime = true;
      let speech = new SpeechSynthesisUtterance();
      speech.lang = "en-US";
      speech.text = this.questions[this.$store.state.questionNumber];
      this.$store.state.questionNumber = this.$store.state.questionNumber + 1;
      speech.volume = 1;
      speech.rate = 1;
      speech.pitch = 1;
      //To change the voice uncomment the code below:
      //var voices = speechSynthesis.getVoices();
      //speech.voice = voices[5];
      window.speechSynthesis.speak(speech);
      this.startcamera();
    },
  },
  created() {
    
    const path = "http://localhost:5000/load_models";
        axios
          .get(path, {
          })
          .then((res) => {
            console.log(res.data);
            
          })
          .catch((error) => {
            console.error(error);
          });
    if ( "PersonalityAssessmentResults" in localStorage && "EmotionExtractionResults_user_Q_1" in localStorage && "EmotionExtractionResults_user_Q_2" in localStorage && "EmotionExtractionResults_user_Q_3" in localStorage)
    {
      this.$router.push("/result");
    }
  },
};
</script>
<style scoped>
video {
  margin-left: auto;
  margin-right: auto;
  display: block;
}

.video-js[tabindex="-1"]{
    outline: 0;
    margin-left: auto;
    margin-right: auto;
    display: block;
    position: relative !important;
    width:70% !important;
    height: auto !important;
    font-size: 15px;
    
}
.back{
  background-color: #d3dbe6;
}
</style>