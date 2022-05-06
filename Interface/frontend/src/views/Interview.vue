<template>
  <v-app>
    <br />
    <h1 class="text-center">
      Congratulation! you have reached the second phase
    </h1>
    <br />
    <div style="margin-left:1rem">
    <h2>Interview phase steps:</h2>
    <ol>
      <li v-for="step in this.Steps" :key="step">
        {{step}}
      </li>
    
    </ol>
    <h2 style="color:red">Important Notes:</h2>
    <ul>
      <li v-for="note in this.Notes" :key="note">
        {{note}}
      </li>
    
    </ul>
    <h3 class="text-center">
     Are you ready?
    </h3>
    </div>
    <br />
    <div class="text-center">
      <v-btn
        v-show="this.firstTime"
        depressed
        color="blue"
        x-large
        @click="VideoProcessing()"
        >{{ bottun[0] }}</v-btn
      >
      <v-btn
        v-show="this.nextTime"
        
        depressed
        color="blue"
        x-large
        @click="VideoProcessing()"
        >{{ bottun[1] }}</v-btn
      >
    </div>
    <br />
    <div class="text-center">
      <p v-if="this.showQuestion">
        Question {{ this.$store.state.questionNumber }}:
        {{ questions[this.$store.state.questionNumber - 1] }}
      </p>
    </div>

    <div>
      <video id="myVideo" playsinline class="video-js vjs-default-skin"></video>
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
      Steps:[
      "When you press the start button, the first question will appear and the black screen will be activated.",
      "You should press on the middle of it when it is activated.",
      "Press the record button when you are ready to start recording.",
      "Press the stop recording button when you finish or it will be automatically sent after 1 min.",
      "Press the next button to go to the next question."
      ],
      Notes:["Each question you will hear it by our machine and will be written infront of you.",
      "You will be given 1 min to answer for each question with camera opened.",
      "Please make your voice clear, loud and in English."],
      firstTime: true,
      nextTime: false,
      bottun: ["Start", "Next"],
      showQuestion: false,
     
      questions: [
        "Please tell us about yourself in 1 min",
        "Why did you decide to apply to this role?",
        "What do you know about our company?",
      ],
    };
  },
  methods: {
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
      let recorded_videos = new Set();
      
      player.on("finishConvert", ()=> {
        if (player.convertedData.length > 0) {
          recorded_videos.add(
            player.convertedData[player.convertedData.length - 1]
          );
        } else {
          recorded_videos.add(player.convertedData);
        }
        if (recorded_videos.size == 3) {
          let index = 0;
          
          recorded_videos.forEach(async (video) => {
            index = index + 1;
            
            
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
      if (this.$store.state.questionNumber > 2) {
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
    /*if ( "PersonalityAssessmentResults" in localStorage && "EmotionExtractionResults_user_Q_1" in localStorage && "EmotionExtractionResults_user_Q_2" in localStorage && "EmotionExtractionResults_user_Q_3" in localStorage)
    {
      this.$router.push("/result");
    }*/
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
    width:80% !important;
    height: auto !important;
    
}
</style>