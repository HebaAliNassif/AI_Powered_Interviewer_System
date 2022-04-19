<template>
  <v-app>
    <br />
    <h1 class="text-center">
      Congratulation! you have reached the second phase
    </h1>
    <br />
    <h2>Interview phase:</h2>
    <p>
      we are going to ask you some questions, each question you will hear it by
      our machine and will appear infront of you and you will be given 1 min to
      answer on each question with camera opened. Are you ready?
    </p>
    <br />
    <div class="text-center">
      <v-btn
        v-show="this.firstTime"
        depressed
        color="blue"
        x-large
        @click="EmotionExtraction()"
        >{{ bottun[0] }}</v-btn
      >
      <v-btn v-show="this.nextTime" depressed color="blue" x-large>{{
        bottun[1]
      }}</v-btn>
    </div>
    <br />
    <div class="text-center">
      <p style="margin: 0rem 0rem 0rem 4rem" v-if="this.question1">Question 1: {{ questions }}</p>
    </div>

    <div >
      <video style="margin: 0rem 0rem 0rem 33rem" id="myVideo" playsinline class="video-js vjs-default-skin"></video>
    </div>
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
import "videojs-wavesurfer/dist/css/videojs.wavesurfer.css";
import Wavesurfer from "videojs-wavesurfer/dist/videojs.wavesurfer.js";
export default {
  components: {},
  data() {
    return {
      video: "",
      firstTime: true,
      nextTime: false,
      bottun: ["Start", "Next"],
      question1: false,
      questions: "Please tell us about yourself in 1 min",
    };
  },
  methods: {
    startcamera() {
      let options = {
        // video.js options
        controls: true,
        bigPlayButton: false,
        loop: false,
        fluid: false,
        width: 400,
        height: 400,
        plugins: {
          // videojs-record plugin options
          record: {
            image: false,
            audio: true,
            video: true,
            maxLength: 60,
            displayMilliseconds: true,
            debug: false,
          },
        },
      };
      let player = videojs("myVideo", options, function () {
        // print version information at startup
        const msg =
          "Using video.js " +
          videojs.VERSION +
          " with videojs-record " +
          videojs.getPluginVersion("record");
        videojs.log(msg);

        console.log("videojs-record is ready!");
      });

      player.on("finishRecord", function () {
        this.video = player.recordedData;
        const path = "http://localhost:5000/video_processing";
        axios
          .post(path, { video: this.video })
          .then((res) => {
            console.log(res.data);
            
          })
          .catch((error) => {
            console.error(error);
          });
        player.record().destroy();
        
      });
      
    },
    EmotionExtraction() {
      this.question1 = true;
      this.firstTime = false;
      //this.nextTime=true;
      let speech = new SpeechSynthesisUtterance();
      speech.lang = "en-US";
      speech.text = this.questions;
      speech.volume = 1;
      speech.rate = 1;
      speech.pitch = 1;
      //var voices = speechSynthesis.getVoices();
      //speech.voice = voices[5];
      //window.speechSynthesis.speak(speech);
      this.startcamera();
    },
  },
  created() {},
};
</script>