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
      answer on each question with camera opened. Note that once you press the
      start button, the first question will appear and the black screen will be
      activated and you should press on the middle of it once it is activated
      then press the record button once you are ready. Note that it will
      automatically send your recording after 1 min or when you press the stop
      recording button then you should press next for the next question to
      appear and so on. Are you ready?
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
      <v-btn
        v-show="this.nextTime"
        depressed
        color="blue"
        x-large
        @click="EmotionExtraction()"
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
      firstTime: true,
      nextTime: false,
      bottun: ["Start", "Next"],
      showQuestion: false,
      questions: [
        "Please tell us about yourself in 1 min",
        "Why did you decide to apply to this role?",
        "What do you know about our company",
      ],
    };
  },
  methods: {
    startcamera() {
      let QN = this.$store.state.questionNumber.toString();
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
      let recorded_videos = new Set();
      player.on("finishRecord", function () {
        if (player.recordedData.length > 0) {
          recorded_videos.add(
            player.recordedData[player.recordedData.length - 1]
          );
        } else {
          recorded_videos.add(player.recordedData);
        }
        if (recorded_videos.size == 3) {
          let index = 0;
          recorded_videos.forEach((video) => {
            index = index + 1;
            let formData = new FormData();
            console.log("sending video " + index);
            console.log(video);
            formData.append("webcam", video);
            formData.append("username", "aya_Q_" + index );
            const path = "http://localhost:5000/video_processing";
            axios
              .post(path, formData, {
                headers: {
                  "Content-Type": "multipart/form-data",
                },
              })
              .then((res) => {
                console.log(res.data);
              })
              .catch((error) => {
                console.error(error);
              });
          });
        }

        player.record().reset();
      });
    },
    EmotionExtraction() {
      if (this.$store.state.questionNumber > 2) {
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
      //var voices = speechSynthesis.getVoices();
      //speech.voice = voices[5];
      window.speechSynthesis.speak(speech);
      this.startcamera();
    },
  },
  created() {},
};
</script>
<style scoped>
video {
  margin-left: auto;
  margin-right: auto;
  display: block;
}
.video-js[tabindex="-1"] {
  outline: 0;
  margin-left: auto;
  margin-right: auto;
  display: block;
}
</style>