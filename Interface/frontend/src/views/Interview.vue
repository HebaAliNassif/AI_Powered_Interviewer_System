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
      our machine and will appear infront of you and you will be given 2 mins to
      answer on each question with camera opened. Are you ready?
    </p>
    <br />
    <div class="text-center">
      <v-btn depressed color="blue" x-large @click="EmotionExtraction()"
        >Start</v-btn
      >
    </div>
     <br />
    <div class="text-center">
      <p v-if="question1">Question 1: {{questions}}</p>
    </div>
    
  </v-app>
</template>
<script>
//import axios from "axios";
export default {
  components: {},
  data() {
    return {
      question1: false,
      questions: "Please tell us about yourself in 2 mins",
    };
  },
  methods: {
    EmotionExtraction() {
      this.question1 = true;
      let speech = new SpeechSynthesisUtterance();
      speech.lang = "en-US";
      speech.text = this.questions;
      speech.volume = 1;
      speech.rate = 1;
      speech.pitch = 1;
      //var voices = speechSynthesis.getVoices();
      //speech.voice = voices[5];
      window.speechSynthesis.speak(speech);
      /*const path = 'http://localhost:5000/ping';
      axios.get(path)
        .then((res) => {
          console.log(res.data);
        })
        .catch((error) => {
  
          console.error(error);
        });*/
    },
  },
  created() {},
};
</script>