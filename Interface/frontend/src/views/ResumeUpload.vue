<template>
  <v-app>
    <br />
    <h1 class="text-center">Welcome to our Recruitment Process</h1>
    <v-img src="../assets/robot.jpg"> </v-img>
    <br />
    <p class="text-center">
      Please upload your resume/cv
    </p>
    
    <br />
    <div align="center">
    <v-file-input
      v-model="files"
      show-size
      outlined
      :rules="rules"
      accept=".pdf"
      color="blue"
      elevation="5"
      label="Upload resume..."
      prepend-icon="mdi-cloud-upload"
      truncate-length="50"
      style="max-width:600px"
    ></v-file-input>
    </div>

    <br />
    
    <div class="text-center">
      <v-btn color="green" elevation="5" x-large plain @click="SubmitResume()"
        >submit</v-btn
      >
    </div>
    <br />
  </v-app>
</template>
<script>
//import axios from "axios";
export default {
  components: {},
  data() {
    return {
      files: [],
      rules: [
      files => !(files.size > 1024 * 1024) || 'File size should be less than 2 MB!',
      files => files.size > 0 || 'Required!'
    ],
    };
  },
  methods: {
    SubmitResume() {
      if (this.files.size>0){
        //send to backend

        //push to second stage if cv is accepted
        this.$router.push('/interview');
      }
        
    },
  },
  created() {},
};
</script>