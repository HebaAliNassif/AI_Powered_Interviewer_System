<template>
  <v-app >
    
    <div class="back">
    <br />
    <div align="center">
    <v-img  src="../assets/GPLogo2.png" width="80%" height="auto"> </v-img>
    </div>
    <br />
    <h2 style="font-family:Times New Roman;color:#443842" class="text-center">Please upload your Resume/CV</h2>

    <br />
    <div align="center">
      <v-file-input
        v-model="files"
        show-size
        outlined
        :rules="rules"
        accept=".pdf"
        color="#5b738c" 
        elevation="5"
        label="Upload resume..."
        prepend-icon="mdi-cloud-upload"
        truncate-length="50"
        style="max-width: 600px"
      ></v-file-input>
    </div>

    <br />

    <div class="text-center">
      <v-btn 
      color="#5b738c" 
      elevation="5"
       x-large 
       dark 
       @click="SubmitResume()"
        >submit</v-btn
      >
    </div>
    <br />
    </div>
  </v-app>
</template>
<script>
import axios from "axios";
export default {
  components: {},
  data() {
    return {
      files: [],
      rules: [
        (files) =>
          !(files.size > 1024 * 1024) || "File size should be less than 2 MB!",
        (files) => files.size > 0 || "Required!",
      ],
    };
  },
  methods: {
    SubmitResume() {
      if (this.files.size > 0) {
        //send to backend
        let formData = new FormData();
        console.log("sending pdf ");
        console.log(this.files);
        formData.append("cv", this.files);
        formData.append("username", "user_cv");//to be edited
        const path = "http://localhost:5000/add_resume";
        axios
          .post(path, formData, {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          })
          .then((res) => {
            console.log(res.data);
            localStorage.setItem('Resume_user',JSON.stringify(res.data));
          })
          .catch((error) => {
            console.error(error);
          });
        
        this.$router.push("/interview");
      }
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
    /*if ( "Resume_user" in localStorage )
    {
      this.$router.push("/result");
    }*/
  },
};
</script>
<style scoped>
.back{
  background-color: #d3dbe6;
}
</style>