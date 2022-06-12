<template>
  <v-app >
    <v-app-bar
      
      app
      color="#5b738c"
      dark
      
    >
    <v-spacer/>
      <h1  style="font-size: 40px;font-family:Times New Roman;">Welcome to our Recruitment Process</h1>
      
      <template v-slot:extension>
        
        <v-tabs fixed-tabs >
          <v-tabs-slider  color="white"></v-tabs-slider>
          <v-tab style="font-size: 15px;font-family:Times New Roman" to="/upload_resume">Phase 1</v-tab>
          <v-tab style="font-size: 15px;font-family:Times New Roman" :disabled="ResumeNotReady" to="/interview">Phase 2</v-tab>
          <v-tab style="font-size: 15px;font-family:Times New Roman" :disabled="ResultsNotReady" to="/result">Result</v-tab>
        </v-tabs>
        
      </template>
      <v-spacer></v-spacer>
      
    </v-app-bar>
   
    <v-main>
      <router-view/>
    </v-main>
  </v-app>
</template>

<script>

export default {
  name: 'App',

  data: () => ({
    drawer: false,
      group: null,
      ResultsNotReady:true,
      ResumeNotReady:true,

    //
  }),
  created() {
    if ( "PersonalityAssessmentResults" in localStorage || "EmotionExtractionResults_user_Q_1" in localStorage || "EmotionExtractionResults_user_Q_2" in localStorage || "EmotionExtractionResults_user_Q_3" in localStorage ||  "Resume_user" in localStorage )
    {
      this.ResultsNotReady=false;
    }
    if ("Resume_user" in localStorage ){
      this.ResumeNotReady=false
    }
  },
};
</script>
<style scoped>

</style>