<template>
  <v-app>
    <v-app-bar
      app
      color="blue"
      dark
      
    >
      
        <h1> AI Powered Interviewer System </h1>
     
      <template v-slot:extension>
        <v-tabs fixed-tabs align-with-title>
          <v-tabs-slider color="white"></v-tabs-slider>
          <v-tab to="/upload_resume">Phase 1</v-tab>
          <v-tab to="/interview">Phase 2</v-tab>
          <v-tab :disabled="ResultsNotReady" to="/result">Result</v-tab>
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

    //
  }),
  created() {
    if ( "PersonalityAssessmentResults" in localStorage || "EmotionExtractionResults_user_Q_1" in localStorage || "EmotionExtractionResults_user_Q_2" in localStorage || "EmotionExtractionResults_user_Q_3" in localStorage ||  "Resume_user" in localStorage )
    {
      this.ResultsNotReady=false;
    }
  },
};
</script>
