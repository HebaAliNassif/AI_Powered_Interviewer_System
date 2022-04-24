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
          <v-tab to="/result">Result</v-tab>
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
    //
  }),
};
</script>
