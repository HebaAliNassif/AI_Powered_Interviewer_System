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
      <v-app-bar-nav-icon @click="drawer = true"></v-app-bar-nav-icon>
    </v-app-bar>
    <v-navigation-drawer
      v-model="drawer"
      absolute
      temporary
      right
    >
      <v-list
        nav
        dense
      >
        <v-list-item-group
          v-model="group"
          active-class="blue--text text--accent-4"
        >
          <v-list-item>
            <v-list-item-icon>
              <v-icon>mdi-home</v-icon>
            </v-list-item-icon>
            <v-list-item-title>Home</v-list-item-title>
          </v-list-item>

          <v-list-item>
            <v-list-item-icon>
              <v-icon>mdi-account</v-icon>
            </v-list-item-icon>
            <v-list-item-title>Account</v-list-item-title>
          </v-list-item>
        </v-list-item-group>
      </v-list>
    </v-navigation-drawer>
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
