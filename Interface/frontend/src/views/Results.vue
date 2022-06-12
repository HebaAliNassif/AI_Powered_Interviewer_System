<template>
  <v-app class="back">
    <br />
    <div class="text-center">
    <v-alert
    style="font-family:Times New Roman;color:#443842"
    v-show="this.waitForResults"
  border="top"
  elevation="9"
  type="info"
>Please wait untill all the results are prepared..Try to refresh the page after some time</v-alert>
    <v-progress-circular a
      v-show="this.waitForResults"
      indeterminate
      color="#443842"
    ></v-progress-circular>
    </div>
    <div align="center">
      <h2 style="font-family:Times New Roman;color:#443842">Data Extracted From Your Resume</h2>
      <br/>
      <v-card style="border:2px solid #5b738c" class="mx-auto" max-width="600"  elevation="5"
  shaped>
  
            <v-list-item color="rgba(0, 0, 0, .4)">
              <v-list-item-content>
                <v-list-item-title class="title" style="font-family:Times New Roman;color:#443842">Email: {{this.Email}}</v-list-item-title>
                <v-list-item-subtitle style="font-family:Times New Roman;color:#443842">Phone Number: {{this.Phone}}</v-list-item-subtitle>
                <br/>
      <v-divider></v-divider>
      <br/>
                <div align="start">
                  <v-list-item-title style="font-family:Times New Roman;color:#443842" class="title">Important Links:</v-list-item-title>
                  
                <br/>
                <ol style="font-size:18px;font-family:Times New Roman;color:#443842">
                   
                  <li v-for="link in this.Links" :key="link">
                   
                    {{link}}
                  

                  </li>
    
                </ol>
                </div>
              </v-list-item-content>
            </v-list-item>
      </v-card>
      <br/>
      <v-divider></v-divider>
      <br/>
    </div>
    <div style="display: flex;
    flex-direction: row;
    font-family:Times New Roman;
    align-self: center;
    ">
    
    <div style="font-family:Times New Roman;color:#443842">
    <h2 style="margin:0px 0px 0px 60px">Emotions Detected</h2>
    <apexchart  type="pie" width="380" :options="chartOptions" :series="series"></apexchart>
    </div>
    <div style="font-family:Times New Roman;color:#443842">
    <h2 style="margin:0px 0px 0px 60px">Personality Assessed</h2>
    <apexchart type="bar" width="380" :options="this.chartOptions2" :series="this.series2"></apexchart>
    </div>
    </div>
    <br/>
      <v-divider></v-divider>
      <br/>
    <div align="center" style="font-family:Times New Roman;color:#443842">
      <h2>Resume Qualifications</h2>
      <br/>
      <v-data-table
      style="border:2px solid #5b738c;max-width:50%"
    :headers="headers"
    :items="Fields"
    :sort-by="['per']"
    :sort-desc="[true, false]"
    :footer-props="{
    'items-per-page-options': [5, 10, 15]
  }"
    :items-per-page="5"
    class="elevation-5"
  ></v-data-table>
    </div>
    <br />
  </v-app>
</template>
<script>
import axios from "axios";
import VueApexCharts from "vue-apexcharts";

export default {
  components: {
    apexchart: VueApexCharts,
  },

  data() {
    return {
      waitForResults: true,
      Email: "",
      Phone: "",
      Links: "",
      headers: [
        {
          text: "Field",
          align: "center",
          sortable: false,
          value: "name",
        },
        { text: "Percentage %", align: "center", value: "per" },
      ],
      Fields: [
        
      ],
      series: [0, 0, 0, 0, 0, 0, 0, 0],

      chartOptions: {
        chart: {
          width: 380,
          type: "pie",
        },
        labels: [
          "Happy",
          "Contempt",
          "Anger",
          "Disgust",
          "Fear",
          "Sad",
          "Surprise",
          "Neutral",
        ],
        responsive: [
          {
            options: {
              chart: {
                width: 200,
              },
              legend: {
                position: "bottom",
              },
            },
          },
        ],
      },
      series2: [
        {
          data: [0, 0, 0, 0, 0],
        },
      ],
      chartOptions2: {
        chart: {
          height: 350,
          type: "bar",
        },
        plotOptions: {
          bar: {
            columnWidth: "45%",
            distributed: true,
            horizontal: true,
          },
        },
        dataLabels: {
          enabled: false,
         
        },
        stroke: {
          show: true,
          width: 1,
          colors: ['#fff']
        },
        legend: {
          show: false,
        },
        xaxis: {
          max: 100,

          categories: [
            ["Openness"],
            ["Conscientiousness"],
            ["Extroversion"],
            ["Agreeableness"],
            ["Neuroticism"],
          ],
          labels: {
            style: {
              colors: ['#000000'],
              fontSize: "12px",
            },
          },
        },
      },
    };
  },
  methods: {},
  created() {
    var PersonalityAssessmentResults;

    var Question1_data;
    var Question2_data;
    var Question3_data;
    if ("Resume_user" in localStorage) {
      //fill the statistics in charts

      let Resume_data = JSON.parse(localStorage.getItem("Resume_user"));
      this.Email = Resume_data.Overview.Email[0];
      this.Phone = Resume_data.Overview.Phone[0];
      this.Links = Resume_data.Overview.Links;
      this.Fields = [
        {
          name: "Sales",
          per: Resume_data.Ranking.sales * 100,
        },
        {
          name: "Accountant",
          per: Resume_data.Ranking.accountant * 100,
        },
        {
          name: "Advocate",
          per: Resume_data.Ranking.advocate * 100,
        },
        {
          name: "Agriculture",
          per: Resume_data.Ranking.agriculture * 100,
        },
        {
          name: "Apparel",
          per: Resume_data.Ranking.apparel * 100,
        },
        {
          name: "Arts",
          per: Resume_data.Ranking.arts * 100,
        },
        {
          name: "Automobile",
          per: Resume_data.Ranking.automobile * 100,
        },
        {
          name: "Aviation",
          per: Resume_data.Ranking.aviation * 100,
        },
        {
          name: "Banking",
          per: Resume_data.Ranking.banking * 100,
        },
        {
          name: "bpo",
          per: Resume_data.Ranking.bpo * 100,
        },
        {
          name: "Business Development",
          per: Resume_data.Ranking.business_development * 100,
        },
        {
          name: "Chef",
          per: Resume_data.Ranking.chef * 100,
        },
        {
          name: "Construction",
          per: Resume_data.Ranking.construction * 100,
        },
        {
          name: "Consultant",
          per: Resume_data.Ranking.consultant * 100,
        },
        {
          name: "Designer",
          per: Resume_data.Ranking.designer * 100,
        },
        {
          name: "Digital Media",
          per: Resume_data.Ranking.digital_media * 100,
        },
        {
          name: "engineering",
          per: Resume_data.Ranking.engineering * 100,
        },
        {
          name: "Finance",
          per: Resume_data.Ranking.finance * 100,
        },
        {
          name: "Fitness",
          per: Resume_data.Ranking.fitness * 100,
        },
        {
          name: "Healthcare",
          per: Resume_data.Ranking.healthcare * 100,
        },
        {
          name: "HR",
          per: Resume_data.Ranking.hr * 100,
        },
        {
          name: "Information Technology",
          per: Resume_data.Ranking.information_technology * 100,
        },
        {
          name: "Public Relations",
          per: Resume_data.Ranking.public_relations * 100,
        },
        {
          name: "Teacher",
          per: Resume_data.Ranking.teacher * 100,
        },
      ];
    }

    if ("PersonalityAssessmentResults" in localStorage) {
      PersonalityAssessmentResults = JSON.parse(
        localStorage.getItem("PersonalityAssessmentResults")
      );
      console.log(PersonalityAssessmentResults);
      this.series2[0].data = [
        JSON.parse(localStorage.getItem("PersonalityAssessmentResults")).cOPN,
        JSON.parse(localStorage.getItem("PersonalityAssessmentResults")).cCON,
        JSON.parse(localStorage.getItem("PersonalityAssessmentResults")).cEXT,
        JSON.parse(localStorage.getItem("PersonalityAssessmentResults")).cAGR,
        JSON.parse(localStorage.getItem("PersonalityAssessmentResults")).cNEU,
      ];
    }
    if (
      "EmotionExtractionResults_user_Q_1" in localStorage &&
      "EmotionExtractionResults_user_Q_2" in localStorage &&
      "EmotionExtractionResults_user_Q_3" in localStorage
    ) {
      if (
        "PersonalityAssessmentResults" in localStorage &&
        "Resume_user" in localStorage
      ) {
        this.waitForResults = false;
      }

      Question1_data = JSON.parse(
        localStorage.getItem("EmotionExtractionResults_user_Q_1")
      );
      Question2_data = JSON.parse(
        localStorage.getItem("EmotionExtractionResults_user_Q_2")
      );
      Question3_data = JSON.parse(
        localStorage.getItem("EmotionExtractionResults_user_Q_3")
      );
      let total_count =
        Question1_data["count"] +
        Question2_data["count"] +
        Question3_data["count"];

      this.series = [
        ((Question1_data["Happy"] +
          Question2_data["Happy"] +
          Question3_data["Happy"]) *
          100) /
          total_count,
        ((Question1_data["Contempt"] +
          Question2_data["Contempt"] +
          Question3_data["Contempt"]) *
          100) /
          total_count,
        ((Question1_data["Anger"] +
          Question2_data["Anger"] +
          Question3_data["Anger"]) *
          100) /
          total_count,
        ((Question1_data["Disgust"] +
          Question2_data["Disgust"] +
          Question3_data["Disgust"]) *
          100) /
          total_count,
        ((Question1_data["Fear"] +
          Question2_data["Fear"] +
          Question3_data["Fear"]) *
          100) /
          total_count,
        ((Question1_data["Sad"] +
          Question2_data["Sad"] +
          Question3_data["Sad"]) *
          100) /
          total_count,
        ((Question1_data["Surprise"] +
          Question2_data["Surprise"] +
          Question3_data["Surprise"]) *
          100) /
          total_count,
        ((Question1_data["Neutral"] +
          Question2_data["Neutral"] +
          Question3_data["Neutral"]) *
          100) /
          total_count,
      ];
    } else {
      this.waitForResults = true;
    }
    /*
      if (Question1_data!== "undefined" && Question2_data!== "undefined" && Question3_data!== "undefined"){
        
      }*/
  },
};
</script>
<style scoped>
.apexcharts-legend-series {
  /* cursor: pointer; */
  /* line-height: normal; */
  margin: inherit;
}
.back{
  background-color: #d3dbe6;
}
</style>
