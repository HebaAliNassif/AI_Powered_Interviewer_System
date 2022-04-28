<template>
  <v-app>
    <br />
    <v-alert
    v-show="this.waitForResults"
  border="top"
  elevation="9"
  type="info"
>Please wait untill all the results are prepared</v-alert>
    <div style="display: flex;
    flex-direction: row;
    font-family: auto;
    align-self: center;">
    
    <div style="margin:0px 0px 0px 20px">
    <h2>Emotion Detected Percentages</h2>
    <apexchart type="pie" width="380" :options="chartOptions" :series="series"></apexchart>
    </div>
    <div >
    <h2>Personality Assessment Percentages</h2>
    <apexchart type="bar" width="380" :options="this.chartOptions2" :series="this.series2"></apexchart>
    </div>
    </div>
    <div align="center" style="font-family: auto;">
      <h2>Resume Statistics</h2>
      
    </div>
    <br />
  </v-app>
</template>
<script>
import axios from "axios";
import VueApexCharts from 'vue-apexcharts'

export default {
  components: {
          apexchart: VueApexCharts,
        },
        
  data() {
    return {
      waitForResults:true,
        series: [],

        chartOptions: {
            chart: {
              width: 380,
              type: 'pie',
            },
            labels: ["Happy","Contempt","Anger","Disgust","Fear","Sad","Surprise","Neutral"],
            responsive: [{
              options: {
                chart: {
                  width: 200
                },
                legend: {
                  position: 'bottom'
                }
              }
            }]
            },
            series2: [{
            data: [29, 55, 15, 70, 90]
          }], 
        chartOptions2: {
            chart: {
              height: 350,
              type: 'bar',
              
            },
            plotOptions: {
              bar: {
                columnWidth: '45%',
                distributed: true,
              }
            },
            dataLabels: {
              enabled: false
            },
            legend: {
              show: false
            },
            xaxis: {
              categories: [
                ['Openness'],
                ['Conscientiousness'],
                ['Extroversion'],
                ['Agreeableness'],
                ['Neuroticism'],
                
              ],
              labels: {
                style: {
                  fontSize: '12px'
                }
              }
            }
          },    
      
  };
  },
  methods: {
  
      
  },
  created() {
    
      let Question1_data=JSON.parse(localStorage.getItem("EmotionExtractionResults_user_Q_1"))
      let Question2_data=JSON.parse(localStorage.getItem("EmotionExtractionResults_user_Q_2"))
      let Question3_data=JSON.parse(localStorage.getItem("EmotionExtractionResults_user_Q_3"))
      
      if (Question1_data!=null && Question2_data!=null && Question3_data!=null){
        let total_count=Question1_data['count']+Question2_data['count']+Question3_data['count'];
        this.waitForResults=false;
        this.series=[
          (Question1_data['Happy']+Question2_data['Happy']+Question3_data['Happy'])*100/total_count,
          (Question1_data['Contempt']+Question2_data['Contempt']+Question3_data['Contempt'])*100/total_count,
          (Question1_data['Anger']+Question2_data['Anger']+Question3_data['Anger'])*100/total_count,
          (Question1_data['Disgust']+Question2_data['Disgust']+Question3_data['Disgust'])*100/total_count,
          (Question1_data['Fear']+Question2_data['Fear']+Question3_data['Fear'])*100/total_count,
          (Question1_data['Sad']+Question2_data['Sad']+Question3_data['Sad'])*100/total_count,
          (Question1_data['Surprise']+Question2_data['Surprise']+Question3_data['Surprise'])*100/total_count,
          (Question1_data['Neutral']+Question2_data['Neutral']+Question3_data['Neutral'])*100/total_count
          ];
      }
      else{
        this.waitForResults=true;
        this.series=[0,0,0,0,0,0,0,0]
         
      }
      
   
  },
};
</script>
<style scoped>
.apexcharts-legend-series {
    /* cursor: pointer; */
    /* line-height: normal; */
    margin: inherit;
}
</style>