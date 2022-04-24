<template>
  <v-app>
    <br />

    <div style="display: flex;
    flex-direction: row;
    font-family: auto;
    align-self: center;">
    <div style="margin:0px 0px 0px 20px">
    <h2>Emotion Detected percentages</h2>
    <apexchart type="pie" width="380" :options="chartOptions" :series="series"></apexchart>
    </div>
    <div >
    <h2>Personality assessment percentages</h2>
    <apexchart type="radialBar" width="380" :options="chartOptions" :series="series"></apexchart>
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
            }
      
  };
  },
  methods: {
  
      
  },
  created() {
      const path = "http://localhost:5000/result";
        axios
          .get(path)
          .then((res) => {
            let l=[res.data.Happy,res.data.Contempt,res.data.Anger,res.data.Disgust,res.data.Fear,res.data.Sad,res.data.Surprise,res.data.Neutral];
            this.series=l;
            
          })
          .catch((error) => {
            console.error(error);
          });
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