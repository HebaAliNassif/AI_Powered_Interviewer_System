import Vue from 'vue'
import VueRouter from 'vue-router'

Vue.use(VueRouter)
import UploadResume from '../views/ResumeUpload.vue'
import Interview from '../views/Interview.vue'
import Results from '../views/Results.vue'

const routes = [
  {
    path: '/',
    name: 'upload_resume',
    component: UploadResume,
    
  },
  {
    path: '/upload_resume',
    name: 'upload_resume',
    component: UploadResume
  },
  {
    path: '/interview',
    name: 'interview',
    component: Interview
  },
  {
    path: '/result',
    name: 'result',
    component: Results
  },
  
  
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router
