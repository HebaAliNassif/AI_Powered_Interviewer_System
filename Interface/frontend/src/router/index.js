import Vue from 'vue'
import VueRouter from 'vue-router'

Vue.use(VueRouter)
import UploadResume from '../views/ResumeUpload.vue'
import Interview from '../views/Interview.vue'

const routes = [
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
  
  
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router
