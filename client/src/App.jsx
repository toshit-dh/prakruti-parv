/* eslint-disable no-unused-vars */
import React from 'react'
import { Register,Login,Home,Education,AnimalTracking,Welcome,IdentifySpecies,Profile,AddProject,ViewProjects,Project, Poaching } from './components/index'
import {BrowserRouter,Routes,Route} from 'react-router-dom'
import ProtectedRoute from './utils/ProtectedRoute'
import ProjectForm from './components/add-project-form/ProjectForm'
import ReportTemplate from './components/fund-report-template/ReportTemplate'
import Myproject from './components/myproject/Myproject'
import Settings from './components/settings/Settings'
import Education2 from './components/education/Education2'
import Explore from './components/explore/Explore'

function App() {


  return (
    <BrowserRouter>
        <Routes>
          <Route path='/register' element={<Register/>}/>
          <Route path='/login' element={<Login/>}/>
          <Route  path='/' element={<Home/>}/>
          <Route path='/education' element={<Education2/>}/>
          <Route path='/welcome' element={<Welcome/>}/>
          <Route path='/identify' element={<IdentifySpecies/>}/>
          <Route path='/poaching-detection' element={<Poaching/>}/>
          <Route path='/profile' element={<Profile/>}/>
          <Route path='/addProject' element={<AddProject/>}/>
          <Route path='/viewProjects' element={<ViewProjects/>}/>
          <Route path='/add-project' element={<ProjectForm/>}/>
          <Route path='/project/:projectId' element={<Myproject/>}/>
          <Route path='/settings' element={<Settings/>}/>
          <Route path='/explore' element={<Explore/>}/>


        </Routes>
    </BrowserRouter>
  )
}

export default App
