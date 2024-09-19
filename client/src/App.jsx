/* eslint-disable no-unused-vars */
import React from 'react'
import { Register,Login,Home,Education,AnimalTracking,Welcome,IdentifySpecies,Profile } from './components/index'
import {BrowserRouter,Routes,Route} from 'react-router-dom'
import ProtectedRoute from './utils/ProtectedRoute'

function App() {


  return (
    <BrowserRouter>
        <Routes>
          <Route path='/register' element={<Register/>}/>
          <Route path='/login' element={<Login/>}/>
          <Route  path='/' element={<Home/>}/>
          <Route path='/education' element={<Education/>}/>
          <Route path='/welcome' element={<Welcome/>}/>
          <Route path='/identify' element={<IdentifySpecies/>}/>
          <Route path='/profile' element={<Profile/>}/>
        </Routes>
    </BrowserRouter>
  )
}

export default App
