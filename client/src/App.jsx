/* eslint-disable no-unused-vars */
import React from 'react'
import { Register,Login,Home,Education,AnimalTracking,Welcome } from './components/index'
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
          <Route path='/' element={<Home/>}/>
          <Route path='/welcome' element={<Welcome/>}/>
        </Routes>
    </BrowserRouter>
  )
}

export default App
