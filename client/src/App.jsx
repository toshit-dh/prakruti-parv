/* eslint-disable no-unused-vars */
import React from 'react'
import './App.css'
import { Register,Login,Home } from './components/index'
import {BrowserRouter,Routes,Route} from 'react-router-dom'

function App() {


  return (
    <BrowserRouter>
        <Routes>
          <Route path='/register' element={<Register/>}/>
          <Route path='/login' element={<Login/>}/>
          <Route  path='/' element={<Home/>}/>
        </Routes>
    </BrowserRouter>
  )
}

export default App
