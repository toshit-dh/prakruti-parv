import './App.css';
import {BrowserRouter,Routes,Route} from 'react-router-dom'
import Home from './pages/Home';
import { useEffect, useState } from 'react';
import SplashScreen from './pages/SplashScreen';
import ProtectedRoute from './utils/ProtectedRoute';

export default function App() {
  const [loading,setLoading] = useState(true)
  return (
    loading ? <SplashScreen setLoading = {setLoading}/> : 
    <BrowserRouter>
    <Routes>
      <Route path='/' element= {
        <ProtectedRoute>
          <Home/>
        </ProtectedRoute>
      }/>
    </Routes>
    </BrowserRouter>
  )
}


