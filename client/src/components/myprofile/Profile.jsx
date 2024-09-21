import React from 'react'
import Navbar from '../navbar/Navbar'
import './Profile.css'
import { useNavigate } from 'react-router-dom'

export default function Profile() {
  const navigate = useNavigate() // Corrected line

  return (
    <div className='profile'>
      <Navbar/>
      <div className="content">
        {/* Your content here */}
      </div>
      <button className='floating-button' onClick={() => navigate('/addProject')}>+</button>
    </div>
  )
}
