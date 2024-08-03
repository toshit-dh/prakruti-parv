/* eslint-disable no-unused-vars */
import React,{useState} from 'react';
import { FaEnvelope, FaLock,FaPaw } from 'react-icons/fa';
import loginpic from '../../assets/loginpic.png';
import './Login.css';
import { ToastContainer,toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';



const Login = () => {

  const navigate=useNavigate();
  const toastOptions= {
    position: "bottom-left",
    autoClose: 5000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "dark",
    }


  
  return (
    <>
        <section className="login-wrapper">
      <div className="form-container">
        <form className="login-form">
          <div className="form-header">
                <motion.h2  initial={{ opacity: 0, scale: 0.5, rotate: -45 }}
                animate={{ opacity: 1, scale: 1, rotate: 0 }}
                transition={{ duration: 1, ease: "easeOut" }}>
                   <FaPaw class name="title-icon" />
                             Login
                </motion.h2>

          </div>
          <div className="form-group">
            <div className="input-icon">
              <FaEnvelope />
              <input type="email" id="email" name="email" placeholder="Email"  />
            </div>
          </div>
          <div className="form-group">
            <div className="input-icon">
              <FaLock />
              <input type="password" id="password" name="password" placeholder="Password"/>
            </div>
          </div>
          <div className="form-group">
            <button type="submit" onClick={()=>{navigate('/')}}>Login</button>
            <span className='redirectRegister'>Don&apos;t have an account? <a href="/register"> Register</a></span>
          </div>
        </form>
        <div className="image-container">
          <img src={loginpic} alt="Login" className="login-image" />
        </div>
      </div>
    </section>
    <ToastContainer/>
    <style>{`
        .Toastify__toast {
          white-space: nowrap;
          text-overflow: ellipsis;
          overflow: hidden;
        }
      `}</style>

    </>
    
  );
};

export default Login;