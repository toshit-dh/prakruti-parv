/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import { FaUser, FaLock, FaPaw } from 'react-icons/fa'; 
import loginpic from '../../assets/loginpic.png';
import './Login.css';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import {LOGIN_ROUTE} from '../../utils/Routes'

const Login = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });

  const navigate = useNavigate();

  const toastOptions = {
    position: 'bottom-left',
    autoClose: 5000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: 'dark',
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post(LOGIN_ROUTE, formData,{
        withCredentials: true
      });
      console.log(response);
      
      if (response.status === 200) {
        toast.success('Login successful!', toastOptions);
        navigate('/');
      } else {
        toast.error(response.data.message || 'Login failed.', toastOptions);
      }
    } catch (error) {
      toast.error(error.message || 'An error occurred.', toastOptions);
    }
  };

  return (
    <>
      <section className="login-wrapper">
        <div className="form-container">
          <form className="login-form" onSubmit={handleSubmit}>
            <div className="form-header">
              <motion.h2
                initial={{ opacity: 0, scale: 0.5, rotate: -45 }}
                animate={{ opacity: 1, scale: 1, rotate: 0 }}
                transition={{ duration: 1, ease: 'easeOut' }}
              >
                <FaPaw className="title-icon" />
                Login
              </motion.h2>
            </div>
            <div className="form-group">
              <div className="input-icon">
                <FaUser /> 
                <input
                  type="text"
                  id="username"
                  name="username"
                  placeholder="Username"
                  value={formData.username}
                  onChange={handleChange}
                />
              </div>
            </div>
            <div className="form-group">
              <div className="input-icon">
                <FaLock />
                <input
                  type="password"
                  id="password"
                  name="password"
                  placeholder="Password"
                  value={formData.password}
                  onChange={handleChange}
                />
              </div>
            </div>
            <div className="form-group">
              <button type="submit">Login</button>
              <span className="redirectRegister">
                Don&apos;t have an account? <a href="/register"> Register</a>
              </span>
            </div>
          </form>
          <div className="image-container">
            <img src={loginpic} alt="Login" className="login-image" />
          </div>
        </div>
      </section>
      <ToastContainer />
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
