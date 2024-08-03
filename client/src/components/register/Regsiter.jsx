/* eslint-disable no-unused-vars */
import React from 'react';
import { FaUser, FaEnvelope, FaLock, FaKey, FaTree } from 'react-icons/fa';
import registerpic from '../../assets/registerpic.png';
import './Register.css';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { motion } from 'framer-motion';

const Register = () => {
  return (
    <>
      <section className="register-wrapper">
        <div className="form-container">
          <form className="register-form">
            <div className="form-header">
              <motion.h2
                initial={{ opacity: 0, scale: 0.5, rotate: -45 }}
                animate={{ opacity: 1, scale: 1, rotate: 0 }}
                transition={{ duration: 1, ease: "easeOut" }}
                className="header-title"
              >
                <FaTree className="title-icon" /> Register
              </motion.h2>
            </div>
            <div className="form-group">
              <div className="input-icon">
                <FaUser />
                <input type="text" id="username" name="username" placeholder="Username" />
              </div>
            </div>
            <div className="form-group">
              <div className="input-icon">
                <FaEnvelope />
                <input type="email" id="email" name="email" placeholder="Email" />
              </div>
            </div>
            <div className="form-group">
              <div className="input-icon">
                <FaLock />
                <input type="password" id="password" name="password" placeholder="Password" />
              </div>
            </div>
            <div className="form-group">
              <div className="input-icon">
                <FaKey />
                <input type="password" id="confirmPassword" name="confirmPassword" placeholder="Confirm Password" />
              </div>
            </div>
            <div className="form-group checkbox-group">
              <input type="checkbox" id="checkbox" name="checkbox" />
              <label htmlFor="checkbox">I agree to the terms and conditions</label>
            </div>
            <div className="form-group">
              <button type="submit">Register</button>
              <span className='redirectLogin'>Already have an account? <a href="/login">Login</a></span>
            </div>
          </form>
          <div className="image-container">
            <img src={registerpic} alt="Register" className="register-image" />
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

export default Register;
