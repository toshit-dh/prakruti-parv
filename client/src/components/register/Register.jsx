/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import { FaUser, FaEnvelope, FaLock, FaKey, FaTree } from 'react-icons/fa';
import registerpic from '../../assets/registerpic.png';
import './Register.css';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { SIGNUP_ROUTE } from '../../utils/Routes';

const Register = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'User', 
    checkbox: false,
  });
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const validateForm = () => {
    const { username, email, password, confirmPassword, role, organization, checkbox } = formData;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const passwordRegex = /^.{8,16}$/;

    if (username.length > 25) {
      toast.error('Username must be between 4 and 8 characters long.');
      return false;
    }

    if (!emailRegex.test(email)) {
      toast.error('Invalid email address.');
      return false;
    }

    if (!passwordRegex.test(password)) {
      toast.error('Password must be between 8 and 16 characters long.');
      return false;
    }

    if (password !== confirmPassword) {
      toast.error('Passwords do not match.');
      return false;
    }

    if (!role) {
      toast.error('Please select a role.');
      return false;
    }

    if (!checkbox) {
      toast.error('You must agree to the terms and conditions.');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    try {
      const response = await axios.post(SIGNUP_ROUTE, formData);
      if (response.status === 201) {
        toast.success('Registration successful!');
        navigate('/login');
      } else {
        toast.error(response.data.message || 'Registration failed.');
      }
    } catch (error) {
      console.log(error);
      toast.error(error.response?.data?.message || 'An error occurred.');
    }
  };

  return (
    <>
      <section className="register-wrapper">
        <div className="register-form-container">
          <form className="register-form" onSubmit={handleSubmit}>
            <div className="register-form-header">
              <motion.h2
                initial={{ opacity: 0, scale: 0.5, rotate: -45 }}
                animate={{ opacity: 1, scale: 1, rotate: 0 }}
                transition={{ duration: 1, ease: 'easeOut' }}
                className="register-header-title"
              >
                <FaTree className="register-title-icon" /> Register
              </motion.h2>
            </div>
            <div className="register-form-group">
              <div className="register-input-icon">
                <FaUser />
                <input
                  type="text"
                  id="username"
                  name="username"
                  placeholder="Username"
                  value={formData.username}
                  onChange={handleChange}
                  className="register-input"
                />
              </div>
            </div>
            <div className="register-form-group">
              <div className="register-input-icon">
                <FaEnvelope />
                <input
                  type="email"
                  id="email"
                  name="email"
                  placeholder="Email"
                  value={formData.email}
                  onChange={handleChange}
                  className="register-input"
                />
              </div>
            </div>
            <div className="register-form-group">
              <div className="register-input-icon">
                <FaLock />
                <input
                  type="password"
                  id="password"
                  name="password"
                  placeholder="Password"
                  value={formData.password}
                  onChange={handleChange}
                  className="register-input"
                />
              </div>
            </div>
            <div className="register-form-group">
              <div className="register-input-icon">
                <FaKey />
                <input
                  type="password"
                  id="confirmPassword"
                  name="confirmPassword"
                  placeholder="Confirm Password"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  className="register-input"
                />
              </div>
            </div>

            <div className="register-form-group">
              <div className="register-input-icon">
                <FaUser />
                <select
                  id="role"
                  name="role"
                  value={formData.role}
                  onChange={handleChange}
                  className="register-select"
                  required
                >{
                  ["User", "Conservationist","Organisation"].map((role) => (
                    <option key={role} value={role}>{role}</option>
                  ))
                }
                </select>
              </div>
            </div>

            <div className="register-form-group checkbox-group">
              <input
                type="checkbox"
                id="checkbox"
                name="checkbox"
                checked={formData.checkbox}
                onChange={handleChange}
                className="register-checkbox"
              />
              <label htmlFor="checkbox" className="register-checkbox-label">
                I agree to the terms and conditions
              </label>
            </div>
            <div className="register-form-group">
              <button type="submit" className="register-submit-button">Register</button>
              <span className="register-redirect-login">
                Already have an account? <a href="/login">Login</a>
              </span>
            </div>
          </form>
          <div className="register-image-container">
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
