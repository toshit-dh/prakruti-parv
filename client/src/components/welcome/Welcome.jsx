/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */
import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { FaTree, FaPaw } from 'react-icons/fa';
import { GiHummingbird } from 'react-icons/gi';
import './Welcome.css';
import logo from '../../assets/logo.png';
import {useNavigate} from 'react-router-dom';

const Welcome = () => {
  const [isLoaded, setIsLoaded] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 1000); 
    const navigationTimer = setTimeout(() => {
       navigate('/'); 
    }, 6000);

    return () => {
      clearTimeout(timer);
      clearTimeout(navigationTimer);  
    };
  }, []);

  return (
    <div className="welcome-page">
      <motion.div
        className="welcome-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: isLoaded ? 1 : 0 }}
        transition={{ duration: 1, ease: 'easeInOut' }}
      >
        <motion.img
          src={logo}
          alt="Logo"
          className="logo"
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ duration: 2, ease: 'easeOut' }}
        />
        <div className="text-container">
          <motion.h1
            className="welcome-title"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1.5, delay: 0.5, ease: 'easeOut' }}
          >
            {['P', 'r', 'a', 'k', 'r', 'u', 't', 'i', ' ', 'P', 'a', 'r', 'v'].map((char, index) => (
              <motion.span
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1, delay: index * 0.2 }}
                className="char"
              >
                {char}
              </motion.span>
            ))}
          </motion.h1>
          <motion.p
            className="welcome-slogan"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1.5, delay: 1.5, ease: 'easeOut' }}
          >
            Preserve the Wild, Sustain the Future
          </motion.p>
        </div>
        <motion.div
          className="icons-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 2, ease: 'easeInOut' }}
        >
          <FaTree size={60} className="icon-style" />
          <FaPaw size={60} className="icon-style" />
          <GiHummingbird size={60} className="icon-style" />
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Welcome;
