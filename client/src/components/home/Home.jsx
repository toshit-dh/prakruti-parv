/* eslint-disable no-unused-vars */
import React, { useEffect, useState,useRef } from 'react';
import './Home.css';
import Navbar from '../navbar/Navbar'; 
import videofile from '../../assets/nature-video.mp4';
import birdChirping from '../../assets/birds-chirping.mp3';

const Home = () => {
  const [scrollY, setScrollY] = useState(0);
  const audioRef = useRef(null);
  

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    audioRef.current = new Audio(birdChirping);
    audioRef.current.play();
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
    };
  }, []);

  return (
    <div className="homeContainer">
      <Navbar />
      <div className="videoContainer" style={{ filter: `blur(${Math.min(scrollY / 100, 10)}px)` }}>
        <video autoPlay muted loop className="backgroundVideo">
          <source src={videofile} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div className="textOverlay">
          <h1 className="mainTitle">Prakruti Parv</h1>
          <p className="tagline">Preserve the wild and sustain the future!!!</p>
        </div>
      </div>
    </div>
  );
};

export default Home;
