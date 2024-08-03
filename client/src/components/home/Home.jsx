/* eslint-disable no-unused-vars */
import React, { useEffect, useState } from 'react';
import './Home.css';
import Navbar from '../navbar/Navbar'; 
import videofile from '../../assets/nature-video.mp4';
import birdChirping from '../../assets/birds-chirping.mp3';

const Home = () => {
  const [scrollY, setScrollY] = useState(0);
  

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const audio = new Audio(birdChirping);
    audio.play();
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
