/* eslint-disable no-unused-vars */
import React, { useEffect, useState, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import './Home.css';
import Navbar from '../../components/navbar/Navbar';
import videofile from '../../assets/nature-video.mp4';
import birdChirping from '../../assets/birds-chirping.mp3';
import { useSelector, useDispatch } from 'react-redux';
import { fetchUser } from '../../redux/slice/UserSlice';

const Home = () => {
  const [scrollY, setScrollY] = useState(0);
  const audioRef = useRef(null);
  const dispatch = useDispatch();
  const user = useSelector((state) => state.user);
  const location = useLocation();

  
  useEffect(() => {
    dispatch(fetchUser());
  }, [dispatch]);

  useEffect(() => {
    audioRef.current = new Audio(birdChirping);
    audioRef.current.play();

    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
      window.removeEventListener('scroll', handleScroll);
    };
  }, [location]);

  return (
    <div className="homeContainer">
      <Navbar />
      <div
        className="videoContainer"
        style={{ filter: `blur(${Math.min(scrollY / 100, 10)}px)` }}
      >
        <video autoPlay muted loop className="backgroundVideo">
          <source src={videofile} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div className="textOverlay">
          <h1 className="mainTitle">Prakruti Parv</h1>
          <p className="tagline">Wild Wonders Await</p>
        </div>
      </div>
    </div>
  );
};

export default Home;
