import React, { useEffect, useState, useRef } from 'react';
import './Home.css';
import Navbar from '../../components/navbar/Navbar'; 
import videofile from '../../assets/nature-video.mp4';
import birdChirping from '../../assets/birds-chirping.mp3';
import { useDispatch } from 'react-redux';
import { fetchUser } from '../../redux/slice/UserSlice';

const Home = () => {
  const dispatch = useDispatch();
  const [scrollY, setScrollY] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [interactionOccurred, setInteractionOccurred] = useState(false); // To track if user has interacted
  const audioRef = useRef(null);

  useEffect(() => {
    dispatch(fetchUser());
  }, [dispatch]);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Function to handle the first user interaction
  const handleUserInteraction = () => {
    if (!interactionOccurred) {
      setInteractionOccurred(true);
    }
  };

  const handleAudioToggle = () => {
    if (!interactionOccurred) {
      handleUserInteraction(); // Ensures first click enables audio
    }

    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play().catch(error => {
          console.error('Audio playback error:', error);
        });
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="homeContainer" onClick={handleUserInteraction}>
      <Navbar />
      <div className="videoContainer" style={{ filter: `blur(${Math.min(scrollY / 100, 10)}px)` }}>
        <video autoPlay muted loop className="backgroundVideo">
          <source src={videofile} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        <div className="textOverlay">
          <h1 className="mainTitle">Prakruti Parv</h1>
          <p className="tagline">Wild Wonders Await</p>
        </div>
      </div>
      {/* Hidden audio element */}
      <audio ref={audioRef} src={birdChirping} />
      {/* Button to toggle audio playback */}
      <button onClick={handleAudioToggle} className="audioToggleButton">
        {isPlaying ? 'Stop Audio' : 'Play Audio'}
      </button>
    </div>
  );
};

export default Home;
