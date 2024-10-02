/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FaUpload, FaSearch} from 'react-icons/fa'; 
import Navbar from '../navbar/Navbar';
import './Poaching.css';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import axios from 'axios';
import loadingGif from '../../assets/load4poach.gif';
import { IDENTIFY_ROUTE, POACH_ROUTE } from '../../utils/Routes';

const Poaching = () => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoUploaded, setVideoUploaded] = useState(false); 
  const [selectedFile, setSelectedFile] = useState(null);
  const [animalInfo, setPoachInfo] = useState({});
  const [loading, setLoading] = useState(false); 

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

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedVideo(URL.createObjectURL(file));
      setVideoUploaded(true);
      setSelectedFile(file);
    } else {
      toast.error('Please upload a video file.', toastOptions);
    }
  };

  const handleBackClick = () => {
    setSelectedVideo(null);
    setVideoUploaded(false);
    setSelectedFile(null);
    setPoachInfo({});
  };

  const handleIdentifyClick = async () => {
    if (!selectedVideo) {
      toast.error('Please upload a video first.', toastOptions);
      return;
    }

    setLoading(true); 

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await axios.post(POACH_ROUTE, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log(response);
      const result = response.data;
      if (response.status === 200) {
        setPoachInfo(result);
      } else {
        toast.error(result.error, toastOptions);
      }
    } catch (error) {
      toast.error('An error occurred. Please try again.', toastOptions);
    } finally {
      setLoading(false); 
    }
  };

  return (
    <div className="poachPage">
      <Navbar />
      <div className="content5">
        <div className="titleContainer">
          <motion.h1
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: 'easeOut' }}
          >
            Unyielding Vigilance Against the Silent Slaughter of Earth's Majestic Creatures
          </motion.h1>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: 'easeOut', delay: 0.5 }}
          >
            Identify Poaching Here
          </motion.h2>
        </div>

        {!videoUploaded && (
          <div className="uploadContainer1">
            <motion.div
              className="uploadBox"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, ease: 'easeOut', delay: 1 }}
            >
              <label htmlFor="upload-input" className="uploadLabel">
                <FaUpload className="uploadIcon" />
                <p>Upload Video Here</p>
                <input
                  id="upload-input"
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  className="uploadInput"
                />
              </label>
            </motion.div>
          </div>
        )}

        {loading && (
          <div className="loadingContainer">
            <img src={loadingGif} alt="Loading..." width={800}/>
            <p>Loading...</p>
          </div>
        )}

        {videoUploaded && !loading && (
          <div className="videoPreviewContainer">
            <motion.video
              src={selectedVideo}
              controls
              autoPlay
              muted
              className="videoPreview"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1, ease: 'easeOut' }}
            />
            <motion.div
              className="buttonGroup" 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, ease: 'easeOut', delay: 0.5 }}
            >
              <button className="backButton" onClick={handleBackClick}>
                Back
              </button>
              <button className="identifyButton" onClick={handleIdentifyClick}>
                <FaSearch className="identifyIcon" /> Identify
              </button>
            </motion.div>
          </div>
        )}
      </div>
      <ToastContainer />
      <style>{`
        .Toastify__toast {
          white-space: nowrap;
          text-overflow: ellipsis;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
};

export default Poaching;
