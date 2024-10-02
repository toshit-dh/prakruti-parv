/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import CountUp from 'react-countup';
import { motion } from 'framer-motion';
import { FaUpload, FaSearch, FaLeaf, FaInfoCircle, FaExclamationCircle, FaSkullCrossbones } from 'react-icons/fa'; 
import Navbar from '../navbar/Navbar';
import './IdentifySpecies.css';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import axios from 'axios';
import loadingGif from '../../assets/loading.gif'
import { IDENTIFY_ROUTE } from '../../utils/Routes';

const IdentifySpecies = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageUploaded, setImageUploaded] = useState(false); 
  const [selectedFile, setSelectedFile] = useState(null);
  const [animalInfo, setAnimalInfo] = useState({});
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

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(URL.createObjectURL(file));
      setImageUploaded(true);
      setSelectedFile(file);
    } else {
      toast.error('Please upload an image file.', toastOptions);
    }
  };

  const handleBackClick = () => {
    setSelectedImage(null);
    setImageUploaded(false);
    setSelectedFile(null);
    setAnimalInfo({});
  };

  const handleIdentifyClick = async() => {
    if (!selectedImage) {
      toast.error('Please upload an image first.', toastOptions);
      return;
    }

    setLoading(true); 

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post(IDENTIFY_ROUTE, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log(response);
      const result = response.data;
      if (response.status === 200) {
        setAnimalInfo(result);
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
    <div className="identifyPage">
      <Navbar />
      <div className="content2">
        <div className="titleContainer">
          <motion.h1
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: 'easeOut' }}
          >
            Identify Wildlife with Our Advanced Model
          </motion.h1>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: 'easeOut', delay: 0.5 }}
          >
            Track and Learn About Diverse Species
          </motion.h2>
        </div>
        <div className="counters">
          <div className="counter">
            <h2>
              <CountUp end={150} duration={2} />+
            </h2>
            <p>Trained on species</p>
          </div>
          <div className="counter">
            <h2>
              <CountUp end={1000} duration={2} />+
            </h2>
            <p>Trusted by users</p>
          </div>
          <div className="counter">
            <h2>
              <CountUp end={2000} duration={2} />+
            </h2>
            <p>Conserved through efforts</p>
          </div>
        </div>

        {!imageUploaded && (
          <div className="uploadContainer">
            <motion.div
              className="uploadBox"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1, ease: 'easeOut', delay: 1 }}
            >
              <label htmlFor="upload-input" className="uploadLabel">
                <FaUpload className="uploadIcon" />
                <p>Upload Image Here</p>
                <input
                  id="upload-input"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="uploadInput"
                />
              </label>
            </motion.div>
          </div>
        )}

        {loading && (
        <div className="loadingContainer">
          <img src={loadingGif} alt="Loading..." width={800} />
          <p>Loading...</p>
        </div>
        )}

        {imageUploaded && !loading && (
          <div className="imagePreviewContainer">
            <motion.img
              src={selectedImage}
              alt="Selected Preview"
              className="imagePreview"
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

            {Object.keys(animalInfo).length > 0 && (
              <div className="infoContainer">
                <div className="infoBox kingdomPhylum slideInFromLeft">
                  <h3>It is <span className='animalName'>{(animalInfo.label).toUpperCase()}</span></h3>
                  <p><FaLeaf /> <strong>Kingdom:</strong> {animalInfo.kingdom}</p>
                  <p><FaLeaf /> <strong>Phylum:</strong> {animalInfo.phylum}</p>
                  <p><FaLeaf /> <strong>Class:</strong> {animalInfo.class}</p>
                  <p><FaLeaf /> <strong>Order:</strong> {animalInfo.order}</p>
                  <p><FaLeaf /> <strong>Family:</strong> {animalInfo.family}</p>
                  <p><FaLeaf /> <strong>Scientific Name:</strong> {animalInfo.scientific_name}</p>
                  <p><FaLeaf /> <strong>Primary Habitat:</strong> {animalInfo.primary_habitat}</p>
                  <p><FaLeaf /> <strong>Geographical Range:</strong> {animalInfo.geographical_range}</p>
                </div>
                <div className="infoBox funFact slideInFromRight">
                  <p><FaInfoCircle /> <strong>Fun Fact:</strong> {animalInfo.fun_fact}</p>
                </div>
                <div className="infoBox legalConsequences slideInFromLeft">
                  <p><FaExclamationCircle /> <strong>Legal Consequence 1:</strong> {animalInfo.crime_1}</p>
                  <p><FaExclamationCircle /> <strong>Legal Consequence 2:</strong> {animalInfo.crime_2}</p>
                </div>
                <div className="infoBox legalConsequences slideInFromRight">
                  <p><FaSkullCrossbones /> <strong>Crime Example 1:</strong> {animalInfo.example_1}</p>
                  <p><FaSkullCrossbones /> <strong>Crime Example 2:</strong> {animalInfo.example_2}</p>
                </div>
              </div>
            )}

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

export default IdentifySpecies;
