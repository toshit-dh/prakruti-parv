import  { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import Navbar from '../../components/navbar/Navbar';
import './IdentifyAudio.css';

const IdentifyAudio = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const fileInputRef = useRef(null);
  const audioRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      // Check if file is audio
      if (!selectedFile.type.startsWith('audio/')) {
        setError('Please upload an audio file');
        setFile(null);
        setFileName('');
        return;
      }
      
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
      setResult(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      
      if (!droppedFile.type.startsWith('audio/')) {
        setError('Please upload an audio file');
        return;
      }
      
      setFile(droppedFile);
      setFileName(droppedFile.name);
      setError('');
      setResult(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Please select an audio file');
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError('');

    const formData = new FormData();
    formData.append('audio', file);

    try {
      const response = await axios.post('http://127.0.0.1:8081/predict-sound', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setResult(response.data);
      setIsLoading(false);
    } catch (err) {
      console.error('Error identifying sound:', err);
      setError('Failed to identify sound. Please try again.');
      setIsLoading(false);
    }
  };

  return (
    <div className="identify-audio-container">
      <Navbar />
      
      <div className="identify-audio-content">
        <motion.h1 
          className="identify-audio-title"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Identify Animal Sounds
        </motion.h1>
        
        <motion.p 
          className="identify-audio-description"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          Upload an audio recording to identify which animal it belongs to
        </motion.p>
        
        <div className="identify-audio-main">
          <motion.div 
            className="upload-section"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <div 
              className="dropzone"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={handleUploadClick}
            >
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="audio/*"
                className="file-input" 
              />
              
              {!file ? (
                <div className="upload-prompt">
                  <div className="upload-icon">
                    <i className="fas fa-microphone"></i>
                  </div>
                  <p>Drag & drop an audio file or click to browse</p>
                  <span>Supports WAV, MP3, and other audio formats</span>
                </div>
              ) : (
                <div className="file-preview">
                  <div className="audio-icon">
                    <i className="fas fa-file-audio"></i>
                  </div>
                  <p className="file-name">{fileName}</p>
                  <div className="audio-controls">
                    {URL.createObjectURL(file) && (
                      <>
                        <audio ref={audioRef} src={URL.createObjectURL(file)} onEnded={() => setIsPlaying(false)} />
                        <button className="play-button" onClick={handlePlayPause}>
                          {isPlaying ? (
                            <i className="fas fa-pause"></i>
                          ) : (
                            <i className="fas fa-play"></i>
                          )}
                        </button>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            {error && <p className="error-message">{error}</p>}
            
            <motion.button 
              className="identify-button"
              onClick={handleSubmit}
              disabled={!file || isLoading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {isLoading ? 'Identifying...' : 'Identify Animal'}
            </motion.button>
          </motion.div>
          
          {result && (
            <motion.div 
              className="result-section"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="result-header">
                <h2>Identification Results</h2>
              </div>
              
              <div className="main-result">
                <motion.div 
                  className="animal-icon"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <img src={`/images/animals/${result.animal.toLowerCase()}.png`} alt={result.animal} 
                       onError={(e) => {e.target.src = '/images/animals/default.png'}} />
                </motion.div>
                <div className="animal-details">
                  <h3>{result.animal}</h3>
                  <div className="confidence-meter">
                    <div className="confidence-bar">
                      <motion.div 
                        className="confidence-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence}%` }}
                        transition={{ duration: 1, delay: 0.3 }}
                      ></motion.div>
                    </div>
                    <span className="confidence-value">{result.confidence.toFixed(1)}% Confidence</span>
                  </div>
                </div>
              </div>
              
              <div className="other-predictions">
                <h4>Other Possibilities</h4>
                <ul className="predictions-list">
                  {result.top_predictions.slice(1).map((prediction, index) => (
                    <motion.li 
                      key={index}
                      className="prediction-item"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: 0.4 + (index * 0.1) }}
                    >
                      <span className="prediction-animal">{prediction[0]}</span>
                      <div className="prediction-bar-container">
                        <motion.div 
                          className="prediction-bar"
                          initial={{ width: 0 }}
                          animate={{ width: `${prediction[1]}%` }}
                          transition={{ duration: 0.8, delay: 0.6 + (index * 0.1) }}
                        ></motion.div>
                        <span className="prediction-value">{prediction[1].toFixed(1)}%</span>
                      </div>
                    </motion.li>
                  ))}
                </ul>
              </div>
              
              <motion.button 
                className="new-upload-button"
                onClick={() => {
                  setFile(null);
                  setFileName('');
                  setResult(null);
                  setIsPlaying(false);
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Upload New Audio
              </motion.button>
            </motion.div>
          )}
        </div>
      </div>
      
      <div className="nature-decorations">
        <div className="leaf leaf-1"></div>
        <div className="leaf leaf-2"></div>
        <div className="leaf leaf-3"></div>
      </div>
    </div>
  );
};

export default IdentifyAudio;
