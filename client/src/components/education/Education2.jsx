import React, { useState } from "react";
import "./Education2.css";
import Navbar from "../navbar/Navbar";
import { Swiper, SwiperSlide } from 'swiper/react';
import { Navigation, Pagination } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/navigation';
import 'swiper/css/pagination';
import { FaSearch, FaSpinner } from 'react-icons/fa'; 
import wiki from 'wikipedia';
import { jsPDF } from 'jspdf';
import axios from 'axios';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { 
    FaPaw, FaLeaf, 
  FaDownload, FaTrash, FaTree, FaFeather,FaVolumeUp,FaVolumeOff
} from 'react-icons/fa';
import { GiElephant, GiTigerHead, GiBirdHouse } from 'react-icons/gi';

const Education2 = () => {
  const [showSearch, setShowSearch] = useState(false);
  const [loading, setLoading] = useState(false); 
  const [species, setSpecies] = useState('');
  const [info, setInfo] = useState(null); 
  const [image, setImage] = useState(''); 
  const [videos, setVideos] = useState([]);
  const [embeddedVideos, setEmbeddedVideos] = useState([]);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audio] = useState(new Audio());

  const toastOptions = {
    position: "bottom-left",
    autoClose: 5000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "dark",
  }

  const handleGetStartedClick = () => {
    setShowSearch(true);
  };
  const fetchYoutubeVideos = async (speciesName) => {
    try {
      const response = await axios.get(`http://127.0.0.1:8081/get-youtube-videos?name=${speciesName}`);
      setVideos(response.data.videos.slice(0,10));
      setEmbeddedVideos(extractVideoIDs(response.data.videos.slice(0,10)));
      console.log(response.data);
      console.log(embeddedVideos);
    } catch (error) {
      toast.error('Failed to fetch videos', toastOptions);
    }
  };
  const extractVideoIDs = (videos) => {
    return videos.map(videoUrl => {
        const urlParams = new URL(videoUrl).searchParams;
        const videoID = urlParams.get("v");
        return `${videoID}`;
    });
   };
   const fetchAnimalSound = async (animalName) => {
    try {
      const response = await axios({
        url: `http://127.0.0.1:8081/get-animal-sound?animal=${animalName}`,
        method: 'GET',
        responseType: 'blob'
      });
      const audioBlob = new Blob([response.data], { type: 'audio/*' });
      const audioUrl = URL.createObjectURL(audioBlob);
      setAudioUrl(audioUrl);
      audio.src = audioUrl;
    } catch (error) {
      console.error('Error fetching sound:', error);
      toast.error('Failed to load animal sound', toastOptions);
    }
  };
  const toggleSound = () => {
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSearch = async () => {
    if (species === '') {
        toast.error('Please enter a species name!!!', toastOptions);
        return;
    }
    setLoading(true); 
    try {
        const page = await wiki.page(species);
        const summary = await page.summary();
        setInfo(summary.extract);
        setImage(summary.originalimage.source);
        await fetchYoutubeVideos(species);
        await fetchAnimalSound(species);
    } catch (error) {
        toast.error(error.message, toastOptions);
    } finally {
        setLoading(false);
    }
  };

  const getBase64Image = async (url) => {
    const response = await axios({ url, responseType: 'blob' });
    const blob = response.data;
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        resolve(reader.result);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const handleClear = () => {
    setSpecies('');
    setInfo(null);
    setImage('');
    setVideos([]);
  };

  const createPdf = async (imageUrl, summary, title) => {
    const doc = new jsPDF();
    const addWatermark = (text) => {
      doc.setTextColor(220, 220, 220); 
      doc.setFontSize(50); 
      doc.setFont("helvetica", "italic"); 
      doc.text(text, 100, 280, {
        align: 'center',
        angle: 0,
        baseline: 'bottom' 
      });
    };
    
    addWatermark('Prakruti Parv');
    doc.setFontSize(24);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(0, 0, 0);
    doc.text(title, 105, 40, { align: 'center' });
    
    const base64Image = await getBase64Image(imageUrl);
    doc.addImage(base64Image, 'JPEG', 15, 50, 180, 160); 
    
    doc.setFontSize(10); 
    doc.setFont('helvetica', 'normal'); 
    doc.setTextColor(0, 0, 0);
    
    const page = await wiki.page(title);
    let content = await page.content();
    content = content.replace(/<\/?[^>]+(>|$)/g, "");
    const lines = doc.splitTextToSize(content, 180);
    const imageBottomY = 50 + 160; 
    const textStartY = imageBottomY + 20;
    const margin = 15;
    const pageHeight = doc.internal.pageSize.height;
    const lineHeight = 10;
    let currentY = textStartY;

    lines.forEach((line, index) => {
      if (currentY + lineHeight > pageHeight - margin) {
        doc.addPage(); 
        addWatermark('Prakruti Parv');
        currentY = margin; 
      }
      doc.setFontSize(10); 
      doc.setFont('helvetica', 'normal'); 
      doc.setTextColor(0, 0, 0);
      doc.text(line, margin, currentY);
      currentY += lineHeight;
    });
    return doc.output('blob');
  };

  const handlePdf = async () => {
    if (species === '' || info === null || image === '') {
        toast.error('Please search for a species first!!!', toastOptions);
        return;
    }
    setLoading(true);
    try {
        const pdfBlob = await createPdf(image, info, species);
        const url = URL.createObjectURL(pdfBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${species}-info.pdf`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        toast.error(error.message, toastOptions);
    } finally {
        setLoading(false);
    }
  };

  return (
    <div className="education2-container">
      <Navbar />
      <div className="leaves-animation" />
      <div className="education2-content">
        {!showSearch ? (
          <div className="education2-welcome-section">
            
            <h1><FaLeaf className="leaf-icon" /> Explore the Wonders of Wildlife</h1>
            <p><FaPaw /> Learn, understand, and protect our Earth's most magnificent creatures <FaPaw /> </p>
            <button className="get-started-btn" onClick={handleGetStartedClick}>
              <FaTree /> Get Started
            </button>
            
          </div>
        ) : (
          <>
            <div className="education2-search-container">
              <GiBirdHouse className="bird-house-icon" size={60} />
              <h2><FaFeather /> Discover Wildlife Wonder</h2>
              <div className="education2-search-form">
                <div className="education2-input-wrapper">
                  <FaPaw className="education2-input-icon" size={30} />
                  <input
                    type="text"
                    className="education2-search-input"
                    placeholder="Enter a species name"
                    value={species}
                    onChange={(e) => setSpecies(e.target.value)}
                  />
                </div>
                <button className="education2-search-btn" onClick={handleSearch}>
                  <FaSearch className="species-search" /> Search
                </button>
                {loading && (
                  <div className="loadingOverlay">
                    <div className="paw-print-loader">
                      <FaPaw />
                      <FaPaw />
                      <FaPaw />
                    </div>
                  </div>
                )}
              </div>
              <div className="actions">
                <button className="clear-btn" onClick={handleClear}>
                  <FaTrash /> Clear
                </button>
                <button className="download-btn" onClick={handlePdf}>
                  <FaDownload /> Download PDF
                </button>
              </div>
            </div>
            {info && image && (
              <div className="education2-result-container">
                {audioUrl && (
                  <button 
                    className="sound-button"
                    onClick={toggleSound}
                    title={isPlaying ? "Stop Sound" : "Play Sound"}
                  >
                    {isPlaying ? <FaVolumeUp /> : <FaVolumeOff />}
                  </button>
                )}
                <GiElephant className="result-icon" />
                <div className="image-section">
                  <img src={image} alt={species} className="species-image" />
                </div>
                <div className="info-section">
                  <h3><FaPaw /> {species}</h3>
                  <p>{info}</p>
                </div>
                {videos.length > 0 && (
                <div className="videos-section">
                  <h3><FaPaw /> Related Videos</h3>
                  <Swiper
                    modules={[Navigation, Pagination]}
                    spaceBetween={20}
                    slidesPerView={3}
                    navigation
                    pagination={{ clickable: true }}
                    className="videos-swiper"
                  >
                    {embeddedVideos.map((video, index) => (
                      <SwiperSlide key={index}>
                        <div className="video-card">
                          <iframe
                            width="100%"
                            height="200"
                            src={`https://www.youtube.com/embed/${video}`}
                            title={`Video ${index + 1}`}
                            frameBorder="0"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                          />
                        </div>
                      </SwiperSlide>
                    ))}
                  </Swiper>
                </div>
              )}
              </div>
            )}
          </>
        )}
      </div>
      <ToastContainer />
    </div>
  );
};

export default Education2;