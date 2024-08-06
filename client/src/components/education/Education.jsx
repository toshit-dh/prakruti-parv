/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import { FaSearch, FaFilePdf,FaTimes,FaSpinner } from 'react-icons/fa'; 
import Navbar from '../navbar/Navbar'; 
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';
import 'swiper/css/pagination';
import 'swiper/css/navigation';
import './Education.css';
import {motion} from 'framer-motion';
import { Autoplay, Pagination, Navigation } from 'swiper/modules';
import wiki from 'wikipedia';
import { ToastContainer,toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { createCanvas, loadImage } from 'canvas';
import { jsPDF } from 'jspdf';
import axios from 'axios';



const Education = () => {
  const [species, setSpecies] = useState('');
  const [info, setInfo] = useState(null); 
  const [image, setImage] = useState(''); 
  const [loading, setLoading] = useState(false); 
  const toastOptions= {
    position: "bottom-left",
    autoClose: 5000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "dark",
    }
  const sliderInfo = [
    {
      name: 'Amur Tiger',
      info: 'Amur tigers are one of the larger tiger subspecies. The average weight for males is 160-190 kg, while females are smaller.',
      img: 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/P.t.altaica_Tomak_Male.jpg/1200px-P.t.altaica_Tomak_Male.jpg'
    },
    {
      name: 'Sumatran Orangutan',
      info: 'The Sumatran orangutan (Pongo abelii) is one of the three species of orangutans. Critically endangered, and found only in the north of the Indonesian island of Sumatra.',
      img: 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Sumatra-Orang-Utan_im_Pongoland.jpg/330px-Sumatra-Orang-Utan_im_Pongoland.jpg'
    },
    {
      name: 'Javan Rhino',
      info: 'The Javan rhinoceros (Rhinoceros sondaicus) is a critically endangered member of the genus Rhinoceros.',
      img: 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Rhinoceros_sondaicus_in_London_Zoo.jpg/330px-Rhinoceros_sondaicus_in_London_Zoo.jpg'
    },
    {
      name: 'Vaquita',
      info: 'The vaquita (Phocoena sinus) is a species of porpoise endemic to the northern end of the Gulf of California in Baja California, Mexico.',
      img: 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/Vaquita4_Olson_NOAA.jpg/330px-Vaquita4_Olson_NOAA.jpg'
    },
    {
      name: 'Hawksbill Turtle',
      info: 'The hawksbill sea turtle (Eretmochelys imbricata) is a critically endangered sea turtle belonging to the family Cheloniidae.',
      img: 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Eretmochelys-imbricata-K%C3%A9lonia-2.JPG/330px-Eretmochelys-imbricata-K%C3%A9lonia-2.JPG'
    }
  ];

  const handleSearch = async() => {
    if(species===''){
        toast.error('Please enter a species name!!!',toastOptions);
        return;
    }
    setLoading(true); 
    try {
        const page = await wiki.page(species);
        const summary = await page.summary();
        setInfo(summary.extract);
        setImage(summary.originalimage.source);
        
    } catch (error) {
        
        toast.error('Some error occured!!!',toastOptions);
        
    }
    finally{
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
  
  const handleClear=()=>{
    setSpecies('');
    setInfo(null);
    setImage('');
  }


  const createPdf = async (imageUrl,summary,title) => {
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
    const contentMargin = 10;
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
  }

  


  const handlePdf= async()=>{
    if(species ==='' || info===null || image===''){
        toast.error('Please search for a species first!!!',toastOptions);
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

  }


  return (
    <div>
      <Navbar /> 
      <div className="educationSection">
        <div className="educationIntroText">
        <motion.h1
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
          >
            Discover Wildlife Knowledge!
          </motion.h1>
          <p>Search for any species and gain valuable insights about them!!!</p>
        </div>

        <div className="educationTrendingSection">
          
          <Swiper
            spaceBetween={30}
            centeredSlides={true}
            autoplay={{
              delay: 2500,
              disableOnInteraction: false,
            }}
            pagination={{
              clickable: true,
            }}
            navigation={true}
            modules={[Autoplay, Pagination, Navigation]}
            className="mySwiper"
          >
            {sliderInfo.map((item, index) => (
              <SwiperSlide key={index}>
                <div className="carouselItem" style={{ backgroundImage: `url(${item.img})` }}>
                  <div className="carouselContent">
                    <h3>{item.name}</h3>
                    <p>{item.info}</p>
                  </div>
                </div>
              </SwiperSlide>
            ))}
          </Swiper>
        </div>
        <div className="educationSearchBarContainer">
          <input
            type="text"
            placeholder="Enter species name..."
            value={species}
            onChange={(e) => setSpecies(e.target.value)}
          />
          <button onClick={handleSearch} disabled={loading}>
            <FaSearch />
            Search
          </button>
        </div>

        <div className="educationInfoSection">
          <div className="educationInfoContent">
            {info ? (
              <div dangerouslySetInnerHTML={{ __html: info }} />
            ) : (
              <p>Get the information here</p>
            )}
          </div>
          <div className="educationInfoImage">
            {image ? (
              <img src={image} alt="Species" />
            ) : (
              <p>Get the image here!!!</p>
            )}
          </div>
        </div>

        <div className="educationButtons">
          <button className="educationDownloadButton" onClick={handlePdf}>
            <FaFilePdf /> Download PDF
          </button>
          <button className="educationClearButton" onClick={handleClear}>
            <FaTimes /> Clear
          </button>
        </div>
        {loading && (
        <div className="loadingOverlay">
          <FaSpinner className="loadingIcon" />
        </div>
      )}
      </div>
      <ToastContainer/>
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

export default Education;
