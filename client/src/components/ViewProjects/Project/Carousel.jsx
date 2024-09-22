import React from 'react';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css"; 
import "slick-carousel/slick/slick-theme.css";

const ImageCarousel = () => {
  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
  };
  const images = []
  return (
    <Slider {...settings}>
      {images.length > 0 ? (
        images.map((image,index) => (
          <div key={index}>
            <img src={image} alt="Project Image" style={{ width: '100%', height: 'auto' }} />
          </div>
        ))
      ) : (
        <div>No images available</div>
      )}
    </Slider>
  );
  
};

export default ImageCarousel;
