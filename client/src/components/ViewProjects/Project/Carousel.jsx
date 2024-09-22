import React from 'react';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css"; 
import "slick-carousel/slick/slick-theme.css";
import './Carousel.css'
const ImageCarousel = ({ images }) => {
  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
  };

  return (
    <Slider {...settings}>
      {images.length > 0 ? (
        images.map((image, index) => (
          <div key={index}>
            <img src={`http://localhost:8080${image.url}`} alt="Project Image" className="carousel-image" />
          </div>
        ))
      ) : (
        <div>No images available</div>
      )}
    </Slider>
  );
};

export default ImageCarousel;
