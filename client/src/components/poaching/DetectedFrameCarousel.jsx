import React from 'react';
import Slider from 'react-slick';
import './DetectedFrameCarousel.css'
const DetectedFramesCarousel = ({ detectedFrames }) => {
    const settings = {
        dots: true,
        infinite: true,
        speed: 500,
        slidesToShow: 1,
        slidesToScroll: 1,
    };
    console.log(detectedFrames);
    
    return (
        <div style={{ width: '80%', margin: 'auto' }}>
            <h2>Detected Frames</h2>
            <Slider {...settings}>
                {detectedFrames.map((frame, index) => (
                    <div key={index}>
                        <img
                            src={frame}
                            alt={`Detected Frame ${index + 1}`}
                            style={{ width: '100%', height: 'auto' }}
                        />
                    </div>
                ))
                }
            </Slider>
        </div>
    );
};

export default DetectedFramesCarousel;
