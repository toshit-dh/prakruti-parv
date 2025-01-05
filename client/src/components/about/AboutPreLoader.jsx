import { IconButton } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import './AboutPreLoader.css';
import { useNavigate } from 'react-router-dom';
import  aboutVideo from '../../assets/about-video.mp4';
import { useSpring, animated } from 'react-spring';
import { useState } from 'react';

const AboutPreLoader = () => {
    const navigate = useNavigate();
    const [isExiting, setIsExiting] = useState(false);

    const fadeOut = useSpring({
        opacity: isExiting ? 0 : 1,
        transform: isExiting ? 'scale(1.5) blur(10px)' : 'scale(1) blur(0px)',
        config: { duration: 1000 },
        onRest: () => {
            if (isExiting) {
                navigate('/about');
            }
        }
    });

    const handleSkip = () => {
        setIsExiting(true);
    };
    return (
        <div className="preloader-container">
            <video 
                className="background-video"
                autoPlay 
                loop 
                playsInline
            >
                <source src={aboutVideo} type="video/mp4" />
            </video>
            
            <div className="button-container">
                <IconButton 
                    className="animate-button skip-button"
                    onClick={handleSkip}
                >
                    <SkipNextIcon />
                </IconButton>
                
                <IconButton 
                    className="animate-button home-button"
                    onClick={() => navigate('/')}
                >
                    <HomeIcon />
                </IconButton>
            </div>
        </div>
    );
};

export default AboutPreLoader;