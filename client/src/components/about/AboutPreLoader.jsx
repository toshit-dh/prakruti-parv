import { IconButton } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import './AboutPreLoader.css';
import { useNavigate } from 'react-router-dom';
import  aboutVideo from '../../assets/about-video.mp4';
const AboutPreLoader = () => {
    const navigate = useNavigate();
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
                    onClick={() => navigate('/about')}
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