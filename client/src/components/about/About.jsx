import React from 'react';
import { motion } from 'framer-motion';
import Navbar from '../navbar/Navbar';
import { Container, Typography, Button, Grid } from '@mui/material';
import styled from '@emotion/styled';
import { useNavigate } from 'react-router-dom';


const AboutContainer = styled(Container)`
  background-color: #f0f8ff;
  padding: 50px;
  border-radius: 10px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  margin-top: 120px;
`;


const Description = styled(Typography)`
  font-size: 1.2rem;
  margin-top: 20px;
`;

const AnimatedButton = styled(motion(Button))`
  margin-top: 30px;
 
`;

const FeatureSection = styled.div`
  margin-top: 40px;
`;

const FeatureTitle = styled(Typography)`
  font-size: 2rem;
  font-weight: bold;
  color: #2e8b57;
  text-align: center;
`;

const FeatureCard = styled(motion.div)`
  background-color: #ffffff;
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 4px 10px rgba(0,0,0,0.1);
  pointer:cursor;
  &:hover {
    transform: translateY(-10px) scale(1.02);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
  }

  img {
    transition: transform 0.3s ease;
  }

  &:hover img {
    transform: scale(1.1);
  }
`;

const Footer = styled.footer`
  background-color: #2e8b57;
  color: white;
  text-align: center;
  padding: 20px;
  margin-top:30px;
`;
const Title = styled(motion.h1)`
  text-align: center;
  font-size: 3rem;
  margin: 2rem 0;
  background: linear-gradient(to right, #2e8b57, #3cb371);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  position: relative;
  cursor: pointer;
  
  &::after {
    content: '';
    position: absolute;
    width: 100%;
    height: 3px;
    bottom: -5px;
    left: 0;
    background: #2e8b57;
    transform: scaleX(0);
    transform-origin: right;
    transition: transform 0.5s ease;
  }
  
  &:hover::after {
    transform: scaleX(1);
    transform-origin: left;
  }
`;

const About = () => {
    const features = [
        { title: 'Poaching Detection', imageUrl: 'https://media0.giphy.com/media/3o6Mba61hJ4LGK9AGI/200w.gif?cid=6c09b952g6wwznrrhlia2cv5tzk8d9v7rxtj1gy66abndzzz&ep=v1_gifs_search&rid=200w.gif&ct=g' },
        { title: 'Animal Identification', imageUrl: 'https://cdn.dribbble.com/users/77598/screenshots/11298534/media/d631f46c6a89600f4d044afdad413c9c.gif' },
        { title: 'Fundraising', imageUrl: 'https://media0.giphy.com/media/3oxHQg1F3nkc7gt9Qc/giphy.gif?cid=6c09b952h8oqux5zk33jt8ojemddi6thjxj5yfwd2nzu7l4b&ep=v1_gifs_search&rid=giphy.gif&ct=g' },
        { title: 'Education & Awareness', imageUrl: 'https://i.pinimg.com/originals/0f/0b/48/0f0b48e71dae625f8f5afe5d1c4dbaec.gif' },
    ];
    
    const navigate=useNavigate();

    return (
        <>
            <Navbar />
            <AboutContainer>
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.7 }}>
                    <Title variant="h1">About Prakruti Parv</Title>
                    <Description>
                        The proposed system, Prakruti Parv, addresses key challenges in wildlife conservation by integrating advanced technologies.
                    </Description>
                    <Description>
                        Utilizing deep learning models, the platform enhances species identification and improves the accuracy of poaching detection through a PyTorch-based neural network.
                    </Description>
                    <Description>
                        This allows users to gain reliable information about endangered wildlife while contributing to the identification of illegal activities.
                    </Description>
                    <AnimatedButton 
                        variant="contained" 
                        color="primary" 
                        whileHover={{ scale: 1.05 }} 
                        whileTap={{ scale: 0.95 }}
                        onClick={()=>{navigate('/')}}
                    >
                        Join Us in Conservation
                    </AnimatedButton>
                </motion.div>

                <FeatureSection>
                    <FeatureTitle variant="h2">Our Features</FeatureTitle>
                    <Grid container spacing={3} style={{ marginTop: '20px' }}>
                        {features.map((feature, index) => (
                            <Grid item xs={12} sm={6} md={6} key={index}>
                                <FeatureCard 
                                    initial={{ scale: 0.9 }} 
                                    animate={{ scale: [1,1.05,1] }} 
                                    transition={{ duration: 0.8 }}
                                >
                                    <img src={feature.imageUrl} alt={feature.title} style={{ width: '100%', borderRadius: '10px' }} />
                                    <Typography variant="h6" style={{ marginTop: '10px', color:'#2e8b57' }}>
                                        {feature.title}
                                    </Typography>
                                </FeatureCard>
                            </Grid>
                        ))}
                    </Grid>
                </FeatureSection>
            </AboutContainer>

            <Footer>
                <Typography variant="body1">© {new Date().getFullYear()} Prakruti Parv. All rights reserved.</Typography>
                <Typography variant="body2">Contact us at prakrutiparv0@gmail.com</Typography>
            </Footer>
        </>
    );
};

export default About;
