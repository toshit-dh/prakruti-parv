/* eslint-disable no-unused-vars */
import React, { useEffect, useState } from "react";
import "./Myproject.css";
import axios from "axios";
import ProjectMap from "../project-map/ProjectMap";



const Myproject = ({project}) => {
  
  const [loading, setLoading] = useState(true);
  const [location, setLocation] = useState(null);
  const [steps, setSteps] = useState([]);
  useEffect(() => {
    console.log(project);
    
    const fetchCoordinates = async (placeName) => {
      try {
        const response = await axios.get(
          `https://nominatim.openstreetmap.org/search?q=${placeName}&format=json&addressdetails=1`
        );
        const data = response.data;
        console.log(data);
        
        if (data.length > 0) {
          const { lat, lon } = data[0];
          setLocation({
            latitude: parseFloat(lat),
            longitude: parseFloat(lon),
          });
        } else {
          console.error("Location not found");
        }
      } catch (error) {
        console.error("Error fetching coordinates:", error);
      }
    };
    fetchCoordinates(project.location)  
    setLoading(false)
  }, [project]);

  if (loading) {
    return <div className="myproject-loading">Loading...</div>;
  }
  
  return (
        <div className="myproject-container">
          {location && (
            <div className="myproject-map-background">
              <ProjectMap location={location} steps={steps} />
            </div>
          )}
        </div>

  );
};

export default Myproject;
