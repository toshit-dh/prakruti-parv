import axios from "axios";
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { GET_PROJECT_BY_ID_ROUTE } from "../../../utils/Routes";
import Navbar from "../../navbar/Navbar";
import "./Project.css";
import ImageCarousel from "./Carousel";
export default function Project() {
  const [project, setProject] = useState({});
  const { id } = useParams();
  useEffect(() => {
    const fetch = async () => {
      try {
        const response = await axios.get(GET_PROJECT_BY_ID_ROUTE(id), {
          withCredentials: true,
        });
        console.log(response.data);
        setProject(response.data);
      } catch (error) {
        console.log(error.message);
      }
    };
    fetch();
  }, []);
  return (
    <div className="content">
      <Navbar />
      <div className="project-card"></div>
    </div>
  );
}
