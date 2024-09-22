import axios from "axios";
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { GET_PROJECT_BY_ID_ROUTE } from "../../../utils/Routes";
import Navbar from "../../navbar/Navbar";
import "./Project.css";
import ImageCarousel from "./Carousel"; // Assuming you have this carousel component

export default function Project() {
  const [project, setProject] = useState(null);
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
  }, [id]); // Pass id as a dependency here

  return (
    <div className="content">
      <Navbar />
      <div className="project-card">
        {project && (
          <>
            <div className="column-1">
              <div className="row row-1-5">{project.title}</div>
              <div className="row row-3-5">
                {
                  project.images && <ImageCarousel images={project.images}/>
                }
              </div>
              <div className="row row-1-5">
                <div className="row row-3">
                  <div className="sub-column">
                    {project.startDate.split("T")[0]}
                  </div>
                  <div className="sub-column">
                    {project.endDate.split("T")[0]}
                  </div>
                  <div className="sub-column">{project.location}</div>
                </div>
              </div>
            </div>
            <div className="column-2">
              <div className="row-col-2">
                <div className="sub-column">
                  <strong>Creator:</strong> {project.creator.username}
                </div>
                <div className="sub-column">
                  <strong>Status:</strong> {project.status}
                </div>
              </div>
              <div className="row">
                <p>
                  <strong>{project.type}</strong>: {project.description}
                  <br />
                  {project.tags &&
                    project.tags.map((tag, index) => (
                      <strong key={index}> {tag} </strong>
                    ))}
                </p>
              </div>
              <div className="row-col-2">
                <div className="sub-column">
                  <strong>Current Amount:</strong> ${project.currentAmount}
                </div>
                <div className="sub-column">
                  <strong>Goal Amount:</strong> ${project.goalAmount}
                </div>
              </div>
              <div className="row-col-2">
                <div className="sub-column">Updates</div>
                <div className="sub-column">Milestones</div>
              </div>
              <div className="row">
                <p>Want to contribute?</p>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
