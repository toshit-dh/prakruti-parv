import React, { useState, useEffect } from "react";
import jspdf from "jspdf";
import ReportTemplate from "../fund-report-template/ReportTemplate";
import stamp from "../../assets/prakruti-parv-stamp.png";
import Myproject from "../myproject/Myproject";
import {
  FaTrash,
  FaTimes,
  FaEdit,
  FaMoneyBill,
  FaBullseye,
  FaLocationArrow,
} from "react-icons/fa";
import Steps from "../myprofile/Steps";
import "./ViewProject.css";
import Navbar from "../navbar/Navbar";
import axios from "axios";
import { GET_ALL_PROJECTS_ROUTE } from "../../utils/Routes";
import { useNavigate } from "react-router-dom";
const ViewProject = () => {
  const navigate = useNavigate()
  const [endDate, setEndDate] = useState("");
  const [status, setStatus] = useState("");
  const [type, setType] = useState("");
  const [tags, setTags] = useState("");
  const [projects, setProjects] = useState([]);
  const [filteredProjects, setFilteredProjects] = useState(projects);
  const [isReportDialogOpen, setReportDialog] = useState(false);
    const [isMapDialogOpen, setIsMapDialogOpen] = useState(false);
    const [isStepsDialogOpen, setIsStepsDialogOpen] = useState(false);
    const [selectedProject,setSelectedProject] = useState(null)

  const handleFilter = () => {
    let filtered = projects;
    if (endDate) {
      filtered = filtered.filter(
        (project) => new Date(project.endDate) <= new Date(endDate)
      );
    }
    if (status) {
      filtered = filtered.filter((project) => project.status === status);
    }
    if (type) {
      filtered = filtered.filter((project) => project.type === type);
    }
    if (tags) {
      filtered = filtered.filter((project) => project.tags.includes(tags));
    }

    setFilteredProjects(filtered);
  };

  useEffect(() => {
    const fetch = async () => {
      const response = await axios.get(GET_ALL_PROJECTS_ROUTE, {
        withCredentials: true,
      });
      console.log(response.data);
      
      setProjects(response.data);
      setFilteredProjects(response.data);
    };
    fetch();
  }, []);

  return (
    <div className="content3">
      <Navbar />
      <div className="projects-container">
      {projects.map((project, index) => (
        <div key={index} className="project-container">
          <div className="project-header">
            <h2>{project.organizationName}</h2>
            <div className="project-dates">
              <div className="dates"> 
                <strong>Start:</strong> {project.startDate.split("T")[0]}
              </div>
              <div className="dates"> 
                <strong>End:</strong> {project.endDate.split("T")[0]}
              </div>
            </div>
            <div className="change-buttons">
              <button>
                Update
                <FaEdit />
              </button>
              <button>
                Delete
                <FaTrash />
              </button>
              <button>
                Close
                <FaTimes />
              </button>
            </div>
          </div>
          <div className="project-row">
            <div className="banner-image">
              <img src={project.bannerImage} alt="Banner" />
            </div>
            <div className="project-description">
              <p>{project.description}</p>
              <div className="more-details">
                <p>
                  <FaBullseye color="blue" /> <strong> Target Amount:₹</strong>{" "}
                  {project?.goalAmount}
                </p>
                <p>
                  <FaMoneyBill color="green" />{" "}
                  <strong> Current Amount: ₹</strong>
                  {project?.currentAmount}
                </p>
                <p>
                  <FaLocationArrow color="red" />
                  <strong> Current Amount: ₹</strong>
                  {project?.currentAmount}
                </p>
              </div>
            </div>
          </div>
          <div className="project-buttons">
            <div className="span">
              <span
                className="generate"
                onClick={() => {
                  let pro;
                  if (setIsMapDialogOpen) {
                    pro = null;
                  } else {
                    pro = project;
                  }
                  setSelectedProject(project);
                  setIsMapDialogOpen(!isMapDialogOpen);
                }}
              >
                View On Map
              </span>
            </div>
            <div className="span">
              <span
                className="generate"
                onClick={() => setIsStepsDialogOpen(!isStepsDialogOpen)}
              >
                View Steps
              </span>
            </div>
            <div className="span">
              <span
                className="generate"
                onClick={() => setReportDialog(!isReportDialogOpen)}
              >
                Generate Report
              </span>
            </div>
          </div>
          {isReportDialogOpen && (
            <div className="dialog">
              <div className="modal">
                <ReportTemplate projectData={project} ref={ref} />
                <div className="dialog-buttons">
                  <button onClick={() => setReportDialog(!isReportDialogOpen)}>
                    Close
                  </button>
                  <button onClick={handleDownload}>Download PDF</button>
                </div>
              </div>
            </div>
          )}
          {isMapDialogOpen && (
            <div className="dialog">
              <div className="modal">
                <Myproject project={selectedProject} />
                <div className="dialog-buttons">
                  <button onClick={() => setIsMapDialogOpen(!isMapDialogOpen)}>
                    Close
                  </button>
                </div>
              </div>
            </div>
          )}
          {isStepsDialogOpen && (
            <div className="dialog">
              <div className="modal">
                <Steps steps={project.steps} />
                <div className="dialog-buttons">
                  <button
                    onClick={() => setIsStepsDialogOpen(!isStepsDialogOpen)}
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
      </div>
    </div>
  );
};

export default ViewProject;
