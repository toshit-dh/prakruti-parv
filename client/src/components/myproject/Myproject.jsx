/* eslint-disable no-unused-vars */
import React, { useEffect, useState, useRef } from "react";
import jspdf from "jspdf";
import { useParams } from "react-router-dom";
import "./Myproject.css";
import axios from "axios";
import { GET_PROJECT_BY_ID_ROUTE } from "../../utils/Routes";
import Navbar from "../navbar/Navbar";
import ProjectMap from "../project-map/ProjectMap";
import stamp from '../../assets/prakruti-parv-stamp.png'
import {
  FaTrash,
  FaTimes,
  FaEdit,
  FaPhone,
  FaMoneyBill,
  FaBullseye,
  FaUsers,
} from "react-icons/fa";
import { useSpring, animated } from "react-spring";
import ReportTemplate from "../fund-report-template/ReportTemplate";

const Myproject = () => {
  const { projectId } = useParams();
  const [project, setProject] = useState(null);
  const [loading, setLoading] = useState(true);
  const [location, setLocation] = useState(null);
  const [steps, setSteps] = useState([]);
  const [isReportDialogOpen, setReportDialog] = useState(false);
  const pdf = new jspdf();
  const ref = useRef();
  useEffect(() => {
    const fetchProject = async () => {
      try {
        const response = await axios.get(GET_PROJECT_BY_ID_ROUTE(projectId), {
          withCredentials: true,
        });
        setProject(response.data);

        const placeName = response.data.location;
        await fetchCoordinates(placeName);

        setSteps(response.data.steps || []);
      } catch (error) {
        console.error("Error fetching project:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchProject();
  }, [projectId]);

  const fetchCoordinates = async (placeName) => {
    try {
      const response = await axios.get(
        `https://nominatim.openstreetmap.org/search?q=${placeName}&format=json&addressdetails=1`
      );
      const data = response.data;

      if (data.length > 0) {
        const { lat, lon } = data[0];
        setLocation({ latitude: parseFloat(lat), longitude: parseFloat(lon) });
      } else {
        console.error("Location not found");
      }
    } catch (error) {
      console.error("Error fetching coordinates:", error);
    }
  };

  const handleUpdate = () => {
    console.log("Update project:", projectId);
  };

  const handleDelete = () => {
    console.log("Delete project:", projectId);
  };

  const handleClose = () => {
    console.log("Close project:", projectId);
  };
  const handleDownload = () => {
    const element = ref.current;
    if (element) {
      console.log(element);
      const doc = new jspdf("p", "mm", "a4");
      doc.html(element, {
        callback: (doc) => {
          doc.addImage(stamp,'PNG', 145, 245, 40, 40)
          doc.save("report.pdf");
        },
        x: 5, 
        y: 5,
        width: 180, 
        windowWidth: 900, 
        margin: [10, 10, 10, 10],
      });
    }
    setReportDialog(!isReportDialogOpen)
  };
  const projectInfoAnimation = useSpring({
    opacity: loading ? 0 : 1,
    transform: loading ? "translateY(-20px)" : "translateY(0)",
    config: { duration: 500 },
  });

  if (loading) {
    return <div className="myproject-loading">Loading...</div>;
  }

  return (
    <>
      <Navbar />
      <animated.div
        style={projectInfoAnimation}
        className="myproject-info-container"
      >
        <h1>{project?.title}</h1>
        <div className="myproject-actions">
          <button onClick={handleDelete} className="myproject-button">
            <FaTrash color="red" /> Delete Project
          </button>
          <button onClick={handleClose} className="myproject-button">
            <FaTimes color="orange" /> Close Project
          </button>
          <button onClick={handleUpdate} className="myproject-button">
            <FaEdit color="green" /> Update Project
          </button>
        </div>
        <div className="myproject-details">
          <p>
            <FaPhone color="brown" /> <strong>Phone:</strong>{" "}
            {project?.contactPhoneNumber}
          </p>
          <p>
            <FaMoneyBill color="green" /> <strong>Current Amount:₹</strong>{" "}
            {project?.currentAmount}
          </p>
          <p>
            <FaBullseye color="blue" /> <strong>Target Amount:₹</strong>{" "}
            {project?.goalAmount}
          </p>
          <p>
            <FaUsers color="purple" /> <strong>Contributors:</strong>{" "}
            {project?.contributors?.length}
          </p>
          <p>
            <strong>Description:</strong> {project?.description}
          </p>
          <div className="wrap-span">
            <span
              className="generate"
              onClick={() => setReportDialog(!isReportDialogOpen)}
            >
              Generate Report
            </span>
          </div>
        </div>
      </animated.div>
      <div className="myproject-container">
        {location && (
          <div className="myproject-map-background">
            <ProjectMap location={location} steps={steps} />
          </div>
        )}
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
    </>
  );
};

export default Myproject;