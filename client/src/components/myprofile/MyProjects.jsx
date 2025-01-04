import React, { useRef, useState } from "react";
import jspdf from "jspdf";
import "./MyProjects.css";
import ReportTemplate from "../fund-report-template/ReportTemplate";
import stamp from "../../assets/prakruti-parv-stamp.png";
import Myproject from "../myproject/Myproject";
import {
  FaTrash,
  FaTimes,
  FaEdit,
  FaMoneyBill,
  FaBullseye,
  FaLocationArrow
} from "react-icons/fa";
import Steps from "./Steps";
const MyProjects = ({ projects }) => {
  const [isReportDialogOpen, setReportDialog] = useState(false);
  const [isMapDialogOpen, setIsMapDialogOpen] = useState(false);
  const [isStepsDialogOpen, setIsStepsDialogOpen] = useState(false);
  const [selectedProject,setSelectedProject] = useState(null)
  const ref = useRef();
  const handleDownload = () => {
    const element = ref.current;
    if (element) {
      console.log(element);
      const doc = new jspdf("p", "mm", "a4");
      doc.html(element, {
        callback: (doc) => {
          doc.addImage(stamp, "PNG", 145, 250, 40, 40);
          doc.save("report.pdf");
        },
        x: 5,
        y: 5,
        width: 180,
        windowWidth: 900,
        margin: [5, 5, 5, 5],
      });
    }
    setReportDialog(!isReportDialogOpen);
  };
  if (projects == null) return;
  console.log(projects);
  return (
    <div className="projects-container">
      {projects.map((project, index) => (
        <div key={index} className="project-container">
          <div className="project-header">
            <h2>{project.organizationName}</h2>
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

          <div className="project-dates">
            <div>
              <strong>Start Date:</strong> {project.startDate.split("T")[0]}
            </div>
            <div>
              <strong>End Date:</strong> {project.endDate.split("T")[0]}
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
                <Steps steps={project.steps}/>
                <div className="dialog-buttons">
                  <button onClick={() => setIsStepsDialogOpen(!isStepsDialogOpen)}>
                    Close
                  </button>
                </div>
              </div>
            </div>
          )
          }
        </div>
      ))}
    </div>
  );
};

export default MyProjects;
