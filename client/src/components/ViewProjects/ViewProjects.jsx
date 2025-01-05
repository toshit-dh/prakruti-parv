import { useState, useEffect } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import { GET_ALL_PROJECTS_ROUTE } from "../../utils/Routes";
import Navbar from "../navbar/Navbar";
import "./ViewProject.css";
import { useNavigate } from "react-router-dom";

const ViewProjects = () => {
  const PROJECT_STATUSES = {
    ALL: "all",
    ACTIVE: "active",
    COMPLETED: "completed",
    PENDING: "pending",
  };

  const [projects, setProjects] = useState([]);
  const [filter, setFilter] = useState(PROJECT_STATUSES.ALL);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const { data } = await axios.get(GET_ALL_PROJECTS_ROUTE, {
          withCredentials: true,
        });
        setProjects(Array.isArray(data) ? data : []);
      } catch (error) {
        toast.error("Failed to fetch projects. Please try again.");
      }
    };

    fetchProjects();
  }, []);

  const filteredProjects = projects.filter(
    (project) => {
        if (filter === PROJECT_STATUSES.ALL) return true;
        return project.status.toLowerCase() === filter.toLowerCase();
    }
  );

  const handleProjectNavigation = (projectId) => {
    navigate(`/project/${projectId}`);
  };

  return (
    <>
      <Navbar />
      <div className="main-container">
        <section className="projects-section">
          <header className="projects-header">
            <h2 className="all-projects">All Projects</h2>
            {projects.length > 0 && (
              <div className="filter-buttons">
                {Object.values(PROJECT_STATUSES).map((status) => (
                  <button
                    key={status}
                    onClick={() => setFilter(status)}
                    className={filter === status ? "active" : ""}
                  >
                    {status.charAt(0).toUpperCase() + status.slice(1)}
                  </button>
                ))}
              </div>
            )}
          </header>
          <div className="projects-list">
            {filteredProjects.map((project) => (
              <div
                key={project._id}
                className="project-item"
                onClick={() => handleProjectNavigation(project._id)}
              >
                <div
                  className="project-banner"
                  style={{ backgroundImage: `url(${project?.bannerImage})` }}
                >
                  <div className="project-overlay">
                    <h3>{project?.title}</h3>
                    <p>{project?.description}</p>
                    <span className="project-status">{project?.status}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
        <ToastContainer />
      </div>
    </>
  );
};

export default ViewProjects;
