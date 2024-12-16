/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react";
import Navbar from "../navbar/Navbar";
import './Myproject.css'
import { FaEdit, FaUser, FaEnvelope, FaMedal, FaPlusCircle } from "react-icons/fa"; 
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from "axios";
import { EDIT_PROFILE_ROUTE, GET_PROJECT_BY_ORGANIZATION_ROUTE } from "../../utils/Routes"; 
import defaultAvatar from '../../assets/default.png';
import { useNavigate } from "react-router-dom";

const PROJECT_STATUSES = {
  ALL: "all",
  ACTIVE: "active",
  COMPLETED: "completed",
  PENDING: "pending",
};

export default function Profile() {
  const [user, setUser] = useState(null);
  const [projects, setProjects] = useState([]);
  const [isEdit, setIsEdit] = useState(false);
  const [avatar, setAvatar] = useState(null);
  const [isChanged, setIsChanged] = useState(false);
  const [filter, setFilter] = useState(PROJECT_STATUSES.ALL); 
  const [showPopup, setShowPopup] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const { data } = await axios.get(EDIT_PROFILE_ROUTE, { withCredentials: true });
        setUser({
          ...data.data.profile,
          username: data.data.username,
          role: data.data.role,
          badges: data.data.badges,
          email: data.data.email,
        });
        setAvatar(`https://res.cloudinary.com/djt5vw5aa/image/upload/v1727512495/user-profiles/${data.data._id || 'default'}.jpg`);
      } catch (error) {
        console.error(error);
        toast.error("Failed to fetch user data. Please try again.");
      }
    };
    fetchUserData();
  }, []); 

  useEffect(() => {
    const fetchProjects = async () => {
      const userId = user?._id;
      if (userId && user.role.toLowerCase() === "organisation") {
        try {
          const { data } = await axios.get(GET_PROJECT_BY_ORGANIZATION_ROUTE(userId), { withCredentials: true });
          console.log(data)
          setProjects(data.projects); 
        } catch (error) {
          console.error(error);
          toast.error("Failed to fetch projects. Please try again.");
        }
      }
    };

    if (user) {
      fetchProjects();
    }
  }, [user]); 
  
  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
      const objectUrl = URL.createObjectURL(file);
      setAvatar(objectUrl);
      setUser((prevUser) => ({
        ...prevUser,
        profile: { ...prevUser.profile, avatarUrl: objectUrl },
      }));
      setIsChanged(true);
    } else {
      toast.error("Please upload an image in .jpg or .png format.");
    }
  };

  const handleChange = (e) => {
    const { id, value } = e.target;
    setUser((prevUser) => ({
      ...prevUser,
      [id]: value,
    }));
    setIsChanged(true);
  };

  const handleSave = async () => {
    const formData = new FormData();
    formData.append("name", user.name);
    formData.append("bio", user.bio || "");
    formData.append("role", user.role);

    const fileInput = document.getElementById("image-upload");
    if (fileInput.files.length > 0) {
      formData.append("image", fileInput.files[0]);
    }

    try {
      await axios.post(EDIT_PROFILE_ROUTE, formData, {
        withCredentials: true,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      toast.success("Profile updated successfully!");
      setIsEdit(false);
      setIsChanged(false);
    } catch (error) {
      console.error("Error response:", error.response ? error.response.data : error.message);
      toast.error(`Error updating profile: ${error.message}`);
    }
  };

  const handleCancel = () => {
    setIsEdit(false);
    setIsChanged(false);
    if (user?.profile.avatarUrl) {
      setAvatar(user.profile.avatarUrl);
    }
    if (avatar?.startsWith("blob:")) {
      URL.revokeObjectURL(avatar);
    }
  };

  const filteredProjects = projects.filter(project => {
    return filter === PROJECT_STATUSES.ALL || project.status === filter; 
  });

  const handleProjectNavigation=(project_id)=>{

      const projectId=project_id;
      navigate(`/project/${projectId}`);
  }
  return (
    <div className="profile">
      <Navbar />
      <ToastContainer />
      <div className="content">
        <div className="profile-header">
          <img className="avatar" src={avatar || defaultAvatar} alt="Profile" />
          <div className="user-info">
            <h2>{user?.username}</h2>
            <div className="user-info-edit">
              <button className="edit" onClick={() => setIsEdit(!isEdit)}>
                <FaEdit />
              </button>
            </div>
            <p><FaUser className="icon" />{user?.role}</p>
            <p><FaMedal className="icon" />{user?.badges.length ? user.badges.join(", ") : "No badges"}</p>
            <p><FaEnvelope className="icon" />{user?.email}</p>
          </div>
        </div>
        
        {isEdit && (
          <div className="edit-section">
            <div className="input-row">
              <label htmlFor="name">Name:</label>
              <input
                type="text"
                id="name"
                value={user.name || ""}
                onChange={handleChange}
                required
              />
            </div>
            <div className="input-row">
              <label htmlFor="bio">Bio:</label>
              <input
                type="text"
                id="bio"
                value={user.bio || ""}
                onChange={handleChange}
              />
            </div>
            <div className="input-row">
              <label htmlFor="image-upload">Profile Image:</label>
              <input
                type="file"
                id="image-upload"
                onChange={handleImageChange}
                accept=".jpg,.png"
              />
            </div>
            <div className="button-group">
              <button
                className={`save-button ${!isChanged && 'no-change'}`}
                onClick={handleSave}
                disabled={!isChanged}
              >
                Save
              </button>
              <button className="cancel-button" onClick={handleCancel}>
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>

      {user?.role.toLowerCase() === "organisation" && (
      <div className="projects-section">
        <h2 className="myprojects">My Projects</h2>
        
        {projects.length > 0 && (
          <div className="filter-buttons">
            {Object.values(PROJECT_STATUSES).map((status) => (
              <button key={status} onClick={() => setFilter(status)}>
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </button>
            ))}
          </div>
        )}
        
        <div className="projects-list">
          {projects.length === 0 ? (
            <div className="no-projects">No projects found for this organization.</div>
          ) : filteredProjects.length === 0 ? (
            <div className="no-projects">No projects match the selected filter.</div>
          ) : (
            filteredProjects.map((project) => (
              <div key={project._id} className="project-item" onClick={()=>handleProjectNavigation(project._id)}>
                <div className="project-banner" style={{ backgroundImage: `url(${project?.bannerImage})` }}>
                  <div className="project-overlay">
                      <h3>{project?.title}</h3>
                      <p>{project?.description}</p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
      )}
      
      <div className="add-project-button" 
      onClick={() => { navigate('/add-project'); }}
      onMouseEnter={() => setShowPopup(true)} 
      onMouseLeave={() => setShowPopup(false)}>
          <FaPlusCircle  size={40}/>
        <div className="popup">
          Add Project Here!
        </div>
      </div>
      
    </div>
  );
}