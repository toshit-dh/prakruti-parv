import React, { useState, useEffect } from "react";
import Navbar from "../navbar/Navbar";
import "./Profile.css";
import {
  FaArrowAltCircleLeft,
  FaArrowAltCircleRight,
  FaPlus,
} from "react-icons/fa";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from "axios";
import {
  EDIT_PROFILE_ROUTE,
  GET_PROJECT_BY_ORGANIZATION_ROUTE,
  UPLOAD_ROUTE,
  GET_UPLOAD_ROUTE
} from "../../utils/Routes";
import prb from "../../assets/prakrutiratna.png";
import pmb from "../../assets/prakrutimudra.png";
import defaultAvatar from "../../assets/default.png";
import { useNavigate } from "react-router-dom";
import MyProjects from "./MyProjects";
import MyUploads from "../myuploads/MyUploads";

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
  const [uploadedItem, setUploadedItem] = useState(null);
  const [fileType, setFileType] = useState("");
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [uploadDialog, setUploadDialog] = useState(false);

  // Form state for upload
  const [uploadData, setUploadData] = useState({
    category: "",
    species: "",
    title: "",
    description: "",
    file: null,
  });

  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const { data } = await axios.get(EDIT_PROFILE_ROUTE, {
          withCredentials: true,
        });
        setUser({
          ...data.data.profile,
          username: data.data.username,
          role: data.data.role,
          badges: data.data.badges,
          email: data.data.email,
          currency: data.data.currency,
        });
        setAvatar(data.data.profile.avatarUrl);
        
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
          const { data } = await axios.get(
            GET_PROJECT_BY_ORGANIZATION_ROUTE(userId),
            { withCredentials: true }
          );
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
      console.error(
        "Error response:",
        error.response ? error.response.data : error.message
      );
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

  const handleProjectNavigation = (project_id) => {
    const projectId = project_id;
    navigate(`/project/${projectId}`);
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedItem(URL.createObjectURL(file));
      const fileExtension = file.type.split("/")[0]; // Extract type (image, video, audio)
      setFileType(fileExtension);
      setUploadData((prevState) => ({ ...prevState, file }));
    }
  };

  const handleUserUpload = async () => {
    const { category, species, title, description, file } = uploadData;

    // Check for missing fields
    if (!category) {
      toast.error("Please select a category.");
      return;
    }

    if (!title) {
      toast.error("Please provide a title.");
      return;
    }

    if (!description) {
      toast.error("Please provide a description.");
      return;
    }

    if (!file) {
      toast.error("Please upload a file.");
      return;
    }


    const formData = new FormData();
    formData.append("category", category);
    formData.append("species", species || "");
    formData.append("title", title);
    formData.append("description", description);
    formData.append("media", file);

    try {
      await axios.post(UPLOAD_ROUTE, formData, {
        withCredentials: true,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log(formData);
      
      toast.success("Upload successful!");
      setUploadDialog(false);
      setUploadData({
        category: "",
        species: "",
        title: "",
        description: "",
        file: null,
      }); // Reset the form
    } catch (error) {
      console.error(
        "Error response:",
        error.response ? error.response.data : error.message
      );
      toast.error("Error uploading content. Please try again.");
    }
  };

  if (user == null) return null;

  return (
    <div className="content">
      <Navbar />
      <ToastContainer />
      <div className={`left ${!isProfileOpen && "close"}`}>
        <div className={`profile  ${isProfileOpen && "closed"}`}>
          {!isProfileOpen && (
            <div className="details">
              <img src={avatar} alt="Avatar" />
              <h2>{user.username}</h2>
              <h4>{user.email}</h4>
              <div className="badge">
                <img src={prb} alt="Badge" className="badge-icon" />
                <h4>Prakruti Ratna: {user.badges}</h4>
              </div>
              <div className="badge">
                <img src={pmb} alt="Badge" className="badge-icon" />
                <h4>Prakruti Mudra: {user.currency}</h4>
              </div>
            </div>
          )}

          <div
            className="toggle-button"
            onClick={() => setIsProfileOpen(!isProfileOpen)}
          >
            {!isProfileOpen ? (
              <FaArrowAltCircleLeft />
            ) : (
              <FaArrowAltCircleRight />
            )}
          </div>
        </div>
      </div>
      <div className="right">
        {user.role.toLowerCase() === "user" ||
        user.role.toLowerCase() === "photographer" ||
        user.role.toLowerCase() === "learner" ? (
          <MyUploads />
        ) : (
          <MyProjects projects={projects} />
        )}
      </div>
      {uploadDialog && (
        <div className="dialog">
          <div className="modal">
            <h2>Upload New Content.</h2>
            <div className="upload-form">
              <label htmlFor="upload-type">Category:</label>
              <select
                id="upload-type"
                value={uploadData.category}
                onChange={(e) =>
                  setUploadData((prevState) => ({
                    ...prevState,
                    category: e.target.value,
                  }))
                }
              >
                <option value="species">Species</option>
                <option value="poaching">Poaching</option>
              </select>
              <label htmlFor="specific-species">
                Specific Species (Optional):
              </label>
              <input
                type="text"
                id="specific-species"
                value={uploadData.species}
                onChange={(e) =>
                  setUploadData((prevState) => ({
                    ...prevState,
                    species: e.target.value,
                  }))
                }
                placeholder="Enter species name..."
              />
              <label htmlFor="title">Title:</label>
              <input
                type="text"
                id="title"
                value={uploadData.title}
                onChange={(e) =>
                  setUploadData((prevState) => ({
                    ...prevState,
                    title: e.target.value,
                  }))
                }
                placeholder="Enter title..."
              />
              <label htmlFor="description">Description:</label>
              <input
                type="text"
                id="description"
                value={uploadData.description}
                onChange={(e) =>
                  setUploadData((prevState) => ({
                    ...prevState,
                    description: e.target.value,
                  }))
                }
                placeholder="Enter description..."
              />
              <label htmlFor="media-upload">Upload Image/Video/Audio:</label>
              <input
                type="file"
                id="media-upload"
                accept="image/*,video/*,audio/*"
                onChange={handleFileChange}
              />

              {/* Display preview */}
              {uploadedItem && fileType === "image" && (
                <img
                  src={uploadedItem}
                  alt="Preview"
                  className="upload-preview"
                />
              )}
              {uploadedItem && fileType === "video" && (
                <video controls className="upload-preview">
                  <source src={uploadedItem} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              )}
              {uploadedItem && fileType === "audio" && (
                <audio controls className="upload-preview">
                  <source src={uploadedItem} type="audio/mpeg" />
                  Your browser does not support the audio tag.
                </audio>
              )}
            </div>
            <div className="dialog-buttons">
              <button onClick={() => setUploadDialog(false)}>Close</button>
              <button onClick={handleUserUpload}>Upload</button>
            </div>
          </div>
        </div>
      )}
      <button className="fab" onClick={() => setUploadDialog(!uploadDialog)}>
        <FaPlus />
      </button>
    </div>
  );
}
