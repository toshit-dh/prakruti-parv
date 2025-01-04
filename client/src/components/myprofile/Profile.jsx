/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react";
import Navbar from "../navbar/Navbar";
import "./Profile.css";
import { FaArrowAltCircleLeft, FaArrowAltCircleRight } from "react-icons/fa";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from "axios";
import {
  EDIT_PROFILE_ROUTE,
  GET_PROJECT_BY_ORGANIZATION_ROUTE,
} from "../../utils/Routes";
import defaultAvatar from "../../assets/default.png";
import { useNavigate } from "react-router-dom";
import MyProjects from "./MyProjects";

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
  const [isProfileOpen, setIsProfileOpen] = useState(false);

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
        });
        setAvatar(
          `https://res.cloudinary.com/djt5vw5aa/image/upload/v1727512495/user-profiles/${
            data.data._id || "default"
          }.jpg`
        );
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

  if (user == null) return;
  return (
    <div className="content">
      <Navbar />
      <ToastContainer />
      <div className={`left ${!isProfileOpen && "close"}`}>
        <div className={`profile  ${isProfileOpen && "closed"}`}>
          {!isProfileOpen && (
            <div className="details">
              <img src={avatar} />
              <h2>{user.username}</h2>
              <h4>Badges: {user.badges}</h4>
              <h4>{user.email}</h4>
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
        <MyProjects projects={projects}/>
      </div>
    </div>
  );
}
