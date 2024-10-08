/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from "react";
import Navbar from "../navbar/Navbar";
import "./Profile.css";
import { FaEdit, FaUser, FaEnvelope, FaMedal } from "react-icons/fa"; 
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from "axios";
import { EDIT_PROFILE_ROUTE } from "../../utils/Routes";
import defaultAvatar from '../../assets/default.png';

export default function Profile() {
  const [user, setUser] = useState(null);
  const [isEdit, setIsEdit] = useState(false);
  const [avatar, setAvatar] = useState(null);
  const [isChanged, setIsChanged] = useState(false);

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
        setAvatar(`https://res.cloudinary.com/djt5vw5aa/image/upload/v1727512495/user-profiles/${data.data._id ? data.data._id : 'default'}.jpg`);
      } catch (e) {
        console.error(e);
        toast.error(e.message);
      }
    };

    fetchUserData();
  }, []);

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
      const response = await axios.post(EDIT_PROFILE_ROUTE, formData, {
        withCredentials: true,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      toast.success("Profile updated successfully!");
      setIsEdit(false);
      setIsChanged(false);
    } catch (e) {
      console.error("Error response:", e.response ? e.response.data : e.message);
      toast.error(`Error updating profile: ${e.message}`);
    }
  };

  const handleCancel = () => {
    setIsEdit(false);
    setIsChanged(false);

    if (user && user.profile.avatarUrl) {
      setAvatar(user.profile.avatarUrl);
    }
    if (avatar && avatar.startsWith("blob:")) {
      URL.revokeObjectURL(avatar);
    }
  };

  return (
    <div className="profile">
      <Navbar />
      <ToastContainer />
      <div className="content">
        <div className="profile-header">
          <img className="avatar" src={avatar ? avatar : defaultAvatar} alt="Profile" />
          <div className="user-info">
            <h2>{user?.username}</h2>
            <div className="user-info-edit">
              <button className="edit" onClick={() => setIsEdit(!isEdit)}>
                <FaEdit />
              </button>
            </div>
            <p>
              <FaUser className="icon" />
              {user?.role}
            </p>
            <p>
              <FaMedal className="icon" />
              {user?.badges.length ? user.badges.join(", ") : "No badges"}
            </p>
            <p>
              <FaEnvelope className="icon" />
              {user?.email}
            </p>
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
      
    </div>
  );
}
