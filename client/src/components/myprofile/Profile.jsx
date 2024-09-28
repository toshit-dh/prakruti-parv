import React, { useState, useEffect } from "react";
import Navbar from "../navbar/Navbar";
import "./Profile.css";
import { FaEdit } from "react-icons/fa";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import axios from 'axios';
import { EDIT_PROFILE_ROUTE } from "../../utils/Routes";

// Dummy user data for fallback
const dummyUser = {
  name: "John Doe",
  username: "john_doe",
  profile: {
    bio: "This is a dummy bio.",
    avatarUrl: "https://example.com/dummy-avatar.png"
  },
  role: "conservationist",
  badges: 5,
  email: "john@example.com"
};

export default function Profile() {
  const [user, setUser] = useState(null);
  const [isEdit, setIsEdit] = useState(false);
  const [avatar, setAvatar] = useState(dummyUser.profile.avatarUrl);
  const [isChanged, setIsChanged] = useState(false);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const response = await axios.get(EDIT_PROFILE_ROUTE, {
          withCredentials: true
        });
        setUser(response.data);
        setAvatar(response.data?.profile?.avatarUrl || dummyUser.profile.avatarUrl);
      } catch (e) {
        console.error(e);
        toast.error(e.message);
      }
    };

    fetchUserData();
  }, []);

  useEffect(() => {
    if (user) {
      const hasChanges =
        user.name !== dummyUser.name ||
        user.username !== dummyUser.username ||
        user.profile?.bio !== dummyUser.profile.bio ||
        user.role !== dummyUser.role ||
        user.email !== dummyUser.email ||
        avatar !== dummyUser.profile.avatarUrl;

      setIsChanged(hasChanges);
    }
  }, [user, avatar]);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
      setAvatar(URL.createObjectURL(file));
      setUser((prevUser) => ({
        ...prevUser,
        profile: { ...prevUser.profile, avatarUrl: URL.createObjectURL(file) },
      }));
    } else {
      toast.error("Please upload an image in .jpg or .png format.");
    }
  };

  const handleChange = (e) => {
    const { id, value } = e.target;
    setUser((prevUser) => ({
      ...prevUser,
      ...(id === "name" && { name: value }),
      ...(id === "bio" && { profile: { ...prevUser.profile, bio: value } }),
      ...(id === "role" && { role: value }),
    }));
  };

  const handleSave = async () => {
    const formData = new FormData();
    formData.append("name", user.name);
    formData.append("bio", user.profile?.bio || "");
    formData.append("role", user.role);

    // Append the actual file object if changed
    const fileInput = document.getElementById('image-upload');
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
      console.log(response);
      toast.success("Profile updated successfully!");
    } catch (e) {
      console.error('Error response:', e.response ? e.response.data : e.message);
      toast.error(`Error updating profile: ${e.message}`);
    }
  };

  const handleCancel = () => {
    setUser(dummyUser);
    setAvatar(dummyUser.profile.avatarUrl);
    setIsEdit(false);
    if (avatar && avatar.startsWith('blob:')) {
      URL.revokeObjectURL(avatar);
    }
  };

  if (!user) {
    return <p>Loading user data...</p>; // Fallback UI while loading
  }

  return (
    <div className="profile">
      <Navbar />
      <div className="content">
        <ToastContainer />
        <div className="profile-header">
          <div className="aside">
            <img src={avatar} alt="Profile" className="avatar" />
            {isEdit && (
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />
            )}
          </div>
          {!isEdit ? (
            <div className="user-info">
              <>
                <h2>{user.name}</h2>
                <p>@{user.username}</p>
                <p>{user.profile.bio}</p>
                <p>Role: {user.role}</p>
                <p>Badges: {user.badges}</p>
                <p>Email: {user.email}</p>
                <button className="edit" onClick={() => setIsEdit(true)}>
                  <FaEdit />
                </button>
              </>
            </div>
          ) : (
            <div className="user-info-edit">
              <div className="input-row">
                <label htmlFor="name">Name: </label>
                <input
                  id="name"
                  type="text"
                  placeholder="Name"
                  value={user.name}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="input-row">
                <label htmlFor="bio">Bio: </label>
                <input
                  id="bio"
                  type="text"
                  placeholder="Bio"
                  value={user.profile.bio}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="input-row">
                <label htmlFor="role">Role: </label>
                <select id="role" value={user.role} onChange={handleChange}>
                  <option value="conservationist">Conservationist</option>
                  <option value="researcher">Researcher</option>
                  <option value="volunteer">Volunteer</option>
                </select>
              </div>
              <div className="button">
                <button
                  className={`save-button ${!isChanged ? 'no-change' : ''}`}
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
    </div>
  );
}
