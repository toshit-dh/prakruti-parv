/* eslint-disable no-unused-vars */
import React, { useState } from "react";
import "./ProjectForm.css"; 
import Navbar from "../navbar/Navbar";
import axios from "axios"; 
import { ToastContainer, toast } from "react-toastify"; 
import 'react-toastify/dist/ReactToastify.css'; 
import { useNavigate } from "react-router-dom"; 
import { CREATE_PROJECT_ROUTE } from "../../utils/Routes";

const ProjectForm = () => {
  const toastOptions = {
    position: 'bottom-left',
    autoClose: 5000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: 'dark',
  };
  const [projectData, setProjectData] = useState({
    organizationName: "",
    bannerFile: null, 
    title: "",
    location: "",
    projectDescription: "",
    targetAmount: "", 
    startDate: "",
    endDate: "",
    steps: [{ description: "", status: "Not Started" }],
    phoneNumber: "",
    email: "",
  });

  const navigate = useNavigate(); 

  const handleChange = (e) => {
    const { id, value, files } = e.target;
    if (id === "bannerFile") {
      setProjectData((prev) => ({
        ...prev,
        [id]: files ? files[0] : prev[id],
      }));
    } else {
      setProjectData((prev) => ({
        ...prev,
        [id]: value,
      }));
    }
  };

  const addStep = () => {
    setProjectData((prev) => ({
      ...prev,
      steps: [...prev.steps, { description: "", status: "Not Started" }],
    }));
  };

  const removeStep = (index) => {
    setProjectData((prev) => ({
      ...prev,
      steps: prev.steps.filter((_, i) => i !== index),
    }));
  };

  const handleStepChange = (index, e) => {
    const { id, value } = e.target;
    const newSteps = [...projectData.steps];
    newSteps[index][id] = value;
    setProjectData((prev) => ({ ...prev, steps: newSteps }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (new Date(projectData.endDate) <= new Date(projectData.startDate)) {
      toast.error("End Date must be after Start Date.");
      return;
    }

    const start = new Date(projectData.startDate);
    const end = new Date(projectData.endDate);
    const durationInDays = Math.ceil((end - start) / (1000 * 60 * 60 * 24));
    const formData = new FormData();
    formData.append("organizationName", projectData.organizationName);
    formData.append("contactPhoneNumber", projectData.phoneNumber);
    formData.append("contactEmail", projectData.email);
    formData.append("title", projectData.title);
    formData.append("description", projectData.projectDescription);
    formData.append("goalAmount", projectData.targetAmount);
    formData.append("endDate", projectData.endDate);
    formData.append("location", projectData.location);
    formData.append("duration", durationInDays);
    formData.append("steps", JSON.stringify(projectData.steps)); 
    formData.append("bannerImage", projectData.bannerFile); 

    try {
      const response = await axios.post(CREATE_PROJECT_ROUTE, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        withCredentials: true, 
      });
      if (response.status === 200) {
        toast.success("Project created successfully!",toastOptions);
        setTimeout(()=>{

          navigate('/profile')
        },3000)
      }
      setProjectData({
        organizationName: "",
        bannerFile: null, 
        title: "",
        location: "",
        projectDescription: "",
        targetAmount: "", 
        startDate: "",
        endDate: "",
        steps: [{ description: "", status: "Not Started" }],
        phoneNumber: "",
        email: "",
      });

    } catch (error) {
      console.error("Error creating project:", error);
      if (error.response && error.response.data && error.response.data.error) {
        toast.error(`Error: ${error.response.data.error}`,toastOptions);
      } else {
        toast.error("An unexpected error occurred. Please try again.",toastOptions);
      }
    }
  };

  return (
    <>
      <Navbar />
      <form className="project-form" onSubmit={handleSubmit} encType="multipart/form-data">
        <h2 className="project-form-title">FundRaising Form</h2>

        <div className="project-form-row">
          <div className="project-input-group">
            <label htmlFor="organizationName">Organization Name:</label>
            <input
              type="text"
              id="organizationName"
              value={projectData.organizationName}
              onChange={handleChange}
              required
            />
          </div>

          <div className="project-input-group">
            <label htmlFor="bannerFile">Banner File:</label>
            <input
              type="file"
              id="bannerFile"
              onChange={handleChange}
              accept=".jpg,.jpeg,.png"
              required
            />
          </div>
        </div>

        <div className="project-form-row">
          <div className="project-input-group">
            <label htmlFor="title">Project Title:</label>
            <input
              type="text"
              id="title"
              value={projectData.title}
              onChange={handleChange}
              placeholder="Enter project title"
              required
            />
          </div>

          <div className="project-input-group">
            <label htmlFor="location">Location:</label>
            <input
              type="text"
              id="location"
              value={projectData.location}
              onChange={handleChange}
              placeholder="Enter project location"
              required
            />
          </div>
        </div>

        <div className="project-input-group">
          <label htmlFor="projectDescription">Project Description:</label>
          <textarea
            id="projectDescription"
            value={projectData.projectDescription}
            onChange={handleChange}
            required
          />
        </div>

        <div className="project-form-row">
          <div className="project-input-group">
            <label htmlFor="targetAmount">Target Amount:</label>
            <input
              type="number"
              id="targetAmount"
              value={projectData.targetAmount}
              onChange={handleChange}
              placeholder="Enter target amount"
              required
              min="1"
            />
          </div>
        </div>

        <div className="project-form-row">
          <div className="project-input-group">
            <label htmlFor="startDate">Start Date:</label>
            <input
              type="date"
              id="startDate"
              value={projectData.startDate}
              onChange={handleChange}
              required
            />
          </div>

          <div className="project-input-group">
            <label htmlFor="endDate">End Date:</label>
            <input
              type="date"
              id="endDate"
              value={projectData.endDate}
              onChange={handleChange}
              required
            />
          </div>
        </div>

        <div className="steps-section">
          <h3>Project Steps</h3>
          {projectData.steps.map((step, index) => (
            <div className="project-add-group" key={index}>
              <input
                type="text"
                id="description"
                value={step.description}
                placeholder={`Step ${index + 1} Description`}
                onChange={(e) => handleStepChange(index, e)}
                required
              />
              <select
                id="status"
                value={step.status}
                onChange={(e) => handleStepChange(index, e)}
                required
              >
                <option value="Not Started">Not Started</option>
                <option value="In Progress">In Progress</option>
                <option value="Completed">Completed</option>
              </select>
              {index === projectData.steps.length - 1 ? (
                <button
                  type="button"
                  className="add-button"
                  onClick={addStep}
                  aria-label="Add Step"
                >
                  +
                </button>
              ) : (
                <button
                  type="button"
                  className="remove-button"
                  onClick={() => removeStep(index)}
                  aria-label="Remove Step"
                >
                  −
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="short-inputs">
          <div className="short-input-group">
            <label htmlFor="phoneNumber">Phone Number:</label>
            <input
              type="tel"
              id="phoneNumber"
              value={projectData.phoneNumber}
              onChange={handleChange}
              placeholder="Enter contact phone number"
              required
              pattern="^\+?[1-9]\d{1,14}$" 
              title="Please enter a valid phone number."
            />
          </div>

          <div className="short-input-group">
            <label htmlFor="email">Email (Gmail):</label>
            <input
              type="email"
              id="email"
              value={projectData.email}
              onChange={handleChange}
              pattern="^[a-zA-Z0-9._%+-]+@gmail\.com$"
              title="Please enter a valid Gmail address."
              placeholder="Enter contact Gmail address"
              required
            />
          </div>
        </div>
        <button type="submit" className="submit-button">
          Submit
        </button>
      </form>
      <ToastContainer /> 
    </>
  );
};

export default ProjectForm;
