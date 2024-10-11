/* eslint-disable no-unused-vars */
import React, { useState } from "react";
import "./ProjectForm.css"; 
import Navbar from "../navbar/Navbar";

const ProjectForm = () => {
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

  const handleSubmit = (e) => {
    e.preventDefault();
    
    console.log(projectData);
  };

  return (

    <>
      <Navbar />
      <form className="project-form" onSubmit={handleSubmit}>
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
    </>
    
  );
};

export default ProjectForm;
