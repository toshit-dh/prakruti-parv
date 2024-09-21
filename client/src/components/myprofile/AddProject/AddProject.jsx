import React, { useState } from 'react';
import axios from 'axios';
import './AddProject.css'; // Import the CSS file

const BASE_URL = "http://localhost:8080";
const PROJECTS_ROUTE = `${BASE_URL}/api/projects`;

export default function AddProject() {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [goalAmount, setGoalAmount] = useState('');
  const [endDate, setEndDate] = useState('');
  const [type, setType] = useState('');
  const [location, setLocation] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    const projectData = {
      title,
      description,
      goalAmount,
      endDate,
      type,
      location
    };
  
    try {
      const response = await axios.post(PROJECTS_ROUTE, projectData, {
        headers: {
          'Content-Type': 'application/json'
        },
        withCredentials: true
      });
      alert('Project added successfully');
      // Reset form
      setTitle('');
      setDescription('');
      setGoalAmount('');
      setEndDate('');
      setType('');
      setLocation('');
    } catch (error) {
      console.error('Error adding project:', error);
      alert('Error adding project');
    }
  };
  

  return (
    <div className="container">
      <h2>Add New Project</h2>
      <form onSubmit={handleSubmit} encType="multipart/form-data">
        <div className="form-group">
          <label htmlFor="title">Title:</label>
          <input type="text" id="title" value={title} onChange={(e) => setTitle(e.target.value)} required />
        </div>
        <div className="form-group">
          <label htmlFor="description">Description:</label>
          <textarea id="description" value={description} onChange={(e) => setDescription(e.target.value)} required />
        </div>
        <div className="form-group">
          <label htmlFor="goalAmount">Goal Amount:</label>
          <input type="number" id="goalAmount" value={goalAmount} onChange={(e) => setGoalAmount(e.target.value)} required />
        </div>
        <div className="form-group">
          <label htmlFor="endDate">End Date:</label>
          <input type="date" id="endDate" value={endDate} onChange={(e) => setEndDate(e.target.value)} required />
        </div>
        <div className="form-group">
          <label htmlFor="type">Type:</label>
          <select id="type" value={type} onChange={(e) => setType(e.target.value)} required>
            <option value="">Select Type</option>
            <option value="Tree">Tree</option>
            <option value="Land">Land</option>
            <option value="Water">Water</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="location">Location:</label>
          <input type="text" id="location" value={location} onChange={(e) => setLocation(e.target.value)} required />
        </div>
        <button type="submit">Add Project</button>
      </form>
    </div>
  );
}

