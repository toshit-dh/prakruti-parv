import React, { useState, useEffect } from "react";
import {
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from "@mui/material";
import "./ViewProject.css";
import Navbar from "../navbar/Navbar";
import axios from "axios";
import { GET_ALL_PROJECTS_ROUTE } from "../../utils/Routes";
import { useNavigate } from "react-router-dom";
const FilterBar = () => {
  const navigate = useNavigate()
  const [endDate, setEndDate] = useState("");
  const [status, setStatus] = useState("");
  const [type, setType] = useState("");
  const [tags, setTags] = useState("");
  const [projects, setProjects] = useState([]);
  const [filteredProjects, setFilteredProjects] = useState(projects);

  const handleFilter = () => {
    let filtered = projects;
    if (endDate) {
      filtered = filtered.filter(
        (project) => new Date(project.endDate) <= new Date(endDate)
      );
    }
    if (status) {
      filtered = filtered.filter((project) => project.status === status);
    }
    if (type) {
      filtered = filtered.filter((project) => project.type === type);
    }
    if (tags) {
      filtered = filtered.filter((project) => project.tags.includes(tags));
    }

    setFilteredProjects(filtered);
  };

  useEffect(() => {
    const fetch = async () => {
      const response = await axios.get(GET_ALL_PROJECTS_ROUTE, {
        withCredentials: true,
      });
      console.log(response.data);
      
      setProjects(response.data);
      setFilteredProjects(response.data);
    };
    fetch();
  }, []);

  return (
    <div className="content3">
      <Navbar />
      <div className="filterBar">
        <TextField
          label="End Date"
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          InputLabelProps={{
            shrink: true,
          }}
        />
        <FormControl>
          <InputLabel>Status</InputLabel>
          <Select value={status} onChange={(e) => setStatus(e.target.value)}>
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value="Active">Active</MenuItem>
            <MenuItem value="Completed">Completed</MenuItem>
            <MenuItem value="Closed">Closed</MenuItem>
          </Select>
        </FormControl>
        <FormControl>
          <InputLabel>Type</InputLabel>
          <Select value={type} onChange={(e) => setType(e.target.value)}>
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value="Tree">Tree</MenuItem>
            <MenuItem value="Land">Land</MenuItem>
            <MenuItem value="Water">Water</MenuItem>
          </Select>
        </FormControl>
        <TextField
          label="Tags"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
        />
        <Button variant="contained" color="primary" onClick={handleFilter}>
          Search
        </Button>
      </div>
      <div className="projectList">
        {filteredProjects.map((project, index) => (
          <div
            key={project.title}
            className="projectItem"
            style={{ backgroundColor: `hsl(${index * 30}, 70%, 80%)` }}
            onClick={()=>navigate(`/project/${project._id}`)}
          >
            <div className="gridContainer">
              <div className="gridItem projectTitle">{project.title}</div>
              <div className="gridItem projectDescription">
                {project.description}
              </div>
              <div className="gridItem projectStatus"> {project.status}</div>
              <div className="gridItem projectType">{project.type}</div>
              <div className="gridItem projectEndDate">
                {project.endDate.split("T")[0]}
              </div>
              <div className="gridItem projectLocation">{project.location}</div>
              <div className="gridItem projectTags">
                {project.tags.join(", ")}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FilterBar;
