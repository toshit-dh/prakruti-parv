const Project = require('../models/ProjectModel');
const User = require('../models/UserModel'); 
const path = require('path');
const fs = require('fs');

exports.createProject = async (req, res) => {
  try {
    const { 
      organizationName, 
      contactPhoneNumber, 
      contactEmail, 
      title, 
      description, 
      goalAmount, 
      endDate, 
      location, 
      duration, 
      steps 
    } = req.body; 

    const creator = req.user.userId; 

    if (!req.file) {
      return res.status(400).json({ error: 'Banner image is required' });
    }

    const bannerImageUrl = req.file.path; 

    let parsedSteps = [];
    if (steps) {
      try {
        parsedSteps = JSON.parse(steps);
        if (!Array.isArray(parsedSteps)) {
          return res.status(400).json({ error: 'Steps must be an array' });
        }
        parsedSteps.forEach((step, index) => {
          if (!step.description) {
            throw new Error(`Step ${index + 1} description is required`);
          }
          if (!["Not Started", "In Progress", "Completed"].includes(step.status)) {
            throw new Error(`Invalid status for step ${index + 1}`);
          }
        });
      } catch (err) {
        return res.status(400).json({ error: 'Invalid steps format: ' + err.message });
      }
    }

    const newProject = new Project({
      organizationName,
      organization_id: creator,
      contactPhoneNumber,
      contactEmail,
      title,
      description,
      bannerImage: bannerImageUrl,
      goalAmount,
      endDate,
      location,
      duration,
      steps: parsedSteps, 
      status: 'Active',
    });

    await newProject.save();
    res.status(200).json({ message: 'Project created successfully', project: newProject });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

exports.getAllProjects = async (req, res) => {
  try {
    const projects = await Project.find()
      .populate('organization_id', 'username ') 
      .exec();
    res.status(200).json(projects);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};


exports.getProjectById = async (req, res) => {
  try {
    const project = await Project.findById(req.params.id)
      .populate('organization_id', 'username ') 
      .exec();
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }
    res.status(200).json(project);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};


exports.updateProject = async (req, res) => {
  const projectId = req.params.id;
  const { 
    title, 
    description, 
    goalAmount, 
    endDate, 
    location, 
    duration, 
    status, 
    steps 
  } = req.body;

  try {
    const project = await Project.findById(projectId);
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }

    if (project.organization_id.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Unauthorized to update this project' });
    }

    if (title) project.title = title;
    if (description) project.description = description;
    if (goalAmount !== undefined) project.goalAmount = goalAmount;
    if (endDate) project.endDate = endDate;
    if (location) {
      project.location = location;
    }
    if (duration !== undefined) project.duration = duration;
    if (status) project.status = status;

    if (steps) {
      let parsedSteps = [];
      try {
        parsedSteps = JSON.parse(steps);
        if (!Array.isArray(parsedSteps)) {
          throw new Error('Steps must be an array');
        }
        parsedSteps.forEach((step, index) => {
          if (!step.description) {
            throw new Error(`Step ${index + 1} description is required`);
          }
          if (!["Not Started", "In Progress", "Completed"].includes(step.status)) {
            throw new Error(`Invalid status for step ${index + 1}`);
          }
        });
      } catch (err) {
        return res.status(400).json({ error: 'Invalid steps format: ' + err.message });
      }

      project.steps = parsedSteps;
    }

    if (req.file) {
      project.bannerImage = req.file.path;
    }

    await project.save();
    res.status(200).json({ message: 'Project updated successfully', project });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error: ' + error.message });
  }
};

exports.deleteProject = async (req, res) => {
  try {
    const project = await Project.findById(req.params.id).exec();
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }
    if (project.organization_id.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Unauthorized to delete this project' });
    }


    await Project.findByIdAndDelete(req.params.id).exec();
    res.status(200).json({ message: 'Project deleted successfully' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};


exports.addMediaToProject = async (req, res) => {
  const projectId = req.params.id;
  const { stepIndex } = req.body; 
  try {
    const project = await Project.findById(projectId);
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }

    if (project.organization_id.toString() !== req.user.userId) {
      return res.status(403).json({ message: 'Unauthorized to add media to this project' });
    }


    const index = parseInt(stepIndex, 10);
    if (isNaN(index) || index < 0 || index >= project.steps.length) {
      return res.status(400).json({ message: 'Invalid step index' });
    }

    if (!req.file) {
      return res.status(400).json({ message: 'No media file uploaded' });
    }

    if (project.steps[index].photo) {
      return res.status(400).json({ message: 'Step already has a photo. Use update instead.' });
    }

 
    const mediaUrl = req.file.path; 

    project.steps[index].photo = mediaUrl;

    await project.save();
    res.status(200).json({ message: 'Step photo added successfully', project });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error: ' + error.message });
  }
};


exports.donateToProject = async (req, res) => {
  try {
    const { amount } = req.body;
    const projectId = req.params.id;
    const userId = req.user.userId;

    const project = await Project.findById(projectId).exec();

    if (!project) {
      return res.status(404).json({ message: 'Project not found'});
    }

    if (amount <= 0) {
      return res.status(400).json({ message: 'Donation amount must be positive' });
    }

    project.currentAmount += amount;
    project.contributors.push({ contributor_id: userId, amount });

    await project.save();

    const user = await User.findById(userId).exec();
    if (user) {
      user.donations.push({ amount, date: new Date(), project: project._id });
      await user.save();
    }

    res.status(200).json({ message: 'Donation added successfully', project });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};


exports.getProjectsByStatus = async (req, res) => {
  try {
    const { status } = req.params;
    const projects = await Project.find({ status })
      .populate('organization_id', 'username name')
      .exec();
    res.status(200).json(projects);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

exports.getProjectsByOrganization = async (req, res) => {
  try {
    const organizationId = req.params.organizationId;
    const projects = await Project.find({ organization_id: organizationId })
      .populate('organization_id', 'username ')
      .exec();

    if (projects.length === 0) {
      return res.status(200).json({ projects });
    }
    res.status(200).json({projects});
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

