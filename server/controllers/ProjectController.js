const Project = require('../models/ProjectModel'); // Adjust the path as needed
const User = require('../models/UserModel'); // Assuming User model is needed for contributor details
const path = require('path');
const fs = require('fs');

// Create a new project
exports.createProject = async (req, res) => {
  try {
    const { title, description, goalAmount, endDate, type, location } = req.body; // Added location
    const creator = req.user.userId; 
    console.log(req.body,creator);
    
    const newProject = new Project({
      title,
      description,
      goalAmount,
      endDate,
      type,
      location, // Added location
      creator,
    });

    await newProject.save();
    res.status(201).json({ message: 'Project created successfully', project: newProject });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Get all projects
exports.getAllProjects = async (req, res) => {
  try {
    const projects = await Project.find().populate('creator', 'username name').exec();
    res.status(200).json(projects);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Get a single project by ID
exports.getProjectById = async (req, res) => {
  try {
    const project = await Project.findById(req.params.id).populate('creator', 'username name').exec();
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }
    res.status(200).json(project);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Update a project by ID
exports.updateProject = async (req, res) => {
  const projectId = req.params.id;
  const { title, description, mediaPaths, tags, updates, milestones } = req.body; // Added tags, updates, milestones

  try {
    const project = await Project.findById(projectId);
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }

    // Update project details
    if (title) project.title = title;
    if (description) project.description = description;
    if (tags) project.tags = tags; // Update tags
    if (updates) project.updates = updates; // Update updates
    if (milestones) project.milestones = milestones; // Update milestones

    // Handle file deletion
    if (mediaPaths && Array.isArray(mediaPaths)) {
      const filesToDelete = mediaPaths.filter(filePath => project.media.includes(filePath));
      filesToDelete.forEach(filePath => {
        fs.unlink(path.join('data/projects/', path.basename(filePath)), (err) => {
          if (err) console.error('Failed to delete file:', filePath);
        });
        // Remove file path from the project media array
        project.media = project.media.filter(mediaPath => mediaPath !== filePath);
      });
    }

    await project.save();
    res.status(200).json({ message: 'Project updated successfully', project });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
};

// Delete a project by ID
exports.deleteProject = async (req, res) => {
  try {
    const deletedProject = await Project.findByIdAndDelete(req.params.id).exec();
    if (!deletedProject) {
      return res.status(404).json({ message: 'Project not found' });
    }
    res.status(200).json({ message: 'Project deleted successfully' });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Add images or videos to a project
exports.addImages = async (req, res) => {
  try {
    const project = await Project.findById(req.params.id).exec();
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }

    const { images } = req.body; // Expecting an array of image/video objects

    if (project.images.length + images.length > 25) {
      return res.status(400).json({ message: 'Cannot add more than 25 images or videos' });
    }

    project.images.push(...images);
    await project.save();
    res.status(200).json({ message: 'Images added successfully', project });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Add a donation to a project
exports.addDonation = async (req, res) => {
  try {
    const { amount } = req.body;
    const project = await Project.findById(req.params.id).exec();

    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }

    if (amount <= 0) {
      return res.status(400).json({ message: 'Donation amount must be positive' });
    }

    project.currentAmount += amount;
    await project.save();

    const user = await User.findById(req.user._id).exec();
    if (user) {
      user.donations.push({ amount, date: new Date(), project: project._id });
      await user.save();
    }

    res.status(200).json({ message: 'Donation added successfully', project });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Get projects by type (Tree, Land, Water)
exports.getProjectsByType = async (req, res) => {
  try {
    const { type } = req.params;
    const projects = await Project.find({ type }).populate('creator', 'username name').exec();
    res.status(200).json(projects);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Get projects by status
exports.getProjectsByStatus = async (req, res) => {
  try {
    const { status } = req.params;
    const projects = await Project.find({ status }).populate('creator', 'username name').exec();
    res.status(200).json(projects);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Add media to a project
exports.addMediaToProject = async (req, res) => {
  const projectId = req.params.id;
  const files = req.files;

  try {
    if (!files || files.length === 0) {
      return res.status(400).json({ message: 'No files uploaded' });
    }

    const filePaths = files.map(file => `data/projects/${file.filename}`);

    const project = await Project.findById(projectId);
    if (!project) {
      return res.status(404).json({ message: 'Project not found' });
    }

    if (project.media.length + files.length > 25) {
      // Delete uploaded files if the limit is exceeded
      files.forEach(file => {
        fs.unlink(path.join('data/projects/', file.filename), (err) => {
          if (err) console.error('Failed to delete file:', file.filename);
        });
      });
      return res.status(400).json({ message: 'Cannot add more than 25 media files' });
    }

    project.media.push(...filePaths);
    await project.save();

    res.status(200).json({ message: 'Media added successfully', files: filePaths });
  } catch (error) {
    if (files) {
      files.forEach(file => {
        fs.unlink(path.join('data/projects/', file.filename), (err) => {
          if (err) console.error('Failed to delete file:', file.filename);
        });
      });
    }
    res.status(500).json({ error: 'Internal server error' });
  }
};
