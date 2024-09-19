const router = require('express').Router();
const {
  createProject,
  getAllProjects,
  getProjectById,
  updateProject,
  deleteProject,
  addMediaToProject,
  donateToProject
} = require('../controllers/ProjectController');
const {upload} = require('../middlewares/FileMiddleware')
const { verifyToken } = require('../middlewares/UserMiddleware');

router.post('/projects', verifyToken, createProject);
router.get('/projects',verifyToken,getAllProjects);
router.get('/projects/:id',verifyToken,getProjectById);
router.put('/projects/:id', verifyToken, updateProject);
router.delete('/projects/:id', verifyToken, deleteProject);
router.post('/projects/:id/media', verifyToken, upload.array('media',25),addMediaToProject);
router.post('/projects/:id/donate', verifyToken, donateToProject);

module.exports = router;
