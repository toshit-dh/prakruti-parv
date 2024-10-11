const router = require('express').Router();
const {
  createProject,
  getAllProjects,
  getProjectById,
  updateProject,
  deleteProject,
  addMediaToProject,
  donateToProject,
  getProjectsByStatus,
  getProjectsByOrganization
} = require('../controllers/ProjectController');
const { uploadProject } = require('../middlewares/FileMiddleware');
const { verifyToken } = require('../middlewares/UserMiddleware');


router.post('/', verifyToken, uploadProject.single('bannerImage'), createProject);
router.get('/', verifyToken, getAllProjects);
router.get('/organization/:organizationId', verifyToken, getProjectsByOrganization);
router.get('/:id', verifyToken, getProjectById);
router.put('/:id', verifyToken, uploadProject.single('bannerImage'), updateProject);
router.delete('/:id', verifyToken, deleteProject);
router.post('/:id/media', verifyToken, uploadProject.single('media'), addMediaToProject);
router.post('/:id/donate', verifyToken, donateToProject);
router.get('/status/:status', verifyToken, getProjectsByStatus);

module.exports = router;
