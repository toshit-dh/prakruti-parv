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

router.post('/', verifyToken, createProject);
router.get('/',verifyToken,getAllProjects);
router.get('/:id',verifyToken,getProjectById);
router.put('/:id', verifyToken, updateProject);
router.delete('/:id', verifyToken, deleteProject);
router.post('/:id/media', verifyToken, upload.array('media',25),addMediaToProject);
//router.post('/:id/donate', verifyToken, donateToProject);

module.exports = router;
