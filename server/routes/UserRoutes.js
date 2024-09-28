const router = require('express').Router()
const {signup,login,verifyEmail,verifyToken,getProfile,editProfile} = require('../controllers/UserController')
const {uploadProfile} = require('../middlewares/FileMiddleware')
router.post('/signup', signup);
router.post('/login', login);
router.get('/verify-email',verifyEmail);
router.get('/verify-token',verifyToken);
router.get('/',verifyToken,getProfile)
router.post('/',verifyToken,uploadProfile.single('image'),editProfile)
module.exports = router;