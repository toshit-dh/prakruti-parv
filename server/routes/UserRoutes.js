const router = require('express').Router()
const {signup,login,verifyEmail,verifyToken,getProfile,editProfile} = require('../controllers/UserController')
const {verifyToken: verifyTokenM} = require('../middlewares/UserMiddleware')
const {uploadProfile} = require('../middlewares/FileMiddleware')
router.post('/signup', signup);
router.post('/login', login);
router.get('/verify-email',verifyEmail);
router.get('/verify-token',verifyToken);
router.get('/',verifyTokenM,getProfile)
router.post('/',verifyTokenM,uploadProfile,editProfile)
module.exports = router;