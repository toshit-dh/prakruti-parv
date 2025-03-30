const router = require('express').Router()
const {upload,getAll,getForUser,reduceCurrency,fetchCurrency} = require('../controllers/UserUploadController')
const {verifyToken} = require('../middlewares/UserMiddleware')
const {uploadUserUploads} = require('../middlewares/FileMiddleware')
router.post('/',verifyToken,uploadUserUploads,upload);
router.get('/',verifyToken,getAll);
router.get('/user',verifyToken,getForUser);
router.get('/reduce',verifyToken,reduceCurrency)
router.get('/currency',verifyToken,fetchCurrency)
module.exports = router;