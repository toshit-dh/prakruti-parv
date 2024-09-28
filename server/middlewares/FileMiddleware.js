const multer = require("multer");
const cloudinary = require('cloudinary').v2
const {CloudinaryStorage} = require("multer-storage-cloudinary");

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.envCLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,

})

const profileStorage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: "user-profiles",
    allowed_formats: ["jpg", "png", "jpeg"],
  },
});

const projectStorage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: "user-projects",
    allowed_formats: ["jpg", "png", "jpeg", "mp4", "avi", "mov"], 
  },
});

const uploadProfile = multer({ storage: profileStorage });
const uploadProject = multer({storage: projectStorage})

module.exports = { uploadProfile,uploadProject};
