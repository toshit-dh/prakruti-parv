const multer = require("multer");
const cloudinary = require("cloudinary").v2;
const { CloudinaryStorage } = require("multer-storage-cloudinary");

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});
const profileStorage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: "user-profiles",
    public_id: (req, file) => req.user.userId,
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

const userUploadStorage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: "user-uploads",
    allowed_formats: [
      "jpg",
      "jpeg",
      "png",
      "webp", 
      "mp4",
      "avi",
      "mov", 
      "mp3",
      "wav",
      "ogg", 
    ],
    public_id: (req, file) => {
      const userId = req.user.userId; 
      const category = req.body.category;
      if (!category || (category !== "species" && category !== "poaching")) {
        throw new Error(
          "Invalid category. It must be either 'species' or 'poaching'."
        );
      }
      req.body.mediaType = file.mimetype.split("/")[0];
      return `${category}/${userId}`; 
    },
  },
});

const uploadProfile = (req, res, next) => {
  const upload = multer({
    storage: profileStorage,
    limits: {
      fileSize: 20 * 1024 * 1024, 
    },
  }).single('image'); 

  upload(req, res, (err) => {
    if (err) {
      console.error('Multer Error:', err);
      if (err instanceof multer.MulterError) {
        return res.status(400).send(err.message); 
      } else {
        return res.status(500).send('An unknown error occurred during the upload.');
      }
    }
    if (!req.file) {
      return res.status(400).send('No file uploaded.'); 
    }
    next();
  });
};

const uploadUserUploads = (req, res, next) => {
  const upload = multer({
    storage: userUploadStorage,
    limits: {
      fileSize: 20 * 1024 * 1024,
    },
  }).single("media");

  upload(req, res, (err) => {
    if (err) {
      console.error("Multer Error:", err);
      if (err instanceof multer.MulterError) {
        return res.status(400).send(err.message);
      } else {
        return res
          .status(500)
          .send("An unknown error occurred during the upload.");
      }
    }
    if (!req.file) {
      console.log("No file uploaded.");
      
      return res.status(400).send("No file uploaded.");
    }
    req.body.mediaUrl = req.file.path;
    next();
  });
}
const uploadProject = multer({ storage: projectStorage })
module.exports = { uploadProfile, uploadProject, uploadUserUploads};
