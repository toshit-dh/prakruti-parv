const mongoose = require("mongoose");

const userUploadSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User", // Assuming you have a User model
      required: true,
    },
    category: {
      type: String,
      enum: ["species", "poaching"], // The category options you mentioned
      required: true,
    },
    species: {
      type: String,
      required: false,
    },
    title: {
      type: String,
      required: true,
    },
    description: {
      type: String,
      required: true,
    },
    mediaUrl: {
      type: String, // URL of the uploaded media (image, video, or audio)
      required: true,
    },
    mediaType: {
      type: String,
      enum: ["image", "video", "audio"], // You can extend this if more types are supported
      required: true,
    },
    likes: {
      type: Number,
      default: 0, // Default to 0 if no likes
    },
    isHonoured: {
      type: Boolean,
      default: false, // Default to false if not set
    },
    postIconUrl: {
      type: String,
      default:
        "https://res.cloudinary.com/djt5vw5aa/image/upload/v1743327381/user-uploads/badges/gz8owa0p54cmkdvztgv2.png", // Default icon URL
    },
    ratnaUrl: {
      type: String,
      default:
        "https://res.cloudinary.com/djt5vw5aa/image/upload/v1743327381/user-uploads/badges/b9hwik8m0ygicbviy56u.png", // Default Ratna URL
    },
    createdAt: {
      type: Date,
      default: Date.now,
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("UserUpload", userUploadSchema);
