const UserUpload = require("../models/UserUploadModel"); // Assuming your UserUpload model is located in the models folder
const User = require("../models/UserModel"); // Assuming your User model is located in the models folder
exports.upload = async (req, res) => {
  try {
    const userId = req.user.userId; 
    const { category, species, title, description, mediaUrl, mediaType } =
      req.body;
      console.log(req.body);
      
    if (!category || !title || !description || !mediaUrl || !mediaType) {
      return res.status(400).json({ error: "All fields are required" });
    }
    const newUpload = new UserUpload({
      userId,
      category,
      species,
      title,
      description,
      mediaUrl,
      mediaType,
    });
    await newUpload.save();
    return res
      .status(201)
      .json({ message: "Upload successful", data: newUpload });
  } catch (error) {
    console.error(error);
    res.status(400).json({ error: error.message });
  }
};

// Get all user uploads (Admin/Global view)
exports.getAll = async (req, res) => {
  try {
    const uploads = await UserUpload.find()
      .populate("userId", "username email") // Populate user details if needed
      .sort({ createdAt: -1 }); // Sort by most recent uploads first

    if (!uploads) {
      return res.status(404).json({ error: "No uploads found" });
    }

    return res
      .status(200)
      .json({ message: "Uploads fetched successfully", data: uploads });
  } catch (error) {
    console.error(error);
    res.status(400).json({ error: error.message });
  }
};

// Get all uploads for a specific user
exports.getForUser = async (req, res) => {
  try {
    const userId = req.user.userId; // Get the user ID from the authenticated user

    // Find all uploads by the specific user
    const uploads = await UserUpload.find({ userId })
      .populate("userId", "username email") // Optionally populate user details
      .sort({ createdAt: -1 }); // Sort by most recent uploads first

    if (!uploads || uploads.length === 0) {
      return res.status(404).json({ error: "No uploads found for this user" });
    }

    return res
      .status(200)
      .json({ message: "User uploads fetched successfully", data: uploads });
  } catch (error) {
    console.error(error);
    res.status(400).json({ error: error.message });
  }
};

exports.reduceCurrency = async (req, res) => {
  try {
    const userId = req.user.userId; 
    const { amount } = req.query; 
    const user = await User.findByIdAndUpdate(
      userId,
      { $inc: { currency: -amount } },
      { new: true }
    );

    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    return res.status(200).json({
      message: "Currency reduced successfully",
      data: user,
    });
  } catch (error) {
    console.error(error);
    res.status(400).json({ error: error.message });
  }
}