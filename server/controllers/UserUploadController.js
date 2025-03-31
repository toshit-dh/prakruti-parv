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
    const { userId } = req.user;
    if (!userId) return res.status(400).json({ error: "User ID is required" });

    const user = await User.findById(userId).lean();
    if (!user) return res.status(404).json({ error: "User not found" });

    const query = user.role === "admin" ? {} : { userId };
    const uploads = await UserUpload.find(query)
      .populate("userId", "username email")
      .sort({ createdAt: -1 });

    if (!uploads.length) return res.status(404).json({ error: "No uploads found" });
    
    res.status(200).json(
      { message: "Uploads fetched successfully",
         data: uploads,
         isAdmin: user.role === "admin"}
    );
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal server error" });
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

exports.fetchCurrency = async (req, res) => {
  try {
    const userId = req.user.userId; 
    const user = await User.findById(userId); 
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    return res.status(200).json({
      message: "Currency fetched successfully",
      data: user.currency,
    });
  } catch (error) {
    console.error(error);
    res.status(400).json({ error: error.message });
  }
}

exports.awardHonour = async (req, res) => {
  try{
    const userId = req.user.userId; 
    if (!userId) return res.status(400).json({ error: "User ID is required" });
    const user = await User.findById(userId);
    if (!user) return res.status(404).json({ error: "User not found" });
    if (user.role != "admin") return res.status(403).json({ error: "Only admin can award honour" });
    const {toUser,uploadId} = req.body;
    if (!uploadId) return res.status(400).json({ error: "Upload ID is required" });
    const upload = await UserUpload.findById(uploadId);
    if (!upload) return res.status(404).json({ error: "Upload not found" });
    if (!toUser) return res.status(400).json({ error: "User ID to award is required" });
    const userToAward = await User.findById(toUser);
    if (!userToAward) return res.status(404).json({ error: "User to award not found" });
    upload.isHonoured = true;
    await upload.save();
    userToAward.badges += 1;
    userToAward.currency += 10; 
    await userToAward.save();
    return res.status(200).json({
      message: "Honour awarded successfully"
    })
  }catch(error){
    console.error(error);
    res.status(400).json({ error: error.message });
  }
}