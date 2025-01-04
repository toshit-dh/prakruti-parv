const User = require('../models/UserModel');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken')
const {sendVerificationEmail} = require('../utils/SendEmail')
const path = require('path')
const APIResponse = require('../utils/APIResponse')
const cloudinary = require('cloudinary').v2;
const JWT_SECRET = process.env.JWT_SECRET 


exports.signup = async (req, res) => {
    const { username,email, password,role } = req.body;
  
    try {
      const existingUser = await User.findOne({ username });
      if (existingUser) return res.status(400).json(new APIResponse(null, 'Username already exists').toJson());
      
      const hashedPassword = await bcrypt.hash(password, 12);
  
      const newUser = new User({
        username,
        email,
        password: hashedPassword,
        role,
        isVerified: false
      });
  
      const verificationToken = newUser.generateVerificationToken();
      await newUser.save();
  
      await sendVerificationEmail(newUser, verificationToken);
  
      res.status(201).json(new APIResponse(null, 'User registered successfully. Please check your email to verify your account.').toJson());
    } catch (error) {
      console.error('Signup error:', error);
      res.status(500).json(new APIResponse(null, error.message).toJson())
    }
  };
  

exports.login = async (req, res) => {
    const { username, password } = req.body;
    try {
      const user = await User.findOne({ username });
      if (!user) return res.status(401).json(new APIResponse(null, 'Invalid username or password').toJson());

      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) return res.status(402).json(new APIResponse(null, 'Invalid username or password').toJson());

      if (!user.isVerified) return res.status(403).json(new APIResponse(null, 'Please verify your email before logging in').toJson());
      
  
      const token = jwt.sign(
        { userId: user._id, role: user.role },
        JWT_SECRET,
        { expiresIn: '1h' } 
      );
  
      res.cookie('token', token, {
        httpOnly: true, 
        secure: process.env.NODE_ENV === 'production', 
      });
  
      res.status(200).json(new APIResponse(null, 'Login successful').toJson());
    } catch (error) {
      console.error('Login error:', error);
      res.status(500).json(new APIResponse(null, 'Internal server error').toJson());
    }
  };

  exports.logout=async(req,res) => {
    await User.findByIdAndUpdate(req.user?._id,{$unset:{refreshToken:1}},{new:true}) 
    return res.status(200).clearCookie('token').json(new APIResponse(null, 'Logout successful').toJson());
    
  }

  exports.verifyEmail = async (req, res) => {
    const { token } = req.query;
  
    try {
      const user = await User.findOne({
        verificationToken: token,
        verificationTokenExpires: { $gt: Date.now() }
      });
  
      if (!user) return res.status(400).json(new APIResponse(null, 'Invalid or expired token').toJson());
  
      user.isVerified = true;
      user.verificationToken = undefined;
      user.verificationTokenExpires = undefined;
      await user.save();
  
      res.status(200).sendFile(path.join(__dirname, '../views/verificationSuccess.html'));
  } catch (error) {
    console.error('Verification error:', error);
    res.status(500).sendFile(path.join(__dirname, '../views/verificationError.html'));
  }
  };
  

  exports.verifyToken = (req, res,next) => {
    try {
      const token = req.cookies.token

      if (!token) return res.status(401).json(new APIResponse(null, 'No token provided').toJson());

      jwt.verify(token, JWT_SECRET, (err, decoded) => {
        if (err) return res.status(401).json(new APIResponse(null, 'Invalid token').toJson());
        res.status(200).json(new APIResponse(decoded, 'Token verified successfully').toJson());
      });
    } catch (error) {
      console.error('Token verification error:', error);
      res.status(500).json(new APIResponse(null, 'Internal server error').toJson());
    }
  }

  exports.getProfile= async(req,res) => {
    try {
      const id = req.user.userId
      console.log(id);
      
      console.log(await User.findById(id));
      
      res.status(200).json(new APIResponse(await User.findById(id)).toJson())
    } catch (e) {
      console.error('Edit Profile Error:', e);
      res.status(500).json(new APIResponse(null, 'Internal server error').toJson());
    }
  }

  exports.editProfile = async (req, res) => {
    let uploadedFileId;
    try {
      const id = req.user.userId
      const profile = req.body;
      if (!profile.name || !profile.role || !profile.bio) return res.status(400).json(new APIResponse(null, 'First name and last name are required').toJson());
      const user = await User.findById(id);
      if (!user) return res.status(404).json(new APIResponse(null, 'User not found').toJson());
      user.profile = profile;
      const updatedUser = await user.save();
      res.status(200).json(new APIResponse(updatedUser, "Profile Updated Successfully"));
    } catch (e) {
      console.error('Edit Profile Error:', e);
      if (uploadedFileId) await cloudinary.uploader.destroy(uploadedFileId);
      res.status(500).json(new APIResponse(null, 'Internal server error').toJson());
    }
  };
  