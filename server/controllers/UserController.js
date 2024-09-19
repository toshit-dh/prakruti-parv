const User = require('../models/UserModel');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken')
const {sendVerificationEmail} = require('../utils/SendEmail')
const path = require('path')
const APIResponse = require('../utils/APIResponse')

const JWT_SECRET = process.env.JWT_SECRET 


exports.signup = async (req, res) => {
    const { username,email, password } = req.body;
  
    try {
      const existingUser = await User.findOne({ username });
      if (existingUser) return res.status(400).json(new APIResponse(null, 'Username already exists').toJson());
      
      const hashedPassword = await bcrypt.hash(password, 12);
  
      const newUser = new User({
        username,
        email,
        password: hashedPassword,
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