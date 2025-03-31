const jwt = require('jsonwebtoken')
const JWT_SECRET = process.env.JWT_SECRET 
const APIResponse = require('../utils/APIResponse')
exports.verifyToken = (req, res,next) => {
  try {
    
    const token = req.cookies.token

    if (!token) return res.status(401).json(new APIResponse(null, 'No token provided').toJson());

    jwt.verify(token, JWT_SECRET, (err, decoded) => {
      if (err) return res.status(401).json(new APIResponse(null, 'Invalid token').toJson());
      req.user = decoded
      console.log("Request user:", req.user);
      next()
    });
  } catch (error) {
    console.error('Token verification error:', error);
    res.status(500).json(new APIResponse(null, 'Internal server error').toJson());
  }
}