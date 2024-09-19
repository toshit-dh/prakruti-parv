const mongoose = require('mongoose');

const projectSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
    trim: true,
  },
  description: {
    type: String,
    required: true,
    trim: true,
  },
  goalAmount: {
    type: Number,
    required: true,
    min: [0, 'Goal amount must be positive'],
  },
  currentAmount: {
    type: Number,
    default: 0,
    min: [0, 'Current amount must be positive'],
  },
  startDate: {
    type: Date,
    default: Date.now,
  },
  endDate: {
    type: Date,
    required: true,
  },
  creator: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User', 
    required: true,
  },
  contributors: [{
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User', 
    },
    amount: {
      type: Number,
      min: [0, 'Contribution amount must be positive'],
    },
  }],
  status: {
    type: String,
    enum: ['Active', 'Completed', 'Closed'],
    default: 'Active',
  },
  type: {
    type: String,
    enum: ['Tree', 'Land', 'Water'],
    required: true,
  },
  images: [{
    url: {
      type: String,
      required: true,
    },
    type: {
      type: String,
      enum: ['image', 'video'],
      required: true,
    },
    description: {
      type: String,
      trim: true,
    },
  }],
  tags: [{
    type: String,
    trim: true,
  }],
  location: {
    type: {
      type: String,
      enum: ['Point'],
      default: 'Point',
    },
    coordinates: {
      type: [Number],
      index: '2dsphere',
    },
  },
  updates: [{
    date: {
      type: Date,
      default: Date.now,
    },
    message: {
      type: String,
      trim: true,
    },
  }],
  milestones: [{
    name: {
      type: String,
      trim: true,
    },
    description: {
      type: String,
      trim: true,
    },
    achievedDate: {
      type: Date,
    },
  }],
}, { timestamps: true });

// Limit the number of images/videos to 25
projectSchema.path('images').validate(function (images) {
  return images.length <= 25;
}, 'You can only add up to 25 images or videos.');

const Project = mongoose.model('Project', projectSchema);

module.exports = Project;
