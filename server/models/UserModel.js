const mongoose = require("mongoose");
const { Schema } = mongoose;
const crypto = require("crypto");
const { type } = require("os");

const userSchema = new Schema({
  username: {
    type: String,
    required: true,
    unique: true,
  },
  name: {
    type: String,
    required: false,
    unique: false,
  },
  email: {
    type: String,
    required: true,
    unique: true,
  },
  isVerified: {
    type: Boolean,
    default: false,
  },
  password: {
    type: String,
    required: true,
  },
  role: {
    type: String,
    enum: ["user", "admin", "conservationist", "photographer", "learner"],
    default: "user",
  },
  badges: {
    type: Number,
    default: 0,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
  donations: [
    {
      amount: Number,
      date: Date,
      project: {
        type: Schema.Types.ObjectId,
        ref: "Project",
      },
    },
  ],
  favorites: [
    {
      type: Schema.Types.ObjectId,
      ref: "Species",
    },
  ],
  profile: {
    type: {
      name: { type: String, default: "" },
      bio: { type: String, default: "" },
      avatarUrl: { type: String, default: "" },
      contactEmail: { type: String, default: "" },
    },
    default: {},
  },
  friends: [
    {
      type: Schema.Types.ObjectId,
      ref: "User",
    },
  ],
  friendRequests: [
    {
      from: {
        type: Schema.Types.ObjectId,
        ref: "User",
      },
    },
  ],
  verificationToken: String, // Added field
  verificationTokenExpires: Date, // Added field
});

userSchema.pre("save", function (next) {
  this.updatedAt = Date.now();
  next();
});

userSchema.methods.generateVerificationToken = function () {
  const token = crypto.randomBytes(32).toString("hex");
  this.verificationToken = token;
  this.verificationTokenExpires = Date.now() + 3600000; // 1 hour
  return token;
};

module.exports = mongoose.model("User", userSchema);
