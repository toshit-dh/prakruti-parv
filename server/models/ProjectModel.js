const mongoose = require("mongoose");

const projectSchema = new mongoose.Schema(
  {
    organizationName: {
      type: String,
      required: true,
      trim: true,
    },
    organization_id: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User", 
      required: true,
    },
    contactPhoneNumber: {
      type: String,
      trim: true,
      match: [/^\+?[1-9]\d{1,14}$/, "Please fill a valid phone number"],
    },
    contactEmail: {
      type: String,
      trim: true,
      lowercase: true,
    },
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
    bannerImage: {
      type: String, 
      required: true,
      trim: true,
      match: [
        /^https?:\/\/.+\.(jpg|jpeg|png|gif)$/i,
        "Please provide a valid image URL",
      ],
    },
    goalAmount: {
      type: Number,
      required: true,
      min: [0, "Goal amount must be positive"],
    },
    currentAmount: {
      type: Number,
      default: 0,
      min: [0, "Current amount must be positive"],
    },
    startDate: {
      type: Date,
      default: Date.now,
    },
    endDate: {
      type: Date,
      required: true,
      validate: {
        validator: function (value) {
          return value > this.startDate;
        },
        message: "End date must be after the start date",
      },
    },
    duration: {
      type: Number, 
      required: true,
      min: [1, "Duration must be at least 1 day"],
    },
    location: {
        type: String,
        required:true,
    },
    steps: {
      type: [
        {
          description: {
            type: String,
            required: true,
            trim: true,
          },
          status: {
            type: String,
            enum: ["Not Started", "In Progress", "Completed"],
            default: "Not Started",
          },
          photo: {
            type: String, 
            trim: true,
          },
        },
      ],
      default: [],
    },
    contributors: {
      type: [
        {
          contributor_id: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User",
            required: true,
          },
          amount: {
            type: Number,
            required: true,
            min: [0, "Contribution amount must be positive"],
          },
        },
      ],
      default: [],
    },

    status: {
      type: String,
      enum: ["Active", "Completed", "Closed"],
      default: "Active",
    },
  },
  { timestamps: true }
);

const Project = mongoose.model("Project", projectSchema);

module.exports = Project;
