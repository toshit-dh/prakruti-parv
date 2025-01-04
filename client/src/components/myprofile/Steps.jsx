import React from "react";
import {
  VerticalTimeline,
  VerticalTimelineElement,
} from "react-vertical-timeline-component";
import "react-vertical-timeline-component/style.min.css";
import {
  FaCircle,
  FaPlay,
  FaSpinner,
  FaHourglassHalf,
  FaCheckCircle,
  FaTimesCircle,
  FaPauseCircle,
  FaQuestionCircle,
  FaClipboardList,
  FaClipboardCheck,
} from "react-icons/fa";
import "./Steps.css";
import image from "../../assets/banner-1.jpg";
export default function Steps() {
  const stepIcons = [
    <FaCircle />,
    <FaPlay />,
    <FaSpinner />,
    <FaHourglassHalf />,
    <FaCheckCircle />,
    <FaTimesCircle />,
    <FaPauseCircle />,
    <FaQuestionCircle />,
    <FaClipboardList />,
    <FaClipboardCheck />,
  ];
const steps = [
  {
    status: "Completed",
    date: "2025-01-01",
    description: "Step 1: Initialization completed",
  },
  {
    status: "Completed",
    date: "2025-01-02",
    description: "Step 2: Configuration finalized",
  },
  {
    status: "Active",
    date: "2025-01-03",
    description: "Step 3: Currently in progress",
  },
  {
    status: "Pending",
    date: "2025-01-04",
    description: "Step 4: Waiting for approval",
  },
  {
    status: "Pending",
    date: "2025-01-05",
    description: "Step 5: Pending review",
  },
  {
    status: "Pending",
    date: "2025-01-06",
    description: "Step 6: Pending resources allocation",
  },
  {
    status: "Pending",
    date: "2025-01-07",
    description: "Step 7: Pending team assignments",
  },
  {
    status: "Pending",
    date: "2025-01-08",
    description: "Step 8: Pending task dependencies",
  },
  {
    status: "Pending",
    date: "2025-01-09",
    description: "Step 9: Pending final review",
  },
  {
    status: "Pending",
    date: "2025-01-10",
    description: "Step 10: Pending completion",
  },
];


  return (
    <div className="steps">
      <h1>Steps Timeline</h1>
      <VerticalTimeline
        animate={true}
        className="vertical-timeline-custom-line"
        lineColor="linear-gradient(to bottom, #A5D6A7, #81C784, #66BB6A)"
      >
        {steps.map((step, index) => (
          <VerticalTimelineElement
            className="vertical-timeline-element--work"
            contentStyle={{
              border: "3px solid #4CAF50",
              color: "#fff",
            }}
            shadowSize="large"
            contentArrowStyle={{ borderRight: "7px solid  #4CAF50" }}
            dateClassName="date"
            date={step.date}
            iconStyle={
              step.status == "Active"
                ? {
                    background: "#228B22",
                    color: "#fff",
                    animation: "colorChange 0.5s infinite",
                  }
                : {
                    background: "#4CAF50",
                    color: "#fff",
                  }
            }
            icon={stepIcons[index % stepIcons.length]}
            iconClassName={step.status}
          >
            <div className="step-content">
              <h3 className="vertical-timeline-element-title">
                <span
                  style={{
                    color: "green",
                  }}
                >
                  Step {index + 1} :{" "}
                </span>
                <span>{step.status}</span>
              </h3>
              <p className="vertical-timeline-element-description">
                {step.description}
              </p>
              {step.status != "Pending" && <img src={image} />}
            </div>
          </VerticalTimelineElement>
        ))}
      </VerticalTimeline>
    </div>
  );
}
