import React from "react";
import "./ReportTemplate.css";
import stampImage from "../../assets/prakruti-parv-stamp.png";
import logo from "../../assets/logo-no-background.png";
import reportbg from "../../assets/reportbg.png";
const ReportTemplate = React.forwardRef((props, ref) => {
  const { projectData } = props;
  if (projectData == null) return null;

  return (
    <div className="fund-report-template" id="report-template" ref={ref}>
      <div className="fund-report-header">
        <img src={logo} alt="App Logo" className="fund-report-logo" />
        <h1 className="fund-report-title">FundRaising Report</h1>
      </div>
      <div className="fund-report-banner-container">
        <img src={reportbg} alt="Banner" className="fund-report-banner" />
      </div>
      <h2 className="fund-report-organization-name">
        {projectData.organizationName}
      </h2>
      <p className="fund-report-project-description">
        {projectData.description}
      </p>
      <div className="fund-report-steps-section">
        <h3 className="fund-report-steps-title">Project Steps</h3>
        <ol className="fund-report-steps-list">
          {projectData.steps.map((step, index) => (
            <li key={index} className="fund-report-step-item">
              {step.description}
            </li>
          ))}
        </ol>
      </div>
      <div className="fund-report-contact-info">
        <div className="fund-report-target-amount">
          <p>
            <strong>Total Amount Targeted:</strong>
          </p>
          <h2 className="fund-report-amount">{projectData.goalAmount}</h2>
        </div>
        <div className="bottom-container">
          <div className="contact">
            <p className="fund-report-email">
              <strong>Email:</strong> {projectData.contactEmail}
            </p>
            <p className="fund-report-phone">
              <strong>Phone:</strong> {projectData.contactPhoneNumber}
            </p>
          </div>
          <div className="fund-report-stamp-container">
            <img
              src={stampImage}
              alt="Prakruti Parv Stamp"
              className="fund-report-stamp"
            />
          </div>
        </div>
      </div>
    </div>
  );
});

export default ReportTemplate;
