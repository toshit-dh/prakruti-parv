/* eslint-disable no-unused-vars */
import React from 'react';
import './ReportTemplate.css'; 
import stampImage from '../../assets/prakruti-parv-stamp.png';
import logo from '../../assets/logo-no-background.png';

const ReportTemplate = () => {
    const projectData = {
        organizationName: "Prakruti Parv Foundation",
        projectDescription: "Our mission is to conserve wildlife and protect endangered species through awareness and action.",
        steps: [
            "Research and identify endangered species.",
            "Create awareness programs.",
            "Collaborate with local communities.",
            "Implement conservation actions.",
            "Monitor and evaluate progress."
        ],
        totalAmount: "₹40,000",
        email: "contact@prakrutiparv.org",
        phone: "+91 98765 43210"
    };

    return (
        <div className="fund-report-template" id="report-template">
            <div className="fund-report-header">
                <img src={logo} alt="App Logo" className="fund-report-logo" />
                <h1 className="fund-report-title">FundRaising Report</h1>
            </div>

            <div className="fund-report-banner-container">
                <img 
                    src="https://static.vecteezy.com/system/resources/previews/011/844/124/non_2x/abstract-banner-with-wildlife-design-concept-vector.jpg" 
                    alt="Banner" 
                    className="fund-report-banner" 
                />
            </div>

            <h2 className="fund-report-organization-name">{projectData.organizationName}</h2>

            <p className="fund-report-project-description">{projectData.projectDescription}</p>

            <div className="fund-report-steps-section">
                <h3 className="fund-report-steps-title">Project Steps</h3>
                <ol className="fund-report-steps-list">
                    {projectData.steps.map((step, index) => (
                        <li key={index} className="fund-report-step-item">{step}</li>
                    ))}
                </ol>
            </div>

            <div className="fund-report-contact-info">
                <div className="fund-report-target-amount">
                    <p><strong>Total Amount Targeted:</strong></p>
                    <h2 className="fund-report-amount">{projectData.totalAmount}</h2>
                </div>
                <p className="fund-report-email"><strong>Email:</strong> {projectData.email}</p>
                <p className="fund-report-phone"><strong>Phone:</strong> {projectData.phone}</p>
            </div>

            <div className="fund-report-stamp-container">
                <img src={stampImage} alt="Prakruti Parv Stamp" className="fund-report-stamp" />
            </div>
        </div>
    );
};

export default ReportTemplate;
