import React from 'react';
import './MyProjects.css';

const MyProjects = () => {
  const projects = [
    {
      organizationName: "VanRaksha-Foundation",
      bannerImage: "https://res.cloudinary.com/djt5vw5aa/image/upload/v1734439386/user-projects/env5mctnwrovstw7ngrf.png",
      description: "Our mission is to protect and conserve the majestic tigers in Ranthambore National Park. This project aims to enhance anti-poaching measures, promote community awareness, and ensure sustainable habitat management to safeguard the tiger population for future generations.",
      startDate: new Date(1728663747161).toLocaleDateString(),
      endDate: new Date(1750118400000).toLocaleDateString(),
    },
    {
      organizationName: "Wildlife Preservation Trust",
      bannerImage: "https://res.cloudinary.com/djt5vw5aa/image/upload/v1734439386/user-projects/env5mctnwrovstw7ngrf.png",
      description: "Dedicated to preserving wildlife habitats and preventing species extinction worldwide through research, advocacy, and sustainable development.",
      startDate: new Date(1738663747161).toLocaleDateString(),
      endDate: new Date(1760118400000).toLocaleDateString(),
    },
    {
      organizationName: "Ocean Conservation Initiative",
      bannerImage: "https://res.cloudinary.com/djt5vw5aa/image/upload/v1734439386/user-projects/env5mctnwrovstw7ngrf.png",
      description: "Working towards ocean cleanup and protecting marine biodiversity through collaborative global initiatives and sustainable practices.",
      startDate: new Date(1748663747161).toLocaleDateString(),
      endDate: new Date(1770118400000).toLocaleDateString(),
    },
    {
      organizationName: "Forest Recovery Foundation",
      bannerImage: "https://res.cloudinary.com/djt5vw5aa/image/upload/v1734439386/user-projects/env5mctnwrovstw7ngrf.png",
      description: "Restoring endangered forests and ensuring sustainable forest management through reforestation projects and local community involvement.",
      startDate: new Date(1758663747161).toLocaleDateString(),
      endDate: new Date(1780118400000).toLocaleDateString(),
    },
    {
      organizationName: "Clean Earth Project",
      bannerImage: "https://res.cloudinary.com/djt5vw5aa/image/upload/v1734439386/user-projects/env5mctnwrovstw7ngrf.png",
      description: "Fighting pollution and promoting clean energy solutionsyviiiiiiiiiiiiiiiiiiiiiiiiiiiillllllllllllllllllllllllllllllb  dr to create a greener, more sustainable future for all.",
      startDate: new Date(1768663747161).toLocaleDateString(),
      endDate: new Date(1790118400000).toLocaleDateString(),
    },
  ];

  return (
    <div className="projects-container">
      {projects.map((project, index) => (
        <div key={index} className="project-container">
          <div className="project-header">
            <h2>{project.organizationName}</h2>
          </div>
          
          <div className="banner-image">
            <img src={project.bannerImage} alt="Banner" />
          </div>

          <div className="project-description">
            <h3>Description:</h3>
            <p>{project.description}</p>
          </div>

          <div className="project-dates">
            <div>
              <strong>Start Date:</strong> {project.startDate}
            </div>
            <div>
              <strong>End Date:</strong> {project.endDate}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default MyProjects;
