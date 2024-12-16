// Navbar.js
import React, { useState } from "react";
import { FaUserGraduate, FaDonate, FaTimes, FaBars,FaGlobe } from "react-icons/fa";
import { NavLink, useNavigate } from "react-router-dom";
import logo from "../../assets/logo.png";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu"; // Use MUI's MenuIcon
import CustomDrawer from "./CustomDrawer"; // Import the new Drawer component
import "./Navbar.css";

const Navbar = () => {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isOpen, setIsOpen] = useState(false); // Added state for menu
  const navigate = useNavigate();

  const toggleDrawer = (open) => (event) => {
    if (
      event.type === "keydown" &&
      (event.key === "Tab" || event.key === "Shift")
    ) {
      return;
    }
    setIsDrawerOpen(open);
  };

  const toggleMenu = () => {
    setIsOpen(!isOpen); // Toggle the menu state
  };

  const handleEducate = () => {
    navigate("/education");
  };

  return (
    <nav className="navContainer">
      <div className="navLogo">
        <img src={logo} alt="logo" width={160} height={80} />
        <IconButton
          onClick={toggleDrawer(true)}
          edge="start"
          sx={{ color: "white" }}
          aria-label="menu"
          className="drawerIcon"
        >
          <MenuIcon />
        </IconButton>
        <CustomDrawer isOpen={isDrawerOpen} toggleDrawer={toggleDrawer} />
      </div>

      <div className={`navLinks ${isOpen ? "showMenu" : ""}`}>
        <NavLink to="/" className="navLink" activeclassName="active">
          Home
        </NavLink>
        <NavLink to="/identify" className="navLink" activeclassName="active">
          Identify-Species
        </NavLink>
        <NavLink
          to="/poaching-detection"
          className="navLink"
          activeclassName="active"
        >
          Poaching-Detection
        </NavLink>
        <NavLink
          to="/animal-tracking"
          className="navLink"
          activeclassName="active"
        >
          Animal-Tracking
        </NavLink>
        <NavLink to="/profile" className="navLink" activeclassName="active">
          My-Profile
        </NavLink>
      </div>
      <div className="navButtons">
        <button className="navEducateButton" onClick={handleEducate}>
          <FaUserGraduate className="navIcon" /> Educate
        </button>
        <button className="navDonateButton">
          <FaDonate className="navIcon" /> Donate
        </button>
      </div>
      <div className="navHamburger" onClick={toggleMenu}>
        {isOpen ? <FaTimes /> : <FaBars />}
      </div>
    </nav>
  );
};

export default Navbar;
