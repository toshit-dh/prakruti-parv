/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import { FaHome, FaBolt, FaUserGraduate, FaDonate, FaTimes, FaBars,FaUser } from 'react-icons/fa';
import { NavLink ,useNavigate} from 'react-router-dom';
import logo from '../../assets/logo.png';
import {useSelector} from 'react-redux'
import './Navbar.css';

const Navbar = () => {
  const navigate = useNavigate()
  const user = useSelector((state)=>state.user)
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  return (
    <nav className="navContainer">
      <div className="navLogo">
        <img src={logo} alt="logo" width={160} height={80} />
      </div>
      <div className={`navLinks ${isOpen ? 'showMenu' : ''}`}>
        <NavLink
          to="/"
          className="navLink"
          style={({ isActive }) => ({
            borderBottom: isActive ? "4px solid var(--primary-color)" : 'none',
            paddingBottom: '8px', 
          })}
        >
          Home
        </NavLink>
        <NavLink
          to="/identify"
          className="navLink"
          style={({ isActive }) => ({
            borderBottom: isActive ? "4px solid var(--primary-color)" : 'none',
            paddingBottom: '8px',
          })}
        >
          Identify-Species
        </NavLink>
        <NavLink
          to="/poaching-detection"
          className="navLink"
          style={({ isActive }) => ({
            borderBottom: isActive ? "4px solid var(--primary-color)" : 'none',
            paddingBottom: '8px',
          })}
        >
          Poaching-Detection
        </NavLink>
        <NavLink
          to="/animal-tracking"
          className="navLink"
          style={({ isActive }) => ({
            borderBottom: isActive ? "4px solid var(--primary-color)" : 'none',
            paddingBottom: '8px',
          })}
        >
          Animal-Tracking
        </NavLink>
        {
          user.isAuthenticated && <NavLink
          to="/myprofile"
          className="navLink"
          style={({ isActive }) => ({
            borderBottom: isActive ? "4px solid var(--primary-color)" : 'none',
            paddingBottom: '8px',
          })}
        >
          My-Profile
        </NavLink>
        }
      </div>
      <div className="navButtons">
      {
        !user.isAuthenticated && <button className="navLoginButton" onClick={()=>{
          navigate('/login')
        }}>
            <FaUser className="navIcon" /> Login
          </button>
      }
        <button className="navEducateButton">
          <FaUserGraduate className="navIcon" /> Learn
        </button>
        <button className="navDonateButton">
          <FaDonate className="navIcon" /> Donate
        </button>
      </div>
      <div className="navHamburger" onClick={toggleMenu}>
        {isOpen ? <FaTimes /> : <FaBars />}
      </div>
      <div className={`navLinksMobile ${isOpen ? 'showMenu' : ''}`}>
        <NavLink
          to="/"
          className="navLinkMobile"
        >
          Home
        </NavLink>
        <NavLink
          to="/identify"
          className="navLinkMobile"
        >
          Identify
        </NavLink>
        <NavLink
          to="/poaching-detection"
          className="navLinkMobile"
        >
          Poaching Detection
        </NavLink>
        <NavLink
          to="/animal-tracking"
          className="navLinkMobile"
        >
          Animal Tracking
        </NavLink>
        <button className="navEducateButtonMobile">
          <FaUserGraduate className="navIcon" /> Learn
        </button>
        <button className="navDonateButtonMobile">
          <FaDonate className="navIcon" /> Donate
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
