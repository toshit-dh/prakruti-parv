import React, { useState } from "react";
import Navbar from "../navbar/Navbar";
import GoogleTranslate from "../google-translate/GoogleTranslate";
import "./Settings.css";

const Settings = () => {
  const [darkMode, setDarkMode] = useState(false);

  const toggleMode = () => {
    setDarkMode(!darkMode);
    document.body.classList.toggle("dark-mode", darkMode);
  };

  return (
    <div className={`settings-page ${darkMode ? "dark-mode" : ""}`}>
      <Navbar />

      <div className="settings-container">
        <div className="settings-header">
          <h2 className="settings-title">Settings</h2>
          <div className="div-for-border">
            <p className="settings-description">
                Customize your preferences below.
            </p>
          </div>
        </div>

        <div className="settings-content">
          <div className="settings-left">
            <div className="setting-item">
              <label htmlFor="mode">Mode</label>
              <div className="toggle-switch">
                <span>Light</span>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={darkMode}
                    onChange={toggleMode}
                  />
                  <span className="slider"></span>
                </label>
                <span>Dark</span>
              </div>
            </div>

            <div className="setting-item">
              <label htmlFor="notifications">Enable Notifications</label>
              <div className="toggle-switch">
                <label className="switch">
                  <input type="checkbox" />
                  <span className="slider"></span>
                </label>
              </div>
            </div>

            <div className="setting-item">
              <label htmlFor="privacy">Privacy Mode</label>
              <div className="toggle-switch">
                <label className="switch">
                  <input type="checkbox" />
                  <span className="slider"></span>
                </label>
              </div>
            </div>
          </div>

          <div className="settings-right">
            <div className="language-selection">
              <h3>Select Language</h3>
              <p>Choose your preferred language for the website:</p>
              <GoogleTranslate />
            </div>

            <div className="setting-item">
              <label htmlFor="notifications">Dark Mode Setting</label>
              <p>Toggle for dark or light mode.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
