import React from "react";
import { useNavigate } from "react-router-dom";
import logo from "../../assets/logo.png";
import {
  Drawer,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from "@mui/material";
import ChatIcon from "@mui/icons-material/Chat";
import PublicIcon from "@mui/icons-material/Public";
import SettingsIcon from "@mui/icons-material/Settings";
import LogoutIcon from "@mui/icons-material/Logout";
import "./CustomDrawer.css";

const CustomDrawer = ({ isOpen, toggleDrawer }) => {
    const navigate = useNavigate()
  const drawerList = () => (
    <div
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
      className="customDrawer"
    >
      <div className="navLogo">
        <img src={logo} alt="logo" width={160} height={80} />
      </div>
      <List>
        {[
          { text: "Connect", icon: <ChatIcon className="drawerIcon" /> },
          { text: "Projects", icon: <PublicIcon className="drawerIcon" /> },
          { text: "Settings", icon: <SettingsIcon className="drawerIcon" /> },
          { text: "Logout", icon: <LogoutIcon className="drawerIcon" /> },
        ].map(({ text, icon }) => (
          <ListItem 
          button key={text} 
          className="listItem"
          onClick={()=>navigate('/viewProjects')}
          >
            <ListItemIcon>{icon}</ListItemIcon>
            <ListItemText primary={text} />
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Drawer anchor="left" open={isOpen} onClose={toggleDrawer(false)}>
      {drawerList()}
    </Drawer>
  );
};

export default CustomDrawer;
