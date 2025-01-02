/* eslint-disable no-unused-vars */
/* eslint-disable react/prop-types */
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
import ExploreIcon from '@mui/icons-material/Explore';
import { useDispatch } from "react-redux";
import "./CustomDrawer.css";
import { useSelector } from "react-redux";
import { logoutUser } from "../../redux/slice/UserSlice";

const CustomDrawer = ({ isOpen, toggleDrawer }) => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { isAuthenticated, status: logoutStatus, error } = useSelector(
    (state) => state.user
  );

  const handleLogout = async () => {
    try {
      const resultAction = await dispatch(logoutUser());
      if (logoutUser.fulfilled.match(resultAction)) {
        navigate("/login"); 
      } else {
        console.error('Logout failed:', resultAction.payload);
      }
    } catch (err) {
      console.error('An error occurred during logout:', err);
    }
  };
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
          { text: "Explore", icon: <ExploreIcon className="drawerIcon" /> },
          { text: "Logout", icon: <LogoutIcon className="drawerIcon" /> },
        ].map(({ text, icon }) => (
          <ListItem
            button
            key={text}
            className="listItem"
            onClick={() => {
              if(text == "Projects") navigate("/viewProjects");
              if(text=='Logout') handleLogout();
              if(text=='Settings') navigate("/settings");
              if(text=='Explore') navigate("/explore");
            }}
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
