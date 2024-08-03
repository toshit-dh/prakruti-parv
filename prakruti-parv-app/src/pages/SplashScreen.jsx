// SplashScreen.js
import React, { useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { setUserDetails, setUserAuthenticated } from '../redux/slice/UserSlice' 

export default function  SplashScreen ({setLoading}) {
  const dispatch = useDispatch();

  useEffect(() => {
    const verifyUser = async () => {
      try {
        // const response = await fetch('https://your-api.com/verify');
        // const data = await response.json();
        if (true) {
          dispatch(setUserAuthenticated(true));
          dispatch(setUserDetails({name: "Toshit"}))
          
        } else {
          dispatch(setUserAuthenticated(false));
        }
      } catch (error) {
        console.error('Error verifying user:', error);
        dispatch(setUserAuthenticated(false));
      } 
    };
    verifyUser();
  }, [dispatch]);

  return <div onClick={()=>setLoading(false)}>Loading...</div>;
};


