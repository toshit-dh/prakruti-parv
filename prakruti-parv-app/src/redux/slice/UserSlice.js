// UserSlice.js
import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  state: {
    user: {},
    isAuthenticated: false,
  },
};

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setUserDetails: (state,action) => {
        state.state.user = action.payload
    },
    setUserAuthenticated: (state, action) => {
      state.state.isAuthenticated = action.payload;
    },
  },
});

export const { setUserDetails, setUserAuthenticated } = userSlice.actions;

export default userSlice.reducer;
