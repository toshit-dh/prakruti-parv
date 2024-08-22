import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';
import { TOKEN_ROUTE } from '../../utils/Routes';

const initialState = {
  user: {},
  isAuthenticated: false,
  status: 'idle', 
  error: null,
};


export const fetchUser = createAsyncThunk(
  'user/fetchUser',
  async (_,{ rejectWithValue }) => {
    try {
      const response = await axios.get(TOKEN_ROUTE,{
        withCredentials: true
      }); 

      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);


const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUser.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchUser.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.user = action.payload;
        state.isAuthenticated = true;
      })
      .addCase(fetchUser.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
        state.isAuthenticated = false;
      });
  },
});



export default userSlice.reducer;
