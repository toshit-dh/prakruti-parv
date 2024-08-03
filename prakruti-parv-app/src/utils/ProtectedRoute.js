import React from 'react'
import {useSelector} from "react-redux"

const ProtectedRoute = ({children}) => {
    const user = useSelector((state) => state.user)
    console.log(user);
    
    if(!user.state.isAuthenticated) return (<h1>login</h1>)
    return children

};

export default ProtectedRoute