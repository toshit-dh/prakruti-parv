import React from 'react'
import {useSelector} from "react-redux"

const ProtectedRoute = ({children}) => {
    const user = useSelector((state) => state.user)
    
    if(!user.isAuthenticated) return (<h1>login first</h1>)
    return children

};

export default ProtectedRoute