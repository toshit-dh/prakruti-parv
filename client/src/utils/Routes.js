const BASE_URL = "http://localhost:8080"
const BASE_URL_2 = "http://localhost:8081"
const ADD = "api"
const USER = "users"

export const SIGNUP_ROUTE = `${BASE_URL}/${ADD}/${USER}/signup`
export const LOGIN_ROUTE = `${BASE_URL}/${ADD}/${USER}/login`
export const TOKEN_ROUTE = `${BASE_URL}/${ADD}/${USER}/verify-token`
export const IDENTIFY_ROUTE = `${BASE_URL_2}/identify`