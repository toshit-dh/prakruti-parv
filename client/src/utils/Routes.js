const BASE_URL = "http://localhost:8080"
const BASE_URL_2 = "http://localhost:8081"
const ADD = "api"
const USER = "users"

export const SIGNUP_ROUTE = `${BASE_URL}/${ADD}/${USER}/signup`
export const LOGIN_ROUTE = `${BASE_URL}/${ADD}/${USER}/login`
export const LOGOUT_ROUTE = `${BASE_URL}/${ADD}/${USER}/logout`
export const TOKEN_ROUTE = `${BASE_URL}/${ADD}/${USER}/verify-token`
export const EDIT_PROFILE_ROUTE = `${BASE_URL}/${ADD}/${USER}/`


export const IDENTIFY_ROUTE = `${BASE_URL_2}/identify`
export const POACH_ROUTE = `${BASE_URL_2}/poach`

const PROJECT_BASE_URL = `${BASE_URL}/${ADD}/projects`;

export const CREATE_PROJECT_ROUTE = `${PROJECT_BASE_URL}`;
export const GET_ALL_PROJECTS_ROUTE = `${PROJECT_BASE_URL}`;
export const GET_PROJECT_BY_ID_ROUTE = (id) => `${PROJECT_BASE_URL}/${id}`;
export const GET_PROJECT_BY_ORGANIZATION_ROUTE=(id)=>`${PROJECT_BASE_URL}/organization/${id}`
export const UPDATE_PROJECT_ROUTE = (id) => `${PROJECT_BASE_URL}/${id}`;
export const DELETE_PROJECT_ROUTE = (id) => `${PROJECT_BASE_URL}/${id}`;
export const ADD_MEDIA_TO_PROJECT_ROUTE = (id) => `${PROJECT_BASE_URL}/${id}/media`;
export const DONATE_TO_PROJECT_ROUTE = (id) => `${PROJECT_BASE_URL}/${id}/donate`;

export const UPLOAD_ROUTE = `${BASE_URL}/${ADD}/uploads/`;
export const GET_UPLOAD_ROUTE = `${BASE_URL}/${ADD}/uploads/user`;
export const FETCH_BALANCE = `${BASE_URL}/${ADD}/uploads/currency`;
export const REDUCE_CURRENCY_ROUTE = (amount) => `${BASE_URL}/${ADD}/uploads/reduce?amount=${amount}`;