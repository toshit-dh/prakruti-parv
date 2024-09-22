const express = require('express')
const cors = require('cors')
const cp = require('cookie-parser')
const morgan = require('morgan')
const mongoose = require("mongoose")
const app = express()
const port = 8080
const path = require('path')

require('dotenv').config()

const userRoutes = require('./routes/UserRoutes')
const projectRoutes = require('./routes/ProjectRoutes')
app.use(cors({
  origin: "http://localhost:5173",
  credentials: true
}))
app.use(cp())
app.use(morgan('dev'))
app.use(express.json())

app.get('/', (_, res) => {
    res.send('WELCOME TO PRAKRUTI PARV- WILDLIFE CONSERVATION APP')
})
app.use(express.static(path.join(__dirname,'data/projects/')))
app.use('/api/users',userRoutes)
app.use('/api/projects',projectRoutes)

mongoose
  .connect(
   process.env.DB_URL
  )
  .then(() => {
    console.log("DB Connected")
  })
  .catch((err) => {
    console.log(err.message)
  })

app.listen(port, () => {
    console.log(port)
});
