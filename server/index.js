const express = require('express')
const cors = require('cors')
const cp = require('cookie-parser')
const morgan = require('morgan')
const mongoose = require("mongoose")
const app = express()
const port = 8080

require('dotenv').config()

const userRoutes = require('./routes/UserRoutes')

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

app.use('/api/users',userRoutes)

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
