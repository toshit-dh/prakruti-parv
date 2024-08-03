const express = require('express')
const cors = require('cors')
const mongoose = require("mongoose")
const app = express()
const port = 3000

require('dotenv').config()

app.use(cors())
app.use(express.json());

app.get('/', (_, res) => {
    res.send('WELCOME TO PRAKRUTI PARV- WILDLIFE CONSERVATION APP')
})


const DB_URL = 
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
