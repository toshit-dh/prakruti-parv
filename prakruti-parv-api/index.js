const express = require('express');
const app = express();
const port = 3000;

app.use(express.json());

app.get('/', (_, res) => {
    res.send('WELCOME TO PRAKRUTI PARV- WILDLIFE CONSERVATION APP');
});

app.listen(port, () => {
    console.log(port);
});
