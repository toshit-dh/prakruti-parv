const nodemailer = require('nodemailer');
const fs = require('fs');
const path = require('path');

const transporter = nodemailer.createTransport({
  service: 'Gmail',
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS
  }
});

const sendVerificationEmail = async (user, token) => {
  const verificationUrl = `http://${process.env.FRONTEND_URL}/api/users/verify-email?token=${token}`;
  
  const filePath = path.join(__dirname, '../views/verificationEmail.html');
  let htmlTemplate = fs.readFileSync(filePath, 'utf8');

  htmlTemplate = htmlTemplate.replace('{{verificationUrl}}', verificationUrl)

  await transporter.sendMail({
    to: user.email,
    subject: 'Verify your email',
    html: htmlTemplate
  });
};

module.exports = {sendVerificationEmail}