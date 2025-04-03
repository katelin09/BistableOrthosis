//server.js
const express = require('express');
const path = require('path');
const { exec } = require('child_process');

const app = express();

app.use(express.static('public'));

app.get('/generate', (req, res) => {

  const cmd = '/Applications/Blender.app/Contents/MacOS/Blender --background --python createMesh.py';
  exec(cmd, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${error}`);
      return res.status(500).send('Error generating mesh');
    }
    console.log(stdout);
    res.send('Mesh generated successfully');
  });
});

app.get('/mesh', (req, res) => {
  res.sendFile(path.join(__dirname, 'brace.stl'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

