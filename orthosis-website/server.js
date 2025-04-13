//server.js
const express = require('express');
const path = require('path');
const { exec } = require('child_process');
const fs = require('fs');

const app = express();
// store latest user input and opt results
let latestUserInput = null;

app.use(express.json());
app.use(express.static('public'));

let latestResults = null;

// optimization endpoint
app.post('/optimize', (req, res) => {
  const inputData = req.body;
  console.log('Received optimization input:', inputData);

  latestUserInput = inputData;
  
  const optimizerParams = {
    naturalAngle: inputData.naturalAngle,
    ptorqueExtend: inputData.forces.external || 30.0,
    atorqueBend: inputData.forces.extend || 50.0,
    atorqueExtend: inputData.forces.bend || 20.0,

    d1: inputData.dimensions.d1,
    d2: inputData.dimensions.d2,
    d3: inputData.dimensions.d3,
    w1: inputData.dimensions.w1,
    w2: inputData.dimensions.w2,
    w3: inputData.dimensions.w3,
    l1: inputData.dimensions.l1,
    l2: inputData.dimensions.l2,
    l3: inputData.dimensions.l3,

    thickness: inputData.thickness
  };
  
  // temp JSON file w params
  const paramsFile = path.join(__dirname, 'optimizer_params.json');
  fs.writeFileSync(paramsFile, JSON.stringify(optimizerParams, null, 2));
  
  const cmd = `python ${path.join(__dirname, 'run_optimizer.py')} --params ${paramsFile}`;
  
  exec(cmd, (error, stdout, stderr) => {
    try {
      fs.unlinkSync(paramsFile);
    } catch (err) {
      console.error('Error deleting temporary params file:', err);
    }
    
    if (error) {
      console.error(`Error running optimizer: ${error}`);
      console.error(`Stderr: ${stderr}`);
      return res.status(500).json({ error: 'Error running opt' });
    }
    
    console.log(`Optimizer output: ${stdout}`);
    
    try {
      // read results
      const resultsFile = path.join(__dirname, 'optimization_results.json');
      if (fs.existsSync(resultsFile)) {
        const resultsData = JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
        latestResults = resultsData; // Store the results
        res.json(resultsData);
      } else {

        const outputLines = stdout.trim().split('\n');
        const results = {};
        

        for (const line of outputLines) {
          if (line.includes('Torque down:')) {
            results.torqueDown = parseFloat(line.split(':')[1].trim());
          } else if (line.includes('Torque up:')) {
            results.torqueUp = parseFloat(line.split(':')[1].trim());
          } else if (line.includes('Model file:')) {
            results.mjcModelFile = line.split(':')[1].trim();
          }
        }
        
        latestResults = results; // store res
        res.json(results);
      }
    } catch (parseError) {
      console.error(`Error parsing optimization results: ${parseError}`);
      return res.status(500).json({ error: 'Error parsing optimization results' });
    }
  });
});

// generate mesh from optimizer result endpoint
app.post('/generate', (req, res) => {
  const { optimizationResults } = req.body;
  
  let combinedParams = {};
  
  try {
    const resultsFilePath = path.join(__dirname, 'optimization_results.json');
    let optimResults = optimizationResults;
    
    if (!optimResults && fs.existsSync(resultsFilePath)) {
      optimResults = JSON.parse(fs.readFileSync(resultsFilePath, 'utf8'));
    }
    
    if (!optimResults) {
      return res.status(400).json({ error: 'No optimization results available' });
    }
    
    const userInput = latestUserInput || (optimResults.dimensions ? {
      dimensions: optimResults.dimensions,
      naturalAngle: optimResults.naturalAngle,
      forces: {
        external: optimResults.ptorqueExtend,
        extend: optimResults.atorqueBend,
        bend: optimResults.atorqueExtend
      }
    } : null);
    
    if (!userInput) {
      return res.status(400).json({ error: 'No user input available' });
    }
    
    combinedParams = {

      dimensions: userInput.dimensions,
      naturalAngle: userInput.naturalAngle,
      

      ptorqueExtend: userInput.forces.external,
      atorqueBend: userInput.forces.extend,
      atorqueExtend: userInput.forces.bend,
      
      // TODO: again this may be wrong for thickness
      thickness: userInput.thickness || 'medium',
      
      torqueDown: optimResults.torqueDown,
      torqueUp: optimResults.torqueUp,
      geometryValues: optimResults.geometryValues,
      mjcModelFile: optimResults.mjcModelFile,
      torqueCurve: optimResults.torqueCurve
    };
    
    console.log('Combined parameters for mesh generation:', combinedParams);
    
    const blenderParamsFile = path.join(__dirname, 'blender_params.json');
    fs.writeFileSync(blenderParamsFile, JSON.stringify(combinedParams, null, 2));
    
    const cmd = `/Applications/Blender.app/Contents/MacOS/Blender --background --python ${path.join(__dirname, 'createMesh.py')} -- --params ${blenderParamsFile}`;
    
    exec(cmd, (error, stdout, stderr) => {

      try {
        fs.unlinkSync(blenderParamsFile);
      } catch (err) {
        console.error('Error deleting temporary blender params file:', err);
      }
      
      if (error) {
        console.error(`Error generating mesh: ${error}`);
        console.error(`Stderr: ${stderr}`);
        return res.status(500).json({ error: 'Error generating mesh' });
      }
      
      console.log(stdout);
      res.json({ message: 'Mesh generated successfully' });
    });
  } catch (error) {
    console.error('Error preparing mesh generation:', error);
    return res.status(500).json({ error: 'Error preparing mesh generation' });
  }
});


// app.get('/mesh', (req, res) => {
//   res.sendFile(path.join(__dirname, 'brace.stl'));
// });

// Add these routes before the existing '/mesh' route
app.get('/finger.stl', (req, res) => {
  res.sendFile(path.join(__dirname, 'finger.stl'));
});

app.get('/brace.stl', (req, res) => {
  res.sendFile(path.join(__dirname, 'brace.stl'));
});

// Modify the existing /mesh route to send the combined file
app.get('/mesh', (req, res) => {
  res.sendFile(path.join(__dirname, 'brace_combined.stl'));
});


app.get('/download', (req, res) => {
  const filePath = path.join(__dirname, 'brace.stl');
  res.download(filePath, 'custom_brace.stl');
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));