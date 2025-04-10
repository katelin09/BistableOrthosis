
// Elements
const viewer = document.getElementById('viewer');
const generateBtn = document.getElementById('generateBtn');
const statusEl = document.getElementById('status');
const loadingEl = document.getElementById('loading');
const downloadBtn = document.querySelector('.download-btn');

// Three.js variables
let scene, camera, renderer, mesh;
let isInitialized = false;
let controls; // for orbit controls
let optimizationResults = null; // Store optimization results

// Initialize Three.js scene
function initScene() {
  if (isInitialized) return;
  
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8f8f8);
  
  // Camera 
  const width = viewer.clientWidth;
  const height = viewer.clientHeight;
  camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  camera.position.set(0, 0, 100);
  
  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputEncoding = THREE.sRGBEncoding;
  
  while (viewer.firstChild) {
    if (viewer.firstChild.id === 'loading') {
      viewer.firstChild.style.display = 'block';
      break;
    } else {
      viewer.removeChild(viewer.firstChild);
    }
  }
  viewer.appendChild(renderer.domElement);
  
  controls = new THREE.TrackballControls(camera, renderer.domElement);

  controls.rotateSpeed = 5.0;
  controls.zoomSpeed = 1.2;
  controls.panSpeed = 0.8;

  controls.noZoom = false;
  controls.noPan = false;
  controls.staticMoving = false;
  controls.dynamicDampingFactor = 0.3;


  addLights();
  
  window.addEventListener('resize', onWindowResize);
  
  isInitialized = true;

}

function addLights() {
  // Ambient light
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);
  
  // Directional light 
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(1, 1, 1);
  scene.add(directionalLight);
  
  // even more lights
  const light1 = new THREE.DirectionalLight(0xffffff, 0.5);
  light1.position.set(-1, 1, 1);
  scene.add(light1);
  
  const light2 = new THREE.DirectionalLight(0xffffff, 0.3);
  light2.position.set(0, -1, 0);
  scene.add(light2);
}

function onWindowResize() {
  if (!camera || !renderer) return;
  
  const width = viewer.clientWidth;
  const height = viewer.clientHeight;
  
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function loadSTL() {
  if (!scene) return;
  
  console.log("Starting to load meshes sequentially...");
  
  // Remove old meshes (if they exist)
  if (mesh) {
    if (mesh.isGroup) {
      mesh.children.forEach(child => scene.remove(child));
    } else {
      scene.remove(mesh);
    }
    mesh = null;
  }
  
  // Clear any other meshes that might be in the scene
  scene.children.forEach(child => {
    if (child.isMesh) {
      scene.remove(child);
    }
  });
  
  loadingEl.style.display = 'block';
  statusEl.textContent = 'Loading finger mesh...';
  
  // Create distinctive materials
  const fingerMaterial = new THREE.MeshPhongMaterial({
    color: 0xC0C0C0,  // Silver color
    specular: 0x777777,
    shininess: 100,
    transparent: true,
    opacity: .8

  });

  const braceMaterial = new THREE.MeshPhongMaterial({
    color: 0x000000,       
    specular: 0x292827,    
    shininess: 5,          
    emissive: 0x000000,     
    flatShading: false,   
  });
  
  // const braceMaterial = new THREE.MeshPhongMaterial({
  //   color: 0x000000,  // Black color
  //   specular: 0x000000,
  //   shininess: 20
  // });
  
  // Create a group to hold both meshes
  const group = new THREE.Group();
  mesh = group;  // Store reference to the group
  scene.add(group);  // Add group to scene immediately
  
  // Load finger mesh first
  const loader = new THREE.STLLoader();
  loader.load(
    '/finger.stl', 
    function(fingerGeometry) {
      console.log("Finger mesh loaded successfully");
      
      // Create and add finger mesh to the group
      const fingerMesh = new THREE.Mesh(fingerGeometry, fingerMaterial);
      fingerMesh.name = "finger";
      // fingerMesh.rotation.z = Math.PI/2;
      // fingerMesh.rotation.x = -0.1;
      fingerMesh.position.y = -4; //move knuckle back and forth
      fingerMesh.position.z =-1; //move up and down
      group.add(fingerMesh);
      
      statusEl.textContent = 'Finger loaded. Now loading brace...';
      
      // Now load the brace mesh 
      loader.load(
        '/brace.stl',
        function(braceGeometry) {
          console.log("Brace mesh loaded successfully");
          
          // Create and add brace mesh to the group
          const braceMesh = new THREE.Mesh(braceGeometry, braceMaterial);
          braceMesh.name = "brace";
          group.add(braceMesh);
          
          // Once both meshes are loaded, center and scale the group
          const box = new THREE.Box3().setFromObject(group);
          const center = new THREE.Vector3();
          box.getCenter(center);
          group.position.sub(center);
          
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 50 / maxDim;
          group.scale.set(scale, scale, scale);
          
          // Position camera
          camera.position.z = maxDim;
          
          loadingEl.style.display = 'none';
          statusEl.textContent = 'Mesh loaded successfully';
          
          animate();
        },
        function(xhr) {
          const percentComplete = (xhr.loaded / xhr.total) * 100;
          statusEl.textContent = `Loading brace: ${Math.round(percentComplete)}%`;
        },
        function(error) {
          console.error('Error loading brace STL:', error);
          
          // Even if brace fails, we still have the finger - just finish up with what we have
          const box = new THREE.Box3().setFromObject(group);
          const center = new THREE.Vector3();
          box.getCenter(center);
          group.position.sub(center);
          
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 50 / maxDim;
          group.scale.set(scale, scale, scale);
          
          camera.position.z = maxDim;
          
          loadingEl.style.display = 'none';
          statusEl.textContent = 'Only finger mesh loaded (brace failed)';
          
          animate();
        }
      );
    },
    function(xhr) {
      const percentComplete = (xhr.loaded / xhr.total) * 100;
      statusEl.textContent = `Loading finger: ${Math.round(percentComplete)}%`;
    },
    function(error) {
      console.error('Error loading finger STL:', error);
      
      // If finger fails, try loading just the brace
      statusEl.textContent = 'Finger failed, trying brace only...';
      
      loader.load(
        '/brace.stl',
        function(braceGeometry) {
          console.log("Brace mesh loaded successfully");
          
          const braceMesh = new THREE.Mesh(braceGeometry, braceMaterial);
          braceMesh.name = "brace";
          group.add(braceMesh);
          
          const box = new THREE.Box3().setFromObject(group);
          const center = new THREE.Vector3();
          box.getCenter(center);
          group.position.sub(center);
          
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 50 / maxDim;
          group.scale.set(scale, scale, scale);
          
          camera.position.z = maxDim;
          
          loadingEl.style.display = 'none';
          statusEl.textContent = 'Only brace mesh loaded (finger failed)';
          
          animate();
        },
        function(xhr) {
          const percentComplete = (xhr.loaded / xhr.total) * 100;
          statusEl.textContent = `Loading brace: ${Math.round(percentComplete)}%`;
        },
        function(braceError) {
          console.error('Error loading brace STL:', braceError);
          loadCombinedMesh();  // Fall back to combined mesh
        }
      );
    }
  );
  
  // Fallback function to load combined mesh if separate meshes fail
  function loadCombinedMesh() {
    console.log('Falling back to combined mesh');
    statusEl.textContent = 'Loading combined mesh...';
    
    loader.load(
      '/mesh', 
      function(geometry) {
        // Clear the group and add the combined mesh
        while (group.children.length > 0) {
          group.remove(group.children[0]);
        }
        
        const combinedMesh = new THREE.Mesh(geometry, braceMaterial);
        combinedMesh.name = "combined";
        group.add(combinedMesh);
        
        // Center and scale
        geometry.computeBoundingBox();
        const boundingBox = geometry.boundingBox;
        const center = new THREE.Vector3();
        boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);
        
        const size = boundingBox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 50 / maxDim;
        
        group.scale.set(scale, scale, scale);
        
        // Position camera
        camera.position.z = maxDim;
        
        loadingEl.style.display = 'none';
        statusEl.textContent = 'Mesh loaded successfully (combined)';
        
        animate();
      },
      function(xhr) {
        const percentComplete = (xhr.loaded / xhr.total) * 100;
        statusEl.textContent = `Loading combined: ${Math.round(percentComplete)}%`;
      },
      function(error) {
        console.error('Error loading combined STL:', error);
        loadingEl.style.display = 'none';
        statusEl.textContent = 'Error loading all meshes. Check console for details.';
      }
    );
  }
}


function animate() {
  requestAnimationFrame(animate);
  controls.update(); 
  renderer.render(scene, camera);
}

// Function to collect all input values
function collectInputData() {
    // Get the selected thickness option
    let thicknessValue = "medium"; // Default to medium
    const thicknessOptions = document.getElementsByName('thickness');
    for (const option of thicknessOptions) {
      if (option.checked) {
        thicknessValue = option.value;
        break;
      }
    }
    
    const data = {
      dimensions: {
        d1: parseFloat(document.getElementById('d1').value) || 0,
        d2: parseFloat(document.getElementById('d2').value) || 0,
        d3: parseFloat(document.getElementById('d3').value) || 0,
        w1: parseFloat(document.getElementById('w1').value) || 0,
        w2: parseFloat(document.getElementById('w2').value) || 0,
        w3: parseFloat(document.getElementById('w3').value) || 0,
        l1: parseFloat(document.getElementById('l1').value) || 0,
        l2: parseFloat(document.getElementById('l2').value) || 0,
        l3: parseFloat(document.getElementById('l3').value) || 0
      },
      naturalAngle: parseFloat(document.getElementById('angle').value) || 0,
      forces: {
        external: parseFloat(document.getElementById('f_external').value) || 0,
        extend: parseFloat(document.getElementById('f_extend').value) || 0,
        bend: parseFloat(document.getElementById('f_bend').value) || 0
      },
      thickness: thicknessValue 
    };
    
    return data;
}

// Function to send data to the optimizer and get results
async function runOptimization(inputData) {
  try {
    const response = await fetch('/optimize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(inputData)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const results = await response.json();
    optimizationResults = results; // Store the results
    
    console.log('Optimization results:', results);
    statusEl.textContent = 'Optimization complete, generating mesh...';
    
    return results;
  } catch (error) {
    console.error('Optimization error:', error);
    loadingEl.style.display = 'none';
    statusEl.textContent = 'Error running optimization. Check console for details.';
    throw error;
  }
}

// async function generateMesh(inputData) {
//   try {
//     // Create dummy optimization results
//     const dummyResults = {
//       torqueDown: 0.5,
//       torqueUp: 0.5,
//       geometryValues: {
//         beamA: 25.0,
//         beamB: 15.0,
//         beamC: 25.0,
//         theta: 30.0,
//         hingeThickness: 0.6,
//         tendonWidth: 1.6,
//         tendonThickness: 1.2,
//         hingeLength: 2.0
//       },
//       mjcModelFile: "dummy.xml",
//       torqueCurve: [] // Empty array if needed
//     };
    
//     // Store dummy results
//     optimizationResults = dummyResults;
    
//     statusEl.textContent = 'Skipping optimization, generating mesh directly...';
    
//     // Call the generate endpoint directly
//     const response = await fetch('/generate', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json'
//       },
//       body: JSON.stringify({
//         optimizationResults: dummyResults
//       })
//     });
    
//     if (!response.ok) {
//       throw new Error(`HTTP error! Status: ${response.status}`);
//     }
    
//     const result = await response.json();
//     console.log('Mesh generation result:', result);
    
//     return result;
//   } catch (error) {
//     console.error('Mesh generation error:', error);
//     loadingEl.style.display = 'none';
//     statusEl.textContent = 'Error generating mesh. Check console for details.';
//     throw error;
//   }
// }

async function generateMesh(inputData) {
  try {
    // Create a more complete dummy results object that includes the user input
    const dummyResults = {
      torqueDown: 0.5,
      torqueUp: 0.5,
      // Include the dimensions from inputData
      dimensions: inputData.dimensions,
      // Include the natural angle
      naturalAngle: inputData.naturalAngle,
      // Include forces mapped to the expected properties
      ptorqueExtend: inputData.forces.external,
      atorqueBend: inputData.forces.extend,
      atorqueExtend: inputData.forces.bend,
      // Include thickness
      thickness: inputData.thickness,
      // Basic geometry values
      geometryValues: {
        beamA: 25.0,
        beamB: 15.0,
        beamC: 25.0,
        theta: 30.0,
        hingeThickness: 0.6,
        tendonWidth: 1.6,
        tendonThickness: 1.2,
        hingeLength: 2.0
      }
    };
    
    // Store dummy results
    optimizationResults = dummyResults;
    
    statusEl.textContent = 'Skipping optimization, generating mesh directly...';
    
    // Call the generate endpoint
    const response = await fetch('/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        optimizationResults: dummyResults
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('Mesh generation result:', result);
    
    return result;
  } catch (error) {
    console.error('Mesh generation error:', error);
    loadingEl.style.display = 'none';
    statusEl.textContent = 'Error generating mesh. Check console for details.';
    throw error;
  }
}

// Modified event listener for the generate button
generateBtn.addEventListener('click', async function() {
  statusEl.textContent = 'Collecting input data...';
  loadingEl.style.display = 'block';
  
  try {
    // Initialize the 3D scene if not already
    initScene();
    
    // Collect all input data
    const inputData = collectInputData();
    console.log('Collected input data:', inputData);
    
    // Skip optimization and go straight to mesh generation
    await generateMesh(inputData);
    
    // Load the STL mesh
    statusEl.textContent = 'Mesh generated, now loading...';
    loadSTL();
  } catch (error) {
    console.error('Error in generate process:', error);
    loadingEl.style.display = 'none';
    statusEl.textContent = 'Error in process. Check console for details.';
  }
});


// //event listeners
// generateBtn.addEventListener('click', async function() {
//   statusEl.textContent = 'Collecting input data...';
//   loadingEl.style.display = 'block';
  
//   try {
//     // Initialize the 3D scene if not already
//     initScene();
    
//     // Collect all input data
//     const inputData = collectInputData();
//     console.log('Collected input data:', inputData);
    
//     // Run optimization with input data
//     statusEl.textContent = 'Running optimization...';
//     const results = await runOptimization(inputData);
    
//     // Generate the mesh with the optimization results
//     statusEl.textContent = 'Optimization complete, generating mesh...';
//     await fetch('/generate', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json'
//       },
//       body: JSON.stringify({
//         optimizationResults: results
//       })
//     });
    
//     // Load the STL mesh
//     statusEl.textContent = 'Mesh generated, now loading...';
//     loadSTL();
//   } catch (error) {
//     console.error('Error in generate process:', error);
//     loadingEl.style.display = 'none';
//     statusEl.textContent = 'Error in process. Check console for details.';
//   }
// });

downloadBtn.addEventListener('click', function() {
  if (!optimizationResults) {
    statusEl.textContent = 'No results available to download';
    return;
  }
  
  const downloadLink = document.createElement('a');
  downloadLink.href = '/download';
  downloadLink.download = 'generated_mesh.stl';
  document.body.appendChild(downloadLink);
  downloadLink.click();
  document.body.removeChild(downloadLink);
  
  statusEl.textContent = 'Downloading STL file';
});