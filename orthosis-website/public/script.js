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
  
  //Orbit controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.25;
  controls.screenSpacePanning = false;
  controls.maxPolarAngle = Math.PI;
  controls.update();
  
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
  
  //remove old mesh (if it exists)
  if (mesh) {
    scene.remove(mesh);
    mesh = null;
  }
  
  loadingEl.style.display = 'block';
  statusEl.textContent = 'Loading mesh...';
  
  // stl loader
  const loader = new THREE.STLLoader();
  
  loader.load(
    '/mesh', 
    function(geometry) {
      //centering
      geometry.computeBoundingBox();
      const boundingBox = geometry.boundingBox;
      const center = new THREE.Vector3();
      boundingBox.getCenter(center);
      geometry.translate(-center.x, -center.y, -center.z);
      
      //rescale so you can see whole thing
      const size = boundingBox.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 50 / maxDim; 
      
      const material = new THREE.MeshPhongMaterial({
        color: 0x7c9cb0,
        specular: 0x111111,
        shininess: 100
      });
      
      mesh = new THREE.Mesh(geometry, material);
      mesh.scale.set(scale, scale, scale);
      scene.add(mesh);
      
      //position camera to see whole model
      camera.position.z = maxDim * 2;
      
      loadingEl.style.display = 'none';
      statusEl.textContent = 'Mesh loaded successfully';
      
      animate();
    },
    function(xhr) {
      const percentComplete = (xhr.loaded / xhr.total) * 100;
      statusEl.textContent = `Loading: ${Math.round(percentComplete)}%`;
    },
    function(error) {
      console.error('Error loading STL:', error);
      loadingEl.style.display = 'none';
      statusEl.textContent = 'Error loading mesh. Check console for details.';
    }
  );
}

function animate() {
  requestAnimationFrame(animate);
  
  if (controls) controls.update();
  
  renderer.render(scene, camera);
}

//event listeners
generateBtn.addEventListener('click', function() {
  statusEl.textContent = 'Generating mesh...';
  loadingEl.style.display = 'block';
  
  initScene();
  
  fetch('/generate')
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.text();
    })
    .then(data => {
      console.log(data);
      statusEl.textContent = 'Mesh generated, now loading...';
      
      loadSTL();
    })
    .catch(error => {
      console.error('Error:', error);
      loadingEl.style.display = 'none';
      statusEl.textContent = 'Error generating mesh. Check console for details.';
    });
});

downloadBtn.addEventListener('click', function() {

  // placeholdeer
  statusEl.textContent = 'not done yet';
});