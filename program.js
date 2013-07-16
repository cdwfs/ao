var container;
var stats;
var scene, camera, renderer;
var cameraControls, effectController;
var clock = new THREE.Clock();
var cubes = [];
var dirLight;
var message, speed, factor;
var testCube;
var cubeUniforms;

var fillScene = function() {
  scene = new THREE.Scene();
  //scene.fog = new THREE.Fog( 0x808080, 5, 5.5 );

  var aoLookupTex    = THREE.ImageUtils.loadTexture( "cube_ao_lookup.png" );
  cubeUniforms = {
    time:       { type: "f",  value: 1.0 },
    resolution: { type: "v2", value: new THREE.Vector2() },
    texture:    { type: "t",  value: aoLookupTex },
  };
  var aoCubeMaterial = new THREE.ShaderMaterial( {
    uniforms: cubeUniforms,
    vertexShader: document.getElementById( 'vertexShader' ).textContent,
    fragmentShader: document.getElementById( 'fragmentShader' ).textContent,
  } );

  testCube = createAoCubeGeometry();
  var cubeMat;
  var x,y,z;
  var cube;
  for(z=-0; z<1; z += 1) {
    for(y=0; y<1; y += 1) {
      for(x=0; x<1; x += 1) {
        cubeMat = aoCubeMaterial;
        cube = new THREE.Mesh(testCube,cubeMat);
        cube.position = new THREE.Vector3(x+0.5, y+0.5, z+0.5);
        cubes.push(cube);
        scene.add(cube);
      }
    }
  }

  updateVertexAo(testCube, CubeAoOctant.pXnYnZ, new THREE.Vector3( 1, 0, 0), 0x0);
  updateVertexAo(testCube, CubeAoOctant.pXnYnZ, new THREE.Vector3( 0,-1, 0), 0x0);
  updateVertexAo(testCube, CubeAoOctant.pXnYnZ, new THREE.Vector3( 0, 0,-1), 0x0);
  updateCubeAo(testCube);

  dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  scene.add(dirLight);

  var axes = new THREE.AxisHelper( 100 );
	axes.material.depthTest = true;
	axes.material.transparent = true;
	axes.matrixAutoUpdate = false;
	axes.visible = true;
  scene.add(axes);
};

var factor0 = 0xF, factor1 = 0xF, factor2 = 0xF, factor3 = 0xF;
var applyAndUpdate = function() {
  updateVertexAo(testCube, CubeAoOctant.nXnYnZ, new THREE.Vector3(-1, 0, 0), factor0);
  updateVertexAo(testCube, CubeAoOctant.nXnYpZ, new THREE.Vector3(-1, 0, 0), factor1);
  updateVertexAo(testCube, CubeAoOctant.nXpYpZ, new THREE.Vector3(-1, 0, 0), factor2);
  updateVertexAo(testCube, CubeAoOctant.nXpYnZ, new THREE.Vector3(-1, 0, 0), factor3);
  updateCubeAo(testCube);
};

function onWindowResize( event ) {

  cubeUniforms.resolution.value.x = 40;//window.innerWidth;
  cubeUniforms.resolution.value.y = 40;//window.innerHeight;

  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  renderer.setSize( window.innerWidth, window.innerHeight );

}


var init = function() {
	var canvasWidth = window.innerWidth;
	var canvasHeight = window.innerHeight;
	var canvasRatio = canvasWidth / canvasHeight;
  renderer = new THREE.WebGLRenderer( { antialias: true } );
	renderer.gammaInput = true;
	renderer.gammaOutput = true;
	renderer.setSize(canvasWidth, canvasHeight);
	renderer.setClearColor( 0x808080, 1.0 );

  container = document.getElementById( 'webgl_container' );
  container.appendChild( renderer.domElement );

  // STATS
  stats = new Stats();
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.top = '50px';
  container.appendChild( stats.domElement );

	// CAMERA
	camera = new THREE.PerspectiveCamera( 30, canvasRatio, 0.1, 1000.0 );
	camera.position.set( -2,-2,-2 );
  camera.lookAt( new THREE.Vector3(2,2,2) );

  cameraControls = new THREE.OrbitAndPanControls(camera, renderer.domElement);
  cameraControls.target.set(0.5,0.5,0.5);

	fillScene();
  message = "Fap";
  speed = 3;
  factor = 5;

  var gui = new dat.GUI();
  gui.add(this, 'factor0', 0, 15).onChange(applyAndUpdate);
  gui.add(this, 'factor1', 0, 15).onChange(applyAndUpdate);
  gui.add(this, 'factor2', 0, 15).onChange(applyAndUpdate);
  gui.add(this, 'factor3', 0, 15).onChange(applyAndUpdate);

  onWindowResize();

  window.addEventListener( 'resize', onWindowResize, false );
};

var render = function() {
	var delta = clock.getDelta();

  cubeUniforms.time.value += delta * 5;

  THREE.AxisHelper(100);

  cameraControls.update(delta);
  renderer.render(scene,camera);
};

var animate = function() {
  window.requestAnimationFrame(animate);
  dirLight.position.copy(cameraControls.object.position)
  render();
  stats.update()
};

init();
animate();
