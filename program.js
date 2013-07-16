var container;
var scene, camera, renderer;
var cameraControls, effectController;
var clock = new THREE.Clock();
var cubes = [];
var dirLight;
var message, speed, factor;
var testCube;

var createAoCubeGeometry = function() {
  var cubeGeom = new THREE.Geometry();
  //    +y
  //    2-----3
  //   /|    /|
  //  6-----7 |
  //  | |   | |
  //  | 0---|-1 +x
  //  |/    |/
  //  4-----5
  // +z
  cubeGeom.vertices.push( new THREE.Vector3(-0.5,-0.5,-0.5));
  cubeGeom.vertices.push( new THREE.Vector3( 0.5,-0.5,-0.5));
  cubeGeom.vertices.push( new THREE.Vector3(-0.5, 0.5,-0.5));
  cubeGeom.vertices.push( new THREE.Vector3( 0.5, 0.5,-0.5));
  cubeGeom.vertices.push( new THREE.Vector3(-0.5,-0.5, 0.5));
  cubeGeom.vertices.push( new THREE.Vector3( 0.5,-0.5, 0.5));
  cubeGeom.vertices.push( new THREE.Vector3(-0.5, 0.5, 0.5));
  cubeGeom.vertices.push( new THREE.Vector3( 0.5, 0.5, 0.5));

  cubeGeom.faces.push( new THREE.Face4(0,4,6,2, new THREE.Vector3(-1, 0, 0)) ); // -X
  cubeGeom.faces.push( new THREE.Face4(1,3,7,5, new THREE.Vector3( 1, 0, 0)) ); // +X
  cubeGeom.faces.push( new THREE.Face4(0,1,5,4, new THREE.Vector3( 0,-1, 0)) ); // -Y
  cubeGeom.faces.push( new THREE.Face4(2,6,7,3, new THREE.Vector3( 0, 1, 0)) ); // +Y
  cubeGeom.faces.push( new THREE.Face4(0,2,3,1, new THREE.Vector3( 0, 0,-1)) ); // -Z
  cubeGeom.faces.push( new THREE.Face4(4,5,7,6, new THREE.Vector3( 0, 0, 1)) ); // +Z

  cubeGeom.faceAoFactors = [];
  for(var iFace=0; iFace<6; iFace += 1) {
    cubeGeom.faceAoFactors.push( [0xF, 0xF, 0xF, 0xF] );
    cubeGeom.faceVertexUvs[0].push( [new THREE.Vector2(0,0), new THREE.Vector2(1,0), new THREE.Vector2(1,1), new THREE.Vector2(0,1)] );
  }

  return cubeGeom;
};

var CubeAoOctant = {
  nXnYnZ: 0,
  pXnYnZ: 1,
  nXpYnZ: 2,
  pXpYnZ: 3,
  nXnYpZ: 4,
  pXnYpZ: 5,
  nXpYpZ: 6,
  pXpYpZ: 7,
};

var updateVertexAo = function(cubeGeom, octant, normal, newAo) {
  if      (octant === CubeAoOctant.nXnYnZ) { // vertex 0
    if      (normal.x < 0) { cubeGeom.faceAoFactors[0][0] = newAo; }
    else if (normal.y < 0) { cubeGeom.faceAoFactors[2][0] = newAo; }
    else if (normal.z < 0) { cubeGeom.faceAoFactors[4][0] = newAo; }
  }
  else if (octant === CubeAoOctant.pXnYnZ) { // vertex 1
    if      (normal.x > 0) { cubeGeom.faceAoFactors[1][0] = newAo; }
    else if (normal.y < 0) { cubeGeom.faceAoFactors[2][1] = newAo; }
    else if (normal.z < 0) { cubeGeom.faceAoFactors[4][3] = newAo; }
  }
  else if (octant === CubeAoOctant.nXpYnZ) { // vertex 2
    if      (normal.x < 0) { cubeGeom.faceAoFactors[0][3] = newAo; }
    else if (normal.y > 0) { cubeGeom.faceAoFactors[3][0] = newAo; }
    else if (normal.z < 0) { cubeGeom.faceAoFactors[4][1] = newAo; }
  }
  else if (octant === CubeAoOctant.pXpYnZ) { // vertex 3
    if      (normal.x > 0) { cubeGeom.faceAoFactors[1][1] = newAo; }
    else if (normal.y > 0) { cubeGeom.faceAoFactors[3][3] = newAo; }
    else if (normal.z < 0) { cubeGeom.faceAoFactors[4][2] = newAo; }
  }
  else if (octant === CubeAoOctant.nXnYpZ) { // vertex 4
    if      (normal.x < 0) { cubeGeom.faceAoFactors[0][1] = newAo; }
    else if (normal.y < 0) { cubeGeom.faceAoFactors[2][3] = newAo; }
    else if (normal.z > 0) { cubeGeom.faceAoFactors[5][0] = newAo; }
  }
  else if (octant === CubeAoOctant.pXnYpZ) { // vertex 5
    if      (normal.x > 0) { cubeGeom.faceAoFactors[1][3] = newAo; }
    else if (normal.y < 0) { cubeGeom.faceAoFactors[2][2] = newAo; }
    else if (normal.z > 0) { cubeGeom.faceAoFactors[5][1] = newAo; }
  }
  else if (octant === CubeAoOctant.nXpYpZ) { // vertex 6
    if      (normal.x < 0) { cubeGeom.faceAoFactors[0][2] = newAo; }
    else if (normal.y > 0) { cubeGeom.faceAoFactors[3][1] = newAo; }
    else if (normal.z > 0) { cubeGeom.faceAoFactors[5][3] = newAo; }
  }
  else if (octant === CubeAoOctant.pXpYpZ) { // vertex 7
    if      (normal.x > 0) { cubeGeom.faceAoFactors[1][2] = newAo; }
    else if (normal.y > 0) { cubeGeom.faceAoFactors[3][2] = newAo; }
    else if (normal.z > 0) { cubeGeom.faceAoFactors[5][2] = newAo; }
  }
};
var updateCubeAo = function(cubeGeom) {
  for(var iFace=0; iFace<6; iFace += 1) {
    var lutX = (1/1024) + ((cubeGeom.faceAoFactors[iFace][0] & 0xF) | ((cubeGeom.faceAoFactors[iFace][1] & 0xF) << 4)) / 256;
    var lutY = (1/1024) + ((cubeGeom.faceAoFactors[iFace][3] & 0xF) | ((cubeGeom.faceAoFactors[iFace][2] & 0xF) << 4)) / 256;
    cubeGeom.faceVertexUvs[0][iFace][0].set( lutX+0,      1.0 - (lutY+0)       );
    cubeGeom.faceVertexUvs[0][iFace][1].set( lutX+(1/512),1.0 - (lutY+0)       );
    cubeGeom.faceVertexUvs[0][iFace][2].set( lutX+(1/512),1.0 - (lutY+(1/512)) );
    cubeGeom.faceVertexUvs[0][iFace][3].set( lutX+0,      1.0 - (lutY+(1/512)) );
  }
  cubeGeom.uvsNeedUpdate = true;
};

var fillScene = function() {
  scene = new THREE.Scene();
  //scene.fog = new THREE.Fog( 0x808080, 5, 5.5 );

  var aoLookupTex    = THREE.ImageUtils.loadTexture( "cube_ao_lookup.png" );
	var aoCubeMaterial = new THREE.MeshBasicMaterial( { map: aoLookupTex } );

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
};

var render = function() {
	var delta = clock.getDelta();

  //cube.rotation.x += 0.01;
  //cube.rotation.y += 0.02;

  THREE.AxisHelper(100);

  cameraControls.update(delta);
  renderer.render(scene,camera);
};

var animate = function() {
  window.requestAnimationFrame(animate);
  dirLight.position.copy(cameraControls.object.position)
  render();
};

init();
animate();
