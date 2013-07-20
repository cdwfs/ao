"use strict";
var container;
var stats;
var scene, camera, renderer;
var cameraControls, effectController;
var clock = new THREE.Clock();
var grid;
var dirLight;
var fillScene = function() {
  scene = new THREE.Scene();
  //scene.fog = new THREE.Fog( 0x808080, 5, 5.5 );

  var urCubeGeom = new THREE.CubeGeometry(1,1,1);
  var urCubeMat = new THREE.ShaderMaterial( {
    uniforms: {},
    vertexShader:   document.getElementById( 'vertexShader' ).textContent,
    fragmentShader: document.getElementById( 'solidFragmentShader' ).textContent,
  });
  var x,y,z;
  grid = new THREE.Object3D();
  grid.cubes = [];
  for(z=-0; z<4; z += 1) {
    for(y=0; y<4; y += 1) {
      for(x=0; x<4; x += 1) {
        var geom = urCubeGeom.clone();
        var mat  = urCubeMat.clone();
        mat.uniforms = {
          fillColor:    { type: "v4",  value: new THREE.Vector4(Math.random(), Math.random(), Math.random(), 1.0) },
        };
        var cube = new THREE.Mesh(geom,mat);
        cube.name = "cube" +x+y+z;
        cube.position.set(x+0.5, y+0.5, z+0.5);
        grid.cubes.push(cube);
        scene.add(cube);
      }
    }
  }
  scene.add(grid);

  //updateVertexAo(testCube, CubeAoOctant.pXnYnZ, new THREE.Vector3( 1, 0, 0), 0x0);
  //updateVertexAo(testCube, CubeAoOctant.pXnYnZ, new THREE.Vector3( 0,-1, 0), 0x0);
  //updateVertexAo(testCube, CubeAoOctant.pXnYnZ, new THREE.Vector3( 0, 0,-1), 0x0);
  //updateCubeAo(testCube);

  dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  scene.add(dirLight);

  var axes = new THREE.AxisHelper( 100 );
	axes.material.depthTest = true;
	axes.material.transparent = true;
	axes.matrixAutoUpdate = false;
	axes.visible = true;
  scene.add(axes);
};

var onWindowResize = function( event ) {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize( window.innerWidth, window.innerHeight );
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

  // STATS
  stats = new Stats();
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.top = '00px';
  container.appendChild( stats.domElement );

	// CAMERA
	camera = new THREE.PerspectiveCamera( 45, canvasRatio, 0.1, 1000.0 );
	camera.position.set( -5,-5,-5 );
  camera.lookAt( new THREE.Vector3(2,2,2) );

  cameraControls = new THREE.OrbitAndPanControls(camera, renderer.domElement);
  cameraControls.target.set(2,2,2);

	fillScene();

  //var gui = new dat.GUI();
  //gui.add(this, 'factor0', 0, 15).onChange(applyAndUpdate);
  //gui.add(this, 'factor1', 0, 15).onChange(applyAndUpdate);
  //gui.add(this, 'factor2', 0, 15).onChange(applyAndUpdate);
  //gui.add(this, 'factor3', 0, 15).onChange(applyAndUpdate);

  onWindowResize();

  window.addEventListener( 'resize', onWindowResize, false );
};

var render = function() {
	var delta = clock.getDelta();

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
