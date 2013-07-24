"use strict";

var aoLookupTex = THREE.ImageUtils.loadTexture( "cube_ao_lookup.png" );
var cubeUniforms = {
  texture:    { type: "t",  value: aoLookupTex },
};
var aoCubeMaterial = new THREE.ShaderMaterial( {
  uniforms:       cubeUniforms,
  vertexShader:   document.getElementById( 'vertexShader' ).textContent,
  fragmentShader: document.getElementById( 'fragmentShader' ).textContent,
} );

var createAoCubeGeometry = function() {
  var Octant = {
    nXnYnZ: 0,
    pXnYnZ: 1,
    nXpYnZ: 2,
    pXpYnZ: 3,
    nXnYpZ: 4,
    pXnYpZ: 5,
    nXpYpZ: 6,
    pXpYpZ: 7,
  };

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

  for(var iFace=0; iFace<6; iFace += 1) {
    cubeGeom.faceVertexUvs[0].push( [new THREE.Vector2(0,0), new THREE.Vector2(1,0), new THREE.Vector2(1,1), new THREE.Vector2(0,1)] );
  }

  return cubeGeom;
};

var createVoxelGrid = function(sizeX,sizeY,sizeZ) {
  var m_urCubeGeom = createAoCubeGeometry();
  var createArray3d = function(sizeX,sizeY, sizeZ, defaultVal) {
    var arr = [];
    for(var z=0; z<sizeZ; ++z) {
      arr[z] = [];
      for(var y=0; y<sizeY; ++y) {
        arr[z][y] = [];
        for(var x=0; x<sizeX; ++x) {
          arr[z][y][x] = defaultVal;
        }
      }
    }
    return arr;
  };

  var m_grid = createArray3d(sizeX,sizeY,sizeZ,-1);
  var m_cubes = [];
  var m_aoGrid = createAoGrid(sizeX,sizeY,sizeZ);

  var m_addCube = function(cx,cy,cz) {
    if (cx < 0 || cx >= sizeX || cy < 0 || cy >= sizeY || cz < 0 || cz >= sizeZ) {
      return this; // out of bounds silently fails
    }
    if (m_grid[cz][cy][cx] != -1) {
      throw {
        name: "VoxelError",
        message: "Voxel cell ["+cx+","+cy+","+cz+"] is already filled!"
      };
    }
    m_aoGrid.setCell(cx,cy,cz,true);
    var newCube = new THREE.Mesh(m_urCubeGeom.clone(), aoCubeMaterial);
    newCube.position.set(cx+0.5,cy+0.5,cz+0.5);
    m_grid[cz][cy][cx] = m_cubes.length;
    m_cubes.push(newCube);
    scene.add(newCube);
    return this;
  };

  var m_removeCube = function(cx,cy,cz) {
    if (cx < 0 || cx >= sizeX || cy < 0 || cy >= sizeY || cz < 0 || cz >= sizeZ) {
      return this; // out of bounds silently fails
    }
    if (m_grid[cz][cy][cx] == -1) {
      throw {
        name: "VoxelError",
        message: "Voxel cell ["+cx+","+cy+","+cz+"] is already empty!"
      };
    }
    m_aoGrid.setCell(cx,cy,cz,false);
    var cubeIndex = m_grid[cz][cy][cx];
    m_grid[cz][cy][cx] = -1;
    var oldCube = m_cubes[cubeIndex];
    var lastCube = m_cubes.pop();
    if (oldCube != lastCube) {
      m_cubes[cubeIndex] = lastCube;
    }
    scene.remove(oldCube);
    return this;
  };

  var m_updateCubeAo = function(cx,cy,cz) {
    if (m_grid[cz][cy][cx] == -1) {
      throw {
        name: "VoxelError",
        message: "Voxel cell ["+cx+","+cy+","+cz+"] is already empty!"
      };
    }
    var cubeGeom = m_cubes[ m_grid[cz][cy][cx] ].geometry;
    var tx = cx+0.5, ty = cy+0.5, tz = cz+0.5;
    var faceNormals = ["nX","pX", "nY", "pY", "nZ", "pZ"];
    var faceAoFactors = new Array(4);
    for(var iFace=0; iFace<6; ++iFace) {
      // Look up AO factor at all four corners
      var face = cubeGeom.faces[iFace];
      var normDir = faceNormals[iFace];
      var faceIndices = [face.a, face.b, face.c, face.d];
      for(var iCorner=0; iCorner<4; ++iCorner) {
        var vert = cubeGeom.vertices[faceIndices[iCorner]];
        faceAoFactors[iCorner] = m_aoGrid.getAoFactor(tx+vert.x, ty+vert.y, tz+vert.z, normDir) * 0xF;
      }
      //console.log("face "+iFace+" factors: "+faceAoFactors[0]+" "+faceAoFactors[1]+" "+faceAoFactors[2]+" "+faceAoFactors[3])
      // Combine AO factors into new UVs for this face
      var lutX = (1/1024) + ((faceAoFactors[0] & 0xF) | ((faceAoFactors[1] & 0xF) << 4)) / 256;
      var lutY = (1/1024) + ((faceAoFactors[3] & 0xF) | ((faceAoFactors[2] & 0xF) << 4)) / 256;
      cubeGeom.faceVertexUvs[0][iFace][0].set( lutX+0,      1.0 - (lutY+0)       );
      cubeGeom.faceVertexUvs[0][iFace][1].set( lutX+(1/512),1.0 - (lutY+0)       );
      cubeGeom.faceVertexUvs[0][iFace][2].set( lutX+(1/512),1.0 - (lutY+(1/512)) );
      cubeGeom.faceVertexUvs[0][iFace][3].set( lutX+0,      1.0 - (lutY+(1/512)) );
    }
    cubeGeom.uvsNeedUpdate = true;
  };

  var m_update = function() {
    if (m_aoGrid.isDirty()) {
      m_aoGrid.update();
      for(var cz=0; cz<sizeZ; ++cz) {
        for(var cy=0; cy<sizeY; ++cy) {
          for(var cx=0; cx<sizeX; ++cx) {
            if (m_grid[cz][cy][cx] != -1) {
              m_updateCubeAo(cx,cy,cz);
            }
          }
        }
      }
    }
  }

  return {
    sizeX: function() { return sizeX; },
    sizeY: function() { return sizeY; },
    sizeZ: function() { return sizeZ; },
    cubes: m_cubes,
    addCube: m_addCube,
    removeCube: m_removeCube,
    update: m_update,
  };
};

var rolloverCubePos = new THREE.Vector3();
var normalMatrix = new THREE.Matrix3();
var updateRolloverCube = function( intersector ) {
  normalMatrix.getNormalMatrix( intersector.object.matrixWorld );

  rolloverCubePos.copy( intersector.face.normal );
  rolloverCubePos.applyMatrix3( normalMatrix ).normalize();
  rolloverCubePos.multiplyScalar(0.5);

  rolloverCubePos.add(intersector.point);

  rolloverCubePos.x = Math.floor( rolloverCubePos.x / 1 ) * 1 + 0.5;
  rolloverCubePos.y = Math.floor( rolloverCubePos.y / 1 ) * 1 + 0.5;
  rolloverCubePos.z = Math.floor( rolloverCubePos.z / 1 ) * 1 + 0.5;
  if (rolloverCubePos.x < 0 || rolloverCubePos.x > voxelGrid.sizeX() ||
      rolloverCubePos.y < 0 || rolloverCubePos.y > voxelGrid.sizeY() ||
      rolloverCubePos.z < 0 || rolloverCubePos.z > voxelGrid.sizeZ()) {
    rolloverCubeMesh.material.color.set(0xFF0000);
  } else {
    rolloverCubeMesh.material.color.set(0x00FF00);
  }
  rolloverCubeMesh.material.visible = true;
}


var container;
var stats;
var scene, camera, renderer;
var cameraControls, effectController;
var clock = new THREE.Clock();
var rolloverCubeMesh;
var voxelGrid;
var cubes = [];
var dirLight;
var mouse = new THREE.Vector3(0,10000,0.5);
var projector = new THREE.Projector();
var isShiftDown = false;
var isCtrlDown = false;

var fillScene = function() {
  scene = new THREE.Scene();
  //scene.fog = new THREE.Fog( 0x808080, 5, 5.5 );

  var rolloverGeom = new THREE.CubeGeometry( 1, 1, 1 );
  var rolloverMat  = new THREE.MeshBasicMaterial( { color: 0xff0000, opacity: 0.5, transparent: true } );
  rolloverCubeMesh = new THREE.Mesh( rolloverGeom, rolloverMat );
  scene.add( rolloverCubeMesh );

  voxelGrid = createVoxelGrid(32,8,32);
  voxelGrid.addCube(0,0,0)
           .addCube(1,0,0)
           .addCube(0,0,1)
           .addCube(1,0,1)
           .addCube(1,1,1)
           .addCube(1,2,1);
  voxelGrid.update();

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
var onDocumentMouseMove = function( event ) {
  event.preventDefault();
  mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
  mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
};
var onDocumentKeyDown = function( event ) {
  switch( event.keyCode ) {
    case 16: isShiftDown = true; break;
    case 17: isCtrlDown = true; break;
  }
};

var onDocumentKeyUp = function( event ) {
  switch ( event.keyCode ) {
    case 16: isShiftDown = false; break;
    case 17: isCtrlDown = false; break;
  }
};
var onDocumentMouseDown = function( event ) {
  var raycaster = projector.pickingRay( mouse.clone(), camera );
  var intersects = raycaster.intersectObjects( voxelGrid.cubes );
  if ( intersects.length > 0 ) {
    var intersector = intersects[0];
    if ( isCtrlDown ) {
      // delete cube
      var pos = intersector.object.position;
      voxelGrid.removeCube(Math.floor(pos.x), Math.floor(pos.y), Math.floor(pos.z));
    } else if (isShiftDown) {
      // create cube
      updateRolloverCube( intersector );
      var pos = rolloverCubePos;
      voxelGrid.addCube(Math.floor(pos.x), Math.floor(pos.y), Math.floor(pos.z));
    }
      voxelGrid.update(); // Update AO to reflect changes.
  }
};
var init = function() {
	var canvasWidth = 800;//window.innerWidth;
	var canvasHeight = 600;//window.innerHeight;
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
	camera.position.set( 20,20,20 );
  camera.lookAt( new THREE.Vector3(0,0,0) );

  cameraControls = new THREE.OrbitAndPanControls(camera, renderer.domElement);
  cameraControls.target.set(0,0,0);

	fillScene();

  //var gui = new dat.GUI();
  //gui.add(this, 'factor0', 0, 15).onChange(applyAndUpdate);
  //gui.add(this, 'factor1', 0, 15).onChange(applyAndUpdate);
  //gui.add(this, 'factor2', 0, 15).onChange(applyAndUpdate);
  //gui.add(this, 'factor3', 0, 15).onChange(applyAndUpdate);

  onWindowResize();

  window.addEventListener( 'resize', onWindowResize, false );
  document.addEventListener( 'mousemove', onDocumentMouseMove, false );
  document.addEventListener( 'mousedown', onDocumentMouseDown, false );
  document.addEventListener( 'keydown', onDocumentKeyDown, false );
  document.addEventListener( 'keyup', onDocumentKeyUp, false );
};

var render = function() {
	var delta = clock.getDelta();

  THREE.AxisHelper(100);

  cameraControls.update(delta);

  // find intersections
  var raycaster = projector.pickingRay( mouse.clone(), camera );
  var intersects = raycaster.intersectObjects( voxelGrid.cubes );
  if ( intersects.length > 0 ) {
    var intersector = intersects[0];
    if ( intersector ) {
      updateRolloverCube( intersector );
      rolloverCubeMesh.position = rolloverCubePos;
    }
  }
  else {
    rolloverCubeMesh.material.visible = false;
  }

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
