var aoLookupTex    = THREE.ImageUtils.loadTexture( "cube_ao_lookup.png" );
var cubeUniforms = {
  texture:    { type: "t",  value: aoLookupTex },
};
var aoCubeMaterial = new THREE.ShaderMaterial( {
  uniforms:       cubeUniforms,
  vertexShader:   document.getElementById( 'vertexShader' ).textContent,
  fragmentShader: document.getElementById( 'fragmentShader' ).textContent,
} );


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


var createVoxelGrid = function(sizeX, sizeY, sizeZ) {
  var m_cells = [];
  var m_setCell = function(x,y,z,val) {
    if (x < 0 || x >= sizeX || y < 0 || y >= sizeY || z < 0 || z >= sizeZ) {
      throw {
        name: "BoundsError",
        message: "setCell: voxel grid index ["+x+","+y+","+z+"] is out of bounds."
      };
    }
    m_cells[z*sizeX*sizeY + y*sizeX + x] = val;
  };
  var m_getCell = function(x,y,z) {
    if (x < 0 || x >= sizeX || y < 0 || y >= sizeY || z < 0 || z >= sizeZ) {
      //throw {
      //  name: "BoundsError",
      //  message: "getCell: voxel grid index ["+x+","+y+","+z+"] is out of bounds."
      //};
      return null;
    }
    return m_cells[z*sizeX*sizeY + y*sizeX + x];
  };
  var m_clear = function() {
    m_cells.length = sizeX*sizeY*sizeZ;
    for(var cz=0; cz<sizeZ; cz += 1) {
      for(var cy=0; cy<sizeY; cy += 1) {
        for(var cx=0; cx<sizeX; cx += 1) {
          m_setCell(cx,cy,cz,false);
        }
      }
    }
  };

  // Offsets from a given vertex (at [0,0,0]) to cells which should be considered for
  // visibility testing in a particular octant. Each cell's visibility corresponds to
  // a bit in the bitmask used to index into m_aoFactors[].
  var m_cellOffsets = [
    {x:0,y:0,z:1},
    {x:0,y:1,z:0},
    {x:1,y:0,z:0},
  ];
  var m_aoFactors = [ // Lookup table mapping cell visibility bitmasks to AO factors
    1.000, // 000
    0.666, // 001
    0.666, // 010
    0.333, // 011
    0.666, // 100
    0.333, // 101
    0.333, // 110
    0.000, // 111
  ];

  var m_getAoFactorForOctant = function(vx,vy,vz, ox,oy,oz) {
    var flipX = (ox >= 0) ? function(n) { return n; } : function(n) { return -n-1; };
    var flipY = (oy >= 0) ? function(n) { return n; } : function(n) { return -n-1; };
    var flipZ = (oz >= 0) ? function(n) { return n; } : function(n) { return -n-1; };
    var bitMask = 0x00000000;
    var offset;
    if ( m_getCell(vx+flipX(0), vy+flipY(0), vz+flipZ(0)) ) {
      return 0.0; // fully occluded
    }
    // TODO: early-out if [0,0,0] is false in all octants. This vertex isn't part
    // of any visible cell.
    for(var iCell=0; iCell<m_cellOffsets.length; iCell += 1) {
      offset = m_cellOffsets[iCell];
      if ( m_getCell(vx+flipX(offset.x), vy+flipY(offset.y), vz+flipZ(offset.z)) ) {
        bitMask |= (1<<iCell);
      }
    }
    return m_aoFactors[bitMask];
  };
    }
  };

  m_clear();
  return {
    clear: m_clear,
    sizeX: function() { return sizeX; },
    sizeY: function() { return sizeY; },
    sizeZ: function() { return sizeZ; },
    setCell: m_setCell,
    getCell: m_getCell,
  };
};

var voxels = createVoxelGrid(4,4,4);
voxels.setCell(1,1,1,true);
voxels.getCell(1,1,1);
try {
  //console.log("[2,2,2] factor is: " + voxels.getAoFactor(2,3,2));
} catch(e) {
  window.alert(e.message);
}
