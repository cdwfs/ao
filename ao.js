"use strict";


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
