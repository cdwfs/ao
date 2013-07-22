"use strict";

var createAoGrid = function(sizeX, sizeY, sizeZ, defaultVal) {
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

  var m_cells = createArray3d(sizeX+0, sizeY+0, sizeZ+0, false); // is each cell filled? (true/false)
  var m_verts = createArray3d(sizeX+1, sizeY+1, sizeZ+1, null);
  var m_isDirty = false;
  var m_setCell = function(cx,cy,cz,val) {
    if (cx < 0 || cx >= sizeX || cy < 0 || cy >= sizeY || cz < 0 || cz >= sizeZ) {
      throw {
        name: "BoundsError",
        message: "setCell: voxel cell index ["+cx+","+cy+","+cz+"] is out of bounds."
      };
    }
    m_cells[cz][cy][cx] = val;
    m_isDirty = true;
    return this;
  };
  var m_getCell = function(cx,cy,cz) {
    if (cx < 0 || cx >= sizeX || cy < 0 || cy >= sizeY || cz < 0 || cz >= sizeZ) {
      return null; // out-of-bounds access are legal; there's nothing there.
    }
    return m_cells[cz][cy][cx];
  };
  var m_setAoFactor = function(vx,vy,vz,dir,aoFactor) {
    if (vx < 0 || vx > sizeX || vy < 0 || vy > sizeY || vz < 0 || vz > sizeZ) {
      throw {
        name: "BoundsError",
        message: "setCell: voxel cell index ["+vx+","+vy+","+vz+"] is out of bounds."
      };
    }
    m_verts[vz][vy][vx][dir] = aoFactor;
    return this;
  };
  var m_getAoFactor = function(vx,vy,vz,dir) {
    if (vx < 0 || vx > sizeX || vy < 0 || vy > sizeY || vz < 0 || vz > sizeZ) {
      return null; // out-of-bounds access are legal; there's nothing there.
    }
    if (m_isDirty) {
      throw {
        name: "DirtyError",
        message: "getAoFactor: AO grid is dirty! AO factor may be out of date; call update first!"
      };
    }
    return m_verts[vz][vy][vx][dir];
  };

  var m_clear = function() {
    for(var cz=0; cz<sizeZ; cz += 1) {
      for(var cy=0; cy<sizeY; cy += 1) {
        for(var cx=0; cx<sizeX; cx += 1) {
          m_setCell(cx,cy,cz,false);
        }
      }
    }
    for(var vz=0; vz<=sizeZ; vz += 1) {
      for(var vy=0; vy<=sizeY; vy += 1) {
        for(var vx=0; vx<=sizeX; vx += 1) {
          m_verts[vz][vy][vx] = null;
        }
      }
    }
    m_isDirty = false;
  };

  // Offsets from a given vertex (at [0,0,0]) to cells which should be considered for
  // visibility testing in a particular octant. Each cell's visibility corresponds to
  // a bit in the bitmask used to index into m_aoFactors[].
  var m_cellOffsets = [
    {x:0,y:0,z:1},
    {x:0,y:1,z:0},
    {x:1,y:0,z:0},
  ];
  var m_aoFactors = [ // Lookup table mapping cell visibility bitmasks to AO factors.
    1.000, // 000
    0.666, // 001
    0.666, // 010
    0.333, // 011
    0.666, // 100
    0.333, // 101
    0.333, // 110
    0.000, // 111
  ];
  var flipToOctant = function(offset,octant) {
    return {
      x: (octant & 0x1) ? offset.x : (-1-offset.x),
      y: (octant & 0x2) ? offset.y : (-1-offset.y),
      z: (octant & 0x4) ? offset.z : (-1-offset.z),
    };
  };
  var m_getAoFactorForOctant = function(gridVert, octant) {
    // ASSUMPTION: the octant is not fully occluded -- cell at flipped offset [0,0,0]
    // is 0/null,
    var bitMask = 0x00000000;
    for(var iCell=0; iCell<m_cellOffsets.length; iCell += 1) {
      var offset = flipToOctant(m_cellOffsets[iCell],octant);
      if ( m_getCell(gridVert.x+offset.x, gridVert.y+offset.y, gridVert.z+offset.z) ) {
        bitMask |= (1<<iCell);
      }
    }
    return m_aoFactors[bitMask];
  };
  var m_updateGridAo = function() {
    if (!m_isDirty) {
      return this; // No updates necessary
    }
    // Iterate over vertices
    for(var vz=0; vz<=sizeZ; vz += 1) {
      for(var vy=0; vy<=sizeY; vy += 1) {
        for(var vx=0; vx<=sizeX; vx += 1) {
          var gridVert = {x:vx,y:vy,z:vz};
          // Query cells at offset [0,0,0] in all octants.
          // Keep track of which are occluded; these are the cells we'll need to update.
          // at the end of the loop.
          var isCellFilled = new Array(8);
          var anyCellFilled = false;
          anyCellFilled = (isCellFilled[Octant.nXnYnZ] = m_getCell(vx-1,vy-1,vz-1)) || anyCellFilled;
          anyCellFilled = (isCellFilled[Octant.pXnYnZ] = m_getCell(vx-0,vy-1,vz-1)) || anyCellFilled;
          anyCellFilled = (isCellFilled[Octant.nXpYnZ] = m_getCell(vx-1,vy-0,vz-1)) || anyCellFilled;
          anyCellFilled = (isCellFilled[Octant.pXpYnZ] = m_getCell(vx-0,vy-0,vz-1)) || anyCellFilled;
          anyCellFilled = (isCellFilled[Octant.nXnYpZ] = m_getCell(vx-1,vy-1,vz-0)) || anyCellFilled;
          anyCellFilled = (isCellFilled[Octant.pXnYpZ] = m_getCell(vx-0,vy-1,vz-0)) || anyCellFilled;
          anyCellFilled = (isCellFilled[Octant.nXpYpZ] = m_getCell(vx-1,vy-0,vz-0)) || anyCellFilled;
          anyCellFilled = (isCellFilled[Octant.pXpYpZ] = m_getCell(vx-0,vy-0,vz-0)) || anyCellFilled;
          var results = {pX:0,nX:0, pY:0,nY:0, pZ:0,nZ:0};
          if (!anyCellFilled) {
            m_verts[vz][vy][vx] = null; // no neighboring cells are filled, so AO is undefined.
            continue;
          }
          // Compute full AO visibility test for all unoccluded octants.
          // Cache results for each of the six axis-aligned half-spaces: +x, -x, etc.
          for(var iOct=0; iOct<8; iOct += 1) {
            var of = 0.25 * (isCellFilled[iOct] ? 0.0 : m_getAoFactorForOctant(gridVert, iOct));
            (iOct & 0x1) ? (results.pX += of) : (results.nX += of);
            (iOct & 0x2) ? (results.pY += of) : (results.nY += of);
            (iOct & 0x4) ? (results.pZ += of) : (results.nZ += of);
          }
          m_verts[vz][vy][vx] = results;
        }
      }
    }
    m_isDirty = false;
    return this;
  };

  return {
    sizeX: function() { return sizeX; },
    sizeY: function() { return sizeY; },
    sizeZ: function() { return sizeZ; },
    isDirty: function() { return m_isDirty; },
    clear: m_clear,
    setCell: m_setCell,
    getCell: m_getCell,
    //setAoFactor: m_setAoFactor,
    getAoFactor: m_getAoFactor,
    update: m_updateGridAo,
  };
};

