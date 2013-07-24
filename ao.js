/*jslint indent: 2, white: true, bitwise: true, continue: true */
function createAoGrid(sizeX, sizeY, sizeZ) {
  "use strict";
  var m_cells, m_verts, m_isDirty = false, m_octant, m_cellOffsets, m_aoFactors;
  // Offsets from a given vertex (at [0,0,0]) to cells which should be considered for
  // visibility testing in a particular octant. Each cell's visibility corresponds to
  // a bit in the bitmask used to index into m_aoFactors[].
  m_cellOffsets = [
    {x:0,y:0,z:1},
    {x:0,y:1,z:0},
    {x:1,y:0,z:0}
  ];
  // Lookup table mapping cell visibility bitmasks to AO factors.
  m_aoFactors = [ 
    1.000, // 000
    0.666, // 001
    0.666, // 010
    0.333, // 011
    0.666, // 100
    0.333, // 101
    0.333, // 110
    0.000  // 111
  ];
  // Fake enum
  m_octant = {
    nXnYnZ: 0,
    pXnYnZ: 1,
    nXpYnZ: 2,
    pXpYnZ: 3,
    nXnYpZ: 4,
    pXnYpZ: 5,
    nXpYpZ: 6,
    pXpYpZ: 7
  };

  function createArray3d(sizeX, sizeY, sizeZ, defaultVal) {
    var x, y, z, arr = [];
    for (z=0; z<sizeZ; z+=1) {
      arr[z] = [];
      for (y=0; y<sizeY; y+=1) {
        arr[z][y] = [];
        for (x=0; x<sizeX; x+=1) {
          arr[z][y][x] = defaultVal;
        }
      }
    }
    return arr;
  }

  m_cells = createArray3d(sizeX,   sizeY,   sizeZ,   false); // is each cell filled? (true/false)
  m_verts = createArray3d(sizeX+1, sizeY+1, sizeZ+1, null);
  function m_setCell(cx,cy,cz,val) {
    if (cx < 0 || cx >= sizeX || cy < 0 || cy >= sizeY || cz < 0 || cz >= sizeZ) {
      throw {
        name: "BoundsError",
        message: "setCell: voxel cell index ["+cx+","+cy+","+cz+"] is out of bounds."
      };
    }
    m_cells[cz][cy][cx] = val;
    m_isDirty = true;
    return this;
  }
  function m_getCell(cx,cy,cz) {
    if (cx < 0 || cx >= sizeX || cy < 0 || cy >= sizeY || cz < 0 || cz >= sizeZ) {
      return null; // out-of-bounds access are legal; there's nothing there.
    }
    return m_cells[cz][cy][cx];
  }
  function m_getAoFactor(vx, vy, vz, dir) {
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
  }

  function m_clear() {
	var cx, cy, cz, vx, vy, vz;
    for(cz=0; cz<sizeZ; cz += 1) {
      for(cy=0; cy<sizeY; cy += 1) {
        for(cx=0; cx<sizeX; cx += 1) {
          m_setCell(cx, cy, cz, false);
        }
      }
    }
    for(vz=0; vz<=sizeZ; vz += 1) {
      for(vy=0; vy<=sizeY; vy += 1) {
        for(vx=0; vx<=sizeX; vx += 1) {
          m_verts[vz][vy][vx] = null;
        }
      }
    }
    m_isDirty = false;
  }

  function flipToOctant(offset,octant) {
    return {
      x: (octant & 0x1) ? offset.x : (-1-offset.x),
      y: (octant & 0x2) ? offset.y : (-1-offset.y),
      z: (octant & 0x4) ? offset.z : (-1-offset.z)
    };
  }
  function m_getAoFactorForOctant(gridVert, octant) {
    // ASSUMPTION: the octant is not fully occluded -- cell at flipped offset [0,0,0]
    // is 0/null,
    var bitMask = 0x00000000, iCell, offset;
    for(iCell=0; iCell<m_cellOffsets.length; iCell += 1) {
      offset = flipToOctant(m_cellOffsets[iCell],octant);
      if ( m_getCell(gridVert.x+offset.x, gridVert.y+offset.y, gridVert.z+offset.z) ) {
        bitMask |= (1<<iCell);
      }
    }
    return m_aoFactors[bitMask];
  }
  function m_updateGridAo() {
	var vx, vy, vz, gridVert, isCellFilled, anyCellFilled, results, iOct, of;
    if (!m_isDirty) {
      return this; // No updates necessary
    }
    // Iterate over vertices
    for(vz=0; vz<=sizeZ; vz += 1) {
      for(vy=0; vy<=sizeY; vy += 1) {
        for(vx=0; vx<=sizeX; vx += 1) {
          gridVert = {x:vx,y:vy,z:vz};
          // Query cells at offset [0,0,0] in all octants.
          // Keep track of which are occluded; these are the cells we'll need to update.
          // at the end of the loop.
          isCellFilled = [];
		  isCellFilled.length = 8;
          isCellFilled[m_octant.nXnYnZ] = m_getCell(vx-1,vy-1,vz-1);
          isCellFilled[m_octant.pXnYnZ] = m_getCell(vx,  vy-1,vz-1);
          isCellFilled[m_octant.nXpYnZ] = m_getCell(vx-1,vy,  vz-1);
          isCellFilled[m_octant.pXpYnZ] = m_getCell(vx,  vy,  vz-1);
          isCellFilled[m_octant.nXnYpZ] = m_getCell(vx-1,vy-1,vz  );
          isCellFilled[m_octant.pXnYpZ] = m_getCell(vx,  vy-1,vz  );
          isCellFilled[m_octant.nXpYpZ] = m_getCell(vx-1,vy,  vz  );
          isCellFilled[m_octant.pXpYpZ] = m_getCell(vx,  vy,  vz  );
          anyCellFilled = false;
		  for(iOct=0; iOct<8; iOct+=1) {
			anyCellFilled = anyCellFilled || isCellFilled[iOct];
		  }
          if (!anyCellFilled) {
            m_verts[vz][vy][vx] = null; // no neighboring cells are filled, so AO is undefined.
            continue;
          }
          // Compute full AO visibility test for all unoccluded octants.
          // Cache results for each of the six axis-aligned half-spaces: +x, -x, etc.
          results = {pX:0,nX:0, pY:0,nY:0, pZ:0,nZ:0};
          for(iOct=0; iOct<8; iOct += 1) {
            of = 0.25 * (isCellFilled[iOct] ? 0.0 : m_getAoFactorForOctant(gridVert, iOct));
            if (iOct & 0x1) {
			  results.pX += of;
			} else {
			  results.nX += of;
			}
            if (iOct & 0x2) {
			  results.pY += of;
			} else {
			  results.nY += of;
			}
            if (iOct & 0x4) {
			  results.pZ += of;
			} else {
			  results.nZ += of;
			}
          }
          m_verts[vz][vy][vx] = results;
        }
      }
    }
    m_isDirty = false;
    return this;
  }

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
    update: m_updateGridAo
  };
}
