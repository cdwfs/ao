#include <cstdio>
#include <cstdint>
#include <cstdlib>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int computeOctantOcclusionLUT(void);

int computeQuadAoTexture(void)
{
	const int texWidth  = 512;
	const int texHeight = 512;
	uint8_t *pixels = (uint8_t*)malloc(texWidth*texHeight*sizeof(uint8_t));
	for(int iY=0; iY<256; ++iY)
	{
		for(int iX=0; iX<256; ++iX)
		{
			pixels[(2*iY+0)*texWidth+(2*iX+0)] = ((iX>>0) & 0x0F) << 4;
			pixels[(2*iY+0)*texWidth+(2*iX+1)] = ((iX>>4) & 0x0F) << 4;
			pixels[(2*iY+1)*texWidth+(2*iX+0)] = ((iY>>0) & 0x0F) << 4;
			pixels[(2*iY+1)*texWidth+(2*iX+1)] = ((iY>>4) & 0x0F) << 4;
		}
	}

	const char *texFilename = "cube_ao_lookup.png";
	printf("Writing '%s'...\n", texFilename);
	int32_t writeError = stbi_write_png(texFilename, texWidth, texHeight, 1, pixels, texWidth*sizeof(uint8_t));
	free(pixels);
	if (writeError == 0)
	{
		printf("Error writing output image '%s'\n", texFilename);
		return -1;
	}
	return 0;
}


int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	computeQuadAoTexture();
}
