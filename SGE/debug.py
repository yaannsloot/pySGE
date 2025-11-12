import pygame
from .pipeline import Mesh

def output_debug_uv(mesh: Mesh, material_id:int, output_resolution=1024, output_file="uv.png"):
    uv = mesh._uv_buf[material_id]
    img = pygame.Surface((output_resolution, output_resolution))
    img.fill((0,0,0))
    points, _ = uv.shape
    for i in range(0, points, 3):
        uv_tri = uv[i:i+3,:].copy()
        uv_tri[:,1] = 1 - uv_tri[:,1]
        uv_tri *= output_resolution
        pygame.draw.aalines(img, (255,255,255), True, uv_tri)
    pygame.image.save(img, output_file)