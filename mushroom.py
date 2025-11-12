from SGE.core import *
from SGE.gl_ffp_pipeline import glFFPRenderingPipeline
import pygame
from pygame.locals import *
import time

def main():
    display = Vector2(1920, 1080)
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    pygame.display.set_mode((int(display.x), int(display.y)), DOUBLEBUF|OPENGL, vsync=1)

    pipeline = glFFPRenderingPipeline()

    pipeline.global_ambient_light = (0.1, 0.1, 0.1)
    pipeline.lighting = True
    
    scene = Scene()
    camera = pipeline.Camera(
        viewport_dims=display,
        ortho_size=1,
        near=1,
        far=100,
        orthographic=False)
    camera.set_parent(scene)
    scene.active_camera = camera
    
    diffuse = pipeline.Texture.from_file("assets/diffuse_mush.png", dynamic=False)
    emission = pipeline.Texture.from_file("assets/emission_mush.png", dynamic=False)
    material = pipeline.Material(base_color=diffuse, emission=emission)
    material.shininess = 32
    mush_mesh = pipeline.Mesh.from_obj_file("assets/Mushroom.obj", dynamic=False)
    mushroom = Object(render_func=pipeline.MeshRenderer(mush_mesh, [material, material]))
    mushroom.set_parent(scene)
    mushroom.local_transform.scale = 10
    mushroom.local_transform.loc.y = -0.4

    light = pipeline.Light()
    light.set_parent(scene)
    light.local_transform.loc.xyz = (3, 3, 3)
    light.intensity = 0.4
    light.color.rgb = (0.8, 0.1, 0.1)

    light2 = pipeline.Light(index=1)
    light2.set_parent(scene)
    light2.local_transform.loc.xyz = (-3, 3, 3)
    light2.intensity = 0.4
    light2.color.rgb = (0.1, 0.1, 0.8)

    camera.local_transform.loc.z = 4
    previous_time = time.perf_counter()
    smoothed_dt = 1/30
    s_alpha = 0.05
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        current_time = time.perf_counter()
        dt = current_time - previous_time
        dt = min(dt, 1/30.0)
        previous_time = current_time
        smoothed_dt = smoothed_dt * (1 - s_alpha) + dt * s_alpha

        mushroom.rotate_local((0,1,0), 20*smoothed_dt)

        pipeline.pre()
        scene.render()
        pipeline.post() # Not needed for this pipeline but might be needed for a different one

        pygame.display.flip()


if __name__ == "__main__":
    main()