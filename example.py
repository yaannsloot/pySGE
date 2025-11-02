from SGE.core import *
from SGE.gl_ffp_pipeline import glFFPRenderingPipeline
import pygame
from pygame.locals import *
import time

def main():
    pygame.init()
    display = (1000, 1000)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL, vsync=1)

    pipeline = glFFPRenderingPipeline()
    
    scene = Scene()
    light = pipeline.Light()
    light2 = pipeline.Light(index=1)
    light.set_parent(scene)
    light2.set_parent(scene)
    light.local_transform.loc.update(-12, 0, -10)
    light.color.rgb = (0, 0, 1)
    light2.local_transform.loc.update(12, 0, 10)
    light3 = pipeline.Light(index=2)
    light3.set_parent(scene)
    light3.directional = True
    light3.color.rgb = (1, 0, 1)
    light3.intensity = 0.2
    light3.local_transform.rot.set_euler(-90, 0, 0)
    light.intensity = 2.5
    light2.intensity = 2.5
    light2.color.rgb = (1, 0, 0)
    print("LOAD teapot.obj")
    mesh = pipeline.Mesh.from_obj_file("teapot.obj")
    mesh_obj = Object(render_func=pipeline.MeshRenderer(mesh, flat_shading=False))
    mesh_obj.set_parent(scene)
    mesh_obj.local_transform.scale *= 0.18
    previous_time = time.perf_counter()
    smoothed_dt = 1/30
    s_alpha = 0.05

    camera = pipeline.Camera(viewport_dims=display)
    camera.set_parent(scene)
    scene.active_camera = camera
    camera.fov = 20
    camera.local_transform.rot.set_euler(-90, 180, 0)
    camera.local_transform.loc.y = 20
    camera.local_transform.loc.z = 1

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

        mesh_obj.rotate_local((0,0,1), 600*smoothed_dt)
        mesh_obj.rotate_local((0,1,0), 100*smoothed_dt)

        pipeline.pre()
        scene.render()
        pipeline.post() # Not needed for this pipeline but might be needed for a different one

        pygame.display.flip()

        #print(1/smoothed_dt) # uncomment for fps
main()
