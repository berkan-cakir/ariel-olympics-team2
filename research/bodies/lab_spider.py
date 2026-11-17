from ariel.body_phenotypes.robogen_lite.modules.core import CORE_DIMENSIONS, CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import body_spider45


def spider_with_blocks_diag() -> CoreModule:
    core = body_spider45()

    cam_name = core.spec.cameras[0].name
    core.spec.delete(core.spec.camera(cam_name))
    core.spec.worldbody.bodies[0].add_camera(
        name=cam_name,
        pos=[0, 0, CORE_DIMENSIONS[0]-0.02],
        euler=[-90, 45, 180],
    )

    return core
