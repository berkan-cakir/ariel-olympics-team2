from ariel.body_phenotypes.robogen_lite.modules.core import CORE_DIMENSIONS, CoreModule


class CoreDiagCam(CoreModule):
    def __init__(self, index: int) -> None:
        super().__init__(index)

        core_name = self.module_type.name.lower()

        # Couldn't figure out how to rotate the exising cam, but this works too.
        self.spec.delete(self.spec.camera(f"{core_name}_mycamera"))
        self.spec.worldbody.bodies[0].add_camera(
            name=f"{core_name}_mycamera",
            pos=[0, 0, CORE_DIMENSIONS[0]-0.02],
            euler=[-90, 45, 180],
        )
