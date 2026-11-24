from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule


def champion() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)

    fr_leg = HingeModule(index=1)
    fr_leg.rotate(-90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=fr_leg.body,
        prefix="fr_leg_",
    )

    fr_foot = BrickModule(index=9)
    fr_leg.sites[ModuleFaces.FRONT].attach_body(
        body=fr_foot.body,
        prefix="fr_foot_",
    )

    fl_leg = HingeModule(index=2)
    fl_leg.rotate(90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=fl_leg.body,
        prefix="fl_leg_",
    )

    fl_foot = BrickModule(index=8)
    fl_leg.sites[ModuleFaces.FRONT].attach_body(
        body=fl_foot.body,
        prefix="fl_foot_",
    )

    neck = HingeModule(index=3)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=neck.body,
        prefix="neck_",
    )
    torso = BrickModule(index=5)
    neck.sites[ModuleFaces.FRONT].attach_body(
        body=torso.body,
        prefix="torso_",
    )
    spine1 = HingeModule(index=6)
    torso.sites[ModuleFaces.FRONT].attach_body(
        body=spine1.body,
        prefix="spine1_",
    )
    spine2 = HingeModule(index=7)
    spine1.sites[ModuleFaces.FRONT].attach_body(
        body=spine2.body,
        prefix="spine2_",
    )
    pelvis = BrickModule(index=10)
    spine2.sites[ModuleFaces.FRONT].attach_body(
        body=pelvis.body,
        prefix="pelvis_",
    )
    bl_leg = HingeModule(index=11)
    bl_leg.rotate(-90)
    pelvis.sites[ModuleFaces.RIGHT].attach_body(
        body=bl_leg.body,
        prefix="bl_leg_",
    )
    bl1_foot = BrickModule(index=17)
    bl1_foot.rotate(90)
    bl_leg.sites[ModuleFaces.FRONT].attach_body(
        body=bl1_foot.body,
        prefix="bl1_foot_",
    )
    bl2_foot = BrickModule(index=18)
    bl1_foot.sites[ModuleFaces.BOTTOM].attach_body(
        body=bl2_foot.body,
        prefix="bl2_foot_",
    )
    bl3_foot = BrickModule(index=19)
    bl2_foot.sites[ModuleFaces.FRONT].attach_body(
        body=bl3_foot.body,
        prefix="bl3_foot_",
    )
    br_leg = HingeModule(index=12)
    br_leg.rotate(90)
    pelvis.sites[ModuleFaces.LEFT].attach_body(
        body=br_leg.body,
        prefix="br_leg_",
    )
    br1_foot = BrickModule(index=13)
    br1_foot.rotate(-90)
    br_leg.sites[ModuleFaces.FRONT].attach_body(
        body=br1_foot.body,
        prefix="br1_foot_",
    )
    br2_foot = BrickModule(index=15)
    br1_foot.sites[ModuleFaces.BOTTOM].attach_body(
        body=br2_foot.body,
        prefix="br2_foot_",
    )
    br3_foot = BrickModule(index=16)
    br2_foot.sites[ModuleFaces.FRONT].attach_body(
        body=br3_foot.body,
        prefix="br3_foot_",
    )
    return core
