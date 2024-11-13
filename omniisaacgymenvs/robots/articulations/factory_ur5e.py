# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema


class FactoryUR5e(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "UniversalRobot",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        self._usd_path = "../Models/ur5e_rl_libre.usd"
        add_reference_to_stage(self._usd_path, prim_path)
     
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "ur5e_link0/ur5e_joint1",
            "ur5e_link1/ur5e_joint2",
            "ur5e_link2/ur5e_joint3",
            "ur5e_link3/ur5e_joint4",
            "ur5e_link4/ur5e_joint5",
            "ur5e_link5/ur5e_joint6",
            "_hand/ur5e_finger_joint1", 
            "_hand/ur5e_finger_joint2",
        ]

        drive_type = ["angular"] * 6 + ["linear"] * 2
        default_dof_pos = [math.degrees(x) for x in [0.00e+00, -2.07e+00, 1.13e+00, -2.07e+00, -1.57e+00, 5.60e-01]] + [0.013, 0.013]
        stiffness = [40 * np.pi / 180] * 6 + [1000] * 2
        damping = [80 * np.pi / 180] * 6 + [20] * 2
        max_force = [87, 87, 87, 87, 87, 12, 200, 200]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.61, 2.61, 2.61]] + [0.2, 0.2]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )