
### How to RUN

All commands have to be executed from `OmniIsaacGymEnvs/omniisaacgymenvs`

To train a policy for the first time, run:
```
PYTHON_PATH scripts/rlgames_train.py task=FactoryUR5eNutBoltUnScrew num_envs=256 

```
To test a trained checkpoint, run:
```
PYTHON_PATH scripts/rlgames_train.py task=FactoryUR5eNutBoltUnScrew num_envs=256 checkpoint=/[your_path]/OmniIsaacGymEnvs/omniisaacgymenvs/runs/WEE/nn/WEE.pth test=True

```

The dof paths have to be updated in OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/factory_ur5e.py and in `OmniIsaacGymEnvs/omniisaacgymenvs/tasks/myfactory/factory_base.py`

Also dof paths of fingers have to be changed in `factory_ur5e_view.py` (robots/articulations/views/)


## HAND-e / 8 DOF 
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

## Robotiq_85 / 12 DOF
dof_paths = [
            "base_link_inertia/shoulder_pan_joint",
            "shoulder_link/shoulder_lift_joint",
            "upper_arm_link/elbow_joint",
            "forearm_link/wrist_1_joint",
            "wrist_1_link/wrist_2_joint",
            "wrist_2_link/wrist_3_joint",
            "robotiq/robotiq_arg2f_base_link/finger_joint",
            "robotiq/robotiq_arg2f_base_link/left_inner_knuckle_joint",
            "robotiq/robotiq_arg2f_base_link/right_inner_knuckle_joint",
            "robotiq/robotiq_arg2f_base_link/right_outer_knuckle_joint",
            "robotiq/left_outer_finger/left_inner_finger_joint",
            "robotiq/right_outer_finger/right_inner_finger_joint",
        ]

dof_paths = [
            "myur5e/base_link/shoulder_pan_joint",
            "myur5e/shoulder_link/shoulder_lift_joint",
            "myur5e/upper_arm_link/elbow_joint",
            "myur5e/forearm_link/wrist_1_joint",
            "myur5e/wrist_1_link/wrist_2_joint",
            "myur5e/wrist_2_link/wrist_3_joint",
            "myur5e/wrist_3_link/robotiq_85_left_inner_knuckle_joint",
            "myur5e/wrist_3_link/robotiq_85_left_knuckle_joint",
            "myur5e/wrist_3_link/robotiq_85_right_inner_knuckle_joint",
            "myur5e/wrist_3_link/robotiq_85_right_knuckle_joint",
            "myur5e/robotiq_85_left_inner_knuckle_link/robotiq_85_left_finger_tip_joint",
            "myur5e/robotiq_85_right_inner_knuckle_link/robotiq_85_right_finger_tip_joint",
        ]
