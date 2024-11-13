
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


