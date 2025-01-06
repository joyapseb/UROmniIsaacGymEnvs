# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for nut-bolt screw task using UR5e.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
PYTHON_PATH omniisaacgymenvs/scripts/rlgames_train.py task=FactoryUR5eNutBoltUnScrew
"""


import hydra
import math
import omegaconf
import torch
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import omni.kit

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.simulation_context import SimulationContext
import omniisaacgymenvs.tasks.myfactory.factory_control as fc
from omniisaacgymenvs.tasks.myfactory.factory_env_nut_bolt import myFactoryEnvNutBolt
from omniisaacgymenvs.tasks.myfactory.factory_schema_class_task import myFactoryABCTask
from omniisaacgymenvs.tasks.myfactory.factory_schema_config_task import (
    myFactorySchemaConfigTask,
)
import rclpy
import threading
from omniisaacgymenvs.sim2real.ur5e import UR5eTraj


class myFactoryTaskNutBoltUnScrew(myFactoryEnvNutBolt, myFactoryABCTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        """Initialize environment superclass. Initialize instance variables."""
        rclpy.init()

        super().__init__(name, sim_config, env)

        self.flag= False

        self.randomization_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._get_task_yaml_params()

        sim2real_config = self._task_cfg['sim2real']
        if sim2real_config['enabled'] and self._num_envs == 1:
            self.real_world_ur5e = UR5eTraj()
            print("init connection")
            self.ros_thread = threading.Thread(target=self.spin_ros)
            self.ros_thread.start()
        return
    
    def spin_ros(self):
        try:
            rclpy.spin(self.real_world_ur5e)
        except KeyboardInterrupt:
            pass
        finally:
            self.real_world_ur5e.destroy_node()
            rclpy.shutdown()


    def _get_task_yaml_params(self) -> None:
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=myFactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self._task_cfg)
        print(omegaconf.OmegaConf.to_yaml(self.cfg_task))
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        asset_info_path = "../tasks/myfactory/yaml/factory_asset_info_nut_bolt.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt[""][""][""]["tasks"][
            "myfactory"
        ][
            "yaml"
        ]  # strip superfluous nesting

        ppo_path = "train/FactoryUR5eNutBoltUnScrewPPO.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def post_reset(self) -> None:
        """Reset the world. Called only once, before simulation begins."""

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        self.acquire_base_tensors()
        self._acquire_task_tensors()

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # Reset all envs
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        asyncio.ensure_future(
            self.reset_idx_async(indices)
        )
        # self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)


    def _acquire_task_tensors(self) -> None:
        """Acquire tensors."""

        target_heights = (
            self.cfg_base.env.table_height
            + self.bolt_head_heights *1.4
            + self.nut_heights * 1.2
        )

        self.targ = target_heights

        self.target_pos =  torch.tensor(
            [0.615, -0.425, target_heights], device=self.device
        ).repeat((self.num_envs, 1))

        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        # print("-------------------",self.actions, self.actions.size())

    def pre_physics_step(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self.world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        reset_buf = self.reset_buf.clone()
        # env= = (metric < thh).mom

        if len(env_ids) > 0:
            self.reset_idx(env_ids)


        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

        if self._dr_randomizer.randomize:
            rand_envs = torch.where(
                self.randomization_buf >= self._dr_randomizer.min_frequency,
                torch.ones_like(self.randomization_buf),
                torch.zeros_like(self.randomization_buf),
            )
            # print(self.randomization_buf, self._dr_randomizer.min_frequency)
            # print("randomization: ", rand_envs[0:10])
            rand_env_ids = torch.nonzero(torch.logical_and(rand_envs, reset_buf))
            self.dr.physics_view.step_randomization(rand_env_ids)
            self.randomization_buf[rand_env_ids] = 0
    
    async def pre_physics_step_async(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self.world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        reset_buf = self.reset_buf.clone()

        if len(env_ids) > 0:
            await self.reset_idx_async(env_ids)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

        if self._dr_randomizer.randomize:
            rand_envs = torch.where(
                self.randomization_buf >= self._dr_randomizer.min_frequency,
                torch.ones_like(self.randomization_buf),
                torch.zeros_like(self.randomization_buf),
            )

            rand_env_ids = torch.nonzero(torch.logical_and(rand_envs, reset_buf))
            self.dr.physics_view.step_randomization(rand_env_ids)
            self.randomization_buf[rand_env_ids] = 0


    def reset_idx(self, env_ids) -> None:
        """Reset specified environments."""

        self._reset_ur5e(env_ids)
        self._reset_object(env_ids)
        self._reset_buffers(env_ids)

    async def reset_idx_async(self, env_ids) -> None:
        """Reset specified environments."""

        self._reset_ur5e(env_ids)
        self._reset_object(env_ids)

        self._reset_buffers(env_ids)

    def _reset_ur5e(self, env_ids) -> None:
        """Reset DOF states and DOF targets of UR5e."""
        self.dof_pos[env_ids] = torch.cat(
            (
                torch.tensor(self.cfg_task.randomize.ur5e_arm_initial_dof_pos,device=self.device,).repeat((len(env_ids), 1)),
                (self.nut_widths_max[env_ids] * 0.5) * 1.0, 
                (self.nut_widths_max[env_ids] * 0.5) * 1.0,  
            ),  # buffer on gripper DOF pos to prevent initial contact
            dim=-1,
        )  
        self.dof_vel[env_ids] = 0.0  
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]
        print(env_ids)
    
        indices = env_ids.to(dtype=torch.int32)
        self.ur5es.set_joint_positions(self.dof_pos[env_ids], indices=indices)
        self.ur5es.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

    def _regrasp_ur5e(self, env_ids, pos ) -> None:
        """Reset DOF states and DOF targets of UR5e."""

        self.dof_pos[env_ids] = torch.cat(
            (
                torch.tensor(pos,device=self.device,),
                self.nut_widths_max[env_ids]  * 1.0, 
                self.nut_widths_max[env_ids]  * 1.0,  
            ),  # buffer on gripper DOF pos to prevent initial contact
            dim=-1,
        )
        velocity_list = [0, 0, 0, 0, 0.0, 0.0,0,0]
        velocity_tensor = torch.tensor(velocity_list, dtype=torch.float32, device=self.device) # Assign the tensor to self.dof_vel[env_ids] 
        self.dof_vel[env_ids] = velocity_tensor

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        j_ids = torch.tensor(0,dtype=torch.int32).unsqueeze(0)
        self.ur5es.set_joint_positions(self.dof_pos[env_ids], indices=j_ids)
        self.ur5es.set_joint_velocities(self.dof_vel[env_ids], indices=j_ids)

    def _close_ur5e(self, env_ids, pos ) -> None:
        """Reset DOF states and DOF targets of UR5e."""

        self.dof_pos[env_ids] = torch.cat(
            (
                torch.tensor(pos,device=self.device,),
                (self.nut_widths_max[env_ids] * 0.5) * 1.0, 
                (self.nut_widths_max[env_ids] * 0.5)* 1.0,  
            ),  # buffer on gripper DOF pos to prevent initial contact
            dim=-1,
        )
        velocity_list = [0, 0, 0, 0, 0, 0.0,0.5,0.5]
        velocity_tensor = torch.tensor(velocity_list, dtype=torch.float32, device=self.device) # Assign the tensor to self.dof_vel[env_ids] 
        self.dof_vel[env_ids] = velocity_tensor

        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        j_ids = torch.tensor(0,dtype=torch.int32).unsqueeze(0)
        self.ur5es.set_joint_positions(self.dof_pos[env_ids], indices=j_ids)
        self.ur5es.set_joint_velocities(self.dof_vel[env_ids], indices=j_ids)

    def _reset_object(self, env_ids) -> None:
        """Reset root state of nut."""

        self.nut_pos[env_ids, :] =  torch.tensor(
            [0.615, -0.425, 0.144], device=self.device #hardcode
        ).repeat(len(env_ids), 1)

        nut_rot = (
            self.cfg_task.randomize.nut_rot_initial
            * torch.ones((len(env_ids), 1), device=self.device)
            * math.pi
            / 180.0  #nut spawn rotation 
        )
        self.nut_quat[env_ids, :] = torch.cat(
            (
                torch.cos(nut_rot * 0.5),
                torch.zeros((len(env_ids), 1), device=self.device),
                torch.zeros((len(env_ids), 1), device=self.device),
                torch.sin(nut_rot * 0.5),
            ),
            dim=-1,
        )

        self.nut_linvel[env_ids, :] = 0.0
        self.nut_angvel[env_ids, :] = 0.0

        indices = env_ids.to(dtype=torch.int32)
        self.nuts.set_world_poses(
            self.nut_pos[env_ids] + self.env_pos[env_ids],
            self.nut_quat[env_ids],
            indices,
        )
        self.nuts.set_velocities(
            torch.cat((self.nut_linvel[env_ids], self.nut_angvel[env_ids]), dim=1),
            indices,
        )

    def _reset_buffers(self, env_ids) -> None:
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale, ignore=False
    ) -> None:
        """Apply actions from policy as position/rotation/force/torque targets."""
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if self.cfg_task.rl.unidirectional_pos and ignore == False:
            pos_actions[:, 2] = (pos_actions[:, 2] + 1.0) * 0.05  # [-1, 0]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.fingertip_midpoint_pos + pos_actions
        )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if self.cfg_task.rl.unidirectional_rot and ignore == False:
            rot_actions[:, 2] = (rot_actions[:, 2] + 1.0) * 0.2  # COUNTER CLOCK-WISE 
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )
       
        # print("dof_posF: ",self.dof_pos[0,5])
        # print("Flag",self.flag)


        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_midpoint_quat
        )

        if self.cfg_ctrl["do_force_ctrl"]:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if self.cfg_task.rl.unidirectional_force and ignore == False:
                force_actions[:, 2] = (force_actions[:, 2] + 1.0) * 0.2  # [-1, 0]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.force_action_scale, device=self.device
                    )
                )

            torque_actions = actions[:, 9:12] 
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.torque_action_scale, device=self.device
                    )
                )

            self.ctrl_target_fingertip_contact_wrench = torch.cat(
                (force_actions, torque_actions), dim=-1
            )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.rotation_speed = rot_actions[:,2]

        self.generate_ctrl_signals()

    def post_physics_step(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1
        
        if self.world.is_playing():

            if self._task_cfg['sim2real']['enabled'] and self.num_envs == 1:
                # Only retrieve the 0-th joint position even when multiple envs are used
                act_indices = torch.arange(8, dtype=torch.int64, device=self.device)
                joint_pos = self.ur5es.get_joint_positions(indices=[0], joint_indices=act_indices)
                self.send_joint_pos(joint_pos, grp_open=False)
                if self.cfg_task.env.sequential_sequence:
                    is_limited = self.dof_pos[:,5] <= -6.0
                    if is_limited.any():
                        if self.cfg_task.env.open_and_rotate:
                            self._rotate_gripper(
                                sim_steps=self.cfg_task.env.num_gripper_rotate_sim_steps
                            )
                            # joint_pos = self.ur5es.get_joint_positions(indices=[0], joint_indices=act_indices)
                            # self.send_joint_pos(joint_pos, grp_open=False)
                            self._close_gripper(
                                sim_steps=self.cfg_task.env.num_gripper_close_sim_steps
                            )
                            # joint_pos = self.ur5es.get_joint_positions(indices=[0], joint_indices=act_indices)
                            # self.send_joint_pos(joint_pos, grp_open=False)
                        else:
                            self.send_joint_pos(joint_pos, grp_open=False)
                    else:
                        self.send_joint_pos(joint_pos, grp_open=False)
                else:
                    self.send_joint_pos(joint_pos, grp_open=False)
                self.send_joint_pos(joint_pos, grp_open=False)




            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    async def post_physics_step_async(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1
        
        if self.world.is_playing():
            if self.cfg_task.env.sequential_sequence:
                is_limited = self.dof_pos[0,5] <= -6.0
                if is_limited:
                    if self.cfg_task.env.open_and_rotate:
                        print("d")
                        # self._open_gripper_async(
                        #     sim_steps=self.cfg_task.env.num_gripper_open_sim_steps
                        # )

                    
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _refresh_task_tensors(self) -> None:
        """Refresh tensors."""

        self.fingerpad_midpoint_pos = fc.translate_along_local_z(
            pos=self.finger_midpoint_pos,#-0.05,  #offset for model
            quat=self.hand_quat,
            offset=self.asset_info_ur5e_table.ur5e_finger_length
            - self.asset_info_ur5e_table.ur5e_fingerpad_length * 0.5,
            device=self.device,
        )

        self.finger_nut_keypoint_dist = self._get_keypoint_dist(body="finger_nut")
        self.nut_keypoint_dist = self._get_keypoint_dist(body="nut")

        self.nut_dist_to_target = torch.norm(
            self.target_pos - self.nut_com_pos, p=2, dim=-1
        )  # distance between nut COM and target

        self.nut_dist_to_fingerpads = torch.norm(
            self.fingerpad_midpoint_pos - self.nut_com_pos, p=2, dim=-1
        )  # distance between nut COM and midpoint between centers of fingerpads
        

        self.was_success = torch.zeros_like(self.progress_buf, dtype=torch.bool)

    def get_observations(self) -> dict:
        """Compute observations."""
        sensor_joint_forces = self.ur5es.get_measured_joint_forces(clone=False) #mimics FT sensors
        print("OBSERVATIONS", sensor_joint_forces)


        # Shallow copies of tensors
        obs_tensors = [
            self.fingertip_midpoint_pos,
            self.fingertip_midpoint_quat,
            self.fingertip_midpoint_linvel,
            self.fingertip_midpoint_angvel,
            self.nut_com_pos,
            self.nut_com_quat,
            self.nut_com_linvel,
            self.nut_com_angvel,
        ]

        if self.cfg_task.rl.add_obs_finger_force:
            obs_tensors += [self.left_finger_force, self.right_finger_force]
        else:
            obs_tensors += [
                torch.zeros_like(self.left_finger_force),
                torch.zeros_like(self.right_finger_force),
            ]
        
        self.obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  # shape = (num_envs, num_observations)

        observations = {self.ur5es.name: {"obs_buf": self.obs_buf}}

        return observations

    def calculate_metrics(self) -> None:
        """Update reset and reward buffers."""

        # Get successful and failed envs at current timestep
        curr_successes = self._get_curr_successes()
        curr_failures = self._get_curr_failures(curr_successes)

        self._update_reset_buf(curr_successes, curr_failures)
        self._update_rew_buf(curr_successes)

        self.randomization_buf += 1 #DOMAIN RANDOMIZATION BUFF

        if torch.any(self.is_expired):
            self.extras["successes"] = torch.mean(curr_successes.float())

    def _update_reset_buf(self,curr_successes, curr_failures) -> None:
        """Assign environments for reset if successful or failed."""

        self.reset_buf[:] = self.is_expired

    def _update_rew_buf(self, curr_successes) -> None:
        """Compute reward at current timestep."""

        keypoint_reward = -(self.nut_keypoint_dist + self.finger_nut_keypoint_dist)
        action_penalty = torch.norm(self.actions, p=2, dim=-1)
    
        # scaled_success_bonus = torch.clamp(
        #     self.cfg_task.rl.success_bonus / (self.nut_dist_to_target + 0.01),
        #     max=self.cfg_task.rl.success_bonus
        # )

        scaled_success_bonus =  -math.log10((self.nut_dist_to_target)+0.001)*2
        print("scale_log",scaled_success_bonus)

        in_target = torch.where( self.nut_dist_to_target <= 0.0001,1,0) #task success

        self.rew_buf[:] = (
            keypoint_reward * self.cfg_task.rl.keypoint_reward_scale
            - action_penalty * self.cfg_task.rl.action_penalty_scale
            + in_target * self.cfg_task.rl.close_bonus
            + curr_successes * scaled_success_bonus
        )


        # print("MIN REW: ",self.rew_buf.min(),"INDEX: ",self.rew_buf.argmin(), self.progress_buf[self.rew_buf.argmin()])
        # print("NUT TARGET MIN_REW: ", self.nut_dist_to_target[self.rew_buf.argmin()])

        print("MAX REW: ",self.rew_buf.max(),"INDEX: ", self.progress_buf[self.rew_buf.argmax()])
        print("NUT TARGET MAX_REW: ", self.nut_dist_to_target[self.rew_buf.argmax()])


    def _get_keypoint_dist(self, body) -> torch.Tensor:
        """Get keypoint distance."""

        axis_length = (
            self.asset_info_ur5e_table.ur5e_hand_length
            + self.asset_info_ur5e_table.ur5e_finger_length
        )

        if body == "finger" or body == "nut":
            # Keypoint distance between finger/nut and target
            if body == "finger":
                self.keypoint1 = self.fingertip_midpoint_pos
                self.keypoint2 = fc.translate_along_local_z(
                    pos=self.keypoint1,
                    quat=self.fingertip_midpoint_quat,
                    offset=-axis_length,
                    device=self.device,
                )

            elif body == "nut":
                self.keypoint1 = self.nut_com_pos
                self.keypoint2 = fc.translate_along_local_z(
                    pos=self.nut_com_pos,
                    quat=self.nut_com_quat,
                    offset=axis_length,
                    device=self.device,
                )

            self.keypoint1_targ = self.target_pos
            self.keypoint2_targ = self.keypoint1_targ + torch.tensor(
                [0.0, 0.0, axis_length], device=self.device
            )
            # print("kp1N: ",self.keypoint1[:,-1].min())
            # print("kp2N: ",self.keypoint2[:,-1].min())

        elif body == "finger_nut":
            # Keypoint distance between finger and nut
            self.keypoint1 = self.fingerpad_midpoint_pos
            self.keypoint2 = fc.translate_along_local_z(
                pos=self.keypoint1,
                quat=self.fingertip_midpoint_quat,
                offset=axis_length, #change
                device=self.device,
            )

            self.keypoint1_targ = self.nut_com_pos
            self.keypoint2_targ = fc.translate_along_local_z(
                pos=self.nut_com_pos,
                quat=self.nut_com_quat,
                offset=axis_length,
                device=self.device,
            )
            # print("kp1F: ",self.keypoint1[:,-1].min())
            # print("kp2F: ",self.keypoint2[:,-1].min())


        self.keypoint3 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 1.0 / 3.0
        self.keypoint4 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 2.0 / 3.0
        self.keypoint3_targ = (self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 1.0 / 3.0)
        self.keypoint4_targ = (self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 2.0 / 3.0)

        keypoint_dist = (
            torch.norm(self.keypoint1_targ - self.keypoint1, p=2, dim=-1)
            + torch.norm(self.keypoint2_targ - self.keypoint2, p=2, dim=-1)
            + torch.norm(self.keypoint3_targ - self.keypoint3, p=2, dim=-1)
            + torch.norm(self.keypoint4_targ - self.keypoint4, p=2, dim=-1)
        )
        # print("DIST: ",keypoint_dist.min(), keypoint_dist.argmin())
        return keypoint_dist

#_______________________________________________________________sync process
    def _rotate_gripper(
            self, sim_steps=450
    ) -> None:
        act_indices = torch.arange(8, dtype=torch.int64, device=self.device)
        joint_objetive = self.ur5es.get_joint_positions(indices=[0], joint_indices=act_indices)
        initial_wrist = joint_objetive[0,5]
        objective = -(joint_objetive[0,5])
        for step in range(sim_steps):
            try:
                print("objective", objective)
                joint_pos = self.ur5es.get_joint_positions(indices=[0], joint_indices=act_indices)
                joint_pos[:5] = joint_objetive[:5]
                
                desired_position = initial_wrist + (objective - initial_wrist) * (step + 1) / sim_steps #interpolation
                joint_pos[0,5] = desired_position
                print("joint_increment", joint_pos)
                pos = joint_pos.clone()
                pos = pos.squeeze().tolist()
                print("pos", pos)
                self._regrasp_ur5e(0, pos[:6])
                self.send_joint_pos(joint_pos, grp_open = True)

                SimulationContext.step(self.world, render=True)
            except Exception as e:
                print(f"Error during rotate execution: {e}")

    def _close_gripper(
            self, sim_steps=20
    ) -> None:
        act_indices = torch.arange(8, dtype=torch.int64, device=self.device)
        joint_pos = self.ur5es.get_joint_positions(indices=[0], joint_indices=act_indices)
        for _ in range(sim_steps):
            try:
                # joint_pos = self.ur5es.get_joint_positions(indices=[0], joint_indices=act_indices)
                pos = joint_pos.clone()
                pos = joint_pos.squeeze().tolist()
                self._close_ur5e(0, pos[:6])
                self.send_joint_pos(joint_pos, grp_open = False)
                SimulationContext.step(self.world, render=True)
            except Exception as e:
                print(f"Error during close execution: {e}")


    def _get_curr_successes(self) -> torch.Tensor:
        """Get success mask at current timestep."""

        curr_successes = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

        # If nut is close enough to target pos
        is_unscrewed = torch.where(
            self.nut_dist_to_target < self.thread_pitches.squeeze(-1) * 5,
            torch.ones_like(curr_successes),
            torch.zeros_like(curr_successes),
        )
        curr_successes = torch.logical_or(curr_successes, is_unscrewed)
        # print(self.thread_pitches.squeeze(-1) * 2)
        return curr_successes

    def _get_curr_failures(self, curr_successes) -> torch.Tensor:
        """Get failure mask at current timestep."""

        curr_failures = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

        # If nut is too far from target pos
        self.is_far = torch.where(
            self.nut_dist_to_target > self.cfg_task.rl.far_error_thresh,
            torch.ones_like(curr_failures),
            curr_failures,
        )

        #If nut has slipped (distance-based definition)
        self.is_slipped = torch.where(
            self.nut_dist_to_fingerpads
            > self.asset_info_ur5e_table.ur5e_fingerpad_length * 3.5
            + self.nut_heights.squeeze(-1) * 3.5,
            torch.ones_like(curr_failures),
            curr_failures,
        )

        # self.is_slipped = torch.logical_and(
        #     self.is_slipped, torch.logical_not(curr_successes)
        # )  # ignore slip if successful

        # If nut has fallen (i.e., if nut XY pos has drifted from center of bolt and nut Z pos has drifted below top of bolt)
        self.is_fallen = torch.logical_and(
            torch.norm(self.nut_com_pos[:, 0:2], p=2, dim=-1)
            > self.bolt_widths.squeeze(-1) * 0.5,
            self.nut_com_pos[:, 2]
            < self.cfg_base.env.table_height
            + self.bolt_head_heights.squeeze(-1)
            + self.bolt_shank_lengths.squeeze(-1)
            + self.nut_heights.squeeze(-1) * 0.5,
        )
        self.is_expired_4 = torch.where( self.nut_dist_to_target <= 0.0001,1,0) #RESTART REACHED TARGET

        #EXPIRED CONDITIONS
        self.is_expired = (self.progress_buf[:] >= self.cfg_task.rl.max_episode_length) #DEFAULT_RESTART
        self.is_expired = torch.logical_or(self.is_expired,self.is_expired_4)   


        #FAILURE LOGIC
        curr_failures = torch.logical_or(curr_failures, self.is_far)
        curr_failures = torch.logical_or(curr_failures, self.is_slipped)
        curr_failures = torch.logical_or(curr_failures, self.is_fallen)
   
        return curr_failures
    

    def send_joint_pos(self, joint_pos, grp_open):
        self.real_world_ur5e.send_joint_pos(joint_pos, grp_open)

