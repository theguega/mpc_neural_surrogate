import mujoco
import numpy as np


class MuJoCoEnvironment:
    def __init__(self, model_path="models/3dof_robot_arm.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), 0, False, {}  # obs, reward, done, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel, self.data.site_xpos[0]])

    def render(self, viewer):
        viewer.sync()

    def get_ee_position(self):
        site_id = self.model.site("ee_site").id
        return self.data.site_xpos[site_id].copy()
