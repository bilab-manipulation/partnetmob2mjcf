import time
import mujoco
import mujoco.viewer
from dm_control import mjcf
from typing import Dict
import sys

mujoco_dir = sys.argv[1]

m = mjcf.from_path(mujoco_dir)

arena = mjcf.RootElement()
arena.worldbody.attach(m)

assets: Dict[str, str] = {}
for asset in arena.asset.all_children():
    if asset.tag in ["mesh", "texture"]:
        f = asset.file
        assets[f.get_vfs_filename()] = asset.file.contents

# import pdb; pdb.set_trace()
for deformable in arena.deformable.all_children():
    if deformable.tag in ["skin"]:
        f = deformable.file
        assets[f.get_vfs_filename()] = deformable.file.contents

xml_string = arena.to_xml_string()
# save xml_string to file
with open("arena.xml", "w") as f:
    f.write(xml_string)

m = mujoco.MjModel.from_xml_string(xml_string, assets)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  while viewer.is_running():
    step_start = time.time()
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)