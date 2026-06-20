# WSL Visualization

This machine reports WSLg-style display variables (`DISPLAY=:0` and
`WAYLAND_DISPLAY=wayland-0`).  Gazebo Classic can sometimes open directly on
Windows 11 WSLg, but PX4 + XTDrone meshes are heavy enough that `gzclient` is
often blank, slow, or unstable.  The recommended path is therefore:

1. run Gazebo headless for physics, PX4 SITL, and MAVROS;
2. use RViz for the polished TRUST-UP experiment view.

Use the prepared environment first:

```bash
source /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/setup_trust_up_noetic.bash
```

Recommended WSL launch:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl_wsl_rviz.launch scenario:=circle
```

For the second paper profile:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl_wsl_rviz.launch scenario:=figure8
```

This launch forces `gui:=false` for Gazebo and starts RViz with conservative
WSL OpenGL settings:

```bash
QT_X11_NO_MITSHM=1
QT_QPA_PLATFORM=xcb
LIBGL_ALWAYS_SOFTWARE=1
```

If RViz works but is very slow, try hardware rendering:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl_wsl_rviz.launch \
  scenario:=circle \
  libgl_always_software:=0
```

Use Gazebo GUI only when you specifically need to inspect the raw Gazebo scene:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl.launch scenario:=circle gui:=true
```

For Windows 10 WSL without WSLg, run an X server such as VcXsrv on Windows and set:

```bash
export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=0
export QT_X11_NO_MITSHM=1
```

The controller can also be checked without any graphics:

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/run_headless_paper_experiment.py \
  --scenario figure8 \
  --output /tmp/trust_up_figure8.csv
```
