# Spatial cell
Molecule-level simulation of a biological cell  

![](docs/spatial-cell.gif)

## Simulator - _spatial_cell_sim_
_(a Visual Studio 2019 project)_
Runs the simulation and stores the frames in reduces form in _results/frames.dat_.  
  
Simulation parameters can be configured via _config.json_.  
The simulated content, though - molecules and stuff - need to be configured via code in _src/setup/setup.cuh_.

### Development
#### Dependencies
Use [vcpkg](https://github.com/Microsoft/vcpkg) to install dependencies (it's excepted to be located in `C:\_code\vcpkg` - see the project's include directories for CUDA)
```shell
vcpkg install jsoncpp:x64-windows
```

#### Radix sort fix
The fix for "uses too much shared data" is to change  
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include\cub\device\dispatch\dispatch_radix_sort.cuh`  
<!-- and reduce the number of threads per blocks from 512 to 384, in the `Policy700`, line 788 - the first of "Downsweep policies"   -->
and set the `Policy200` and the default policy on line 805: `typedef Policy200 MaxPolicy;`
(and do a clean rebuild of the project)

## Renderer - _spatial_cell_vis_
_(a Unity project)_

Lazily reads the simulation results from _spatial_cell_sim/results/frames.dat_.

#### Hotkeys
_Space_ - play/pause the simulation  
_Left/right arrows_ - step 1 (+Shift - 5) frame backward/forward  
_0 - 8_ - toggle rendering of a specific particle type (0 - lipids, 1 - dummy particles)  
_x_ - toggle between billboard and 3D model rendering of particles  
_m_ - toggle visibility of metabolic particles (a.k.a. green fog)  


## Data-based configurator - _universe-config_
_(a collection of Jupyter notebooks)_

_chemistry-config.ipynb_ - messing around with PathwayCommons data and generating a few generic particle types in the end  
_complexification-config.ipynb_ - conversion of PDBe molecular complexes into a bunch of connected particles (with some state-machine logic for the ribosome)  
