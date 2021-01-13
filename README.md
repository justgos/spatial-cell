## Spatial cell sim

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