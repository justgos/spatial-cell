#pragma once

#include "types.cuh"


time_point
now() {
    return std::chrono::high_resolution_clock::now();
}

double
getDuration(time_point start, time_point end) {
    return (std::chrono::duration_cast<std::chrono::duration<double>>(end - start)).count();
}
