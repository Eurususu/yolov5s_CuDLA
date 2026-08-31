/*
 * Redirect the legacy NVTX v1 header to the NVTX v3 implementation shipped
 * with CUDA 12.x.
 *
 * MatX 0.4.1 includes <nvToolsExt.h> (NVTX v1) from matx/core/nvtx.h, while
 * CUB bundled with CUDA 12.6 uses NVTX v3. Including both versions in one
 * translation unit is a hard #error in nvToolsExt.h. The nvtx3 drop-in
 * implements the exact same v1 C API (nvtxRangeStartEx, nvtxRangeEnd, ...),
 * so including it first on behalf of MatX resolves the conflict.
 */
#ifndef MATX_REFORMAT_NVTX_SHIM
#define MATX_REFORMAT_NVTX_SHIM
#include <nvtx3/nvToolsExt.h>
#endif
