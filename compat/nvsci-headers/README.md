# NvSci headers (restored)

`nvscibuf.h`, `nvscisync.h`, `nvscierror.h`, `nvsciipc.h`, `nvscievent.h` are
needed to compile `src/cudla_context_standalone.cpp` (cuDLA standalone mode).

JetPack ships the NvSci **runtime libraries** (`nvidia-l4t-nvsci` package) but
**not the development headers** — those are only distributed inside the DRIVE OS
SDK. These copies were reconstructed from the public NVIDIA DRIVE OS 6.0.9
documentation (doxygen `_source.html` pages, which reproduce the headers
verbatim):

    https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/api_reference/nvscibuf_8h_source.html
    .../nvscisync_8h_source.html
    .../nvscierror_8h_source.html
    .../nvsciipc_8h_source.html
    .../nvscievent_8h_source.html

Reconstruction script (doxygen line `<div>` → C source, strip line-number
anchors and `&#160;` entities) lives in this directory's history; the result
passes `g++ -fsyntax-only` and links/ABI-matches the JetPack 6.2
(`L4T r36.5.2`) runtime libraries.

Copyright (c) NVIDIA Corporation — same license terms as the DRIVE OS SDK
apply to these headers. Use them on your own hardware accordingly; if you have
a real DRIVE OS install, prefer the originals.

Wired into the build via `-I ./compat/nvsci-headers` in the root Makefile.
