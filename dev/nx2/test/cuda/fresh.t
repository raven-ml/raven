The digest covers each source's contents. A cubin made from older sources
fails before the host library embeds it.

  $ printf 'source\n' > kernel.cu
  $ printf '\000nx_cuda_sources_80900781866cecf3a3961ff757b13858\000' > kernel.cubin
  $ ../../lib/cuda/kernels/fresh.exe kernel.cubin kernel.cu

  $ printf 'changed\n' > kernel.cu
  $ ../../lib/cuda/kernels/fresh.exe kernel.cubin kernel.cu
  kernel.cubin was built from other sources: rebuild it with dev/nx2/lib/cuda/kernels/build.sh
  [1]

Both ends of the function name must match, as in the cubin's symbol table.

  $ printf 'source\n' > kernel.cu
  $ printf 'xnx_cuda_sources_80900781866cecf3a3961ff757b13858\000' > kernel.cubin
  $ ../../lib/cuda/kernels/fresh.exe kernel.cubin kernel.cu
  kernel.cubin was built from other sources: rebuild it with dev/nx2/lib/cuda/kernels/build.sh
  [1]
  $ printf '\000nx_cuda_sources_80900781866cecf3a3961ff757b13858x\000' > kernel.cubin
  $ ../../lib/cuda/kernels/fresh.exe kernel.cubin kernel.cu
  kernel.cubin was built from other sources: rebuild it with dev/nx2/lib/cuda/kernels/build.sh
  [1]
