let cuda = Result.to_option (Nx_cuda_device.get 0)
let nv = Result.to_option (Nx_nv_device.get ~interface:Kernel 0)
