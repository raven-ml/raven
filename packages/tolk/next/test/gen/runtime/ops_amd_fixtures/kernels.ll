; The kernels of the AMD code object fixtures that are not simple_add_gfx1100.hsaco,
; compiled once for gfx1100, gfx1201 and gfx942 by tinygrad's AMDLLVMCompiler
; (LLVM 21's AMDGPU backend) into simple_add_<arch>.o and scratch_<arch>.o, and
; for gfx1100 into lds_gfx1100.o: three modules, each compiled on its own.

; simple_add: its arguments are those of simple_add_gfx1100.hsaco.

target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @simple_add(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %a, ptr addrspace(1) noalias %b, i32 %n) {
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %t
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %t
  %x = load float, ptr addrspace(1) %ap
  %y = load float, ptr addrspace(1) %bp
  %s = fadd float %x, %y
  %op = getelementptr float, ptr addrspace(1) %out, i32 %t
  store float %s, ptr addrspace(1) %op
  ret void
}

; scratch: it reads its dispatch packet (its work-group size) and scratch memory
; (an array indexed by a loaded value).
declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @E_4(ptr addrspace(1) noalias %data0, ptr addrspace(1) noalias %data1) {
  %arr = alloca [64 x i32], align 4, addrspace(5)
  %dp = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %wp = getelementptr i8, ptr addrspace(4) %dp, i64 4
  %w = load i16, ptr addrspace(4) %wp
  %wi = zext i16 %w to i32
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  %ip = getelementptr i32, ptr addrspace(1) %data1, i32 %t
  %i = load volatile i32, ptr addrspace(1) %ip
  %j = and i32 %i, 63
  %sp = getelementptr [64 x i32], ptr addrspace(5) %arr, i32 0, i32 %j
  store volatile i32 %wi, ptr addrspace(5) %sp
  %kp = getelementptr [64 x i32], ptr addrspace(5) %arr, i32 0, i32 %t
  %k = load volatile i32, ptr addrspace(5) %kp
  %op = getelementptr i32, ptr addrspace(1) %data0, i32 %t
  store i32 %k, ptr addrspace(1) %op
  ret void
}

; lds: it reads and writes 1 KiB of local memory.
@shared = internal addrspace(3) global [256 x float] poison, align 4
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @lds(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %a, ptr addrspace(1) noalias %b, i32 %n) {
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %t
  %x = load float, ptr addrspace(1) %ap
  %sp = getelementptr [256 x float], ptr addrspace(3) @shared, i32 0, i32 %t
  store volatile float %x, ptr addrspace(3) %sp
  %u = xor i32 %t, 255
  %rp = getelementptr [256 x float], ptr addrspace(3) @shared, i32 0, i32 %u
  %y = load volatile float, ptr addrspace(3) %rp
  %op = getelementptr float, ptr addrspace(1) %out, i32 %t
  store float %y, ptr addrspace(1) %op
  ret void
}
