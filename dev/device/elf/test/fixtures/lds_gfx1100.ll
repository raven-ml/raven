target triple = "amdgcn-amd-amdhsa"
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
