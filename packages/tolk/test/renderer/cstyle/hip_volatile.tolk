# tolk D342 golden=90f22c465112b2cd3b5c3a2accbe7ea2

extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 1))) volatile_add(volatile int* data0_4, volatile int* data1_4) {
  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    int val0 = (*(data1_4+Lidx0));
    *(data0_4+Lidx0) = ((int)((((unsigned int)(val0))+1u)));
  }
}
