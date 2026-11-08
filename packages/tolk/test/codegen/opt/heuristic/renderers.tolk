# tolk D89 D187 golden=48b312639e9141f6ac346e5bd319e5a6
renderer	device	arch	has_local	has_shared	shared_max	tensor_cores	registers	per_byte_first
cpu	CPU	x86_64,x86-64	False	True	32768	0	16x16	64
avx2	CPU	x86_64,x86-64-v3	False	True	32768	0	16x32	64
metal	METAL	Apple9	True	True	32768	5		64
cuda	CUDA	sm_89	True	True	49152	8		64
amd	AMD	gfx1100	True	True	65536	4		64
