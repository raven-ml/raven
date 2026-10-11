# AMD driver fixtures

Code objects for gfx1201, each made from its source by `gen.sh`.

- `kernels_gfx1201.hsaco`, of `kernels.cl`, holds the kernels `empty`,
  `double_index`, `spin` and `wild`.
- `other_gfx1201.hsaco`, of `other.cl`, holds `kernels_gfx1201.hsaco`'s
  kernels with a `double_index` that computes `3 * i`. It is an object of
  the same size, for code placed where another object's ran.
- `work_gfx1201.hsaco`, of `work.cl`, holds the kernels `copy`, which copies
  words after a delay, `shared`, which takes 256 bytes of local data share,
  and `inc`, which adds 1 to each word.
- `launch_gfx1201.hsaco`, of `launch.cl`, holds the kernels the suites
  launch. `ids` and `twice` are the conformance laws'
  (`Rig_gpu_support.Conformance.launch_binary`) and learn their grid from
  the implicit arguments. `lds` takes dynamic LDS after 256 bytes of its
  own. `scratch` takes 4100 bytes of scratch per work-item. `packet` reads
  its dispatch packet, and `packet_scratch` does both.
- `launch_gfx942.hsaco` holds the same kernels for gfx942, a GPU of several
  dies, whose queue reads AQL packets.
- `hidden_gfx1201.hsaco`, of `hidden.s`, which the ABI suite reads too,
  holds two kernels and their metadata. `every` names each implicit argument
  of code object version 5, and bounds its workgroups at 128 work-items.
  `plain` names none.

## Discovery tables

`r9700.bin`, `r9700_wide.bin` and `r9700_fused.bin` are discovery tables as an
AMD GPU's firmware leaves them in its memory, laid out as the Linux kernel's
`include/discovery.h` describes them. The ABI and PCI suites read them.
`gen.sh` makes them from `r9700.txt` with `discovery.py`, whose header says
what each holds.

`r9700.txt` lists the blocks of a Radeon AI PRO R9700 (gfx1201) as the amdgpu
driver reports them under `ip_discovery`, one block instance a line. The
command that read it is in `discovery.py`'s header.
