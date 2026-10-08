# Fixtures

## Discovery tables

`r9700.bin`, `r9700_wide.bin` and `r9700_fused.bin` are discovery tables as an
AMD GPU's firmware leaves them in its memory, laid out as the Linux kernel's
`include/discovery.h` describes them. They are made in this directory by
`uv run discovery.py`, whose header says what each holds, from `r9700.txt`.

`r9700.txt` lists the blocks of a Radeon AI PRO R9700 (gfx1201) as the amdgpu
driver reports them under `ip_discovery`, one block instance a line. The
command that read it is in `discovery.py`'s header.
