# Fixtures

`vbios.rom` and its variants `vbios_no_bit.rom`, `vbios_debug.rom` and
`vbios_no_mapper.rom` are VBIOS images laid out as NVIDIA's RM reads them,
made in this directory by `uv run vbios.py`, whose header says what each
holds. No real VBIOS is here: reading one needs root.

No NVIDIA firmware is here: its licence allows its redistribution only for
use on open-source operating systems and forbids taking it apart.
