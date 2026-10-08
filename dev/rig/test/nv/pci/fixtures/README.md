# Fixtures

## VBIOS images

`vbios.rom` and its variants `vbios_no_bit.rom`, `vbios_debug.rom` and
`vbios_no_mapper.rom` are VBIOS images laid out as NVIDIA's RM reads them,
made in this directory by `uv run vbios.py`, whose header says what each
holds. No real VBIOS is here: reading one needs root.

## Firmware containers

No NVIDIA firmware is here: its licence allows its redistribution only for
use on open-source operating systems and forbids taking it apart. These are
containers laid out as the firmware is, with bytes of their own:

- `booter.bin`, `bootloader.bin` and `fmc.elf`, made in this directory by
  `uv run firmware.py`, whose header says what each holds;
- `sections.elf`, an ELF object whose sections are named as the GSP's
  firmware names them, each holding its own name, made by
  `uv run sections.py`.
