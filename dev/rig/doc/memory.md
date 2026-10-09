# How rig reclaims memory

`rig.mli` states what reclamation promises: an allocation that a budget
or a driver refuses runs rounds of reclamation before it raises
`Out_of_memory`, buffer memory paces the collector, and a wait returns
to OCaml often enough for Ctrl-C. This note records how, with the
implementation's constants, which may change without a change to the
interface.

## Rounds of reclamation

An allocation that the budget or the driver refuses runs rounds for the
budget that refused: the device's or, for memory the host's budget
counts, the host's. It tries again after each round, four tries in all,
before it raises `Out_of_memory`.

A round:

1. returns every cached memory, on any device, that counts in that
   budget, once its device's submitted work is done, and for the host
   its kept buffers;
2. waits for the submitted work of every device whose work holds back
   the return of other such memory, such as memory collected over the
   budget, so a device's loss raises `Lost`;
3. drains every device, the host included;
4. from the second round on, collects unreachable buffers and drains
   every device again.

A copy whose device's driver refuses to map the staging memory runs the
same rounds for that device.

## The host's cache

The host keeps the memory of collected buffers of 64 KiB or more in a
cache for the next buffers of their sizes. It returns what the cache
holds beyond a major cycle's share of the program's memory, or beyond
32 MiB where that is more, as it keeps a buffer and at the end of each
major cycle.

## Pacing the collector

The host memory of buffers of 64 KiB or more paces the collector's
major cycles by the program's memory: its OCaml heap and the host
memory its live buffers hold. A cycle is due once the memory allocated
since the last one reaches `custom_major_ratio / 150` of it
(`Gc.control`), 29% by default. Unreachable buffers then hold at most
three such shares, 88% of the program's memory by default.

The memory of another device's buffers paces major cycles by the room
left in that device's budget, the same share of it, and at least a
page: a device that fills up runs cycles more often before it refuses
an allocation. Smaller host buffers pace the collector as any bigarray
does.

## Waits

A wait for a device whose host writes the word, or whose word the host
does not address, blocks in the driver's `sleep` from the first read of
the word. For another device it reads the word for up to 4 us holding
the domain lock, so that the domain's other threads wait at most that
long, then spins on it, yielding the processor with the domain lock
released, and blocks in the driver between reads once the word stood
still for the still interval (200 ms).

## Staging memory

A copy that stages goes through two slots of 64 MiB of host memory,
made at the first copy that needs them and kept for the life of the
process. A copy holds a slot while it waits for devices, and a copy that
finds both slots held waits for one. A device of this machine that maps
no host memory stages through two slots of its own `Pinned` memory
instead, made at its first copy that needs them and kept until it is
lost. A staged copy moves in pieces of half a slot, 32 MiB, which the
slot's two halves take in turn, so one half fills while the other
drains.
