# Drivers

This page traces one submission from `Rig.submit` through a driver to the
hardware and back. What a driver must do, and what rig guarantees it, is in
[`rig_edge.mli`](../lib/edge/rig_edge.mli): the facts a device declares, its
timeline word and how the word advances, the three C functions `room`,
`submit` and `commit` with the properties a driver keeps, and the counted
calls that its stop never overlaps.

## The shape of a driver

A driver is a library that links `rig.edge` and never `rig`, as the preamble of
[`rig_edge.mli`](../lib/edge/rig_edge.mli) states. It allocates a device's
memory, loads code on it, and places work on its queues. rig reads the
device's facts once, when it opens, and calls three C functions of the
driver to run work, with the OCaml runtime released. Below, `submit` alone
names the driver's C function, and `Rig.submit` the OCaml call.

The device reports progress in its timeline word, which only the driver
writes. A store on a queue writes it, the host writes it from a handler, or
the driver writes it once an object of its API completes. A wait reads the
word directly, or through the driver's `signaled` where the word has no host
address. Three properties of a driver carry every rule of `Rig`'s Order
section: the work handed to each queue runs in order, a value's work starts
after the previous value's work has completed, and the host learns through
the word which values have completed, or that the device faulted. Device
order is rule 1 of the Order section seen from the driver.

## One submission

This traces a launch on an AMD GPU through `Rig_amd`, the driver that every
way of opening an AMD GPU builds its device with. Its word is host
memory that the GPU's release packet writes (`Store`), and whose `submit`
also commits. The steps before the driver's `submit` are the same for every
driver.

1. **Checks and claims.** `Rig.submit` checks the run and its buffers, and
   takes a read claim on every allocation the work uses, so that no donation
   writes one in place while the submit runs.
2. **Points to follow.** It reads the stamps of every allocation the work
   uses and collects the points the work must follow, by the rules of
   `Rig`'s Order section, with those of `~waits`, leaving out the points
   already reached.
3. **Producers.** A producer is a device whose point another device's work
   waits for (`Rig_edge.waits`). For each point of a producer, `Rig.submit`
   commits the producer's work through the point. Where the device's facts
   say its queues can wait for that producer, it keeps the point for the
   driver's `submit` to place as a queue wait, and otherwise it waits for
   the point on the host.
4. **Room.** In C, without running OCaml, it waits until no other call is
   submitting or committing on the device, then calls the driver's `room`
   with the parts, the run's blocks and the submission's *scratch*,
   `RIG_SCRATCH` bytes per part that the driver may write during the call
   (the C functions section of [`rig_edge.mli`](../lib/edge/rig_edge.mli)).
   If the parts fit only later, it gives the device back, waits until the
   device has reached one more value, and calls `room` again. A driver may
   instead block in its `submit` until the work fits, as Metal's does.
5. **The value.** It assigns *v*, one more than the device's submitted value,
   and calls the driver's `submit` with *v*, the waits, the parts and their
   scratch, the run's blocks and the address of each of the run's buffers.
   No OCaml code runs between the two, so the driver receives *v*'s work or
   the device is lost.
6. **The driver's `submit`.** `Rig_amd` first decides, into the scratch,
   which of its rings runs each part. A large copy within the GPU's memory
   runs as a kernel on the compute ring, and every other part, the launch
   here, on its own queue's ring. `room` decided the same, to count the
   space each ring needs. On each queue the submission uses, `Rig_amd`
   places a wait for *v - 1* where the other queue completed it, which keeps
   device order across its two queues, and on the compute queue the waits for
   other devices' words. For the launch it copies the parameters into memory of
   its own, adding each buffer's address at its refs, and places a dispatch
   packet. On the last queue it places a release that writes *v* into the
   timeline word. It then flushes the host's writes through the GPU's memory
   window, if any `Mapped` memory is live on the device or mapped by it, sets
   each queue's write position and rings its doorbell, and returns that every
   value up to *v* is committed.
7. **The stamps.** Back in rig, whatever the driver's `submit` returned,
   `Rig.submit` adds `(d, v)` to the stamps of every allocation the work uses
   and, if the submission keeps a value, to the stamps that keep it. It then
   releases its claims and returns the point `(d, v)`. It raises `Rig.Lost`
   instead if the driver's `submit` failed, or if the device was lost before
   `Rig.submit` returns.
8. **The hardware.** The GPU runs the dispatch, then the release writes back
   its caches and stores *v* into the word.
9. **The wait.** `Point.wait (d, v)` reads the word. While it stands below
   *v*, the wait spins, then sleeps in the driver's `sleep` until an event
   of the device or a timeout, and reads again. Once the word reads *v* or
   more, the wait returns, and the host follows the work.
10. **Return.** A later `Buffer.create` on the device finds the collected
    memory whose points the word shows reached, and moves it to the cache
    (`Rig`'s Reclamation section).

A driver whose `submit` does not commit splits step 6. Metal's `submit`
encodes the work into an open command buffer, which runs nothing until it is
committed. A wait commits it, and so does the driver itself when the GPU
runs short of work or the buffer holds enough values. Metal's completion
handler then writes the word (`Host`).
