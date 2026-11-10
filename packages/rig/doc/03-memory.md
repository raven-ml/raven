# Memory

This page traces memory through its life, from a buffer's creation, through
its use on several devices and the host, to its return. The rules are in
[`rig.mli`](../lib/rig.mli): views, borrows and their mappings in the
preamble of its `Buffer` module, host access and copying in that module's
sections, and the return of memory, budgets and reclamation in its
Reclamation section.

## A buffer's life

Only `Buffer.free` frees a buffer early. A buffer is a range of bytes in an
allocation, and
views and borrows add further buffers over the same allocation, sharing its
stamps and claims. Once no buffer over the allocation, and no array that
`Buffer.read`, `Buffer.write` or `Buffer.export` made of it, is reachable,
the OCaml collector collects it. It waits for the work that used it, then
enters its device's cache, from which a later `Buffer.create` reuses it.
The cache returns memory to the driver when the program frees it, lowers
the budget, or an allocation needs the room. The Reclamation section
states each step, and the calls inside which collected memory moves.

```
   buffers, views, borrows and bigarrays reachable
                         |
                  all unreachable
                         |
                         v
              waiting for every point
                         |
          every point reached, or dropped
                         |
          +--------------+-----------------+
          |                                |
   within its budget             over its budget,
          |                       or io memory
          v                                |
        cache ------- cache freed -------> driver
          |
    Buffer.create
          |
          v
      new buffer
```

## Reclamation

Live buffers are freed only by `Buffer.free`. A program that keeps every
buffer reachable
runs out of memory however large its budget, and one that lets them go can
allocate far more than its budget over its life:

```ocaml
# open Rig;;
# let () =
    let d = Result.get_ok (software "M") in
    let kib = 1024 in
    set_budget d (1024 * kib);
    for _ = 1 to 100 do
      ignore (Sys.opaque_identity (Buffer.create d (256 * kib)))
    done;
    print_endline "100 buffers of 256 KiB allocated, none kept";
    let live = List.init 4 (fun _ -> Buffer.create d (256 * kib)) in
    (match Buffer.create d (256 * kib) with
     | _ -> print_endline "allocated"
     | exception Out_of_memory (_, n) ->
         Printf.printf "out of memory: %d KiB\n" (n / kib));
    ignore (Sys.opaque_identity live);;
100 buffers of 256 KiB allocated, none kept
out of memory: 256 KiB
```

The 100 buffers fit a budget of 1 MiB because each allocation the budget
refuses reclaims, and from its second round reclamation runs the collector,
which finds the buffers let go before it. The four live buffers fill the
budget, and reclamation finds nothing to take for a fifth.

The Reclamation section states when an allocation reclaims before it raises
`Rig.Out_of_memory`, and how much the host keeps in its own cache.

## Host access

The host reads and writes a buffer's bytes in place where it addresses the
buffer's memory. `Buffer.read` and `Buffer.write` give those bytes without a
copy to a function, once the work the access follows is done, and
`Buffer.export` waits once and hands the bytes out for good. These three
need memory the host addresses. The blits and `Buffer.fill` reach any
buffer and wait themselves. Where the host does not address the buffer,
the blits' bytes pass through staging memory, which a device copies to or
from the buffer, as `Buffer.copy` does between memories no device copies
directly, and the device's copy engine sets a fill's bytes itself. The `Buffer` module's Host access and Copying sections of
[`rig.mli`](../lib/rig.mli) state these rules, and its Loss section what an
exported array reads after a device is lost.

## Stamps through one program

This traces the GPU program of the [quick start](index.md#on-a-gpu) on a GPU
*d* whose memory the host does not address and which maps host memory, as a
CUDA GPU does. Neither copy names a borrow of `host` on *d*, so each passes
through staging memory. The host moves the bytes between `host` and staging
memory, and *d*'s copy engine between staging memory and `x`. *H* is
`host`'s allocation and *X* `x`'s. Each cell gives the last write, then the
use by *d*. The host's own writes, such as `a.{i} <- …` and its moves of a
copy's bytes, take no value and add no point.

| After | Value | *H*: write; use | *X*: write; use |
|---|---|---|---|
| `Buffer.create Rig.host (4 * n)` | | none; none | |
| `Buffer.create d (4 * n)` | | none; none | none; none |
| `Buffer.copy ~src:host ~dst:x` | `d:1` | none; none | `d:1`; `d:1` |
| `submit` in `double` | `d:2` | none; none | `d:2`; `d:2` |
| `Buffer.copy ~src:x ~dst:host` | `d:3` | none; none | `d:2`; `d:3` |

The first copy's transfer on *d* writes *X*, which has no use. The launch
reads and writes *X*, so it follows `d:1`, which device order covers
already. The copy back reads *X*, so it follows *X*'s write `d:2`, and the
host moves its bytes into *H* once *d* has reached `d:3`. Each copy returns
once the bytes are in its destination, and `Point.wait` once *d* has reached
`d:2`. Once `x` is unreachable, the next call that moves memory on *d* puts
*X* in *d*'s cache, since its points are reached. Once `host` is
unreachable, the collection that frees it returns *H*, since no device maps
it.

A program that copies `host` to and from *d* many times borrows it on *d*
once, `Buffer.borrow d host`, and copies through the borrow, which *d*'s
copy engine reads and writes in place (the `Buffer` module's Copying
section). The copies then give *H* *d*'s points. The copy back's `d:3`
becomes *H*'s write, and *d* releases its mapping of *H* once the borrow
is gone and *d* has run the work submitted until then.

On Metal, whose memory the host addresses, the host runs both copies, which
take no value. The launch is `d:1`, *X*'s only write and use, and *H* gets no
point.

Say *d* is lost after the launch was submitted, with `d:1` reached and `d:2`
not. The Loss section of `rig.mli` says which uses then raise `Rig.Lost`:

| Call | Result |
|---|---|
| `Point.wait` in `double` | raises `Lost (d, why)` |
| `Buffer.copy ~src:x ~dst:host` | raises `Lost`, since *X* is *d*'s memory |
| `Buffer.blit_to_bytes host …` | copies the bytes, since *H* has no point |
| `Buffer.read host …` | reads `65535.`, the value the host wrote |

*H* has no point of *d*, so every use of *H* stays valid, writes and claims
included. *X* raises `Lost` at every use, and returns once *d* has stopped.

If instead *d* is lost during the copy back, with `d:3` submitted and not
reached, the copy raises `Lost` before the host moves any byte into *H*,
which keeps the values the host wrote. Had the copy named a borrow of *H*,
*H*'s last write would be `d:3`, which *d* never reached, so every read or
write of *H* would raise `Lost` for the rest of the process. An array that
`Buffer.export` made of *H* before the loss would read whatever bytes are
there without an error, which may be the values the host wrote, the doubled
ones, or a mix.

Neither case shows memory that the lost device only read while its use was
unreached. Such memory still serves reads, and raises `Lost` for the uses that
depend on every point until the device has stopped, which the Loss section of
`rig.mli` lists.
