# Loss

This page shows what a program sees when a device fails. The rules are in
the Loss section of [`rig.mli`](../lib/rig.mli): what loses a device, which
work counts as done, which uses of memory raise `Rig.Lost`, how hangs are
bounded, when a lost device's memory returns, and what a fork does.

## The story

A loss is final. Once a device is lost, rig counts as done only the values
the device had reached when the loss was recorded. The work of later values
may have run, in part or not at all, and no program can tell which, so rig
refuses every use whose validity depends on that work. Every use of the lost
device and of its memory raises `Rig.Lost`. Other memory raises it only for
the uses that depend on a point the lost device never reached, and the Loss
section lists what each use depends on. The
[worked trace](03-memory.md#stamps-through-one-program) shows both a buffer
that stays usable and one that raises for the rest of the process.

Other devices go on, unless their queues wait for a value the lost device
never reached. A lost device's memory and the values its submissions keep
return only once the device has stopped, as the Stops section of `rig.mli`
defines it, which may be never.

## What a program sees

`Rig.Lost (d, why)` carries the device and its reason, and `Rig.lost d`
returns the same reason without raising. The reasons are listed at
`Rig.Lost`. The process's failure, `Rig.failure`, is its first loss that
`Rig.close` did not cause, or the reason given to `Rig.fail`. A job that runs
on several machines fails as a whole
([`Rig_remote`](../lib/remote/rig_remote.mli)).

Closing a device loses it with the reason `"closed"`, and opening its name
again makes a new device:

```ocaml
# open Rig;;
# let () =
    let a = Result.get_ok (memory_device "A") in
    let b = Result.get_ok (memory_device "B") in
    let x = Buffer.create a 4 in
    close a;
    (match Buffer.copy ~src:x ~dst:(Buffer.create b 4) with
     | () -> print_endline "copied"
     | exception (Lost _ as e) -> print_endline (Printexc.to_string e));
    Printf.printf "lost: %s, failure: %b\n"
      (Option.value (lost a) ~default:"no")
      (Option.is_some (failure ()));
    let a' = Result.get_ok (memory_device "A") in
    Printf.printf "A again: same device %b, submitted %d\n"
      (equal a a') (submitted a');;
A lost: closed
lost: closed, failure: false
A again: same device false, submitted 0
```
