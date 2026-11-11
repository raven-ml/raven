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
never reached. A lost device's stop runs only when the program closes the
device or opens its name again, never inside the call that met the loss,
so a program that catches `Rig.Lost` closes the device. Until its stop the
device keeps its memory and the values its submissions keep, and a GPU
driven over PCI may go on running the work it took. Once the device has
stopped, as the Stops section of `rig.mli` defines it, which may be never,
its memory returns, and memory it only read is ordinary again. Memory whose
last write it never reached raises `Rig.Lost` for the rest of the process.

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
    let a = Result.get_ok (software "A") in
    let b = Result.get_ok (software "B") in
    let x = Buffer.create a 4 in
    close a;
    (match Buffer.copy ~src:x ~dst:(Buffer.create b 4) with
     | () -> print_endline "copied"
     | exception (Lost _ as e) -> print_endline (Printexc.to_string e));
    Printf.printf "lost: %s, failure: %b\n"
      (Option.value (lost a) ~default:"no")
      (Option.is_some (failure ()));
    let a' = Result.get_ok (software "A") in
    Printf.printf "A again: same device %b, submitted %d\n"
      (equal a a') (submitted a');;
A lost: closed
lost: closed, failure: false
A again: same device false, submitted 0
```

## Giving up on work

rig loses a device for the time its work runs only where the device's
driver declares a hang bound, as a GPU driven over PCI does. A program
bounds a wait itself with `Rig.Point.wait_for`, which answers `false` once
its time has passed and leaves the work running. To give up on the work,
the program loses the device with `Rig.lose` and closes it: the close runs
the device's stop, which ends the work as the driver's documentation
states. `Rig.lose` takes no lock and waits for nothing, so another thread
may call it while a wait blocks on the device, and the wait then raises
`Rig.Lost`.

A close bounded in time waits for the device's last point, then gives the
device up if its work is still running:

```ocaml
# let close_within ~ns d =
    (match Point.wait_for ~ns (Point.submitted d) with
     | true -> ()
     | false -> lose d "its work did not end in time"
     | exception Lost _ -> ());
    close d;;
val close_within : ns:int -> t -> unit = <fun>
# let () =
    let c = Result.get_ok (software "C") in
    close_within ~ns:1_000_000_000 c;
    print_endline (Option.value (lost c) ~default:"live");;
closed
```

A device whose work ended in time closes as any other, with the reason
`"closed"`. One given up keeps the reason given to `Rig.lose`, and its loss
is the process's failure if none came before.
