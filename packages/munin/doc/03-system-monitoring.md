# System monitoring

A run records its machine beside its metrics: how busy the CPUs were, how
much memory the process held, what its GPUs drew. A reading reads such
facts now and gives them as pairs of a key and a number, and a session
samples readings on a schedule into its run. The `munin.sys` library makes
readings of the machine and of the process, and the program builds the
readings of its devices from what their library reports.

The examples on this page share one store:

```ocaml
open Munin

let store = Store.open_ ()
```

## Readings

A reading is a function, `unit -> (string * float) list`. Its keys follow
the rules of a metric's key, and in one run a key is a reading's or a
metric's, never both. `munin.sys` gives two kinds of fact
([`munin_sys.mli`](../lib/sys/munin_sys.mli), its preamble). A level is a
value at the call, such as the memory in use. A rate is a counter's change
per second over the window since the reading's previous call, such as the
bytes read from disk per second. The first call starts the window and
gives nothing:

```ocaml
# let host = Munin_sys.host ();;
val host : unit -> (string * float) list = <fun>
# host ();;
- : (string * float) list = []
# Unix.sleepf 0.2;
  let pairs = host () in
  ( List.mem_assoc "sys/mem_total_bytes" pairs,
    List.assoc "sys/interval_s" pairs > 0. );;
- : bool * bool = (true, true)
```

`Munin_sys.host ()` reads the machine and `Munin_sys.process ()` reads
this process. Their keys start with `sys/`, and the Machine and This
process sections of [`munin_sys.mli`](../lib/sys/munin_sys.mli) list each
key, its unit and the system it is read on. A key the system cannot read
is left out of the pairs, such as the load average on Windows.

Each reading keeps the counters of its previous call, so each one belongs
to one sampler.

## Sampling a run

`Session.sample s ~every readings` calls each reading now, then every
`every` seconds on a thread of its own, until the session finishes. Each
call's pairs become one `reading` record, at the time of the call and the
session's latest step, the step of its last metrics or resume, so that a
reader can align the machine's state with the training (the sampling
section of [`session.mli`](../lib/session.mli)).

```ocaml
let sampled =
  Session.within (Session.start ~store ~experiment:"monitoring" ()) @@ fun s ->
  Session.sample s ~every:0.1 [ Munin_sys.host (); Munin_sys.process () ];
  let loss = Session.metric s "train/loss" in
  for step = 0 to 4 do
    Unix.sleepf 0.1;
    Session.log loss ~step (Float.of_int (5 - step))
  done;
  Session.id s
```

```ocaml
# let h = History.read (Option.get (Store.run store sampled)) in
  ( History.metrics h,
    List.for_all (String.starts_with ~prefix:"sys/") (History.readings h) );;
- : string list * bool = (["train/loss"], true)
```

A history keeps a reading's series and statistics as it keeps a metric's.
The greatest sample of `sys/proc_rss_bytes` is the process's peak resident
memory as sampled, and its series against the steps shows where in the
training it grew. Samples of readings are never orphans, since the machine
was in that state whatever a resume later did to the steps.

`finish` stops the samplers, waiting for a reading in progress. A reading
that raises `Sys_error` gives nothing that time, and the sampler goes on.
Any other exception stops the sampler, and `finish` raises it after
writing the run's end.

## Devices

munin reads no device. The program reads its devices through their own
library and gives what it read to `Munin_sys.device`, which turns a
`Munin_sys.device` record into pairs under `dev/<name>/`. A field that is
`None` gives no pair:

```ocaml
# let gpu =
    Munin_sys.device "CUDA:0" (fun () ->
        {
          Munin_sys.unknown with
          used = Some 1_073_741_824;
          temperature = Some 61.;
        })
  in
  gpu ();;
- : (string * float) list =
[("dev/CUDA:0/used_bytes", 1073741824.); ("dev/CUDA:0/temperature_c", 61.)]
```

The Devices section of [`munin_sys.mli`](../lib/sys/munin_sys.mli) states
each field's quantity and key. Every device gives its quantities under the
same keys, whichever vendor made it.

A program that computes on a set of rig devices reads each one's usage,
its own allocations and its budget from rig. This glue samples the
machine, the process and every device of a set `devices` every 15
seconds:

<!-- $MDX skip -->
```ocaml
let every = 15.

let usage d () : Munin_sys.device =
  match (Rig.usage d ~window:every, Rig.allocated d, Rig.budget d) with
  | exception Rig.Lost (_, why) -> raise (Sys_error why)
  | Error why, _, _ -> raise (Sys_error why)
  | Ok u, allocated, budget ->
      {
        allocated = Some allocated;
        budget = (if budget = max_int then None else Some budget);
        memory = u.memory;
        used = u.used;
        busy = u.busy;
        activity = u.activity;
        temperature = u.temperature;
        power = u.power;
      }

let monitor s devices =
  let reading d = Munin_sys.device (Rig.name d) (usage d) in
  Session.sample s ~every
    (Munin_sys.host () :: Munin_sys.process ()
    :: List.map reading (Nx.Devices.to_list devices))
```

A device that the program closed, or that rig lost, raises `Rig.Lost`
from any of the three calls, and the glue turns it into `Sys_error`, so
the sampler skips that device that time and keeps sampling the others.
rig gives a device without a bound on its memory a budget of `max_int`,
which the glue records as no budget.
