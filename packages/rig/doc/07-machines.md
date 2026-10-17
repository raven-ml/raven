# Machines

A program uses the devices of other computers through `Rig` as it uses its
own. This page follows one job across two machines: its processes, one call
travelling to another machine and back, the `rig run` and `rig agent`
commands, a failure, and what the job trusts.
[`Rig_remote`](../lib/remote/rig_remote.mli) states the rules this page
mentions, and the manual pages of `rig run`, `rig agent` and `rig firmware`
state the commands' own (`rig run --help`).

## Jobs

Take two machines, `h100-a` and `h100-b`. A program on `h100-a` uses the
GPUs of `h100-b` in a *job*: the program is the job's *controller*, and an
*agent* on `h100-b` opens that machine's devices for it and runs the work
the controller submits to them ([`Rig_remote`](../lib/remote/rig_remote.mli)
defines machine, job, controller, agent and proxy in its preamble).

The controller gets `h100-b`'s host, and a *proxy* for each device the agent
opens, such as `CUDA:3@h100-b` ([`Rig_remote`](../lib/remote/rig_remote.mli),
"Jobs"). [`Rig.host_of`](../lib/rig.mli) of each proxy is `h100-b`'s host.
No device of `h100-a` reaches the memory of a device of `h100-b`
([`Rig.reaches`](../lib/rig.mli)), so bytes cross between the machines
through `Buffer.copy`.

Add a third machine, `h100-c`, and a loop whose every step exchanges the
same bytes between `h100-b` and `h100-c`. `Buffer.copy` cannot join two
other machines, so each step would carry the bytes through the controller
([`Rig_remote`](../lib/remote/rig_remote.mli), "Order and costs"). A *rail*
carries such a fixed set of transfers directly between the two machines,
step after step. A program makes one through the capability record of a
machine's host, and orders its transfers by the rail's counts
([`Rig_remote_abi`](../lib/remote/abi/rig_remote_abi.mli), "Rails"). A
rig.program program reaches its machine's end of a rail as `Rail` memory
([`Rig_program`](../lib/program/rig_program.mli), "Rails").

## One call, traced

Follow one copy from this process's memory to `CUDA:3@h100-b`:

<!-- $MDX skip -->
```ocaml
Rig.Buffer.copy ~src:here ~dst:there
```

[`Rig_remote`](../lib/remote/rig_remote.mli), "Order and costs", states
each step below.

1. The proxy's driver sends the copy as one message on the controller's
   connection to `h100-b`'s agent, with the bytes of `here` inside it
   ([`Wire`](../lib/remote/proxy/wire.mli), "Messages").
2. The agent applies the message after every message the controller sent it
   before, through its own rig, and copies the bytes into `there`.
3. Once the copy is done, the agent reports its value of the proxy's
   timeline.
4. The controller's receiving thread writes that value into the proxy's
   timeline word, and `Buffer.copy` returns.

A copy from `there` back into this process's memory takes the same path,
with one more message. The agent sends the copy's bytes first, in a message
of their own, and the receiving thread writes them before the value, so a
wait that returns finds them in place.

Since the agent finishes each submission's work before it applies the next
message, work submitted to `h100-b` follows all the work submitted to it
before, whichever of its devices runs it, and rig orders work by the memory
it uses across the two machines as on one
([`Rig_remote`](../lib/remote/rig_remote.mli), "Order and costs").

## `rig run` and `rig agent`

`rig run` starts the job from `h100-a`:

<!-- $MDX skip -->
```sh
$ rig run --on h100-a,h100-b -- ./finetune.exe --config 70b.toml
```

It runs the program on `h100-a`, starts `rig agent` on `h100-b` over `ssh`
with a fresh key, and passes the agent's address and the key to the program
in its environment (`rig run --help`). The `rig` on `h100-b` must be this
machine's release, and `--rig` names it where the login shell there does
not find it. The program joins the job with
`Rig_remote.launched` ([`Rig_remote`](../lib/remote/rig_remote.mli),
"Launchers"):

<!-- $MDX skip -->
```ocaml
let () =
  match Rig_remote.launched () with
  | Some (Ok job) ->
      let n = 4096 in
      let copy h =
        let here = Rig.Buffer.create Rig.host n
        and there = Rig.Buffer.create h n in
        Rig.Buffer.copy ~src:here ~dst:there
      in
      List.iter copy (Rig_remote.hosts job);
      Rig_remote.close job
  | Some (Error why) -> prerr_endline why
  | None -> prerr_endline "not started by rig run"
```

On `h100-b`, `rig agent` serves the job with the machine's host and its
GPUs, those detached from their kernel driver included
([GPUs without a driver](06-gpus-without-a-driver.md), and
`rig agent --help`). A program that starts its own agents calls
`Rig_remote.connect` in place of `launched`.

## When the job fails

Say a GPU of `h100-b` faults. The job fails as a whole, with the fault as
its root cause, and so would it for a connection that broke or fell silent
([`Rig_remote`](../lib/remote/rig_remote.mli), "Failure"). Each agent that
still runs fails its process, as `Rig.fail` does ([`Rig`](../lib/rig.mli),
"Loss"), and returns from `Rig_remote.serve`. The controller keeps its own
devices. The hosts and devices of the job's other machines raise `Rig.Lost`
with the root cause from their next use, and the controller may start
another job. `rig run` names the cause and starts the job again with a fresh
agent and a fresh key, until it gives up as `rig run --help` states.

## Trust

The job's processes prove to each other that they hold the job's key, and
nothing after the proofs is authenticated or encrypted. An agent runs the
code the controller sends it, host code included. So a job belongs on a
network that only its machines read and write, such as a cluster's own
([`Rig_remote`](../lib/remote/rig_remote.mli), "Keys and trust", and
`rig run --help`).
