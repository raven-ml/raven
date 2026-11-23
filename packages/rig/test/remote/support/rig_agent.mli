(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rig's test agent, which a suite of any package starts as a process. *)

val main : unit -> unit
(** [main ()] is the agent for the suites, run as a program's main: with
    arguments [KEYFILE [MODE]], it listens on the loopback at a port the system
    chooses, prints it, and serves one job, then prints "closed" and exits 0
    once the job closed, or prints "failed: WHY" and exits 2 once it failed.
    After "closed" it prints "open: NAME" for each device it opened that is not
    closed.

    In mode "again" it then listens at the same port and serves a second job
    likewise. In mode "linger" it waits 2 s before it exits. In mode "twice" it
    then calls serve again, and prints "serve raised" if that raised
    Invalid_argument. In mode "helper" it starts a process, the program with the
    argument [helper], that prints "helper ready" and waits for a line or the
    end of stdin. It reaps that process before exiting. In mode "deaf" it is no
    agent of rig.remote: it proves the key to the controller, then stops
    listening, so no other agent of the job reaches it, and waits for the job to
    end. In mode "crowded" it limits its descriptors to 256 and, while it
    serves, takes lines on stdin: "fill" opens descriptors until the system
    refuses one, then prints "full"; "free" closes them, then prints "freed". In
    mode "high" it holds 1100 descriptors before it listens, its limit raised to
    4096, so that every socket it makes has a number past 1100. In mode "held"
    it is reached through a relay, whose port it prints, that passes the first
    connection, the controller's, at once, and holds each later one, an agent's
    dial, until a line comes on stdin: it prints "dialled" as it takes each, so
    the dials in flight at once can be counted. It exits once the relay has
    passed every byte both ways, the agent's close included.

    It serves fourteen kinds. "MEM" is two software devices. "POLLED" is two
    Polled devices, which launch Polled's host functions. "TIGHT" is a Polled
    device whose queue holds 3 parts. "DISK" is the io device of files. "HIDDEN"
    is a Polled device whose [Device] memory the host does not address.
    "MAPLESS" is a Polled device that maps no host memory. "UNTIMED" is a Polled
    device that times no work. "BOUNDED" is a Polled device of 2{^ 20} bytes of
    [Device] memory, its budget, which reports its capacity. "FAULTY" is a
    device whose next fallible call faults with "the agent's device faulted",
    which loses it. "METAL" is the machine's Metal GPU, an error off a Mac or
    without one. "NONE" is an opener that fails with "no such hardware".
    "COLLECT" runs a full major collection and opens no device. "RAISE" is an
    opener that raises [Failure "the opener raised"]. "LOUD" prints "opening
    LOUD" and opens no device. *)
