(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { run : Run.t; mutex : Mutex.t; mutable closed : bool }

let create ?base_dir ?experiment ?(tags = []) ?(config = []) () =
  let run = Run.create ?base_dir ?experiment ~tags ~config () in
  { run; mutex = Mutex.create (); closed = false }

let write_scalar t ~step ~epoch ~tag value =
  let event =
    Event.Scalar { step; epoch; tag; value; wall_time = Unix.gettimeofday () }
  in
  Mutex.lock t.mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock t.mutex)
    (fun () -> if not t.closed then Run.append_event t.run event)

let log_scalar t ~step ?epoch ~tag value =
  write_scalar t ~step ~epoch ~tag value

let log_scalars t ~step ?epoch pairs =
  List.iter (fun (tag, value) -> write_scalar t ~step ~epoch ~tag value) pairs

let run_id t = Run.run_id t.run
let run_dir t = Run.dir t.run

let close t =
  Mutex.lock t.mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock t.mutex)
    (fun () -> t.closed <- true)
