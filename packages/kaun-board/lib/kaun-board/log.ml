(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { run : Run.t; mutex : Mutex.t; mutable closed : bool }

let create ?base_dir ?experiment ?(tags = []) ?(config = []) () =
  let run = Run.create ?base_dir ?experiment ~tags ~config () in
  { run; mutex = Mutex.create (); closed = false }

let write_scalar t ~step ~epoch ~tag ~minimize value =
  let event =
    Event.Scalar
      {
        step;
        epoch;
        tag;
        value;
        wall_time = Unix.gettimeofday ();
        minimize;
      }
  in
  Mutex.lock t.mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock t.mutex)
    (fun () -> if not t.closed then Run.append_event t.run event)

let scalars t ~step ?epoch triples =
  List.iter
    (fun (tag, value, minimize) ->
      write_scalar t ~step ~epoch ~tag ~minimize value)
    triples

let run_id t = Run.run_id t.run
let run_dir t = Run.dir t.run

let close t =
  Mutex.lock t.mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock t.mutex)
    (fun () -> t.closed <- true)
