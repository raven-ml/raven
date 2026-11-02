(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Readings of the machine and the process, sampled every half second into a run
   while a CPU-bound computation logs its progress. *)

open Munin

(* Counts the primes below [n] by trial division. *)
let count_primes n =
  let count = ref 0 in
  for i = 2 to n do
    let prime = ref true in
    let j = ref 2 in
    while !j * !j <= i && !prime do
      if i mod !j = 0 then prime := false;
      incr j
    done;
    if !prime then incr count
  done;
  !count

let () =
  let store = Store.open_ () in
  let id =
    Session.within
      (Session.start ~store ~experiment:"compute" ~name:"primes"
         ~params:[ ("limit", `Int 5_000_000) ]
         ~steps:10 ())
    @@ fun s ->
    Session.sample s ~every:0.5 [ Munin_sys.host (); Munin_sys.process () ];
    let found = Session.metric s "primes_found" in
    for step = 1 to 10 do
      Session.log found ~step (Float.of_int (count_primes (step * 500_000)))
    done;
    Session.id s
  in
  let h = History.read (Option.get (Store.run store id)) in
  Printf.printf "run: %s\n" id;
  List.iter
    (fun key ->
      let st = Option.get (History.stats h key) in
      Printf.printf "  %s: %d samples, last %g\n" key st.count st.last.value)
    (History.readings h)
