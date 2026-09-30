(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interception: an interpreter receives each operation its extent performs,
   once, and none from elsewhere; its own operations reach the interpretation
   around it; what it raises, the performer sees; and no result depends on
   whether an interpreter is installed anywhere. *)

open Windtrap
module E = Nx.Op

let x = Nx.create Nx.float32 [| 3 |] [| 1.; -2.; 3. |]

(* An interpreter that records the name of each operation it receives. *)
let recording () =
  let names = ref [] in
  let run op =
    names := E.name op :: !names;
    E.eval op
  in
  ({ E.run }, names)

let names = list string

let extent =
  group "extent"
    [
      test "each operation reaches the interpreter once" (fun () ->
          let i, seen = recording () in
          ignore (E.intercept i (fun () -> Nx.neg (Nx.add x x)));
          equal names [ "neg"; "add" ] !seen);
      test "operations outside the extent do not" (fun () ->
          let i, seen = recording () in
          ignore (E.intercept i (fun () -> Nx.add x x));
          ignore (Nx.neg x);
          equal names [ "add" ] !seen);
      test "the interpreter's operations reach the enclosing one" (fun () ->
          let outer, seen = recording () in
          let run op =
            ignore (Nx.neg x);
            E.eval op
          in
          ignore
            (E.intercept outer (fun () ->
                 E.intercept { run } (fun () -> Nx.add x x)));
          equal names [ "add"; "neg" ] !seen);
      test "a domain spawned inside the extent is outside it" (fun () ->
          let i, seen = recording () in
          let y =
            E.intercept i (fun () ->
                Domain.join (Domain.spawn (fun () -> Nx.add x x)))
          in
          equal names [] !seen;
          equal (array float_exact) [| 2.; -4.; 6. |] (Nx.to_array y));
    ]

let asking =
  group "intercepted"
    [
      test "is false outside every extent" (fun () ->
          is_false (E.intercepted ()));
      test "is true inside one" (fun () ->
          is_true (E.intercept { run = E.eval } E.intercepted));
      test "is false inside the only interpreter" (fun () ->
          let answer = ref true in
          let run op =
            answer := E.intercepted ();
            E.eval op
          in
          ignore (E.intercept { run } (fun () -> Nx.add x x));
          is_false !answer);
    ]

let failing =
  group "exceptions"
    [
      test "one the interpreter raises reaches the performer" (fun () ->
          let run _ = raise Exit in
          is_true
            (E.intercept { run } (fun () ->
                 match Nx.add x x with _ -> false | exception Exit -> true)));
      test "one that ends the extent ends the interception" (fun () ->
          (match E.intercept { run = E.eval } (fun () -> raise Exit) with
          | () -> ()
          | exception Exit -> ());
          is_false (E.intercepted ()));
    ]

(* A program over many kinds of operation: elementwise, comparison, selection,
   reduction, scan, product, movement, sorting and conversion. *)
let program () =
  let m = Nx.reshape [| 3; 1 |] x in
  let p = Nx.matmul m (Nx.transpose m) in
  let s = Nx.where (Nx.less p (Nx.zeros_like p)) (Nx.neg p) (Nx.exp p) in
  let c = Nx.cumsum ~axis:1 s in
  let at = Nx.cast Nx.float32 (Nx.argmax x) in
  Nx.concatenate ~axis:0
    [
      Nx.sum ~axes:[ 0 ] c;
      fst (Nx.sort x);
      Nx.broadcast_to [| 3 |] (Nx.reshape [| 1 |] at);
    ]

let unobservable =
  group "the gate"
    [
      test "an identity interpreter changes no result" (fun () ->
          equal (array float_exact)
            (Nx.to_array (program ()))
            (Nx.to_array (E.intercept { run = E.eval } program)));
      test "an interpreter on another domain changes no result" (fun () ->
          let entered = Atomic.make false and finished = Atomic.make false in
          let other =
            Domain.spawn (fun () ->
                E.intercept { run = E.eval } (fun () ->
                    Atomic.set entered true;
                    while not (Atomic.get finished) do
                      Domain.cpu_relax ()
                    done))
          in
          while not (Atomic.get entered) do
            Domain.cpu_relax ()
          done;
          let y =
            Fun.protect ~finally:(fun () -> Atomic.set finished true) program
          in
          Domain.join other;
          equal (array float_exact) (Nx.to_array (program ())) (Nx.to_array y));
    ]

let () = exit (run "nx interception" [ extent; asking; failing; unobservable ])
