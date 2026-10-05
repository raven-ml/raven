(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The words reverse mode and a compiled call's replay allocate on the minor
   heap over one-element values, whose kernels have almost nothing to do: the
   cost of recording and transposing each operation, and of keying and running a
   program, exactly. Counts are deterministic, so each is pinned: a change that
   lowers one updates it here, in the same commit. *)

open Windtrap

(* The fewest words any of ten calls allocates, after a warm-up call: a
   finaliser that happens to run inside one call cannot inflate the count. *)
let words f =
  ignore (Sys.opaque_identity (f ()));
  let fewest = ref max_int in
  for _ = 1 to 10 do
    let before = Gc.minor_words () in
    ignore (Sys.opaque_identity (f ()));
    fewest := min !fewest (int_of_float (Gc.minor_words () -. before))
  done;
  !fewest

let x = Nx.create Nx.float32 [| 1 |] [| 0.5 |]
let w = Nx.create Nx.float32 [| 1; 1 |] [| 0.5 |]
let row = Nx.reshape [| 1; 1 |] x

(* A selection: [x] where it is positive, [0.] elsewhere. *)
let relu x =
  let zero = Nx.scalar_like x 0. in
  Nx.where (Nx.less_equal x zero) zero x

let reverse =
  group "reverse mode"
    [
      test "grad of a product" (fun () ->
          equal int 1338
            (words (fun () -> Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) x)));
      test "grad of a selection" (fun () ->
          equal int 1695
            (words (fun () -> Rune.grad' (fun x -> Nx.sum (relu x)) x)));
      test "grad of a matrix product" (fun () ->
          equal int 1212
            (words (fun () -> Rune.grad' (fun w -> Nx.sum (Nx.matmul row w)) w)));
    ]

(* A compiled call that replays its program: the cost of keying, binding and
   running it, with the settings it keys on read as they are. *)
let neg = Rune.jit' Nx.neg

let replay =
  group "jit replay"
    [
      test "neg of one element" (fun () ->
          equal int 828 (words (fun () -> neg x)));
    ]

let () = exit (run "rune allocation" [ reverse; replay ])
