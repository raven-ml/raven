(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The words an operation allocates on the minor heap, around a kernel with
   almost nothing to do: the dispatch cost's allocation, exactly. Counts are
   deterministic, so each is pinned: a change that lowers one updates it here,
   in the same commit. *)

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

let one () = Nx.create Nx.float32 [| 1 |] [| 0.5 |]
let a = one ()
let b = one ()
let mask = Nx.less a b
let mat = Nx.create Nx.float32 [| 1; 1 |] [| 0.5 |]
let row = Nx.create Nx.float32 [| 1024 |] (Array.make 1024 0.5)

let dispatch =
  group "one-element operations"
    [
      test "add" (fun () -> equal int 103 (words (fun () -> Nx.add a b)));
      test "less" (fun () -> equal int 103 (words (fun () -> Nx.less a b)));
      test "where" (fun () ->
          equal int 106 (words (fun () -> Nx.where mask a b)));
      test "sum" (fun () -> equal int 138 (words (fun () -> Nx.sum a)));
      test "matmul" (fun () ->
          equal int 122 (words (fun () -> Nx.matmul mat mat)));
      test "zeros" (fun () ->
          equal int 120 (words (fun () -> Nx.zeros Nx.float32 [| 1 |])));
      test "shape of a vector" (fun () ->
          equal int 9 (words (fun () -> Nx.shape row)));
    ]

let () = exit (run "nx allocation" [ dispatch ])
