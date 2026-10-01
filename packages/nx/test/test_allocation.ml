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
      test "add" (fun () -> equal int 64 (words (fun () -> Nx.add a b)));
      test "less" (fun () -> equal int 64 (words (fun () -> Nx.less a b)));
      test "where" (fun () ->
          equal int 64 (words (fun () -> Nx.where mask a b)));
      test "sum" (fun () -> equal int 114 (words (fun () -> Nx.sum a)));
      test "matmul" (fun () ->
          equal int 90 (words (fun () -> Nx.matmul mat mat)));
      test "zeros" (fun () ->
          equal int 77 (words (fun () -> Nx.zeros Nx.float32 [| 1 |])));
      test "shape of a vector" (fun () ->
          equal int 2 (words (fun () -> Nx.shape row)));
    ]

let per_element =
  group "allocation per element"
    [
      (* Both lengths give arrays of more than 64 KiB, which the host allocates
         on a page and by a path of its own. *)
      cases "arange allocates as many words for 2^20 elements as for 2^15"
        ~name:(fun (name, _) -> name)
        [
          ("int64", fun n -> ignore (Nx.arange Nx.int64 0 n 1));
          ("int32, through a cast", fun n -> ignore (Nx.arange Nx.int32 0 n 1));
        ]
        (fun (_, arange) ->
          equal int
            (words (fun () -> arange (1 lsl 15)))
            (words (fun () -> arange (1 lsl 20))));
    ]

(* With no interception anywhere, a host operation allocates its result's array,
   which nx.cpu's kernel writes, and the two words of the [Host] block around
   it. *)
let host_path =
  group "host path"
    [
      test "add allocates its result" (fun () ->
          let shape = [| 1 |] in
          let alloc () =
            let view = Nx_array.View.create shape in
            let buffer = Nx_array.Elements.create Nx.float32 1 in
            { Nx_array.dtype = Nx.float32; view; buffer }
          in
          let x = alloc () and y = alloc () in
          let kernel () =
            let dst = alloc () in
            Nx_cpu.binary Add x y ~dst;
            dst
          in
          equal int (words kernel + 2) (words (fun () -> Nx.add a b)));
    ]

let () = exit (run "nx allocation" [ dispatch; per_element; host_path ])
