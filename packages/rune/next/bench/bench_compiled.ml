(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune.next's Compiled backend on the host, kernel by kernel. A compiled gather
   costs one load per output only where tolk folds the one-hot sum its lowering
   builds into a load at the computed index. Without the fold, the 1Mi row is a
   sum of 2^40 terms, so its baseline guards the fold. *)

let kernels = Nx_backend.kernels Rune_next.Compiled.backend

let host_array x =
  match Nx.Repr.v (Nx.copy x) with
  | Host a -> a
  | Placed _ | Traced _ -> invalid_arg "bench_compiled: not a host value"

(* [n] float32 values gathered at [n] uniform indices in [0, n). Compilation
   happens in the warm-up. *)
let gather id n =
  let (module K : Nx_backend.S) = kernels in
  Thumper.bench_with_setup ~id
    ~setup:(fun () ->
      let st = Random.State.make [| 15 |] in
      let x = Nx.init Nx.float32 [| n |] (fun _ -> Random.State.float st 1.) in
      let i =
        Nx.init Nx.int32 [| n |] (fun _ -> Int32.of_int (Random.State.int st n))
      in
      (host_array i, host_array x, host_array (Nx.zeros Nx.float32 [| n |])))
    id
    (fun (i, x, dst) ->
      K.gather ~axis:0 i x ~dst;
      Nx_device.synchronize Nx_device.host)

let () =
  Thumper.run "compiled"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    [
      Thumper.group ~id:"gather" "gather"
        [ gather "float32-1Mi-from-1Mi-host" (1 lsl 20) ];
    ]
