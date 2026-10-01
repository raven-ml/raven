(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's array primitives at a talon morsel, 4e4 rows, and at 1e7 rows, under the
   ids bench_twins.py measures numpy, pandas and polars by. Inputs come from
   OCaml's [Random], so they are the same in every run and do not depend on
   [Nx.Rng].

   [--transient] prints, instead of timing, the bytes of host arrays each row's
   call allocates beyond its result's, or [-] for a call that allocates enough
   on the OCaml heap to collect. *)

type row =
  | Row : {
      id : string;
      rows : int;
      setup : unit -> 'e;
      run : 'e -> ('a, 'b) Nx.t;
    }
      -> row

let row id rows setup run = Row { id; rows; setup; run }
let state () = Random.State.make [| 15 |]

let tensor kind n f =
  let ba = Bigarray.Array1.init kind Bigarray.c_layout n f in
  Nx.of_bigarray (Bigarray.genarray_of_array1 ba)

let uniform_float64 n =
  let st = state () in
  tensor Bigarray.float64 n (fun _ -> Random.State.float st 1.)

let uniform_float32 n =
  let st = state () in
  tensor Bigarray.float32 n (fun _ -> Random.State.float st 1.)

let words n =
  let st = state () in
  Nx.bitcast Nx.uint64
    (tensor Bigarray.int64 n (fun _ -> Random.State.bits64 st))

let int64s n bound =
  let st = state () in
  tensor Bigarray.int64 n (fun _ -> Int64.of_int (Random.State.int st bound))

let indices n bound =
  let st = state () in
  tensor Bigarray.int32 n (fun _ -> Int32.of_int (Random.State.int st bound))

let monotone n = fst (Nx.sort (indices n n))

(* The sizes a row is named after. *)
let s = 40_000
let l = 10_000_000

let arange =
  [
    row "int64-4e4" s ignore (fun () -> Nx.arange Nx.int64 0 s 1);
    row "int64-1e7" l ignore (fun () -> Nx.arange Nx.int64 0 l 1);
    row "int32-1e7" l ignore (fun () -> Nx.arange Nx.int32 0 l 1);
  ]

let argsort =
  [
    row "float64-4e4" s (fun () -> uniform_float64 s) Nx.argsort;
    row "float64-1e7" l (fun () -> uniform_float64 l) Nx.argsort;
    row "uint64-4e4" s (fun () -> words s) Nx.argsort;
    row "uint64-1e7" l (fun () -> words l) Nx.argsort;
  ]

let cumsum =
  [
    row "float64-4e4" s (fun () -> uniform_float64 s) Nx.cumsum;
    row "float64-1e7" l (fun () -> uniform_float64 l) Nx.cumsum;
    row "int64-1e7" l (fun () -> int64s l (1 lsl 20)) Nx.cumsum;
  ]

let scatter =
  let updates id mode n m =
    row id n
      (fun () -> (uniform_float64 n, indices n m, Nx.zeros Nx.float64 [| m |]))
      (fun (values, indices, base) ->
        Nx.scatter ~mode ~axis:0 ~indices ~values base)
  in
  [
    updates "add-float64-4e4-into-1e2" `Add s 100;
    updates "add-float64-1e7-into-1e2" `Add l 100;
    updates "add-float64-1e7-into-1e6" `Add l 1_000_000;
    updates "set-float64-4e4-into-4e4" `Set s s;
    updates "set-float64-1e7-into-1e6" `Set l 1_000_000;
  ]

let gather =
  let take ~axis (x, indices) = Nx.take ~axis ~indices x in
  let rows n = (Nx.reshape [| n; 8 |] (uniform_float32 (8 * n)), indices n n) in
  [
    row "float32-4e4-monotone" s
      (fun () -> (uniform_float32 s, monotone s))
      (take ~axis:0);
    row "float32-1e7-monotone" l
      (fun () -> (uniform_float32 l, monotone l))
      (take ~axis:0);
    row "float64-1e7-random" l
      (fun () -> (uniform_float64 l, indices l l))
      (take ~axis:0);
    row "float32-rows-4e4x8" s (fun () -> rows s) (take ~axis:0);
    row "float32-rows-1.25e6x8" 1_250_000
      (fun () -> rows 1_250_000)
      (take ~axis:0);
  ]

let groups =
  [
    ("arange", arange);
    ("argsort", argsort);
    ("cumsum", cumsum);
    ("scatter", scatter);
    ("gather", gather);
  ]

(* The bytes of host arrays [f] allocates beyond its result's: the rises of the
   host's allocated bytes during the call, less the result's bytes, or [None] if
   a collection ran during the call. A collection returns dead arrays' bytes
   within the next rise, so the call runs with the collector held off. *)
let transient f =
  let host = Nx_device.host in
  let collections () =
    let s = Gc.quick_stat () in
    (s.minor_collections, s.major_collections)
  in
  let gc = Gc.get () in
  Gc.full_major ();
  let base = Nx_device.Stats.allocated (Nx_device.stats host) in
  Gc.set
    {
      gc with
      space_overhead = 1_000_000;
      custom_major_ratio = 1_000_000;
      custom_minor_ratio = 1_000_000;
    };
  let before = collections () in
  let p = Nx_device.Profile.start () in
  let r = f () in
  let events = Nx_device.Profile.stop p in
  let collected = collections () <> before in
  Gc.set gc;
  if collected then None
  else
    let rise (level, total) = function
      | Nx_device.Profile.Allocation { device; allocated; _ }
        when Nx_device.equal device host ->
          (allocated, total + Int.max 0 (allocated - level))
      | _ -> (level, total)
    in
    Some (snd (List.fold_left rise (base, 0) events) - Nx.nbytes r)

let print_transient () =
  List.iter
    (fun (group, rows) ->
      List.iter
        (fun (Row { id; rows; setup; run }) ->
          let env = setup () in
          match transient (fun () -> run env) with
          | None -> Printf.printf "%s/%s\t-\t-\n%!" group id
          | Some bytes ->
              Printf.printf "%s/%s\t%d\t%.1f\n%!" group id bytes
                (float_of_int bytes /. float_of_int rows))
        rows)
    groups

let () =
  match Sys.argv with
  | [| _; "--transient" |] -> print_transient ()
  | _ ->
      (* A row of 1e7 elements takes up to a second a call, so its 20 batches
         take longer than the default deadline. *)
      Thumper.run "nx_primitives"
        ~config:Thumper.Config.(default |> deadline 600.)
        ~budgets:
          [
            Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
            Thumper.Budget.no_more_alloc_than 0.01;
          ]
        (List.map
           (fun (group, rows) ->
             Thumper.group ~id:group group
               (List.map
                  (fun (Row { id; setup; run; _ }) ->
                    Thumper.bench_with_setup ~id ~setup id run)
                  rows))
           groups)
