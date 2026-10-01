(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's array primitives at a talon morsel, 4e4 rows, and at 1e7 rows, under the
   ids bench_twins.py measures numpy, pandas and polars by. Inputs come from
   OCaml's [Random], so they are the same in every run and do not depend on
   [Nx.Rng].

   [--transient] prints, instead of timing, the bytes of host arrays each row's
   call allocates beyond its results', or [-] for a call that allocates enough
   on the OCaml heap to collect. [--transient] counts host arrays only: a
   kernel's C scratch, such as narrow scatter Add's 5 bytes per position
   (nx_c.h), is not counted. *)

type row =
  | Row : {
      id : string;
      rows : int;
      setup : unit -> 'e;
      run : 'e -> 'r;
      results : 'r -> Nx.packed list;  (** The tensors a call returns. *)
    }
      -> row

let row id rows setup run =
  Row { id; rows; setup; run; results = (fun r -> [ Nx.P r ]) }

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
  tensor Bigarray.int64 n (fun _ -> Int64.of_int (Random.State.int st bound))

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
  let updates_of dtype id mode n m =
    row id n
      (fun () ->
        (Nx.cast dtype (uniform_float64 n), indices n m, Nx.zeros dtype [| m |]))
      (fun (values, indices, base) ->
        Nx.scatter ~mode ~axis:0 ~indices ~values base)
  in
  let updates id = updates_of Nx.float64 id in
  [
    updates "add-float64-4e4-into-1e2" `Add s 100;
    updates "add-float64-1e7-into-1e2" `Add l 100;
    updates "add-float64-1e7-into-1e6" `Add l 1_000_000;
    updates "set-float64-4e4-into-4e4" `Set s s;
    updates "set-float64-1e7-into-1e6" `Set l 1_000_000;
    updates "max-float64-4e4-into-1e2" `Max s 100;
    updates "max-float64-1e7-into-1e2" `Max l 100;
    updates "max-float64-1e7-into-1e6" `Max l 1_000_000;
    updates_of Nx.float16 "add-float16-1e7-into-1e6" `Add l 1_000_000;
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

let mask n = Nx.less_s (uniform_float64 n) 0.5

let positions =
  [
    row "mask50-4e4" s (fun () -> mask s) Nx.positions;
    row "mask50-1e7" l (fun () -> mask l) Nx.positions;
    row "counts-1e7" l (fun () -> int64s l 4) Nx.positions;
  ]

let compress =
  let masked id n =
    row id n
      (fun () -> (uniform_float64 n, mask n))
      (fun (x, condition) -> Nx.compress ~condition x)
  in
  [ masked "float64-4e4-mask50" s; masked "float64-1e7-mask50" l ]

(* Ascending by an int64 key of 1000 values, then by a float64 one. *)
let lexsort =
  let two_keys n () = (int64s n 1000, uniform_float64 n) in
  let sort (a, b) =
    Nx.lexsort (Nx.stack ~axis:1 [ Nx.order_key a; Nx.order_key b ])
  in
  [
    row "int64-float64-4e4" s (two_keys s) sort;
    row "int64-float64-1e7" l (two_keys l) sort;
  ]

let searchsorted =
  let into m () = (fst (Nx.sort (uniform_float64 m)), uniform_float64 l) in
  let search (knots, q) = Nx.searchsorted ~side:`Right knots q in
  [
    row "float64-1e7-into-1e3" l (into 1_000) search;
    row "float64-1e7-into-1e6" l (into 1_000_000) search;
  ]

let unique =
  let keys id n d =
    Row
      {
        id;
        rows = n;
        setup = (fun () -> int64s n d);
        run = Nx.unique;
        results = (fun (g : Nx.groups) -> Nx.[ P g.ids; P g.first; P g.counts ]);
      }
  in
  [
    keys "int64-4e4-1e2" s 100;
    keys "int64-4e4-1e4" s 10_000;
    keys "int64-1e7-1e2" l 100;
    keys "int64-1e7-1e6" l 1_000_000;
  ]

let quantile =
  let box = Nx.quantile [| 0.; 0.25; 0.5; 0.75; 1. |] in
  [
    row "float64-4e4" s (fun () -> uniform_float64 s) box;
    row "float64-1e7" l (fun () -> uniform_float64 l) box;
  ]

let bits =
  let mask () =
    let st = state () in
    Nx.cast Nx.bool
      (tensor Bigarray.int8_unsigned l (fun _ -> Random.State.int st 2))
  in
  let bitmap () = Nx.Bits.of_bool (mask ()) in
  [
    row "of_bool-1e7" l mask (fun m -> fst (Nx.Bits.bytes (Nx.Bits.of_bool m)));
    row "to_bool-1e7" l bitmap Nx.Bits.to_bool;
    row "count-1e7" l bitmap Nx.Bits.count;
  ]

(* [n] strings of [w] random lowercase letters, as offsets and bytes, and a
   permutation of them. *)
let strings n w =
  let st = state () in
  let offsets =
    Bigarray.Array1.init Bigarray.int64 Bigarray.c_layout (n + 1) (fun i ->
        Int64.of_int (i * w))
  in
  let bytes =
    Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout (n * w)
      (fun _ -> 97 + Random.State.int st 26)
  in
  let perm =
    Bigarray.Array1.init Bigarray.int64 Bigarray.c_layout n Int64.of_int
  in
  for i = n - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let t = perm.{i} in
    perm.{i} <- perm.{j};
    perm.{j} <- t
  done;
  (offsets, bytes, perm)

let nx_of a = Nx.of_bigarray (Bigarray.genarray_of_array1 a)

external run_copy_total :
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int = "bench_run_copy_total"
[@@noalloc]

external run_copy :
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  unit = "bench_run_copy"
[@@noalloc]

(* The run-copy twin allocates its results as host buffers, as nx does. *)
let copy_runs (offsets, bytes, perm) =
  let host = Nx_device.host in
  let buffer s n = Nx_device.Buffer.create host s n in
  let total = run_copy_total offsets perm in
  let out = buffer Nx_dtype.Scalar.UInt8 total in
  let out_offsets =
    buffer Nx_dtype.Scalar.Int64 (Bigarray.Array1.dim perm + 1)
  in
  run_copy offsets bytes perm
    (Nx_device.Buffer.bigarray Bigarray.int64 out_offsets)
    (Nx_device.Buffer.bigarray Bigarray.int8_unsigned out);
  Nx.of_buffer Nx.uint8 [| total |] out

let ragged_take =
  let ragged n w () =
    let offsets, bytes, perm = strings n w in
    (Nx.Ragged.v ~offsets:(nx_of offsets) (nx_of bytes), nx_of perm)
  in
  let take (r, indices) = Nx.Ragged.values (Nx.Ragged.take ~indices r) in
  [
    row "strings6-4e4-permuted" s (ragged s 6) take;
    row "strings12-1e7-permuted" l (ragged l 12) take;
    row "strings6-4e4-permuted-runcopy" s (fun () -> strings s 6) copy_runs;
    row "strings12-1e7-permuted-runcopy" l (fun () -> strings l 12) copy_runs;
  ]

let ragged_rows =
  let ragged n w () =
    let offsets, bytes, _ = strings n w in
    Nx.Ragged.v ~offsets:(nx_of offsets) (nx_of bytes)
  in
  let parts n w () =
    let offsets, bytes, _ = strings n w in
    (nx_of offsets, nx_of bytes)
  in
  let lengths n w () =
    ( Nx.full Nx.int64 [| n |] (Int64.of_int w),
      let _, bytes, _ = strings n w in
      nx_of bytes )
  in
  [
    (* [v] returns its operands, so it has no result bytes of its own. *)
    Row
      {
        id = "v-strings12-1e7";
        rows = l;
        setup = parts l 12;
        run = (fun (offsets, bytes) -> Nx.Ragged.v ~offsets bytes);
        results = (fun _ -> []);
      };
    row "of_lengths-strings12-1e7" l (lengths l 12) (fun (lengths, bytes) ->
        Nx.Ragged.offsets (Nx.Ragged.of_lengths lengths bytes));
    row "concat-strings12-2x5e6" l
      (fun () -> (ragged (l / 2) 12 (), ragged (l / 2) 12 ()))
      (fun (a, b) -> Nx.Ragged.values (Nx.Ragged.concat [ a; b ]));
    row "ids-strings6-4e4" s (ragged s 6) Nx.Ragged.ids;
    row "ids-strings12-1e7" l (ragged l 12) Nx.Ragged.ids;
    row "rank-strings12-1e7" l (ragged l 12) Nx.Ragged.rank;
    row "quantile-float64-1e7-into-1e3" l
      (fun () ->
        Nx.Ragged.of_ids ~segments:1000 (int64s l 1000) (uniform_float64 l))
      (Nx.Ragged.quantile [| 0.5 |]);
  ]

let groups =
  [
    ("arange", arange);
    ("argsort", argsort);
    ("cumsum", cumsum);
    ("scatter", scatter);
    ("gather", gather);
    ("positions", positions);
    ("compress", compress);
    ("lexsort", lexsort);
    ("searchsorted", searchsorted);
    ("unique", unique);
    ("quantile", quantile);
    ("bits", bits);
    ("ragged-take", ragged_take);
    ("ragged", ragged_rows);
  ]

(* The bytes of host arrays [f] allocates beyond its results': the rises of the
   host's allocated bytes during the call, less the bytes of the tensors
   [results] finds in its value, or [None] if a collection ran during the call.
   A collection returns dead arrays' bytes within the next rise, so the call
   runs with the collector held off. *)
let transient f results =
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
    let kept =
      List.fold_left (fun n (Nx.P t) -> n + Nx.nbytes t) 0 (results r)
    in
    Some (snd (List.fold_left rise (base, 0) events) - kept)

let print_transient () =
  List.iter
    (fun (group, rows) ->
      List.iter
        (fun (Row { id; rows; setup; run; results }) ->
          let env = setup () in
          match transient (fun () -> run env) results with
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
