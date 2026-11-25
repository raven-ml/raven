(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtypes =
  Nx.
    [
      Dtype float64;
      Dtype float32;
      Dtype float16;
      Dtype bfloat16;
      Dtype float8_e4m3fn;
      Dtype float8_e5m2;
      Dtype float4_e2m1fn;
    ]

let dtype_name (Dtype dt) = Nx.Dtype.name dt
let floats t = Nx.to_array (Nx.cast Nx.float64 t)
let of_floats dt shape xs = Nx.cast dt (Nx.create Nx.float64 shape xs)

(* A tensor as its shape and its elements' bits. *)
let value = pair (array int) (array float_exact)
let value_of t = (Nx.shape t, floats t)
let embedding64 = Nx.Ptree.instantiate (module Embedding)

(* Two software devices, minted when a test first needs them. *)
let two =
  lazy
    (Nx.devices
       (List.map
          (fun n -> Result.get_ok (Rig.software n))
          [ "kaun-embedding-0"; "kaun-embedding-1" ]))

(* A witness for placements of one set: each set has its own brand. *)
let placement () =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)

(* An initializer that records the fans it is applied with. *)
let recording fans key ~fan_in ~fan_out dt shape =
  fans := (fan_in, fan_out) :: !fans;
  Init.zeros key ~fan_in ~fan_out dt shape

(* The rows [apply] gathers, read from [table]'s elements, [vocab] rows of
   [dim]: row [id] for an id in [0, vocab), zeros otherwise. *)
let lookup ~vocab ~dim table ids =
  Array.concat
    (List.map
       (fun id ->
         if id < 0L || id >= Int64.of_int vocab then Array.make dim 0.
         else Array.sub table (Int64.to_int id * dim) dim)
       (Array.to_list ids))

(* An id near [0, vocab), or anywhere among the int64s. *)
let id ~vocab =
  Gen.frequency
    [ (4, Gen.int64_range (-2L) (Int64.of_int (vocab + 1))); (1, Gen.int64) ]

let out_of_range ~vocab ids =
  Array.exists (fun i -> i < 0L || i >= Int64.of_int vocab) ids

(* A table at one of [dtypes] and ids of rank 0 to 3. *)

type case = {
  dtype : dtype;
  vocab : int;
  dim : int;
  table : float array;
  shape : int array;
  ids : int64 array;
}

let pp_case ppf c =
  let pp_array pp = Format.(pp_print_array ~pp_sep:pp_print_space pp) in
  Format.fprintf ppf "@[<v>%s, vocab %d, dim %d@,table [%a]@,ids [%a] [%a]@]"
    (dtype_name c.dtype) c.vocab c.dim
    (pp_array Format.pp_print_float)
    c.table
    (pp_array Format.pp_print_int)
    c.shape
    (pp_array (fun ppf i -> Format.fprintf ppf "%LdL" i))
    c.ids

let case =
  let open Gen in
  with_pp pp_case
  @@ let* dtype = of_list dtypes
     and+ vocab = int_range 1 5
     and+ dim = int_range 1 3
     and+ shape = array ~size:(int_range 0 3) (int_range 0 3) in
     let n = Array.fold_left ( * ) 1 shape in
     let+ table = array ~size:(constant (vocab * dim)) any_float
     and+ ids = array ~size:(constant n) (id ~vocab) in
     { dtype; vocab; dim; table; shape; ids }

(* Rows 0 to 3 of [[3i; 3i + 1; 3i + 2]]. *)
let rows_4x3 ids =
  {
    dtype = Dtype Nx.float32;
    vocab = 4;
    dim = 3;
    table = Array.init 12 float_of_int;
    shape = [| Array.length ids |];
    ids;
  }

let apply_examples =
  [
    rows_4x3 [| 1L; 3L |];
    (* 6a5a4f00e3 *)
    rows_4x3 [| 4L; 1L; -1L |];
    rows_4x3 [| Int64.min_int; Int64.max_int; 0L |];
    { (rows_4x3 [| 2L |]) with shape = [||] };
  ]

let law_gathers c =
  let (Dtype dt) = c.dtype in
  let table = of_floats dt [| c.vocab; c.dim |] c.table in
  let ids = Nx.create Nx.int64 c.shape c.ids in
  List.iter (fun d -> cover ("at " ^ dtype_name d) (d == c.dtype)) dtypes;
  cover "a scalar id" (c.shape = [||]);
  cover "no id" (Array.length c.ids = 0);
  cover "an id outside [0, vocab)" (out_of_range ~vocab:c.vocab c.ids);
  cover "an id that occurs twice"
    (Array.length c.ids
    > List.length (List.sort_uniq compare (Array.to_list c.ids)));
  equal value
    ( Array.append c.shape [| c.dim |],
      lookup ~vocab:c.vocab ~dim:c.dim (floats table) c.ids )
    (value_of (Embedding.apply { table } ids))

(* The compiled lookup in a table of [[4; 3]], three ids at a time: one program,
   which every case calls. *)
let compiled_lookup :
    (Nx.host Nx.float32_t -> Nx.host Nx.int64_t -> Nx.host Nx.float32_t) Lazy.t
    =
  lazy
    (Rune.jit
       Nx.Ptree.(tensor @-> tensor @-> returns tensor)
       (fun table ids -> Embedding.apply { table } ids))

let law_compiled ids =
  Compiled.here ();
  let table = Array.init 12 float_of_int in
  let ids' = Nx.create Nx.int64 [| 3 |] ids in
  cover "an id below 0" (Array.exists (fun i -> i < 0L) ids);
  cover "an id at or above vocab" (Array.exists (fun i -> i >= 4L) ids);
  cover "an id in [0, vocab)" (Array.exists (fun i -> i >= 0L && i < 4L) ids);
  let evaluated = Rune.evaluations () in
  let y =
    Lazy.force compiled_lookup (Nx.create Nx.float32 [| 4; 3 |] table) ids'
  in
  equal ~msg:"calls evaluated by nx" int evaluated (Rune.evaluations ());
  equal value ([| 3; 3 |], lookup ~vocab:4 ~dim:3 table ids) (value_of y)

(* The gradient of [Σ apply p ids * c], [c] small integers: each row's cotangent
   summed once per occurrence of its id, exact in float64. *)
let law_gradient (vocab, ids, c) =
  let dim = 2 in
  let expected = Array.make (vocab * dim) 0. in
  Array.iteri
    (fun k id ->
      if id >= 0L && id < Int64.of_int vocab then
        for j = 0 to dim - 1 do
          let r = (Int64.to_int id * dim) + j in
          expected.(r) <- expected.(r) +. c.((k * dim) + j)
        done)
    ids;
  let n = Array.length ids in
  let ids' = Nx.create Nx.int64 [| n |] ids in
  let c' = Nx.create Nx.float64 [| n; dim |] c in
  let p = { Embedding.table = Nx.full Nx.float64 [| vocab; dim |] 0.5 } in
  let g =
    Rune.grad embedding64
      (fun p -> Nx.sum (Nx.mul (Embedding.apply p ids') c'))
      p
  in
  cover "an id that occurs twice"
    (n > List.length (List.sort_uniq compare (Array.to_list ids)));
  cover "an id outside [0, vocab)" (out_of_range ~vocab ids);
  equal value ([| vocab; dim |], expected) (value_of g.table)

let gradient_case =
  let open Gen in
  let* vocab = int_range 1 4 and+ n = int_range 0 6 in
  let+ ids = array ~size:(constant n) (id ~vocab)
  and+ c =
    array ~size:(constant (n * 2)) (map float_of_int (int_range (-4) 4))
  in
  (vocab, ids, c)

(* t *)

let test_shape () =
  let p = Embedding.init ~key:(Nx.Rng.key 0) ~vocab:7 ~dim:4 in
  equal (array int) [| 7; 4 |] (Nx.shape p.table)

(* make *)

let test_fans () =
  let fans = ref [] in
  ignore
    (Embedding.make ~init:(recording fans) ~key:(Nx.Rng.key 0) ~vocab:7 ~dim:4
       Nx.float32);
  equal (list (pair int int)) [ (4, 4) ] !fans

let test_default () =
  let key = Nx.Rng.key 1 in
  let p = Embedding.make ~key ~vocab:7 ~dim:4 Nx.float64 in
  equal value
    (value_of
       (Init.normal ~stddev:1. key ~fan_in:4 ~fan_out:4 Nx.float64 [| 7; 4 |]))
    (value_of p.table)

let test_placed_key () =
  let (module Two) = Lazy.force two in
  let key = Nx.Rng.key 3 in
  let host = Embedding.init ~key ~vocab:5 ~dim:2 in
  let placed = Embedding.init ~key:(Nx.Rng.place Two.on key) ~vocab:5 ~dim:2 in
  equal (placement ()) Two.on (Nx.placement placed.table);
  equal ~msg:"the values the host key draws" (array float_exact)
    (floats host.table) (floats placed.table)

let test_make_refuses =
  cases ~id:fst "raises if vocab or dim is not positive"
    [ ("vocab 0", (0, 3)); ("dim 0", (3, 0)); ("dim -1", (3, -1)) ]
    (fun (_, (vocab, dim)) ->
      raises_match (Exn.invalid_arg ~sub:"Embedding.make") (fun () ->
          Embedding.make ~key:(Nx.Rng.key 0) ~vocab ~dim Nx.float32))

(* init *)

let test_init () =
  let key = Nx.Rng.key 2 in
  equal value
    (value_of (Embedding.make ~key ~vocab:7 ~dim:4 Nx.float32).table)
    (value_of (Embedding.init ~key ~vocab:7 ~dim:4).table)

(* walk *)

let test_visits () =
  equal (list string) [ "table: a leaf" ]
    (List.map
       (Format.asprintf "%a" Nx.Ptree.pp_visit)
       (Nx.Ptree.visits embedding64
          (Embedding.make ~key:(Nx.Rng.key 0) ~vocab:5 ~dim:2 Nx.float64)))

let test_cast () =
  let p = Embedding.init ~key:(Nx.Rng.key 0) ~vocab:5 ~dim:2 in
  equal string "bfloat16"
    (Nx.Dtype.name
       (Nx.dtype (Nx.Ptree.cast (module Embedding) Nx.bfloat16 p).table))

let () =
  exit
    (run "kaun embedding"
       [
         group "t" [ test "table is [vocab; dim]" test_shape ];
         group "make"
           [
             test "init gets dim as both fans" test_fans;
             test "init defaults to Init.normal ~stddev:1.0" test_default;
             test "a table drawn from a placed key lies there" test_placed_key;
             test_make_refuses;
           ];
         group "init" [ test "is make at float32" test_init ];
         group "apply"
           [
             prop ~count:200 ~examples:apply_examples
               "slice (i, ..., :) is row ids.(i, ...), or zeros for an id \
                outside [0, vocab)"
               case law_gathers;
             prop
               ~examples:
                 [
                   (* 6a5a4f00e3 *)
                   [| 4L; 1L; -1L |];
                   [| Int64.min_int; Int64.max_int; 3L |];
                 ]
               "under Rune.jit, an id outside [0, vocab) also yields zeros"
               Gen.(array ~size:(constant 3) (id ~vocab:4))
               law_compiled;
             prop
               ~examples:[ (3, [| 0L; 2L; 0L |], [| 1.; 1.; 1.; 1.; 1.; 1. |]) ]
               "the table's gradient accumulates each row's cotangent once per \
                occurrence of its id"
               gradient_case law_gradient;
           ];
         group "walk"
           [
             test "visits table" test_visits;
             test "Nx.Ptree.cast converts the table" test_cast;
           ];
       ])
