(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sort through Nx.Prim: the order and stability Nx_kernel.S.sort states,
   against a stable sort of the slices, sub-byte dtypes in one kernel call, its
   rule before any kernel, a split placement, and nx's expansion of a sort. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module C = Nx_support.Counting

let m = Nx_support.memory

module Count = (val Nx.devices ~kernels:(module C) [ m 0 ])
module Count2 = (val Nx.devices ~kernels:(module C) [ m 0; m 1 ])

let host_array x = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x))
let read x = A.to_array (host_array x)
let value a = Nx.Repr.of_array Nx.Host.v a

let sort ?(descending = false) ?k axis x =
  Nx.Prim.eval ~by:"t" (Sort { axis; descending; k; x })

let bits = Int64.bits_of_float
let nan_with k = Int64.float_of_bits (Int64.logor 0x7ff8_0000_0000_0000L k)

(* The order *)

let test_float_order () =
  let a = nan_with 3L and b = nan_with 5L in
  let xs =
    [| 1.; a; -0.; Float.neg_infinity; 0.; b; Float.infinity; -0.; -2. |]
  in
  let v, p = sort 0 (value (A.of_array D.Float64 [| 9 |] xs)) in
  equal ~msg:"ascending, NaNs last, each its bits, in position order"
    (array int64)
    (Array.map bits
       [| Float.neg_infinity; -2.; -0.; -0.; 0.; 1.; Float.infinity; a; b |])
    (Array.map bits (read v));
  equal ~msg:"positions" (array int64)
    [| 3L; 8L; 2L; 7L; 4L; 0L; 6L; 1L; 5L |]
    (read p);
  let v, p =
    sort ~descending:true 0 (value (A.of_array D.Float64 [| 9 |] xs))
  in
  equal ~msg:"descending keeps equal elements in increasing position"
    (array int64)
    (Array.map bits
       [| a; b; Float.infinity; 1.; 0.; -0.; -0.; -2.; Float.neg_infinity |])
    (Array.map bits (read v));
  equal ~msg:"descending positions" (array int64)
    [| 1L; 5L; 6L; 0L; 4L; 2L; 7L; 8L; 3L |]
    (read p)

let test_k () =
  let x =
    value (A.of_array D.Int32 [| 2; 4 |] [| 5l; 1l; 4l; 1l; 0l; 9l; 9l; 3l |])
  in
  let v, p = sort ~descending:true ~k:2 1 x in
  equal ~msg:"shape" (array int) [| 2; 2 |] (Nx.shape v);
  equal ~msg:"the first two of each row" (array int32) [| 5l; 4l; 9l; 9l |]
    (read v);
  equal ~msg:"positions" (array int64) [| 0L; 2L; 1L; 2L |] (read p);
  let v, p = sort ~k:0 1 x in
  equal ~msg:"none kept"
    (pair (array int) (array int))
    ([| 2; 0 |], [| 2; 0 |])
    (Nx.shape v, Nx.shape p)

(* A stable sort of each slice along [axis], as positions. *)
let reference descending s axis xs =
  let r = Array.length s in
  let n = s.(axis) in
  let inner = Array.fold_left ( * ) 1 (Array.sub s (axis + 1) (r - axis - 1)) in
  let outer = Array.fold_left ( * ) 1 (Array.sub s 0 axis) in
  let values = Array.copy xs and positions = Array.make (Array.length xs) 0L in
  for o = 0 to outer - 1 do
    for i = 0 to inner - 1 do
      let at j = (((o * n) + j) * inner) + i in
      let order =
        List.stable_sort
          (fun j l ->
            let c = Int32.compare xs.(at j) xs.(at l) in
            if descending then -c else c)
          (List.init n Fun.id)
      in
      List.iteri
        (fun t j ->
          values.(at t) <- xs.(at j);
          positions.(at t) <- Int64.of_int j)
        order
    done
  done;
  (values, positions)

let case =
  let open Gen in
  let* s = array ~size:(int_range 1 3) (int_range 0 5) in
  let* axis = int_range 0 (Array.length s - 1) in
  let* descending = bool in
  let+ xs =
    array
      ~size:(constant (Array.fold_left ( * ) 1 s))
      (map Int32.of_int (int_range (-3) 3))
  in
  (s, axis, descending, xs)

let law_stable (s, axis, descending, xs) =
  cover "an empty slice" (Array.exists (( = ) 0) s);
  cover "descending" descending;
  let v, p = sort ~descending axis (value (A.of_array D.Int32 s xs)) in
  let want_v, want_p = reference descending s axis xs in
  equal ~msg:"values" (array int32) want_v (read v);
  equal ~msg:"positions" (array int64) want_p (read p)

let test_sub_byte () =
  let check (type v s) name (dt : (v, s) D.t) (pp : v testable) (xs : v array)
      (want : v array) (positions : int64 array) =
    let x = Nx.place Count.on (value (A.of_array dt [| Array.length xs |] xs)) in
    C.reset ();
    let v, p = sort 0 x in
    equal ~msg:(name ^ " in one kernel call") int 1 (C.calls ());
    equal ~msg:(name ^ " values") (array pp) want (read v);
    equal ~msg:(name ^ " positions") (array int64) positions (read p)
  in
  check "int4" D.Int4 int [| 3; -8; 7; 0; -8 |] [| -8; -8; 0; 3; 7 |]
    [| 1L; 4L; 3L; 0L; 2L |];
  check "uint4" D.Uint4 int [| 15; 0; 9 |] [| 0; 9; 15 |] [| 1L; 2L; 0L |];
  check "bit" D.Bit bool [| true; false; true |] [| false; true; true |]
    [| 1L; 0L; 2L |];
  let x =
    Nx.place Count.on
      (value (A.of_array D.Float4_e2m1fn [| 4 |] [| 0.; -0.; 6.; -1.5 |]))
  in
  C.reset ();
  let v, p = sort 0 x in
  equal ~msg:"float4_e2m1fn in one kernel call" int 1 (C.calls ());
  equal ~msg:"float4_e2m1fn, -0 below +0" (array int64)
    (Array.map bits [| -1.5; -0.; 0.; 6. |])
    (Array.map bits (read v));
  equal ~msg:"float4_e2m1fn positions" (array int64) [| 3L; 1L; 0L; 2L |]
    (read p)

(* Rules and placements *)

let test_rules () =
  let x =
    Nx.place Count.on (value (A.of_array D.Int32 [| 2; 3 |] (Array.make 6 0l)))
  in
  C.reset ();
  raises_match (Exn.invalid_arg ~substring:"t: ") (fun () -> sort ~k:4 1 x);
  raises_match (Exn.invalid_arg ~substring:"t: ") (fun () -> sort 2 x);
  raises_match (Exn.invalid_arg ~substring:"t: ") (fun () -> sort ~k:(-1) 0 x);
  equal ~msg:"kernel calls" int 0 (C.calls ())

let test_split () =
  let xs = [| 4l; 1l; 3l; 3l; 0l; 2l; 7l; -1l |] in
  let x = value (A.of_array D.Int32 [| 4; 2 |] xs) in
  let one = sort 1 (Nx.place Count.on x) in
  let two = sort 1 (Nx.place (Count2.split ~axis:0) x) in
  equal ~msg:"values" (array int32) (read (fst one)) (read (fst two));
  equal ~msg:"positions" (array int64) (read (snd one)) (read (snd two));
  equal ~msg:"split as its operand" bool true
    (Option.equal Nx.Placement.equal
       (Nx.placement (fst two))
       (Some (Count2.split ~axis:0)))

(* Expansion *)

(* [expanded f] is [f ()] in an interpretation that computes each sort by its
   expansion ({!Nx.Prim.expand}), and every other operation as eager execution
   does. *)
let repeat n step init =
  let rec go k c =
    if k = n then c
    else go (k + 1) (step (Nx.scalar Nx.int64 (Int64.of_int k)) c)
  in
  go 0 init

let expanded f =
  let rule (type r) i ~by (op : r Nx.Prim.t) : r =
    let expansion =
      match op with Sort _ -> Nx.Prim.expand i ~by ~repeat op | _ -> None
    in
    match expansion with Some r -> r | None -> Nx.Prim.eval ~by op
  in
  Nx.Prim.interpret ~name:"test.expand" Extent rule (fun _ -> f ())

(* Each element's bits, read as unsigned integers of its width: two per
   complex128. *)
let bits_of (type v s) (a : (v, s) A.t) : int64 array =
  let a = A.copy a in
  let read (type w r) (u : (w, r) D.t) (f : w -> int64) =
    Array.map f (A.to_array (Option.get (A.bitcast u a)))
  in
  match D.bits (A.dtype a) with
  | 8 -> read D.Uint8 Int64.of_int
  | 16 -> read D.Uint16 Int64.of_int
  | 32 -> read D.Uint32 (fun w -> Int64.logand (Int64.of_int32 w) 0xffff_ffffL)
  | _ -> read D.Uint64 Fun.id

(* A dtype with elements that break an order: NaNs of two payloads and both
   signs, both zeros, the infinities, the integer extremes. *)
type elts = Elts : ('v, 's) D.t * 'v Gen.t -> elts

let floats =
  Gen.of_list
    [
      nan_with 3L;
      nan_with 5L;
      Float.neg Float.nan;
      -0.;
      0.;
      Float.infinity;
      Float.neg_infinity;
      1.;
      -1.;
      2.5;
      -2.5;
      1e-310;
    ]

let ints lo hi = Gen.of_list [ lo; hi; -1; 0; 1; 2; hi - 1 ]

let dtypes =
  let open Gen in
  let complexes =
    let+ re = floats and+ im = floats in
    { Complex.re; im }
  in
  [
    Elts (D.Float32, floats);
    Elts (D.Float64, floats);
    Elts (D.Float16, floats);
    Elts (D.Bfloat16, floats);
    Elts (D.Float8_e4m3fn, floats);
    Elts (D.Complex64, complexes);
    Elts (D.Complex128, complexes);
    Elts (D.Int8, ints (-128) 127);
    Elts (D.Uint8, of_list [ 0; 1; 2; 127; 128; 255 ]);
    Elts (D.Int16, ints (-32768) 32767);
    Elts (D.Int32, map Int32.of_int (ints (-5) 5));
    Elts (D.Uint32, of_list [ 0l; 1l; Int32.max_int; Int32.min_int; -1l ]);
    Elts
      ( D.Int64,
        of_list
          [
            Int64.min_int;
            Int64.max_int;
            -1L;
            0L;
            1L;
            0x1_0000_0000L;
            -0x1_0000_0000L;
            0xffff_ffffL;
          ] );
    Elts (D.Uint64, of_list [ 0L; 1L; -1L; Int64.min_int; 0xffff_ffffL ]);
    Elts (D.Bool, bool);
  ]

type expansion =
  | Case :
      ('v, 's) D.t * int array * int * bool * int option * 'v array
      -> expansion

let pp_expansion ppf (Case (dt, s, axis, descending, k, _)) =
  Format.fprintf ppf "%s [%s] axis %d%s%s" (D.name dt)
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))
    axis
    (if descending then " descending" else "")
    (match k with None -> "" | Some k -> Printf.sprintf " k %d" k)

let expansion_case =
  let open Gen in
  with_pp pp_expansion
  @@ let* (Elts (dt, elt)) = of_list dtypes in
     let* s = array ~size:(int_range 1 3) (int_range 0 4) in
     let* axis = int_range 0 (Array.length s - 1) in
     (* A long axis takes the network through several stages. *)
     let* long =
       frequency [ (3, constant None); (1, map Option.some (int_range 5 40)) ]
     in
     let s =
       match long with
       | None -> s
       | Some n -> Array.mapi (fun i e -> if i = axis then n else e) s
     in
     let* descending = bool in
     let* k = option (int_range 0 s.(axis)) in
     let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) elt in
     Case (dt, s, axis, descending, k, xs)

let law_expansion (Case (dt, s, axis, descending, k, xs)) =
  cover "a complex dtype"
    (match D.kind dt with D.Complex -> true | _ -> false);
  cover "two keys or more" (D.bits dt >= 64);
  cover "an axis of more than 16" (s.(axis) > 16);
  cover "k" (k <> None);
  let a = A.of_array dt s xs in
  let v, p = sort ~descending ?k axis (Nx.place Count.on (value a)) in
  let v', p' = expanded (fun () -> sort ~descending ?k axis (value a)) in
  equal ~msg:"values, bit for bit" (array int64)
    (bits_of (host_array v))
    (bits_of (host_array v'));
  equal ~msg:"positions" (array int64) (read p) (read p')

let () =
  exit
    (run "nx sort"
       [
         group "order"
           [
             test "floats: -inf < -0 < +0 < +inf < NaN, stable" test_float_order;
             test "k keeps the first elements" test_k;
             prop "each slice is a stable sort" case law_stable;
             test "sub-byte dtypes sort in one kernel call" test_sub_byte;
           ];
         group "rules and placements"
           [
             test "the rule raises before any kernel" test_rules;
             test "a split value sorts as on one device" test_split;
           ];
         group "expansion"
           [
             prop "the expansion sorts as nx.cpu" expansion_case law_expansion;
           ];
       ])
