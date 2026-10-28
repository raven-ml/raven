(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's reductions and scans on the host. On int32 each is the fold of its
   elements, an order-free reference; on every dtype each computes the dtypes
   its doc states, as the host's kernel over the same axes (the narrower floats
   at float32, rounded once, a NaN result any NaN), and raises naming itself
   elsewhere. A NaN term makes a NaN result on every path: the host's kernels
   and values cut along the reduced axis. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog

type host = Nx.host

let value a : (_, _, host) Nx.t = Nx.Repr.of_array Nx.Host.v a
let host_array x = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x))

let names name f =
  raises_match (Exn.invalid_arg ~substring:("Nx." ^ name ^ ": ")) f

(* Shapes and axes *)

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency [ (1, Gen.constant 0); (6, Gen.int_range 1 4) ])

(* Axes of a value of rank [r]: [None] for all, or some, a few negative. *)
let axes_of r =
  let open Gen in
  if r = 0 then constant None
  else
    let* picks = array ~size:(constant r) bool in
    let+ negative = bool in
    let axes =
      List.concat
        (List.mapi
           (fun a p -> if p then [ (if negative then a - r else a) ] else [])
           (Array.to_list picks))
    in
    if axes = [] then None else Some axes

let normal r axes =
  match axes with
  | None -> List.init r Fun.id
  | Some l ->
      List.sort Int.compare (List.map (fun a -> if a < 0 then a + r else a) l)

let reduced s axes =
  Array.of_list
    (List.filteri (fun a _ -> not (List.mem a axes)) (Array.to_list s))

let kept s axes = Array.mapi (fun a n -> if List.mem a axes then 1 else n) s

(* Indices of [s] in C order. *)
let indices s =
  let n = Array.fold_left ( * ) 1 s in
  List.init n (fun i ->
      let idx = Array.make (Array.length s) 0 and k = ref i in
      for a = Array.length s - 1 downto 0 do
        idx.(a) <- !k mod s.(a);
        k := !k / s.(a)
      done;
      idx)

(* int32: the fold of the elements *)

let int32_case =
  let open Gen in
  with_pp
    (fun ppf (s, axes, keepdims, _) ->
      Format.fprintf ppf "%a over %s%s" pp_ints s
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l))
        (if keepdims then ", kept" else ""))
    (let* s = shape in
     let* axes = axes_of (Array.length s) in
     let* keepdims = bool in
     let n = Array.fold_left ( * ) 1 s in
     let+ xs =
       array ~size:(constant n)
         (frequency
            [
              (4, map Int32.of_int (int_range (-5) 5));
              ( 1,
                of_list
                  ~pp:(fun ppf x -> Format.fprintf ppf "%ld" x)
                  [ Int32.max_int; Int32.min_int ] );
            ])
     in
     (s, axes, keepdims, xs))

(* The fold of [f] from [init] of [xs] of shape [s] along [axes], in C order of
   the result. *)
let fold f init s axes xs =
  let out = reduced s axes in
  let acc = Hashtbl.create 16 in
  List.iteri
    (fun i idx ->
      let key =
        Array.of_list
          (List.filteri (fun a _ -> not (List.mem a axes)) (Array.to_list idx))
      in
      let prev = Option.value ~default:init (Hashtbl.find_opt acc key) in
      Hashtbl.replace acc key (f prev xs.(i)))
    (indices s);
  Array.of_list
    (List.map
       (fun idx -> Option.value ~default:init (Hashtbl.find_opt acc idx))
       (indices out))

let law_int32 (s, axes, keepdims, xs) =
  let r = Array.length s in
  let ax = normal r axes in
  cover "an empty axis" (List.exists (fun a -> s.(a) = 0) ax);
  cover "kept" keepdims;
  let x = value (A.of_array D.Int32 (Iarray.of_array s) xs) in
  let check name f init g =
    let y = host_array (g ?axes ~keepdims x) in
    equal ~msg:(name ^ ": shape") (array int)
      (if keepdims then kept s ax else reduced s ax)
      (Iarray.to_array (L.shape (A.layout y)));
    equal ~msg:name (array int32) (fold f init s ax xs) (A.to_array y)
  in
  check "sum" Int32.add 0l (fun ?axes ~keepdims x -> Nx.sum ?axes ~keepdims x);
  check "prod" Int32.mul 1l (fun ?axes ~keepdims x -> Nx.prod ?axes ~keepdims x);
  let elements = Array.fold_left ( * ) 1 (reduced s ax) in
  if List.exists (fun a -> s.(a) = 0) ax && elements > 0 then begin
    names "max" (fun () -> Nx.max ?axes x);
    names "min" (fun () -> Nx.min ?axes x)
  end
  else begin
    check "max" Stdlib.max Int32.min_int (fun ?axes ~keepdims x ->
        Nx.max ?axes ~keepdims x);
    check "min" Stdlib.min Int32.max_int (fun ?axes ~keepdims x ->
        Nx.min ?axes ~keepdims x)
  end

(* The running fold along [axis]. *)
let running f s axis xs =
  let out = Array.copy xs in
  List.iteri
    (fun i idx ->
      if idx.(axis) > 0 then begin
        let prev = Array.copy idx in
        prev.(axis) <- idx.(axis) - 1;
        let j =
          Array.fold_left ( + ) 0
            (Array.mapi
               (fun a v ->
                 v
                 * Array.fold_left ( * ) 1
                     (Array.sub s (a + 1) (Array.length s - a - 1)))
               prev)
        in
        out.(i) <- f out.(j) xs.(i)
      end)
    (indices s);
  out

let law_int32_scan (s, _, _, xs) =
  let r = Array.length s in
  let n = Array.length xs in
  let x = value (A.of_array D.Int32 (Iarray.of_array s) xs) in
  List.iter
    (fun (name, f, (g : ?axis:int -> _ -> _)) ->
      let flat = host_array (g x) in
      equal ~msg:(name ^ ": shape") (array int) s
        (Iarray.to_array (L.shape (A.layout flat)));
      equal ~msg:(name ^ " in C order") (array int32) (running f [| n |] 0 xs)
        (A.to_array flat);
      if r > 0 then
        equal
          ~msg:(name ^ " along the last axis")
          (array int32)
          (running f s (r - 1) xs)
          (A.to_array (host_array (g ~axis:(-1) x))))
    [
      ("cumsum", Int32.add, fun ?axis x -> Nx.cumsum ?axis x);
      ("cumprod", Int32.mul, fun ?axis x -> Nx.cumprod ?axis x);
      ("cummax", Stdlib.max, fun ?axis x -> Nx.cummax ?axis x);
      ("cummin", Stdlib.min, fun ?axis x -> Nx.cummin ?axis x);
    ]

(* Every dtype: the host's kernel *)

type case = Case : ('v, 's) A.t -> case

let dtype = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all

let drawn (type v s) (dt : (v, s) D.t) s : (v, s) A.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  A.v dt (L.contiguous (Iarray.of_array s))
    (Rig.Buffer.of_string (if data = "" then "\000" else data))

let any_case =
  let open Gen in
  with_pp
    (fun ppf (Case a, axes) ->
      Format.fprintf ppf "%a %a over %s" D.pp (A.dtype a) pp_ints
        (Iarray.to_array (L.shape (A.layout a)))
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l)))
    (let* (D.Any dt) = dtype in
     let* s = shape in
     let* a = drawn dt s in
     let+ axes = axes_of (Array.length s) in
     (Case a, axes))

let bits_of (type v s) (a : (v, s) A.t) =
  let a = A.copy a in
  match D.bits (A.dtype a) with
  | 1 -> Array.map Bool.to_int (A.to_array (A.expect D.Bit (A.Any a)))
  | 4 -> A.to_array (Option.get (A.bitcast D.Uint4 a))
  | _ -> A.to_array (Option.get (A.bitcast D.Uint8 a))

(* [a] at [dt], as nx.cpu casts. *)
let cast (type v s w r) (dt : (w, r) D.t) (a : (v, s) A.t) : (w, r) A.t =
  let dst = A.create Rig.host dt (L.shape (A.layout a)) in
  match Nx_kernels_support.map1 (module Nx_cpu) Cast ~dst a with
  | A.Done -> dst
  | _ -> failwith "the host's kernel declined a cast"

(* Each element's bits, as the integers of its bytes. *)
let element_bits (type v s) (a : (v, s) A.t) =
  let w = Int.max 1 (D.bits (A.dtype a) / 8) in
  let b = bits_of a in
  Array.init (Array.length b / w) (fun i -> Array.sub b (i * w) w)

(* Whether each element of [a] is a NaN, a complex number where a part is. *)
let nans (type v s) (a : (v, s) A.t) =
  match D.kind (A.dtype a) with
  | D.Float -> Array.map Float.is_nan (A.to_array (cast D.Float64 a))
  | D.Complex ->
      Array.map
        (fun (c : Complex.t) -> Float.is_nan c.re || Float.is_nan c.im)
        (A.to_array (cast D.Complex128 a))
  | D.Signed | D.Unsigned | D.Boolean -> Array.make (L.numel (A.layout a)) false

(* Each element's bits, or [None] for a NaN: a NaN result promises a NaN,
   never which one. Under [~parts], a complex number's parts are each so,
   as a sum adds them apart. *)
let rec blind : type v s. ?parts:bool -> (v, s) A.t -> int array option array
    =
 fun ?(parts = false) a ->
  match A.dtype a with
  | D.Complex64 when parts ->
      blind (Option.get (A.bitcast D.Float32 (A.copy a)))
  | D.Complex128 when parts ->
      blind (Option.get (A.bitcast D.Float64 (A.copy a)))
  | _ ->
      let n = nans a in
      Array.mapi (fun i e -> if n.(i) then None else Some e) (element_bits a)

let blind_bits = array (option (array int))

let identity dt = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |]

(* The host kernel's reduction [m] of [a] along [axes]: at float32 for the
   floats narrower than 32 bits, at the byte-wide dtype for the sub-byte ones,
   rounded back once; a complex sum over the parts. [None] where the kernel
   declines. *)
let rec reference : type v s.
    Nx_kernel.monoid -> (v, s) A.t -> int array -> (v, s) A.t option =
 fun m a axes ->
  let dt = A.dtype a in
  let through (type w r) (acc : (w, r) D.t) =
    Option.map (cast dt) (reference m (cast acc a) axes)
  in
  let pairs (type w r) (part : (w, r) D.t) =
    Option.map
      (fun p -> A.copy (Option.get (A.bitcast dt (A.copy p))))
      (reference m (Option.get (A.bitcast part (A.copy a))) axes)
  in
  match (dt, m) with
  | ( ( D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
      | D.Float4_e2m1fn ),
      _ ) ->
      through D.Float32
  | D.Int4, _ -> through D.Int8
  | D.Uint4, _ -> through D.Uint8
  | D.Bit, _ -> through D.Bool
  | D.Complex64, Nx_kernel.Sum -> pairs D.Float32
  | D.Complex128, Nx_kernel.Sum -> pairs D.Float64
  | _ -> (
      let s = Iarray.to_array (L.shape (A.layout a)) in
      let out =
        Array.of_list
          (List.filteri (fun i _ -> not (Array.mem i axes)) (Array.to_list s))
      in
      let dst = A.create Rig.host dt (Iarray.of_array out) in
      match
        Nx_cpu.reduce (identity dt) ~shape:s ~axes:(Iarray.of_array axes)
          [| (Monoid m, 0) |]
          ~dsts:[| A.Any dst |]
          [| Plain a |]
      with
      | A.Done -> Some dst
      | _ -> None)

(* What each reduction's doc states it takes. *)
let documented (type v s) (m : Nx_kernel.monoid) (dt : (v, s) D.t) =
  match (D.kind dt, m) with
  | (D.Float | D.Signed | D.Unsigned), _ -> true
  | D.Boolean, (Max | Min) -> true
  | D.Complex, (Sum | Prod | Max | Min) -> true
  | D.Boolean, (Sum | Prod) -> false

let reductions =
  [
    ("sum", (Sum : Nx_kernel.monoid), fun ?axes x -> Nx.sum ?axes x);
    ("prod", Prod, fun ?axes x -> Nx.prod ?axes x);
    ("max", Max, fun ?axes x -> Nx.max ?axes x);
    ("min", Min, fun ?axes x -> Nx.min ?axes x);
  ]

let law_any (Case a, axes) =
  let dt = A.dtype a in
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let ax = Array.of_list (normal (Array.length s) axes) in
  (* An element of the result of no term: an extreme there raises. *)
  let no_term =
    Array.exists (fun i -> s.(i) = 0) ax
    && Array.fold_left ( * ) 1 (reduced s (Array.to_list ax)) > 0
  in
  List.iter
    (fun (name, m, f) ->
      let f () = host_array (f ?axes (value a)) in
      let parts = m = Nx_kernel.Sum in
      let extreme =
        match (m : Nx_kernel.monoid) with Max | Min -> true | _ -> false
      in
      if documented m dt && not (no_term && extreme) then begin
        cover "computed" true;
        match reference m a ax with
        | Some want ->
            equal ~msg:name blind_bits (blind ~parts want) (blind ~parts (f ()))
        | None -> fail (name ^ ": the host's kernel declined a stated dtype")
      end
      else begin
        cover "raised" true;
        names name (fun () -> ignore (f ()))
      end)
    reductions

(* Errors *)

let test_errors () =
  let x = value (A.of_array D.Float32 [| 2; 3 |] (Array.make 6 1.)) in
  names "sum" (fun () -> Nx.sum ~axes:[ 2 ] x);
  names "sum" (fun () -> Nx.sum ~axes:[ 0; -2 ] x);
  names "max" (fun () -> Nx.max (value (A.of_array D.Float32 [| 0 |] [||])));
  names "cumsum" (fun () -> Nx.cumsum ~axis:2 x);
  names "sum" (fun () ->
      Nx.sum (value (A.of_array D.Bool [| 2 |] [| true; false |])));
  names "mean" (fun () ->
      Nx.mean (value (A.of_array D.Bool [| 2 |] [| true; false |])));
  names "cumsum" (fun () ->
      Nx.cumsum (value (A.of_array D.Bool [| 2 |] [| true; false |])))

(* A refused axis is printed as the caller passed it. *)
let test_axis_printed () =
  let x = value (A.of_array D.Float32 [| 2; 3; 4 |] (Array.make 24 1.)) in
  let says sub f = raises_match (Exn.invalid_arg ~substring:sub) f in
  says "Nx.sum: -5 is not an axis of float32 [2; 3; 4]" (fun () ->
      Nx.sum ~axes:[ -5 ] x);
  says "Nx.sum: axis -1 of float32 [2; 3; 4] repeats" (fun () ->
      Nx.sum ~axes:[ 2; -1 ] x)

let test_mean () =
  let x = value (A.of_array D.Float32 [| 2; 2 |] [| 1.; 2.; 3.; 6. |]) in
  equal (array float_exact) [| 2.; 4. |]
    (A.to_array (host_array (Nx.mean ~axes:[ 0 ] x)));
  let all = host_array (Nx.mean ~keepdims:true x) in
  equal (array int) [| 1; 1 |] (Iarray.to_array (L.shape (A.layout all)));
  equal (array float_exact) [| 3. |] (A.to_array all);
  let none = value (A.of_array D.Float64 [| 0 |] [||]) in
  equal bool true (Float.is_nan (A.to_array (host_array (Nx.mean none))).(0))

(* Integer means *)

(* An integer's bits: an unsigned 32-bit one zero-extended. *)
let to_int64 (type v s) (dt : (v, s) D.t) (v : v) : int64 =
  match dt with
  | D.Int64 -> v
  | D.Uint64 -> v
  | D.Int32 -> Int64.of_int32 v
  | D.Uint32 -> Int64.logand (Int64.of_int32 v) 0xFFFF_FFFFL
  | D.Int16 -> Int64.of_int v
  | D.Uint16 -> Int64.of_int v
  | D.Int8 -> Int64.of_int v
  | D.Uint8 -> Int64.of_int v
  | D.Int4 -> Int64.of_int v
  | D.Uint4 -> Int64.of_int v
  | _ -> invalid_arg "to_int64: not an integer dtype"

(* The value of [dt] whose low bits are [x]'s. *)
let of_int64 (type v s) (dt : (v, s) D.t) (x : int64) : v =
  let b = D.bits dt in
  let x =
    if b = 64 then x
    else if D.is D.Signed dt then
      Int64.shift_right (Int64.shift_left x (64 - b)) (64 - b)
    else Int64.shift_right_logical (Int64.shift_left x (64 - b)) (64 - b)
  in
  match dt with
  | D.Int64 -> x
  | D.Uint64 -> x
  | D.Int32 -> Int64.to_int32 x
  | D.Uint32 -> Int64.to_int32 x
  | D.Int16 -> Int64.to_int x
  | D.Uint16 -> Int64.to_int x
  | D.Int8 -> Int64.to_int x
  | D.Uint8 -> Int64.to_int x
  | D.Int4 -> Int64.to_int x
  | D.Uint4 -> Int64.to_int x
  | _ -> invalid_arg "of_int64: not an integer dtype"

(* An integer's value as limbs [(hi, lo)], [hi * 2^32 + lo], [lo] in
   [[0, 2^32)]: sums of them are exact in [int]. *)
let limbs (type v s) (dt : (v, s) D.t) (v : v) =
  let x = to_int64 dt v in
  let hi =
    if D.is D.Signed dt then Int64.shift_right x 32
    else Int64.shift_right_logical x 32
  in
  (Int64.to_int hi, Int64.to_int (Int64.logand x 0xFFFF_FFFFL))

let add_limbs (h, l) (h', l') = (h + h', l + l')

(* The sum [(h, l)] of [n] integers divided by [n], truncated toward zero, by
   long division of its magnitude in base 2^32: its bits, and whether it is
   negative and inexact. *)
let divided n (h, l) =
  let h = h + (l asr 32) and l = l land 0xFFFF_FFFF in
  let negative = h < 0 in
  let a, b =
    if not negative then (h, l)
    else if l = 0 then (-h, 0)
    else (-h - 1, 0x1_0000_0000 - l)
  in
  let low = ((a mod n) lsl 32) + b in
  let q = Int64.(add (shift_left (of_int (a / n)) 32) (of_int (low / n))) in
  ((if negative then Int64.neg q else q), negative && low mod n <> 0)

let integer_dtypes =
  List.filter
    (fun (D.Any dt) -> D.is D.Signed dt || D.is D.Unsigned dt)
    D.all

let int_case =
  let open Gen in
  with_pp
    (fun ppf (D.Any dt, s, axes, _) ->
      Format.fprintf ppf "%a %a over %s" D.pp dt pp_ints s
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l)))
    (let* (D.Any dt) =
       of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) integer_dtypes
     in
     let* s = shape in
     let* axes = axes_of (Array.length s) in
     let extremes =
       [ to_int64 dt (D.min_value dt); to_int64 dt (D.max_value dt) ]
     in
     let+ xs =
       array
         ~size:(constant (Array.fold_left ( * ) 1 s))
         (frequency
            [
              (3, map Int64.of_int (int_range (-5) 5));
              (2, of_list extremes);
              (1, int64);
            ])
     in
     (D.Any dt, s, axes, xs))

let law_int_mean (D.Any dt, s, axes, xs) =
  let ax = normal (Array.length s) axes in
  let xs = Array.map (of_int64 dt) xs in
  let x = value (A.of_array dt (Iarray.of_array s) xs) in
  let n = List.fold_left (fun n a -> n * s.(a)) 1 ax in
  if n = 0 && Array.fold_left ( * ) 1 (reduced s ax) > 0 then
    names "mean" (fun () -> Nx.mean ?axes x)
  else begin
    let sums = fold add_limbs (0, 0) s ax (Array.map (limbs dt) xs) in
    let want = Array.map (divided n) sums in
    let got =
      Array.map (to_int64 dt) (A.to_array (host_array (Nx.mean ?axes x)))
    in
    cover "64 bits" (D.bits dt = 64);
    cover "a negative mean rounded toward zero" (Array.exists snd want);
    equal (array int64) (Array.map fst want) got
  end

(* Means the spec gives: the exact mean rounded toward zero, between the least
   and greatest element. *)
let test_int_mean () =
  let mean dt xs =
    let x = value (A.of_array dt [| Array.length xs |] xs) in
    to_int64 dt (Nx.item [] (Nx.mean x))
  in
  equal ~msg:"int32 3.5" int64 3L (mean D.Int32 [| 3l; 4l |]);
  equal ~msg:"int32 -3.5" int64 (-3L) (mean D.Int32 [| -3l; -4l |]);
  equal ~msg:"int32 -1/3" int64 0L (mean D.Int32 [| -1l; 0l; 0l |]);
  equal ~msg:"int8 -127.5" int64 (-127L) (mean D.Int8 [| -128; -127 |]);
  equal ~msg:"int4 -7.5" int64 (-7L) (mean D.Int4 [| -8; -7 |]);
  equal ~msg:"uint4 14.5" int64 14L (mean D.Uint4 [| 15; 14 |]);
  equal ~msg:"uint8 254.5" int64 254L (mean D.Uint8 [| 255; 254 |]);
  equal ~msg:"uint32 2^32 - 1.5" int64 0xFFFF_FFFEL
    (mean D.Uint32 [| -1l; -2l |]);
  let i64 = Int64.(max_int, min_int) in
  equal ~msg:"int64 max" int64 Int64.max_int
    (mean D.Int64 [| fst i64; fst i64 |]);
  equal ~msg:"int64 min" int64 Int64.min_int
    (mean D.Int64 [| snd i64; snd i64 |]);
  equal ~msg:"int64 (max + min) / 2 = -1/2" int64 0L
    (mean D.Int64 [| fst i64; snd i64 |]);
  equal ~msg:"int64 (2 min - 1) / 3" int64 (-6148914691236517205L)
    (mean D.Int64 [| snd i64; snd i64; -1L |]);
  equal ~msg:"uint64 2^64 - 1" int64 (-1L) (mean D.Uint64 [| -1L; -1L |]);
  equal ~msg:"uint64 (2^64 - 1) / 2" int64 Int64.max_int
    (mean D.Uint64 [| -1L; 0L |]);
  equal ~msg:"uint64 (2^65 - 1) / 3" int64 (-6148914691236517206L)
    (mean D.Uint64 [| -1L; -1L; 1L |]);
  let x = value (A.of_array D.Int32 [| 2; 0 |] [||]) in
  names "mean" (fun () -> Nx.mean ~axes:[ 1 ] x);
  equal ~msg:"an empty axis of an empty result" (array int) [| 0 |]
    (Nx.shape (Nx.mean ~axes:[ 0 ] x));
  let x = value (A.of_array D.Int16 [| 2; 2 |] [| 1; 2; 4; 7 |]) in
  let kept = host_array (Nx.mean ~axes:[ 1 ] ~keepdims:true x) in
  equal ~msg:"kept" (array int) [| 2; 1 |]
    (Iarray.to_array (L.shape (A.layout kept)));
  equal ~msg:"kept" (array int) [| 1; 5 |] (A.to_array kept)

(* Scans of complex numbers and narrow floats: a complex sum adds the parts,
   and a narrow float accumulates at float32, rounding each prefix once. *)
let test_scan_parts () =
  let z =
    A.of_array D.Complex64 [| 3 |]
      Complex.
        [|
          { re = 1.; im = -1. }; { re = 2.; im = 0.5 }; { re = -4.; im = 2. };
        |]
  in
  equal ~msg:"complex64 cumsum"
    (array
       (Testable.make
          ~pp:(fun ppf (c : Complex.t) -> Format.fprintf ppf "%g%+gi" c.re c.im)
          ~equal:( = )))
    Complex.
      [|
        { re = 1.; im = -1. }; { re = 3.; im = -0.5 }; { re = -1.; im = 1.5 };
      |]
    (A.to_array (host_array (Nx.cumsum (value z))));
  let xs = Array.init 300 (fun i -> 1. +. (Float.of_int i /. 7.)) in
  let b = cast D.Bfloat16 (A.of_array D.Float32 [| 300 |] xs) in
  let wide = A.to_array (host_array (Nx.cumsum (value (cast D.Float32 b)))) in
  let want =
    A.to_array (cast D.Bfloat16 (A.of_array D.Float32 [| 300 |] wide))
  in
  equal ~msg:"bfloat16 cumsum, rounded once" (array float_exact) want
    (A.to_array (host_array (Nx.cumsum (value b))))

(* Reductions and scans through Nx.Prim. *)

let reduce1 r axes x =
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout = L.contiguous (Iarray.of_array (Nx.shape x));
           axes;
           prog = identity (Nx.dtype x);
           reductions = [ r ];
           loads = [| Plain x |];
         })
  in
  y

let read x = A.to_array (host_array x)
let bits = Int64.bits_of_float

(* Floats of the same bits, or both NaN. *)
let same_float =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
    ~equal:(fun a b ->
      Int64.equal (bits a) (bits b) || (Float.is_nan a && Float.is_nan b))
let nan_with k = Int64.float_of_bits (Int64.logor 0x7ff8_0000_0000_0000L k)

(* Each result's terms, in C order of the results and of the reduced indices. *)
let terms s axes xs =
  Array.map List.rev (fold (fun acc x -> x :: acc) [] s axes xs)

let scan1 r axis x =
  Nx.Prim.eval ~by:"t"
    (Scan
       {
         layout = L.contiguous (Iarray.of_array (Nx.shape x));
         axis;
         prog = identity (Nx.dtype x);
         reduction = r;
         loads = [| Plain x |];
         carry = None;
       })

(* Each slice of [s] along [axis], in C order of the other axes, as the
   position in C order of its index [j]. *)
let slices s axis =
  let n = s.(axis) in
  let r = Array.length s in
  let after = Array.sub s (axis + 1) (r - axis - 1) in
  let stride = Array.fold_left ( * ) 1 after in
  let total = Array.fold_left ( * ) 1 s in
  if n = 0 then []
  else
    List.init (total / n) (fun o ->
        let base = (o / stride * n * stride) + (o mod stride) in
        fun j -> base + (j * stride))

(* Each prefix of each slice of [s] along [axis]: its terms' positions in C
   order, the last one the result's. *)
let prefixes s axis =
  List.concat_map
    (fun at -> List.init s.(axis) (fun j -> List.init (j + 1) at))
    (slices s axis)

(* The most combinations output [i] of a scan's network passes through,
   [D_i = max(0, 2⌈log₂(i+1)⌉ - 1)], for [n = i + 1] terms. *)
let depth n =
  let rec log2_up b = if 1 lsl b >= n then b else log2_up (b + 1) in
  Float.of_int (Int.max 0 ((2 * log2_up 0) - 1))

(* [a] by [on], read through a transposed view where [strided]: the elements
   of [a], held in a layout that runs its axes backwards. *)
let transposed on strided a =
  let s = Iarray.to_array (L.shape (A.layout a)) in
  let r = Array.length s in
  if (not strided) || r < 2 then on a
  else
    let back = Array.init r (fun i -> s.(r - 1 - i)) in
    let xs = A.to_array a in
    let at idx =
      Array.fold_left (fun p a -> (p * s.(a)) + idx.(r - 1 - a)) 0
        (Array.init r Fun.id)
    in
    let held = Array.of_list (List.map (fun i -> xs.(at i)) (indices back)) in
    Nx.transpose (on (A.of_array (A.dtype a) (Iarray.of_array back) held))

(* A shape of rank at least one and an axis of it, at times long enough for
   several levels of the network. *)
let along =
  let open Gen in
  let* s = shape in
  let s = if Array.length s = 0 then [| 3 |] else s in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ n = frequency [ (3, constant s.(axis)); (1, int_range 5 40) ] in
  let s = Array.copy s in
  s.(axis) <- n;
  (s, axis)

(* [v] rounded to [dt], as a store rounds. *)
let rounded (type s) (dt : (float, s) D.t) v =
  (A.to_array (A.of_array dt [| 1 |] [| v |])).(0)

(* The bound nx.mli states on nx.cpu: at [n] terms through at most
   [D] combinations, [D] the result's {!depth}, within
   [(7 D + 6 ln n) u + u |r|] in the accumulator; then rounded once, so
   between the roundings of the bound's ends. The reference, float64 over
   each prefix from its maximum ({!lse_ref}), adds its own error. *)
let lse_bound u n r =
  (((7. *. depth n) +. (6. *. log (Float.of_int n))) *. u)
  +. (u *. Float.abs r)

(* [lse_ref m terms] is [m + log Σ exp (t - m)] over [terms] of the finite
   maximum [m], at float64, and a bound on its error: each [t - m] split
   exactly into its rounding and a remainder (two-sum), each [exp] taken of
   the rounding and corrected by the remainder, and the exps summed
   compensated (Neumaier), so the sum's relative error is a few units
   whatever the count. The bound adds a unit for each [exp], two for the sum,
   one for the [log] and its own [ulp], and half one for the last addition,
   libm's [exp] and [log] within one [ulp]. *)
let lse_ref m terms =
  let sum = ref 0. and c = ref 0. in
  let add e =
    let s = !sum +. e in
    let lost =
      if Float.abs !sum >= Float.abs e then !sum -. s +. e else e -. s +. !sum
    in
    c := !c +. lost;
    sum := s
  in
  List.iter
    (fun t ->
      let d = t -. m in
      if Float.is_finite d then begin
        let back = d -. t in
        let rest = t -. (d -. back) +. (-.m -. back) in
        add (exp d *. (1. +. rest))
      end)
    terms;
  let s = !sum +. !c in
  let ls = log s in
  let r = m +. ls in
  let err =
    (epsilon_float *. (4. +. Float.abs ls)) +. (ldexp 1. (-53) *. Float.abs r)
  in
  (r, err)

(* Complex products, extremes and their first positions, reduced and scanned
   through Nx on the host, against folds of their terms. A number with a NaN
   part is a NaN. A product's terms are Gaussian integers whose parts are 0 or
   ± a power of two, so every order of multiplication gives the exact product
   but for the sign of a zero part: products compare by value, extremes by
   bits, and a NaN as any NaN. *)

type zcase =
  | Zcase : {
      dt : (Complex.t, 's) D.t;
      s : int array;
      axes : int list option;
      strided : bool;
      xs : Complex.t array;
    }
      -> zcase

let znan (c : Complex.t) = Float.is_nan c.re || Float.is_nan c.im
let zbits (c : Complex.t) = (bits c.re, bits c.im)
let pp_z ppf (c : Complex.t) = Format.fprintf ppf "%h%+hi" c.re c.im
let z_bits =
  Testable.make ~pp:pp_z ~equal:(fun a b ->
      zbits a = zbits b || (znan a && znan b))

let z_value =
  Testable.make ~pp:pp_z ~equal:(fun (a : Complex.t) b ->
      if znan a || znan b then znan a && znan b
      else a.re = b.re && a.im = b.im)

(* A NaN of payload [k] in bits float32 keeps. *)
let payload k = nan_with (Int64.shift_left (Int64.of_int k) 29)

(* A NaN in one part or both. *)
let nan_term =
  let open Gen in
  let* k = int_range 1 1000 in
  let+ which = int_range 0 2 in
  match which with
  | 0 -> { Complex.re = payload k; im = 1. }
  | 1 -> { Complex.re = -0.; im = payload k }
  | _ -> { Complex.re = payload k; im = payload (k + 1) }

(* Shapes of a few elements, and long ones: an output of more than a block's
   1024 terms, a scan longer than a chunk's 4096, rows of many outputs. *)
let zshape =
  let open Gen in
  frequency
    [
      (3, shape);
      (1, map (fun n -> [| n |]) (int_range 3000 9000));
      ( 1,
        let* a = int_range 2 5 in
        let+ b = int_range 17 1100 in
        [| a; b |] );
      ( 1,
        let* a = int_range 17 1100 in
        let+ b = int_range 2 40 in
        [| a; b |] );
    ]

(* [xs] with at most [most] of its elements replaced by draws of [rare]. *)
let sprinkle xs ~most rare =
  let open Gen in
  let n = Array.length xs in
  if n = 0 then constant xs
  else
    let+ picks =
      list ~size:(int_range 0 most) (pair (int_range 0 (n - 1)) rare)
    in
    let xs = Array.copy xs in
    List.iter (fun (i, x) -> xs.(i) <- x) picks;
    xs

(* Terms of a product: units, zeros where [zeros], at most 40 of [1 ± i] and
   [2], and NaNs where [nans]. Extremes: parts of a few values, so that ties
   are common, signed zeros and infinities, and NaNs where [nans]. *)
let zcase ~product =
  let open Gen in
  let z re im = { Complex.re; im } in
  let make (type s) (dt : (Complex.t, s) D.t) =
    let* s = zshape in
    let* axes = axes_of (Array.length s) in
    let* strided = bool in
    let* nans = bool in
    let n = Array.fold_left ( * ) 1 s in
    let* xs =
      if product then
        let* zeros = bool in
        let unit = of_list [ z 1. 0.; z (-1.) 0.; z 0. 1.; z 0. (-1.) ] in
        let* xs =
          array ~size:(constant n)
            (frequency
               [ (30, unit); ((if zeros then 1 else 0), constant (z 0. 0.)) ])
        in
        sprinkle xs ~most:40 (of_list [ z 1. 1.; z 1. (-1.); z 2. 0. ])
      else
        let part =
          frequency
            [
              (8, of_list [ -1.; -0.; 0.; 1.; 2. ]);
              (1, of_list [ Float.infinity; Float.neg_infinity ]);
            ]
        in
        array ~size:(constant n)
          (let* re = part in
           let+ im = part in
           z re im)
    in
    let+ xs = if nans then sprinkle xs ~most:3 nan_term else constant xs in
    (* The terms as stored. *)
    let xs = A.to_array (A.of_array dt (Iarray.of_array s) xs) in
    Zcase { dt; s; axes; strided; xs }
  in
  with_pp
    (fun ppf (Zcase { dt; s; axes; strided; _ }) ->
      Format.fprintf ppf "%a %a over %s%s" D.pp dt pp_ints s
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l))
        (if strided then ", transposed" else ""))
    (let* c128 = bool in
     if c128 then make D.Complex128 else make D.Complex64)

(* A product starts from its first term, and is 1 for none. *)
let product ts =
  match (List.find_opt znan ts, ts) with
  | Some t, _ -> t
  | None, [] -> Complex.one
  | None, t :: ts -> List.fold_left Complex.mul t ts

(* Whether [t] is past [b] in Max's order, or Min's where [flip]: real parts,
   then imaginary parts, [-0] below [+0], and a NaN past every number. *)
let zbeats flip (t : Complex.t) (b : Complex.t) =
  if znan t || znan b then znan t && not (znan b)
  else
    let part x y =
      if x = y then Bool.compare (Float.sign_bit y) (Float.sign_bit x)
      else Float.compare x y
    in
    let c = part t.re b.re in
    let c = if c <> 0 then c else part t.im b.im in
    if flip then c < 0 else c > 0

(* The first extreme of [ts] and its position, in one pass. *)
let zextreme flip ts =
  let _, best, at =
    List.fold_left
      (fun (i, best, at) t ->
        if i = 0 || zbeats flip t best then (i + 1, t, i)
        else (i + 1, best, at))
      (0, Complex.zero, 0) ts
  in
  (best, Int64.of_int at)

(* Steps of a running extreme and a running product: a NaN stays. *)
let zrunning beats acc t =
  if znan acc then acc else if beats t acc then t else acc

let zprod acc t =
  if znan acc then acc else if znan t then t else Complex.mul acc t

let law_complex_prod (Zcase { dt; s; axes; strided; xs }) =
  let ax = normal (Array.length s) axes in
  let ts = terms s ax xs in
  cover "more than a block of terms"
    (Array.exists (fun t -> List.length t > 1024) ts);
  cover "a NaN result" (Array.exists (fun t -> List.exists znan t) ts);
  cover "a product other than a unit"
    (Array.exists (fun t -> Complex.norm (product t) > 1.) ts);
  cover "transposed" (strided && Array.length s > 1);
  let x = transposed value strided (A.of_array dt (Iarray.of_array s) xs) in
  equal ~msg:"prod" (array z_value) (Array.map product ts)
    (read (Nx.prod ?axes x));
  let axis = match axes with Some (a :: _) -> Some a | _ -> None in
  let n = Array.length xs in
  let want =
    match axis with
    | None -> running zprod [| n |] 0 xs
    | Some a ->
        running zprod s (List.hd (normal (Array.length s) (Some [ a ]))) xs
  in
  cover "a scan longer than a chunk" (n > 4096 && axis = None);
  equal ~msg:"cumprod" (array z_value) want (read (Nx.cumprod ?axis x))

let law_complex_extremes (Zcase { dt; s; axes; strided; xs }) =
  let r = Array.length s in
  let ax = normal r axes in
  let ts = terms s ax xs in
  let x = transposed value strided (A.of_array dt (Iarray.of_array s) xs) in
  cover "a NaN result" (Array.exists (fun t -> List.exists znan t) ts);
  cover "transposed" (strided && r > 1);
  let empty = List.exists (fun a -> s.(a) = 0) ax in
  List.iter
    (fun (name, flip, f, arg, scan) ->
      if not empty then begin
        cover "more than a block of terms"
          (Array.exists (fun t -> List.length t > 1024) ts);
        equal ~msg:name (array z_bits)
          (Array.map (fun t -> fst (zextreme flip t)) ts)
          (read (f ?axes x))
      end;
      let axis = match axes with Some (a :: _) -> Some a | _ -> None in
      let along = Option.map (fun a -> if a < 0 then a + r else a) axis in
      let arg_terms =
        match along with
        | None -> [| Array.to_list xs |]
        | Some a -> terms s [ a ] xs
      in
      let terms_along =
        Option.fold ~none:true ~some:(fun a -> s.(a) > 0) along
      in
      if Array.length xs > 0 && terms_along then
        equal ~msg:("arg" ^ name) (array int64)
          (Array.map (fun t -> snd (zextreme flip t)) arg_terms)
          (read (arg ?axis x));
      let want =
        match along with
        | None -> running (zrunning (zbeats flip)) [| Array.length xs |] 0 xs
        | Some a -> running (zrunning (zbeats flip)) s a xs
      in
      equal ~msg:("cum" ^ name) (array z_bits) want (read (scan ?axis x)))
    [
      ( "max",
        false,
        (fun ?axes x -> Nx.max ?axes x),
        (fun ?axis x -> Nx.argmax ?axis x),
        fun ?axis x -> Nx.cummax ?axis x );
      ( "min",
        true,
        (fun ?axes x -> Nx.min ?axes x),
        (fun ?axis x -> Nx.argmin ?axis x),
        fun ?axis x -> Nx.cummin ?axis x );
    ]

let test_complex_cases () =
  let z re im = { Complex.re; im } in
  let v dt xs = value (A.of_array dt [| Array.length xs |] xs) in
  equal ~msg:"a product of no term is 1 + 0i" (array z_bits) [| z 1. 0. |]
    (read (Nx.prod (v D.Complex64 [||])));
  (* (∞ + 0i)(0 + i) is NaN + ∞i, as Mul's textbook product gives it. *)
  equal ~msg:"a NaN product of numbers is a NaN" (array z_bits)
    [| z nan Float.infinity |]
    (read (Nx.prod (v D.Complex128 [| z Float.infinity 0.; z 0. 1. |])));
  (* 1 + 0i times either would make a NaN part or flip a zero's sign. *)
  equal ~msg:"a product of one term is that term" (array z_bits)
    [| z Float.infinity 0.; z 3. (-0.) |]
    (read
       (Nx.prod ~axes:[ 1 ]
          (value
             (A.of_array D.Complex64 [| 2; 1 |]
                [| z Float.infinity 0.; z 3. (-0.) |]))));
  equal ~msg:"and so is a scan's first result" (array z_bits)
    [| z (-0.) (-0.); z 0. (-0.) |]
    (read (Nx.cumprod (v D.Complex128 [| z (-0.) (-0.); z 1. 0. |])));
  let ws = [| z 1. 2.; z 2. (-5.); z 2. (-7.); z 2. (-5.) |] in
  equal ~msg:"max orders real parts, then imaginary parts" (array z_bits)
    [| z 2. (-5.) |]
    (read (Nx.max (v D.Complex128 ws)));
  equal ~msg:"argmax: the first of the greatest" (array int64) [| 1L |]
    (read (Nx.argmax (v D.Complex128 ws)));
  equal ~msg:"min: a -0 part below +0" (array z_bits) [| z 1. (-0.) |]
    (read (Nx.min (v D.Complex64 [| z 1. 0.; z 1. (-0.) |])));
  names "max" (fun () -> Nx.max (v D.Complex64 [||]));
  names "argmin" (fun () -> Nx.argmin (v D.Complex128 [||]))

let complex_group =
  group "complex reductions"
    [
      prop "prod and cumprod are the products of their terms"
        (zcase ~product:true) law_complex_prod;
      prop "max, min, their args and scans are the first extremes"
        (zcase ~product:false) law_complex_extremes;
      test "complex reductions' cases" test_complex_cases;
    ]

(* Logsumexp and Moments take floats alone; an extreme of no term is
   refused. *)
let test_optional_rules () =
  let vector dt xs = value (A.of_array dt [| Array.length xs |] xs) in
  let x = vector D.Int32 [| 1l; 2l |] in
  raises (Invalid_argument "t: Logsumexp does not take int32") (fun () ->
      scan1 (Logsumexp (0, Nx.int32)) 0 x);
  raises (Invalid_argument "t: Moments does not take int32") (fun () ->
      reduce1 (Moments (0, Nx.int32)) [| 0 |] x);
  raises (Invalid_argument "t: Moments does not take int32") (fun () ->
      scan1 (Moments (0, Nx.int32)) 0 x);
  raises (Invalid_argument "t: an extreme of no term: no element along axis 0 of [0]") (fun () ->
      reduce1 (Arg (Max, 0, Nx.int32)) [| 0 |] (vector D.Int32 [||]))

(* NaN results on every path *)

let mem = Nx_support.memory

module Cpu3 = (val Nx.devices [ mem 0; mem 1; mem 3 ])

(* Rows of terms of a float or complex dtype, as their bytes. *)
type nan_case =
  | Nan_case : {
      dt : ('v, 's) D.t;
      rows : int;
      n : int;
      bytes : string;
    }
      -> nan_case

let string_of_bytes b = String.init (Array.length b) (fun i -> Char.chr b.(i))

let part_bytes w c =
  String.init w (fun j ->
      Char.chr (Int64.to_int (Int64.shift_right_logical c (8 * j)) land 0xFF))

(* The bytes of each of [xs] stored in [dt]. *)
let encoded (type s) (dt : (float, s) D.t) xs =
  let a = A.of_array dt [| List.length xs |] (Array.of_list xs) in
  Array.to_list (Array.map string_of_bytes (element_bits a))

let numbers = [ 0.; -0.; 1.; -1.; 2.; 0.5; 3.; infinity; neg_infinity ]

(* A float's or a complex part's bytes: numbers, and NaNs signalling and
   quiet, of both signs and with payloads, from their bits. *)
let pools (type v s) (dt : (v, s) D.t) =
  let nans w l = List.map (part_bytes w) l in
  match dt with
  | D.Float64 | D.Complex128 ->
      ( encoded D.Float64 numbers,
        nans 8
          [
            0x7FF0000000000001L; 0xFFF4000000000005L; 0x7FF8000000000123L;
            0xFFF8000000000000L;
          ] )
  | D.Float32 | D.Complex64 ->
      ( encoded D.Float32 numbers,
        nans 4 [ 0x7F800001L; 0xFFA00005L; 0x7FC00123L; 0xFFC00000L ] )
  | D.Float16 ->
      (encoded D.Float16 numbers, nans 2 [ 0x7C01L; 0xFD05L; 0x7E23L; 0xFE00L ])
  | D.Bfloat16 ->
      ( encoded D.Bfloat16 numbers,
        nans 2 [ 0x7F81L; 0xFFA5L; 0x7FC3L; 0xFFC0L ] )
  | D.Float8_e5m2 ->
      (encoded D.Float8_e5m2 numbers, nans 1 [ 0x7DL; 0xFDL; 0x7EL; 0xFFL ])
  | D.Float8_e4m3fn ->
      ( encoded D.Float8_e4m3fn [ 0.; -0.; 1.; -1.; 2.; 0.5; 3.; 448.; -448. ],
        nans 1 [ 0x7FL; 0xFFL ] )
  | _ -> invalid_arg "pools: not a float or complex dtype"

let nan_dtypes =
  D.
    [
      Any Float16; Any Bfloat16; Any Float8_e5m2; Any Float8_e4m3fn;
      Any Float32; Any Float64; Any Complex64; Any Complex128;
    ]

let parts_of (type v s) (dt : (v, s) D.t) =
  match D.kind dt with D.Complex -> 2 | _ -> 1

let nan_array (type v s) (dt : (v, s) D.t) rows n bytes : (v, s) A.t =
  A.v dt
    (L.contiguous [| rows; n |])
    (Rig.Buffer.of_string (if bytes = "" then "\000" else bytes))

let pp_nan_case ppf (Nan_case { dt; rows; n; bytes }) =
  Format.fprintf ppf "%a [%d; %d] %s" D.pp dt rows n
    (String.concat " "
       (List.init (String.length bytes) (fun i ->
            Printf.sprintf "%02x" (Char.code bytes.[i]))))

(* Rows of one to nine terms, a third of their floats or parts NaNs. *)
let nan_case =
  let open Gen in
  with_pp pp_nan_case
    (let* (D.Any dt) = of_list nan_dtypes in
     let numbers, nans = pools dt in
     let* rows = int_range 1 3 in
     let* n = int_range 1 9 in
     let part = frequency [ (2, of_list numbers); (1, of_list nans) ] in
     let+ ps = list ~size:(constant (rows * n * parts_of dt)) part in
     Nan_case { dt; rows; n; bytes = String.concat "" ps })

(* [y]'s elements' bytes, read on the host. *)
let read_bytes (type v s d) (y : (v, s, d) Nx.t) =
  Array.map string_of_bytes (element_bits (host_array y))

(* What a result is: one per row, of the row's terms; running along the
   row, of the terms up to it; or a position along the row. *)
type result = Folded | Running | Position

(* An operation along a row's axis, its results' bytes; [floats] whether it
   takes floats alone and [apart] whether a complex number's parts fold
   apart, as floats. *)
type nan_op = {
  oname : string;
  result : result;
  floats : bool;
  apart : bool;
  rounds : bool;
      (* Computed over a cut axis by partials that round, which state no
         NaN's bits. *)
  whole : bool;
      (* Reads its axis whole, so that over a cut axis it raises. *)
  run : 'v 's 'd. ('v, 's, 'd) Nx.t -> string array;
}

(* [g] of [x], of a float dtype, read. *)
let on_floats (type v s d)
    (g : 's 'd. (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t)
    (x : (v, s, d) Nx.t) : string array =
  match Nx.dtype x with
  | D.Float64 -> read_bytes (g x)
  | D.Float32 -> read_bytes (g x)
  | D.Float16 -> read_bytes (g x)
  | D.Bfloat16 -> read_bytes (g x)
  | D.Float8_e4m3fn -> read_bytes (g x)
  | D.Float8_e5m2 -> read_bytes (g x)
  | _ -> invalid_arg "on_floats: not a float dtype"

(* [r] of [x] along its rows' axis, through Nx.Prim. *)
let along_rows (type v s d a) (x : (v, s, d) Nx.t) (r : (d, a) Nx.Prim.reduction)
    : a =
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout = L.contiguous (Iarray.of_array (Nx.shape x));
           axes = [| 1 |];
           prog = identity (Nx.dtype x);
           reductions = [ r ];
           loads = [| Plain x |];
         })
  in
  y

(* [r] of [x] running along its rows' axis, through Nx.Prim. *)
let running_rows (type v s d a) (x : (v, s, d) Nx.t)
    (r : (d, a) Nx.Prim.reduction) : a =
  Nx.Prim.eval ~by:"t"
    (Scan
       {
         layout = L.contiguous (Iarray.of_array (Nx.shape x));
         axis = 1;
         prog = identity (Nx.dtype x);
         reduction = r;
         loads = [| Plain x |];
         carry = None;
       })

let row_axes = [ 1 ]
let row_axis = 1

(* The operations nx.mli's reductions section promises a NaN of where a NaN
   term reaches: the monoids and their scans, the positions of the extremes
   and the extremes Arg returns, the mean, the variance, the standard
   deviation, logsumexp and the running moments, on one device, and over
   an axis cut across devices where partials keep bits. *)
let nan_ops =
  [
    {
      oname = "sum";
      rounds = true;
      whole = false;
      result = Folded;
      floats = false;
      apart = true;
      run = (fun x -> read_bytes (Nx.sum ~axes:row_axes x));
    };
    {
      oname = "prod";
      rounds = true;
      whole = false;
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.prod ~axes:row_axes x));
    };
    {
      oname = "max";
      rounds = false;
      whole = false;
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.max ~axes:row_axes x));
    };
    {
      oname = "min";
      rounds = false;
      whole = false;
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.min ~axes:row_axes x));
    };
    {
      oname = "cumsum";
      rounds = true;
      whole = false;
      result = Running;
      floats = false;
      apart = true;
      run = (fun x -> read_bytes (Nx.cumsum ~axis:row_axis x));
    };
    {
      oname = "cumprod";
      rounds = true;
      whole = false;
      result = Running;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.cumprod ~axis:row_axis x));
    };
    {
      oname = "cummax";
      rounds = false;
      whole = false;
      result = Running;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.cummax ~axis:row_axis x));
    };
    {
      oname = "cummin";
      rounds = false;
      whole = false;
      result = Running;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.cummin ~axis:row_axis x));
    };
    {
      oname = "argmax";
      rounds = false;
      whole = false;
      result = Position;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.argmax ~axis:row_axis x));
    };
    {
      oname = "argmin";
      rounds = false;
      whole = false;
      result = Position;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.argmin ~axis:row_axis x));
    };
    {
      oname = "Arg Max's extreme";
      rounds = false;
      whole = false;
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (fst (along_rows x (Arg (Max, 0, Nx.dtype x)))));
    };
    {
      oname = "Arg Min's extreme";
      rounds = false;
      whole = false;
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (fst (along_rows x (Arg (Min, 0, Nx.dtype x)))));
    };
    {
      oname = "running Arg Max's extreme";
      rounds = false;
      whole = false;
      result = Running;
      floats = false;
      apart = false;
      run =
        (fun x -> read_bytes (fst (running_rows x (Arg (Max, 0, Nx.dtype x)))));
    };
    {
      oname = "running Arg Min's extreme";
      rounds = false;
      whole = false;
      result = Running;
      floats = false;
      apart = false;
      run =
        (fun x -> read_bytes (fst (running_rows x (Arg (Min, 0, Nx.dtype x)))));
    };
    {
      oname = "mean";
      rounds = true;
      whole = false;
      result = Folded;
      floats = true;
      apart = false;
      run = (fun x -> on_floats (fun x -> Nx.mean ~axes:row_axes x) x);
    };
    {
      oname = "var";
      rounds = true;
      whole = false;
      result = Folded;
      floats = true;
      apart = false;
      run = (fun x -> on_floats (fun x -> Nx.var ~axes:row_axes x) x);
    };
    {
      oname = "std";
      rounds = true;
      whole = false;
      result = Folded;
      floats = true;
      apart = false;
      run = (fun x -> on_floats (fun x -> Nx.std ~axes:row_axes x) x);
    };
    {
      oname = "logsumexp";
      rounds = true;
      whole = false;
      result = Folded;
      floats = true;
      apart = false;
      run = (fun x -> on_floats (fun x -> Nx.logsumexp ~axes:row_axes x) x);
    };
    {
      oname = "running mean";
      rounds = false;
      whole = true;
      result = Running;
      floats = true;
      apart = false;
      run =
        (fun x ->
          on_floats (fun x -> fst (running_rows x (Moments (0, Nx.dtype x)))) x);
    };
    {
      oname = "running variance";
      rounds = false;
      whole = true;
      result = Running;
      floats = true;
      apart = false;
      run =
        (fun x ->
          on_floats (fun x -> snd (running_rows x (Moments (0, Nx.dtype x)))) x);
    };
  ]

(* How a value computes: by the host's nx.cpu, on the host or over three
   devices split along the rows' axis. *)
type placing = {
  pname : string;
  cut : bool;
  go : 'v 's. nan_op -> ('v, 's) A.t -> string array;
}

let placings =
  [
    { pname = "nx.cpu"; cut = false; go = (fun o a -> o.run (value a)) };
    {
      pname = "nx.cpu over three devices";
      cut = true;
      go = (fun o a -> o.run (Nx.place (Cpu3.split ~axis:row_axis) (value a)));
    };
  ]

(* The NaN-ness of each part of each element of [a], a float or complex
   array. *)
let nan_parts (type v s) (a : (v, s) A.t) : bool array array =
  match D.kind (A.dtype a) with
  | D.Complex ->
      let a = A.expect D.Complex128 (A.Any (cast D.Complex128 a)) in
      Array.map
        (fun (c : Complex.t) -> [| Float.is_nan c.re; Float.is_nan c.im |])
        (A.to_array a)
  | _ ->
      Array.map (fun x -> [| Float.is_nan x |]) (A.to_array (cast D.Float64 a))

(* The results [es] of [o] over [dt], each as its parts' bytes, a NaN part
   as ["nan"] where the parts fold apart, else a number with a NaN part as
   ["nan"]: a NaN result promises a NaN, never which one. *)
let blinded (type v s) o (dt : (v, s) D.t) (es : string array) =
  if o.result = Position then Array.map (fun e -> [| e |]) es
  else
    Array.map
      (fun e ->
        let one = A.v dt (L.contiguous [| 1 |]) (Rig.Buffer.of_string e) in
        let nan = (nan_parts one).(0) in
        if o.apart && Array.length nan = 2 then
          let pw = String.length e / 2 in
          Array.init 2 (fun p ->
              if nan.(p) then "nan" else String.sub e (p * pw) pw)
        else if Array.exists Fun.id nan then [| "nan" |]
        else [| e |])
      es

(* [want], each result the host gives as {!blinded} reads it, with each part
   that a NaN term reaches made ["nan"]: the whole number where a complex
   number with a NaN part is a NaN, the part where the parts fold apart, and
   a position the first NaN term's. *)
let nan_want o rows n (nan : bool array array) (want : string array array) =
  let parts = Array.length nan.(0) in
  let want = Array.copy want in
  let first r upto test =
    let rec go j = if j > upto then None else if test (r * n + j) then Some j else go (j + 1) in
    go 0
  in
  let reached r upto test = first r upto test <> None in
  let any i = Array.exists Fun.id nan.(i) in
  let settle k r upto =
    match o.result with
    | Position -> (
        match first r upto any with
        | Some j -> want.(k) <- [| part_bytes 8 (Int64.of_int j) |]
        | None -> ())
    | Folded | Running ->
        if o.apart && parts = 2 then
          List.iter
            (fun p ->
              if reached r upto (fun i -> nan.(i).(p)) then begin
                let e = Array.copy want.(k) in
                e.(p) <- "nan";
                want.(k) <- e
              end)
            [ 0; 1 ]
        else if reached r upto any then want.(k) <- [| "nan" |]
  in
  for r = 0 to rows - 1 do
    match o.result with
    | Folded | Position -> settle r r (n - 1)
    | Running ->
        for j = 0 to n - 1 do
          settle ((r * n) + j) r j
        done
  done;
  want

let law_nans (Nan_case { dt; rows; n; bytes }) =
  let a = nan_array dt rows n bytes in
  let raw = Array.map string_of_bytes (element_bits a) in
  let quiet = Array.map string_of_bytes (element_bits (Nx_support.quieted a)) in
  let nan = nan_parts a in
  let any i = Array.exists Fun.id nan.(i) in
  let firsts =
    List.filter_map
      (fun r -> List.find_opt any (List.init n (fun j -> (r * n) + j)))
      (List.init rows Fun.id)
  in
  cover "a signalling NaN first in a row"
    (List.exists (fun i -> raw.(i) <> quiet.(i)) firsts);
  cover "a quiet NaN first in a row"
    (List.exists (fun i -> raw.(i) = quiet.(i)) firsts);
  cover "a row of no NaN" (List.length firsts < rows);
  cover "a NaN first past the first device's window"
    (List.exists (fun i -> i mod n >= (n + 2) / 3) firsts);
  cover "an uneven cut" (n mod 3 <> 0);
  cover "an empty window" (n < 3);
  cover "a narrow float" (D.bits dt < 32);
  cover "a complex number" (parts_of dt = 2);
  cover "a complex number of one NaN part"
    (parts_of dt = 2
    && Array.exists (fun p -> p.(0) <> p.(1)) nan);
  List.iter
    (fun o ->
      if (not o.floats) || parts_of dt = 1 then begin
        let host = blinded o dt ((List.hd placings).go o a) in
        let want = nan_want o rows n nan host in
        List.iter
          (fun p ->
            if p.cut && o.whole then
              raises ~msg:(o.oname ^ ", " ^ p.pname)
                (Invalid_argument
                   "t: operand 0 lies at split ~axis:1 [m0; m1; m3] and is \
                    read at on [m0; m1; m3]; Nx.place (Placement.on s) x")
                (fun () -> p.go o a)
            else if not (p.cut && o.rounds) then
              equal ~msg:(o.oname ^ ", " ^ p.pname) (array (array string))
                want (blinded o dt (p.go o a)))
          placings
      end)
    nan_ops

(* An extreme along an empty axis is refused only where its result has an
   element: one with none is the empty result, on one device or cut over
   three along either axis. *)
let test_empty_extremes () =
  let empty s = value (A.of_array D.Float32 (Iarray.of_array s) [||]) in
  let shape x = Iarray.to_array (L.shape (A.layout (host_array x))) in
  equal ~msg:"max" (array int) [| 0 |]
    (shape (Nx.max ~axes:[ 1 ] (empty [| 0; 0 |])));
  equal ~msg:"min, kept" (array int) [| 0; 1 |]
    (shape (Nx.min ~axes:[ 1 ] ~keepdims:true (empty [| 0; 0 |])));
  equal ~msg:"argmax" (array int) [| 0 |]
    (shape (Nx.argmax ~axis:1 (empty [| 0; 0 |])));
  equal ~msg:"argmin" (array int) [| 3; 0 |]
    (shape (Nx.argmin ~axis:1 (empty [| 3; 0; 0 |])));
  List.iter
    (fun axis ->
      let x = Nx.place (Cpu3.split ~axis) (empty [| 0; 0 |]) in
      equal ~msg:"max, cut" (array int) [| 0 |] (shape (Nx.max ~axes:[ 1 ] x));
      equal ~msg:"argmin, cut" (array int) [| 0 |]
        (shape (Nx.argmin ~axis:1 x)))
    [ 0; 1 ];
  names "max" (fun () -> Nx.max ~axes:[ 1 ] (empty [| 2; 0 |]));
  names "min" (fun () -> Nx.min (empty [| 0; 0 |]));
  names "argmax" (fun () -> Nx.argmax ~axis:1 (empty [| 2; 0 |]));
  names "argmin" (fun () -> Nx.argmin (empty [| 0; 0 |]))

(* A scan with a carry continues each slice from the carry's element. Over a
   value cut along its axis, each window's results take the terms before it,
   so it has the bits of the scan on one device. *)
let test_carried_cut () =
  let s : int iarray = [| 5; 3 |] in
  let xs = Array.init 15 (fun i -> Int32.of_int ((i * 7) - 40)) in
  let carried x c : (int32, D.int32_elt, _) Nx.t =
    Nx.Prim.eval ~by:"t"
      (Scan
         {
           layout = L.contiguous s;
           axis = 0;
           prog = identity D.Int32;
           reduction = Monoid (Sum, 0, D.Int32);
           loads = [| Plain x |];
           carry = Some (Nx.Prim.Any c);
         })
  in
  let x = value (A.of_array D.Int32 s xs) in
  let c = value (A.of_array D.Int32 [| 1; 3 |] [| 100l; -7l; 2l |]) in
  let want = A.to_array (host_array (carried x c)) in
  let cut = Nx.place (Cpu3.split ~axis:0) x and on = Nx.place Cpu3.on c in
  raises
    (Invalid_argument
       "t: operand 0 lies at split ~axis:0 [m0; m1; m3] and is read at on [m0; \
        m1; m3]; Nx.place (Placement.on s) x")
    (fun () -> carried cut on);
  let whole = Nx.place Cpu3.on cut in
  equal (array int32) want (A.to_array (host_array (carried whole on)))

(* Nx.logcumsumexp: within nx_cpu.mli's bound of a float64
   log-sum-exp over each prefix, specials and prefixes bit for bit, a NaN as
   any NaN, and the host's bits wherever the value lies. *)

type lcse_case =
  | Lcse_case :
      (float, 's) D.t * int array * int option * bool * float array
      -> lcse_case

(* A float dtype and moderate terms, at times infinite, [-0.] or a NaN, or in
   a quarter of the cases mostly those, along an axis or over the elements,
   read through a transposed view or not. *)
let lcse_case =
  let open Gen in
  let floats (type s) (dt : (float, s) D.t) =
    let element moderate =
      frequency
        [
          (moderate, float_range (-5.) 5.);
          (1, constant (-0.));
          (1, constant Float.neg_infinity);
          (1, constant Float.infinity);
          (1, map (fun k -> nan_with (Int64.of_int k)) (int_range 1 1000));
        ]
    in
    let* moderate = frequency [ (3, constant 40); (1, constant 1) ] in
    let element = element moderate in
    let* s, axis = along in
    let* flat = frequency [ (3, constant false); (1, constant true) ] in
    let* strided = bool in
    let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) element in
    Lcse_case (dt, s, (if flat then None else Some axis), strided, xs)
  in
  with_pp
    (fun ppf (Lcse_case (dt, s, axis, strided, _)) ->
      Format.fprintf ppf "%a %a %s%s" D.pp dt pp_ints s
        (match axis with
        | Some a -> Printf.sprintf "along %d" a
        | None -> "in C order")
        (if strided then ", transposed" else ""))
    (one_of
       [
         floats D.Float64;
         floats D.Float32;
         floats D.Float16;
         floats D.Bfloat16;
         floats D.Float8_e4m3fn;
         floats D.Float8_e5m2;
         floats D.Float4_e2m1fn;
       ])

let law_lcse (Lcse_case (dt, s, axis, strided, xs)) =
  let u = match dt with D.Float64 -> ldexp 1. (-53) | _ -> ldexp 1. (-24) in
  let x = transposed value strided (A.of_array dt (Iarray.of_array s) xs) in
  let n = Array.length xs in
  cover "float64" (D.name dt = "float64");
  cover "a float narrower than float32" (D.bits dt < 32);
  cover "over the elements" (axis = None && Array.length s > 1);
  cover "transposed" (strided && Array.length s > 1);
  cover "an axis of more than 4"
    (match axis with Some a -> s.(a) > 4 | None -> false);
  let y = Nx.logcumsumexp ?axis x in
  equal ~msg:"shape" (array int) s (Nx.shape y);
  let ts = read x in
  let got = read y in
  let ps =
    match axis with Some a -> prefixes s a | None -> prefixes [| n |] 0
  in
  List.iter
    (fun ps ->
      let at = List.nth ps (List.length ps - 1) in
      let terms = List.map (fun p -> ts.(p)) ps in
      let g = got.(at) in
      let m = List.fold_left Float.max Float.neg_infinity terms in
      if List.exists Float.is_nan terms then begin
        cover "a NaN term" true;
        equal ~msg:"a NaN" bool true (Float.is_nan g)
      end
      else if not (Float.is_finite m) then begin
        cover "+inf" (m > 0.);
        cover "every term -inf" (m < 0.);
        equal ~msg:"an infinity" float_exact m g
      end
      else begin
        cover "finite" true;
        cover "a -inf term among finite ones"
          (List.exists (fun t -> t = Float.neg_infinity) terms);
        let want, err = lse_ref m terms in
        let bound = lse_bound u (List.length terms) want +. err in
        let msg = Printf.sprintf "%h against %h" g want in
        at_least ~msg float_exact ~than:(rounded dt (want -. bound)) g;
        at_most ~msg float_exact ~than:(rounded dt (want +. bound)) g
      end)
    ps

(* The logcumsumexp of a prefix of the axis is the prefix of the
   logcumsumexp, bit for bit or both NaN; over the elements, it is the
   logcumsumexp of the elements in C order. Cut over three devices along
   another axis than the one it scans, it has the host's bits. It reads the
   axis it scans whole, the elements' order included, so a cut along it
   raises where the value has elements, and placed whole as the refusal
   writes, it has the host's bits. *)
let law_lcse_bits ((Lcse_case (dt, s, axis, strided, xs)), cut, keep) =
  let x = transposed value strided (A.of_array dt (Iarray.of_array s) xs) in
  let r = Array.length s in
  let y = Nx.logcumsumexp ?axis x in
  let bits_of y = blind (host_array y) in
  (match axis with
  | Some a ->
      let k = keep mod (s.(a) + 1) in
      cover "a proper prefix" (k < s.(a));
      cover "an empty prefix" (k = 0 && s.(a) > 0);
      let prefix t =
        Nx.slice (List.init r (fun i -> if i = a then Nx.R (0, k) else Nx.A)) t
      in
      equal ~msg:"the prefix" blind_bits
        (bits_of (prefix y))
        (bits_of (Nx.logcumsumexp ~axis:a (prefix x)))
  | None ->
      let flat = Nx.reshape [| Array.length xs |] x in
      equal ~msg:"C order" blind_bits
        (bits_of (Nx.reshape [| Array.length xs |] y))
        (bits_of (Nx.logcumsumexp ~axis:0 flat)));
  let c = cut mod r in
  cover "cut along the scanned axis" (axis = Some c);
  cover "cut along another axis" (axis <> None && axis <> Some c);
  cover "cut, in C order" (axis = None && Array.length xs > 0);
  let placed = Nx.place (Cpu3.split ~axis:c) x in
  let placed =
    if (axis = None || axis = Some c) && Array.length xs > 0 then begin
      raises_match
        (Exn.invalid_arg
           ~substring:
             "and is read at on [m0; m1; m3]; Nx.place (Placement.on s) x")
        (fun () -> Nx.logcumsumexp ?axis placed);
      Nx.place Cpu3.on placed
    end
    else placed
  in
  equal ~msg:"cut over three devices" blind_bits (bits_of y)
    (bits_of (Nx.logcumsumexp ?axis placed))

let lcse_bits_case =
  Gen.(triple lcse_case (int_range 0 7) (int_range 0 40))

let test_lcse_cases () =
  let f64 xs = value (A.of_array D.Float64 [| Array.length xs |] xs) in
  let f32 xs = value (A.of_array D.Float32 [| Array.length xs |] xs) in
  let lcse x = read (Nx.logcumsumexp x) in
  let inf = Float.infinity and ninf = Float.neg_infinity in
  let ln2 = Float.log 2. in
  let big = [| 1000.; 1000.; 999. |] in
  equal ~msg:"the premise: log (cumsum (exp x)) overflows at 1000"
    (array float_exact) [| inf; inf; inf |]
    (read (Nx.log (Nx.cumsum (Nx.exp (f64 big)))));
  equal ~msg:"float64 terms of 1000" (array (float 1e-12))
    [| 1000.; 1000. +. ln2; 1000. +. Float.log (2. +. exp (-1.)) |]
    (lcse (f64 big));
  equal ~msg:"float32 terms of 100, whose exp overflows" (array (float 1e-4))
    [| 100.; 100. +. ln2 |]
    (lcse (f32 [| 100.; 100. |]));
  let empty s axis =
    let x = value (A.of_array D.Float32 (Iarray.of_array s) [||]) in
    equal ~msg:"an empty value's shape" (array int) s
      (Nx.shape (Nx.logcumsumexp ?axis x))
  in
  empty [| 2; 0 |] (Some 1);
  empty [| 0; 3 |] (Some 1);
  empty [| 0 |] None;
  equal ~msg:"one element along the axis: itself, -0 as +0" (array int64)
    (Array.map bits [| 2.5; 0.; ninf |])
    (Array.map bits
       (read
          (Nx.logcumsumexp ~axis:1
             (value (A.of_array D.Float64 [| 3; 1 |] [| 2.5; -0.; ninf |])))));
  equal ~msg:"-inf terms" (array float_exact) [| ninf; ninf; 0.; 0. |]
    (lcse (f64 [| ninf; ninf; 0.; ninf |]));
  equal ~msg:"+inf from it on" (array float_exact) [| 1.; inf; inf; inf |]
    (lcse (f64 [| 1.; inf; 2.; ninf |]));
  equal ~msg:"a NaN, from the first on" (array same_float)
    [| 1.; nan; nan; nan |]
    (lcse (f64 [| 1.; nan_with 7L; nan_with 9L; inf |]));
  let x = value (A.of_array D.Float32 [| 2; 3 |] (Array.make 6 1.)) in
  names "logcumsumexp" (fun () -> Nx.logcumsumexp ~axis:2 x)

let lcse_group =
  group "logcumsumexp"
    [
      prop
        "logcumsumexp is log (cumsum (exp x)) within its stated bound at \
         every float dtype, specials exact"
        lcse_case law_lcse;
      prop
        "logcumsumexp's prefixes, C order and cuts over devices keep its bits"
        lcse_bits_case law_lcse_bits;
      test "logcumsumexp's cases" test_lcse_cases;
    ]

let reductions_group =
  group "reductions"
    [
      prop "int32 reductions are the folds of their elements" int32_case
        law_int32;
      prop "int32 scans are the running folds, in C order or along an axis"
        int32_case law_int32_scan;
      prop "each reduction computes its stated dtypes as the host's kernel"
        any_case law_any;
      test
        "bad axes, empty extremes and refused dtypes raise naming the function"
        test_errors;
      test "a refused axis is printed as passed" test_axis_printed;
      test "an extreme of no term raises only where its result has an element"
        test_empty_extremes;
      test "mean divides the sum by the count" test_mean;
      prop "an integer mean is the exact mean truncated toward zero" int_case
        law_int_mean;
      test "integer means of extremes and empty axes" test_int_mean;
      test "a complex cumsum adds the parts, a narrow float's rounds once"
        test_scan_parts;
      test
        "a carried scan over a value cut along its axis raises, and placed \
         whole is the whole's"
        test_carried_cut;
      test "Logsumexp, Moments and Arg's rules raise naming the caller"
        test_optional_rules;
      prop
        "a NaN term makes a NaN result on the host and, where partials keep \
         bits, cut along the reduced axis"
        nan_case law_nans;
    ]

let () =
  exit
    (run "nx reduce"
       [ reductions_group; complex_group; lcse_group ])
