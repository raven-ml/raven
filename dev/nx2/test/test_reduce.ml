(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's reductions and scans on the host. On int32 each is the fold of its
   elements, an order-free reference; on every dtype each computes the dtypes
   its doc states, as the host's kernel over the same axes (the narrower floats
   at float32, rounded once, a NaN result the first NaN element quieted), and
   raises naming itself elsewhere. A NaN result is the first NaN quieted on
   every path: the host's kernels, nx's expansions, and values cut along the
   reduced axis. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog

type host = Nx.host

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

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
  let x = value (A.of_array D.Int32 s xs) in
  let check name f init g =
    let y = host_array (g ?axes ~keepdims x) in
    equal ~msg:(name ^ ": shape") (array int)
      (if keepdims then kept s ax else reduced s ax)
      (L.shape (A.layout y));
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
  let x = value (A.of_array D.Int32 s xs) in
  List.iter
    (fun (name, f, (g : ?axis:int -> _ -> _)) ->
      let flat = host_array (g x) in
      equal ~msg:(name ^ ": shape") (array int) s (L.shape (A.layout flat));
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
  A.v dt (L.contiguous s)
    (Rig.Buffer.of_string (if data = "" then "\000" else data))

let any_case =
  let open Gen in
  with_pp
    (fun ppf (Case a, axes) ->
      Format.fprintf ppf "%a %a over %s" D.pp (A.dtype a) pp_ints
        (L.shape (A.layout a))
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
  match Nx_cpu.apply1 Cast ~dst a with
  | A.Done -> dst
  | _ -> failwith "the host's kernel declined a cast"

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
      let s = L.shape (A.layout a) in
      let out =
        Array.of_list
          (List.filteri (fun i _ -> not (Array.mem i axes)) (Array.to_list s))
      in
      let dst = A.create Rig.host dt out in
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

(* [want], the bytes of [a]'s reduction along [axes], with each result that
   a NaN element reaches made the first one's bytes, quieted: its quiet bit
   set, its sign and the rest of its payload kept. *)
let first_nans (type v s) (a : (v, s) A.t) axes want =
  match D.kind (A.dtype a) with
  | D.Float when D.bits (A.dtype a) >= 8 ->
      let w = D.bits (A.dtype a) / 8 in
      let bytes = bits_of (Nx_support.quieted a) in
      let nan = Array.map Float.is_nan (A.to_array (cast D.Float32 a)) in
      let first p i =
        match p with None when nan.(i) -> Some i | _ -> p
      in
      let firsts =
        fold first None (L.shape (A.layout a)) (Array.to_list axes)
          (Array.init (Array.length nan) Fun.id)
      in
      Array.concat
        (Array.to_list
           (Array.mapi
              (fun o f ->
                match f with
                | Some i -> Array.sub bytes (i * w) w
                | None -> Array.sub want (o * w) w)
              firsts))
  | _ -> want

let law_any (Case a, axes) =
  let dt = A.dtype a in
  let s = L.shape (A.layout a) in
  let ax = Array.of_list (normal (Array.length s) axes) in
  (* An element of the result of no term: an extreme there raises. *)
  let no_term =
    Array.exists (fun i -> s.(i) = 0) ax
    && Array.fold_left ( * ) 1 (reduced s (Array.to_list ax)) > 0
  in
  List.iter
    (fun (name, m, f) ->
      let f () = bits_of (host_array (f ?axes (value a))) in
      let extreme =
        match (m : Nx_kernel.monoid) with Max | Min -> true | _ -> false
      in
      if documented m dt && not (no_term && extreme) then begin
        cover "computed" true;
        match reference m a ax with
        | Some want ->
            equal ~msg:name (array int) (first_nans a ax (bits_of want)) (f ())
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
  equal (array int) [| 1; 1 |] (L.shape (A.layout all));
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
  let x = value (A.of_array dt s xs) in
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
  equal ~msg:"kept" (array int) [| 2; 1 |] (L.shape (A.layout kept));
  equal ~msg:"kept" (array int) [| 1; 5 |] (A.to_array kept)

(* Scans of complex numbers and narrow floats: a complex sum adds the parts,
   and a narrow float accumulates at float32, rounding each prefix once. *)
let test_scan_expansions () =
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

(* Logsumexp, Moments and Arg through Nx.Prim, by nx's expansion, against
   references over each result's terms. *)

(* [f ()] with each reduction and scan computed by its expansion where it has
   one. *)
let expanded f =
  Nx_support.expanded
    (fun (type r) (op : r Nx.Prim.t) ->
      match op with Reduce _ | Scan _ -> true | _ -> false)
    f

(* [law] whose operations compute by their expansions. *)
let expanding law c = expanded (fun () -> law c)

let reduce1 r axes x =
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout = L.contiguous (Nx.shape x);
           axes;
           prog = identity (Nx.dtype x);
           reductions = [ r ];
           loads = [| Plain x |];
         })
  in
  y

let read x = A.to_array (host_array x)
let bits = Int64.bits_of_float
let nan_with k = Int64.float_of_bits (Int64.logor 0x7ff8_0000_0000_0000L k)

(* Each result's terms, in C order of the results and of the reduced indices. *)
let terms s axes xs =
  Array.map List.rev (fold (fun acc x -> x :: acc) [] s axes xs)

let float64_case =
  let open Gen in
  let element =
    frequency
      [
        (24, float_range (-30.) 30.);
        (1, constant Float.neg_infinity);
        (1, constant Float.infinity);
        (1, map (fun k -> nan_with (Int64.of_int k)) (int_range 1 1000));
      ]
  in
  with_pp
    (fun ppf (s, axes, _) ->
      Format.fprintf ppf "%a over %s" pp_ints s
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l)))
    (let* s = shape in
     let* axes = axes_of (Array.length s) in
     let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) element in
     (s, axes, xs))

let law_logsumexp (s, axes, xs) =
  let ax = normal (Array.length s) axes in
  let x = value (A.of_array D.Float64 s xs) in
  let got =
    read (reduce1 (Logsumexp (0, Nx.float64)) (Iarray.of_list ax) x)
  in
  Array.iteri
    (fun i ts ->
      let g = got.(i) in
      match List.find_opt Float.is_nan ts with
      | Some nan ->
          cover "a NaN term" true;
          equal ~msg:"the first NaN, its bits" int64 (bits nan) (bits g)
      | None ->
          let m = List.fold_left Float.max Float.neg_infinity ts in
          if Float.is_finite m then begin
            cover "finite" true;
            let s = List.fold_left (fun s t -> s +. exp (t -. m)) 0. ts in
            let want = m +. log s in
            let n = Float.of_int (List.length ts) in
            at_most
              ~msg:(Printf.sprintf "%h against %h" g want)
              float_exact
              ~than:((n +. 8.) *. epsilon_float *. (1. +. Float.abs want))
              (Float.abs (g -. want))
          end
          else begin
            cover "no term, or an infinite maximum" true;
            equal ~msg:"an infinity" int64 (bits m) (bits g)
          end)
    (terms s ax xs)

(* float32 terms near an offset [c] within [w], whose large ratio of mean to
   spread a formula reading [Σ x²] would lose, and the mean's own error too
   where the spread is small. *)
let moments_case =
  let open Gen in
  with_pp
    (fun ppf (s, axes, c, _) ->
      Format.fprintf ppf "%a over %s near %g" pp_ints s
        (match axes with
        | None -> "all"
        | Some l -> String.concat "," (List.map string_of_int l))
        c)
    (let* s = shape in
     let* axes = axes_of (Array.length s) in
     let* c, w = of_list [ (0., 10.); (1e3, 10.); (1e5, 10.); (1e4, 1e-2) ] in
     let+ xs =
       array
         ~size:(constant (Array.fold_left ( * ) 1 s))
         (map
            (fun d -> Int32.float_of_bits (Int32.bits_of_float (c +. d)))
            (float_range (-.w) w))
     in
     (s, axes, c, xs))

(* The bounds nx's expansion of Moments states, at float32's unit roundoff: the
   mean within [γ(n) M], [M = Σ|x| / n], the variance [V] within [(γ(n + 4) +
   2γ(n)²) V + 2γ(n + 6) γ(n) M √V + 4γ(n + 4) γ(n)² M²]. The references are
   float64 two-pass sums, whose own error is the slack. *)
let law_moments (s, axes, _, xs) =
  let ax = normal (Array.length s) axes in
  let x = value (A.of_array D.Float32 s xs) in
  let mean, var = reduce1 (Moments (0, Nx.float32)) (Iarray.of_list ax) x in
  let mean = read mean and var = read var in
  let u = ldexp 1. (-24) in
  let gamma k = Float.of_int k *. u /. (1. -. (Float.of_int k *. u)) in
  Array.iteri
    (fun i ts ->
      let n = List.length ts in
      if n = 0 then begin
        cover "no term" true;
        equal ~msg:"mean" bool true (Float.is_nan mean.(i));
        equal ~msg:"variance" bool true (Float.is_nan var.(i))
      end
      else begin
        cover "terms" true;
        let nf = Float.of_int n in
        let m = List.fold_left ( +. ) 0. ts /. nf in
        let v =
          List.fold_left (fun a t -> a +. ((t -. m) *. (t -. m))) 0. ts /. nf
        in
        let abs_mean =
          List.fold_left (fun a t -> a +. Float.abs t) 0. ts /. nf
        in
        let e = 4. *. nf *. epsilon_float in
        let slack = (e *. v) +. (e *. abs_mean *. (1. +. (e *. abs_mean))) in
        at_most ~msg:"mean" float_exact
          ~than:((gamma n *. abs_mean) +. slack)
          (Float.abs (mean.(i) -. m));
        let g = gamma n in
        at_most ~msg:"variance" float_exact
          ~than:
            (((gamma (n + 4) +. (2. *. g *. g)) *. v)
            +. (2. *. gamma (n + 6) *. g *. abs_mean *. Float.sqrt v)
            +. (4. *. gamma (n + 4) *. g *. g *. abs_mean *. abs_mean)
            +. slack)
          (Float.abs (var.(i) -. v))
      end)
    (terms s ax xs)

(* Moments scans: by the expansion, within its stated bounds of float64
   references over each prefix, and by nx.cpu on the host, with the
   expansion's bits. *)

let scan1 r axis x =
  Nx.Prim.eval ~by:"t"
    (Scan
       {
         layout = L.contiguous (Nx.shape x);
         axis;
         prog = identity (Nx.dtype x);
         reduction = r;
         loads = [| Plain x |];
         carry = None;
       })

(* Where a Moments scan computes: by the expansion, by nx.cpu, on the
   host. *)
type path = Expansion | Cpu

let path_name = function Expansion -> "expansion" | Cpu -> "nx.cpu"

(* The Moments scan of [a] along [axis] by [path], read. *)
let scan_moments (type v s) path axis (a : (v, s) A.t) =
  let dt = A.dtype a in
  match path with
  | Expansion ->
      expanded (fun () ->
          let m, v = scan1 (Moments (0, dt)) axis (value a) in
          (read m, read v))
  | Cpu ->
      let m, v = scan1 (Moments (0, dt)) axis (value a) in
      (read m, read v)

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

(* The most combinations output [i] of a scan's expansion passes through,
   [D_i = max(0, 2⌈log₂(i+1)⌉ - 1)], for [n = i + 1] terms. *)
let depth n =
  let rec log2_up b = if 1 lsl b >= n then b else log2_up (b + 1) in
  Float.of_int (Int.max 0 ((2 * log2_up 0) - 1))

(* Values near [N(0, 1)]: a sum of three uniforms. *)
let gauss =
  let open Gen in
  let u = float_range (-1.) 1. in
  map (fun (a, b, c) -> a +. b +. c) (triple u u u)

let float32 v = Int32.float_of_bits (Int32.bits_of_float v)

(* What a Moments scan's terms are: near an offset, as {!moments_case}'s; a
   first term of [1e4] above [N(0, 1)]; runs of [±1e3]; [N(1e4, 1)]; and
   [N(0, 1)] past nx.cpu's chunks of 4096. *)
type terms = Near of float | Outlier | Runs of int | Offset | Long

let terms_name = function
  | Near c -> Printf.sprintf "near %g" c
  | Outlier -> "an outlier first"
  | Runs l -> Printf.sprintf "runs of %d of ±1e3" l
  | Offset -> "N(1e4, 1)"
  | Long -> "N(0, 1), long"

(* float32 terms of a kind of {!terms} along one axis, at times long enough
   for several levels of the expansion's recursion. *)
let moments_scan_case =
  let open Gen in
  let along_n n =
    let* outer = int_range 1 2 in
    let+ axis = int_range 0 1 in
    ((if axis = 0 then [| n; outer |] else [| outer; n |]), axis)
  in
  let terms_of kind s axis =
    let size = constant (Array.fold_left ( * ) 1 s) in
    let n = s.(axis) in
    let r = Array.length s in
    let after = Array.sub s (axis + 1) (r - axis - 1) in
    let stride = Array.fold_left ( * ) 1 after in
    let index p = p / stride mod n in
    match kind with
    | Near c -> array ~size (map (fun d -> c +. d) (float_range (-10.) 10.))
    | Outlier ->
        let+ xs = array ~size gauss in
        Array.mapi (fun p x -> if index p = 0 then 1e4 else x) xs
    | Runs l ->
        let+ xs = array ~size gauss in
        Array.mapi
          (fun p x -> if index p / l mod 2 = 0 then 1e3 +. x else x -. 1e3)
          xs
    | Offset -> array ~size (map (fun x -> 1e4 +. x) gauss)
    | Long -> array ~size gauss
  in
  let case kind (s, axis) =
    let+ xs = terms_of kind s axis in
    (kind, s, axis, Array.map float32 xs)
  in
  with_pp
    (fun ppf (kind, s, axis, _) ->
      Format.fprintf ppf "%a along %d, %s" pp_ints s axis (terms_name kind))
    (one_of
       [
         (let* s = shape in
          let s = if Array.length s = 0 then [| 3 |] else Array.copy s in
          let* axis = int_range 0 (Array.length s - 1) in
          let* n = frequency [ (3, constant s.(axis)); (1, int_range 5 40) ] in
          s.(axis) <- n;
          let* c = of_list [ 0.; 1e3; 1e5 ] in
          case (Near c) (s, axis));
         (let* n = int_range 64 1024 in
          let* sa = along_n n in
          case Outlier sa);
         (let* l = int_range 16 128 in
          let* n = int_range (4 * l) 1024 in
          let* sa = along_n n in
          case (Runs l) sa);
         (let* n = int_range 64 1024 in
          let* sa = along_n n in
          case Offset sa);
         (let* n = int_range 4096 4200 in
          let* sa = along_n n in
          case Long sa);
       ])

(* The expansion's bounds, with [D] the depth of the result and [R] the
   terms' range: the mean within [(4 D + 1) u R + u|μ|], the variance [V]
   within [(7 D + 1) u V + (4 D² - 2 D) u R √V]. The references run
   Welford's update in float64 over the terms less the slice's first, whose
   own error, within [n ε (R + V + R²)] with room to spare, is the
   slack. *)
let law_moments_scan (kind, s, axis, xs) =
  cover "near an offset" (match kind with Near _ -> true | _ -> false);
  cover "an outlier first" (kind = Outlier);
  cover "runs of ±1e3" (match kind with Runs _ -> true | _ -> false);
  cover "N(1e4, 1)" (kind = Offset);
  cover "N(0, 1) past 4096 terms" (kind = Long);
  let x = A.of_array D.Float32 s xs in
  let u = ldexp 1. (-24) in
  let bound n ~m ~v ~range =
    let d = depth n in
    ( (((4. *. d) +. 1.) *. u *. range) +. (u *. Float.abs m),
      (((7. *. d) +. 1.) *. u *. v)
      +. (((4. *. d *. d) -. (2. *. d)) *. u *. range *. Float.sqrt v) )
  in
  let mean, var = scan_moments Expansion axis x in
  List.iter
    (fun at ->
      let c = xs.(at 0) in
      let mu = ref 0. and q = ref 0. in
      let lo = ref c and hi = ref c in
      for j = 0 to s.(axis) - 1 do
        let t = xs.(at j) in
        let n = j + 1 in
        let nf = Float.of_int n in
        let d = t -. c in
        let delta = d -. !mu in
        mu := !mu +. (delta /. nf);
        q := !q +. (delta *. (d -. !mu));
        lo := Float.min !lo t;
        hi := Float.max !hi t;
        let m = c +. !mu and v = !q /. nf and range = !hi -. !lo in
        let bm, bv = bound n ~m ~v ~range in
        let slack =
          8. *. nf *. epsilon_float *. (range +. v +. (range *. range))
        in
        cover "one term" (n = 1);
        cover "terms" (n > 1);
        at_most ~msg:"mean" float_exact
          ~than:(bm +. slack +. (epsilon_float *. Float.abs m))
          (Float.abs (mean.(at j) -. m));
        at_most ~msg:"variance" float_exact ~than:(bv +. slack)
          (Float.abs (var.(at j) -. v))
      done)
    (slices s axis)

(* Both paths: no result along an empty axis; one term's own mean and a
   variance of [+0]; a NaN's bits from it on; an infinity's mean and a NaN
   variance; a NaN term's bits past the NaN of opposite infinities; a
   narrow float at float32, rounded once. *)
let test_moments_scan_cases () =
  let both name f =
    List.iter (fun p -> f (name ^ ", " ^ path_name p) p) [ Expansion; Cpu ]
  in
  let moments p dt s xs =
    scan_moments p (Array.length s - 1) (A.of_array dt s xs)
  in
  both "an empty axis" (fun msg p ->
      let mean, var = moments p D.Float64 [| 2; 0 |] [||] in
      equal ~msg (pair int int) (0, 0) (Array.length mean, Array.length var));
  both "one term" (fun msg p ->
      let mean, var = moments p D.Float64 [| 3; 1 |] [| 2.5; -0.; 1e300 |] in
      equal ~msg (pair (array float_exact) (array float_exact))
        ([| 2.5; 0.; 1e300 |], [| 0.; 0.; 0. |])
        (mean, var));
  both "a NaN" (fun msg p ->
      let a = nan_with 7L in
      let xs = [| 1.; a; nan_with 9L; 2. |] in
      let mean, var = moments p D.Float64 [| 4 |] xs in
      equal ~msg
        (pair (array int64) (array int64))
        ( [| bits 1.; bits a; bits a; bits a |],
          [| bits 0.; bits a; bits a; bits a |] )
        (Array.map bits mean, Array.map bits var));
  both "infinities" (fun msg p ->
      let xs = [| 1.; Float.infinity; 2.; Float.neg_infinity |] in
      let mean, var = moments p D.Float64 [| 4 |] xs in
      equal ~msg:(msg ^ ", mean") (array float_exact)
        [| 1.; Float.infinity; Float.infinity |]
        (Array.sub mean 0 3);
      equal ~msg:(msg ^ ", NaNs") (list bool)
        [ true; true; true; true ]
        (List.map Float.is_nan [ mean.(3); var.(1); var.(2); var.(3) ]));
  both "opposite infinities, then a NaN" (fun msg p ->
      let a = nan_with 7L and made = 0x7ff8_0000_0000_0000L in
      let xs = [| Float.infinity; Float.neg_infinity; a; 2. |] in
      let mean, var = moments p D.Float64 [| 4 |] xs in
      equal ~msg
        (pair (array int64) (array int64))
        ( [| bits Float.infinity; made; bits a; bits a |],
          [| made; made; bits a; bits a |] )
        (Array.map bits mean, Array.map bits var));
  both "float16" (fun msg p ->
      let mean, var = moments p D.Float16 [| 4 |] [| 1.; 2.; 3.; 4. |] in
      let want = A.of_array D.Float32 [| 4 |] [| 0.; 0.25; 2. /. 3.; 1.25 |] in
      equal ~msg (pair (array float_exact) (array float_exact))
        ([| 1.; 1.5; 2.; 2.5 |], A.to_array (cast D.Float16 want))
        (mean, var))

(* The extreme and the first position of its bits, by the reference's strict
   [better]. *)
let first_extreme better ts =
  let best = ref 0 in
  List.iteri (fun i t -> if better t (List.nth ts !best) then best := i) ts;
  (List.nth ts !best, Int64.of_int !best)

let law_arg (s, axes, _, xs) =
  let ax = normal (Array.length s) axes in
  if not (List.exists (fun a -> s.(a) = 0) ax) then begin
    let x = value (A.of_array D.Int32 s xs) in
    List.iter
      (fun (name, e, better) ->
        let v, p = reduce1 (Arg (e, 0, Nx.int32)) (Iarray.of_list ax) x in
        let want = Array.map (first_extreme better) (terms s ax xs) in
        equal ~msg:(name ^ " extremes") (array int32) (Array.map fst want)
          (read v);
        equal ~msg:(name ^ " positions") (array int64) (Array.map snd want)
          (read p))
      [
        ("max", (Max : Nx_kernel.extreme), fun t b -> Int32.compare t b > 0);
        ("min", Min, fun t b -> Int32.compare t b < 0);
      ]
  end

let vector dt xs = value (A.of_array dt [| Array.length xs |] xs)

let test_arg_cases () =
  let arg (type v s) e (dt : (v, s) D.t) axes shape (xs : v array) =
    let v, p = reduce1 (Arg (e, 0, dt)) axes (value (A.of_array dt shape xs)) in
    (read v, read p)
  in
  let f64 e xs =
    let v, p = arg e D.Float64 [| 0 |] [| Array.length xs |] xs in
    (bits v.(0), p.(0))
  in
  let found = pair int64 int64 in
  equal ~msg:"max of signed zeros: +0, the first one" found
    (bits 0., 1L)
    (f64 Max [| -0.; 0.; -0.; 0. |]);
  equal ~msg:"min of signed zeros: -0, the first one" found
    (bits (-0.), 0L)
    (f64 Min [| -0.; 0.; -0. |]);
  let a = nan_with 7L and b = nan_with 9L in
  equal ~msg:"a NaN is the extreme: the first, its bits" found
    (bits a, 1L)
    (f64 Max [| 1.; a; 5.; b |]);
  equal ~msg:"and for min" found (bits a, 1L) (f64 Min [| 1.; a; -5.; b |]);
  equal ~msg:"ties: the first" found (bits 7., 1L) (f64 Max [| 2.; 7.; 7. |]);
  let grid = [| 1.; 9.; 3.; 9.; 0.; 2. |] in
  equal ~msg:"over both axes: positions in C order of the reduced indices"
    (pair (array float_exact) (array int64))
    ([| 9. |], [| 1L |])
    (arg Max D.Float64 [| 0; 1 |] [| 2; 3 |] grid);
  equal ~msg:"along axis 0"
    (pair (array float_exact) (array int64))
    ([| 9.; 9.; 3. |], [| 1L; 0L; 0L |])
    (arg Max D.Float64 [| 0 |] [| 2; 3 |] grid);
  equal ~msg:"int4"
    (pair (array int) (array int64))
    ([| 7 |], [| 2L |])
    (arg Max D.Int4 [| 0 |] [| 4 |] [| 3; -8; 7; 7 |]);
  equal ~msg:"bool"
    (pair (array bool) (array int64))
    ([| true |], [| 1L |])
    (arg Max D.Bool [| 0 |] [| 3 |] [| false; true; true |]);
  equal ~msg:"float16 signed zeros" (array int64) [| 1L |]
    (snd (arg Max D.Float16 [| 0 |] [| 3 |] [| -0.; 0.; 0. |]));
  equal ~msg:"float4_e2m1fn signed zeros" (array int64) [| 2L |]
    (snd (arg Min D.Float4_e2m1fn [| 0 |] [| 3 |] [| 1.; 0.; -0. |]))

(* Integers [offset + k], [k] within 1 and [offset] above the float's [2^p] so
   that a step is its last bit, in steps of a power of two [step]: with [step²]
   the largest float over [2^10], Σ d² stays finite where (Σ d)², n² times the
   mean's rounding squared, may overflow. *)
let overflow_case offset step =
  let open Gen in
  let* n = int_range 100 200 in
  let+ ks = array ~size:(constant n) (int_range (-1) 1) in
  (offset, step, ks)

(* The bound of {!law_moments}, in steps, of the exact variance of the
   integers. *)
let law_overflow (type s) (dt : (float, s) D.t) u (offset, step, ks) =
  let xs = Array.map (fun k -> Float.of_int (offset + k) *. step) ks in
  let mean, var = reduce1 (Moments (0, dt)) [| 0 |] (vector dt xs) in
  let n = Array.length ks in
  let nf = Float.of_int n in
  let c = ((read mean).(0) /. step) -. Float.of_int offset in
  let s1 = Array.fold_left (fun a k -> a +. (Float.of_int k -. c)) 0. ks in
  let s2 =
    Array.fold_left (fun a k -> a +. ((Float.of_int k -. c) ** 2.)) 0. ks
  in
  cover "the square of the deviations' sum overflows"
    ((s1 *. s1 >= 0x1p10) && s2 < 0x1p10);
  let s = Array.fold_left ( + ) 0 ks
  and q = Array.fold_left (fun a k -> a + (k * k)) 0 ks in
  let v = Float.of_int ((n * q) - (s * s)) /. (nf *. nf) in
  let m = Float.of_int ((n * offset) + s) /. nf in
  let gamma k = Float.of_int k *. u /. (1. -. (Float.of_int k *. u)) in
  let g = gamma n in
  at_most ~msg:"variance" float_exact
    ~than:
      (((gamma (n + 4) +. (2. *. g *. g)) *. v)
      +. (2. *. gamma (n + 6) *. g *. m *. Float.sqrt v)
      +. (4. *. gamma (n + 4) *. g *. g *. m *. m))
    (Float.abs (((read var).(0) /. (step *. step)) -. v))

let test_moments_cases () =
  let mean, var =
    reduce1 (Moments (0, Nx.float64)) [| 0 |] (vector D.Float64 [||])
  in
  equal ~msg:"no term: NaN" (pair bool bool) (true, true)
    (Float.is_nan (read mean).(0), Float.is_nan (read var).(0));
  let a = nan_with 7L in
  let mean, var =
    reduce1
      (Moments (0, Nx.float64))
      [| 0 |]
      (vector D.Float64 [| 1.; a; nan_with 9L |])
  in
  equal ~msg:"a NaN term: the first, its bits" (pair int64 int64)
    (bits a, bits a)
    (bits (read mean).(0), bits (read var).(0));
  let mean, var =
    reduce1
      (Moments (0, Nx.float16))
      [| 0 |]
      (vector D.Float16 [| 1.; 2.; 3.; 4. |])
  in
  equal ~msg:"float16"
    (pair float_exact float_exact)
    (2.5, 1.25)
    ((read mean).(0), (read var).(0));
  (* Finite terms whose squared deviations overflow: their sum, +inf, where
     the square of the deviations' sum overflows too. *)
  let overflow (type s) (dt : (float, s) D.t) xs =
    let _, var = reduce1 (Moments (0, dt)) [| 0 |] (vector dt xs) in
    (read var).(0)
  in
  equal ~msg:"float32 overflow" int64 (bits Float.infinity)
    (bits (overflow D.Float32 [| 8.25e37; -8.25e37; 8.25e37 |]));
  equal ~msg:"float32 overflow of the mean" int64 (bits Float.infinity)
    (bits (overflow D.Float32 [| 3.3e38; 3.3e38 |]));
  equal ~msg:"float64 overflow" int64 (bits Float.infinity)
    (bits (overflow D.Float64 [| 4.25e307; -4.25e307; 4.25e307 |]));
  equal ~msg:"bfloat16 overflow" int64 (bits Float.infinity)
    (bits (overflow D.Bfloat16 [| 8.25e37; -8.25e37; 8.25e37 |]));
  (* 3000 terms cross a block of the sum's lane order. *)
  let xs = Array.init 3000 (fun i -> 1e4 +. Float.of_int (i mod 7)) in
  let mean, var =
    reduce1 (Moments (0, Nx.float64)) [| 0 |] (vector D.Float64 xs)
  in
  let m = Array.fold_left ( +. ) 0. xs /. 3000. in
  let v =
    Array.fold_left (fun a x -> a +. ((x -. m) *. (x -. m))) 0. xs /. 3000.
  in
  equal ~msg:"3000 terms"
    (pair (float 1e-9) (float 1e-9))
    (m, v)
    ((read mean).(0), (read var).(0))

let test_logsumexp_cases () =
  let lse (type v s) (dt : (v, s) D.t) (xs : v array) =
    read (reduce1 (Logsumexp (0, dt)) [| 0 |] (vector dt xs))
  in
  equal ~msg:"no term" int64 (bits Float.neg_infinity)
    (bits (lse D.Float64 [||]).(0));
  equal ~msg:"terms all -inf" int64 (bits Float.neg_infinity)
    (bits (lse D.Float64 [| Float.neg_infinity; Float.neg_infinity |]).(0));
  equal ~msg:"+inf beside -inf" int64 (bits Float.infinity)
    (bits (lse D.Float64 [| Float.neg_infinity; Float.infinity; 1. |]).(0));
  let wide = A.of_array D.Float32 [| 1 |] [| Float.log 2. |] in
  equal ~msg:"bfloat16 at float32, rounded once" (array float_exact)
    (A.to_array (cast D.Bfloat16 wide))
    (lse D.Bfloat16 [| 0.; 0. |])

(* Logsumexp and Arg scans: by the expansion and by nx.cpu, on the host,
   against references over each prefix. *)

(* [a] by [on], read through a transposed view where [strided]: the elements
   of [a], held in a layout that runs its axes backwards. *)
let transposed on strided a =
  let s = L.shape (A.layout a) in
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
    Nx.transpose (on (A.of_array (A.dtype a) back held))

(* [f] by the expansion then by nx.cpu, given the path's name and its placement
   of an array. *)
let each_path
    (f :
      'd.
      string -> ('v 's. bool -> ('v, 's) A.t -> ('v, 's, 'd) Nx.t) -> unit) =
  expanded (fun () ->
      f "expansion" (fun strided a -> transposed value strided a));
  f "nx.cpu" (fun strided a -> transposed value strided a)

(* Each element's bits, as the integers of its bytes. *)
let element_bits (type v s) (a : (v, s) A.t) =
  let w = Int.max 1 (D.bits (A.dtype a) / 8) in
  let b = bits_of a in
  Array.init (Array.length b / w) (fun i -> Array.sub b (i * w) w)

(* A shape of rank at least one and an axis of it, at times long enough for
   several levels of the expansion's recursion. *)
let along =
  let open Gen in
  let* s = shape in
  let s = if Array.length s = 0 then [| 3 |] else s in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ n = frequency [ (3, constant s.(axis)); (1, int_range 5 40) ] in
  let s = Array.copy s in
  s.(axis) <- n;
  (s, axis)

(* Covers what a scan's law needs: a long axis, a strided view. *)
let cover_scan s axis strided =
  cover "an axis of more than 4" (s.(axis) > 4);
  cover "transposed" (strided && Array.length s > 1)

type float_case =
  | Float_case :
      (float, 's) D.t * int array * int * bool * float array
      -> float_case

(* A float dtype and terms of it along an axis, spread over a range wide
   enough that a slice's early prefixes sit far below its later maximum, with
   infinities and NaNs, read through a transposed view or not. *)
let float_scan_case =
  let open Gen in
  let floats (type s) (dt : (float, s) D.t) wide =
    let element =
      frequency
        [
          (12, float_range (-.wide) wide);
          (6, float_range (-3.) 3.);
          (1, constant (-0.));
          (1, constant Float.neg_infinity);
          (1, constant Float.infinity);
          (1, map (fun k -> nan_with (Int64.of_int k)) (int_range 1 1000));
        ]
    in
    let* s, axis = along in
    let* strided = bool in
    let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) element in
    Float_case (dt, s, axis, strided, xs)
  in
  with_pp
    (fun ppf (Float_case (dt, s, axis, strided, _)) ->
      Format.fprintf ppf "%a %a along %d%s" D.pp dt pp_ints s axis
        (if strided then ", transposed" else ""))
    (one_of
       [
         floats D.Float64 1000.;
         floats D.Float32 1000.;
         floats D.Float16 1000.;
         floats D.Bfloat16 1000.;
         floats D.Float8_e4m3fn 100.;
         floats D.Float8_e5m2 1000.;
         floats D.Float4_e2m1fn 4.;
       ])

(* [v] rounded to [dt], as a store rounds. *)
let rounded (type s) (dt : (float, s) D.t) v =
  (A.to_array (A.of_array dt [| 1 |] [| v |])).(0)

(* nx_kernel.mli's bound, one for both paths: at [n] terms through at most
   [D] combinations, [D] the result's {!depth}, within
   [(7 D + 6 ln n) u + u |r|] in the accumulator; then rounded once, so
   between the roundings of the bound's ends. The reference, float64 over
   each prefix from its maximum, adds its own [2n] units. *)
let lse_bound u n r =
  (((7. *. depth n) +. (6. *. log (Float.of_int n))) *. u)
  +. (u *. Float.abs r)

let law_logsumexp_scan (Float_case (dt, s, axis, strided, xs)) =
  let u = match dt with D.Float64 -> ldexp 1. (-53) | _ -> ldexp 1. (-24) in
  let a = A.of_array dt s xs in
  cover_scan s axis strided;
  each_path (fun by place ->
      let x = place strided a in
      let ts = read x and tb = element_bits (host_array x) in
      let y = scan1 (Logsumexp (0, dt)) axis x in
      let got = read y and gb = element_bits (host_array y) in
      List.iter
        (fun ps ->
          let at = List.nth ps (List.length ps - 1) in
          let terms = List.map (fun p -> ts.(p)) ps in
          let g = got.(at) in
          match List.find_opt (fun p -> Float.is_nan ts.(p)) ps with
          | Some p ->
              cover "a NaN term" true;
              equal
                ~msg:(by ^ ": the first NaN, its bits")
                (array int) tb.(p) gb.(at)
          | None ->
              let m = List.fold_left Float.max Float.neg_infinity terms in
              if not (Float.is_finite m) then begin
                cover "an infinite maximum" true;
                equal ~msg:(by ^ ": an infinity") float_exact m g
              end
              else begin
                cover "finite" true;
                cover "a term far below the maximum"
                  (List.exists (fun t -> t < m -. 100.) terms);
                let n = Float.of_int (List.length terms) in
                let sum =
                  List.fold_left (fun a t -> a +. exp (t -. m)) 0. terms
                in
                let want = m +. log sum in
                let own = lse_bound u (List.length terms) want in
                let bound =
                  own +. (2. *. n *. epsilon_float *. (Float.abs want +. 1.))
                in
                let msg = Printf.sprintf "%s: %h against %h" by g want in
                at_least ~msg float_exact ~than:(rounded dt (want -. bound)) g;
                at_most ~msg float_exact ~than:(rounded dt (want +. bound)) g
              end)
        (prefixes s axis))

let test_logsumexp_scan_cases () =
  each_path (fun by place ->
      let lse (type s) (dt : (float, s) D.t) s xs =
        let x = place false (A.of_array dt s xs) in
        read (scan1 (Logsumexp (0, dt)) (Array.length s - 1) x)
      in
      let msg m = m ^ ", " ^ by in
      let inf = Float.infinity and ninf = Float.neg_infinity in
      equal ~msg:(msg "an empty axis") int 0
        (Array.length (lse D.Float64 [| 2; 0 |] [||]));
      equal
        ~msg:(msg "one term: itself, -0 as +0")
        (array int64)
        (Array.map bits [| 2.5; 0.; ninf |])
        (Array.map bits (lse D.Float64 [| 3; 1 |] [| 2.5; -0.; ninf |]));
      let ln2 = Float.log 2. in
      equal
        ~msg:(msg "prefixes far below the later maximum, float32")
        (array (float 1e-3))
        [| -1000.; -1000. +. ln2; 0.; 1. +. Float.log1p (exp (-1.)) |]
        (lse D.Float32 [| 4 |] [| -1000.; -1000.; 0.; 1. |]);
      equal
        ~msg:(msg "prefixes far below the later maximum, float64")
        (array (float 1e-12))
        [| -2000.; -2000. +. ln2; 0. |]
        (lse D.Float64 [| 3 |] [| -2000.; -2000.; 0. |]);
      let a = nan_with 7L in
      equal
        ~msg:(msg "a NaN: the first, its bits, from it on")
        (array int64)
        [| bits 1.; bits a; bits a; bits a |]
        (Array.map bits
           (lse D.Float64 [| 4 |] [| 1.; a; nan_with 9L; inf |]));
      equal ~msg:(msg "infinities") (array float_exact)
        [| ninf; ninf; 1.; inf; inf |]
        (lse D.Float64 [| 5 |] [| ninf; ninf; 1.; inf; ninf |]);
      let wide = A.of_array D.Float32 [| 2 |] [| 0.; ln2 |] in
      equal
        ~msg:(msg "float16 at float32, rounded once")
        (array float_exact)
        (A.to_array (cast D.Float16 wide))
        (lse D.Float16 [| 2 |] [| 0.; 0. |]))

type arg_case =
  | Arg_case : {
      dt : ('v, 's) D.t;
      key : 'v -> float;
      s : int array;
      axis : int;
      strided : bool;
      xs : 'v array;
    }
      -> arg_case

(* Terms of a dtype Arg takes along an axis, from few values so that ties are
   common, with signed zeros, infinities and NaNs where the dtype has them,
   read through a transposed view or not. [key] orders them as Max does. *)
let arg_scan_case =
  let open Gen in
  let make (type v s) (dt : (v, s) D.t) (key : v -> float) (element : v Gen.t)
      =
    let* s, axis = along in
    let* strided = bool in
    let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) element in
    Arg_case { dt; key; s; axis; strided; xs }
  in
  let float =
    frequency
      [
        (8, map Float.of_int (int_range (-3) 3));
        (1, constant (-0.));
        (1, constant Float.infinity);
        (1, constant Float.neg_infinity);
        (1, map (fun k -> nan_with (Int64.of_int k)) (int_range 1 1000));
      ]
  in
  let extremes small least most =
    frequency
      [ (6, map small (int_range (-3) 3)); (1, of_list [ least; most ]) ]
  in
  with_pp
    (fun ppf (Arg_case { dt; s; axis; strided; _ }) ->
      Format.fprintf ppf "%a %a along %d%s" D.pp dt pp_ints s axis
        (if strided then ", transposed" else ""))
    (one_of
       [
         make D.Float64 Fun.id float;
         make D.Float32 Fun.id float;
         make D.Float16 Fun.id float;
         make D.Int32 Int32.to_float
           (extremes Int32.of_int Int32.min_int Int32.max_int);
         make D.Int64 Int64.to_float
           (extremes Int64.of_int Int64.min_int Int64.max_int);
         make D.Uint8 Float.of_int (of_list [ 0; 1; 2; 255 ]);
         make D.Int4 Float.of_int (int_range (-8) 7);
         make D.Bool (fun b -> if b then 1. else 0.) bool;
       ])

(* Whether [t] is past [b] in Max's order, or Min's where [flip]: [-0] below
   [+0], and a NaN past every number either way. *)
let beats flip t b =
  let nan = Float.is_nan in
  if nan t || nan b then nan t && not (nan b)
  else
    let c =
      if t = b then Bool.compare (Float.sign_bit b) (Float.sign_bit t)
      else Float.compare t b
    in
    if flip then c < 0 else c > 0

let law_arg_scan (Arg_case { dt; key; s; axis; strided; xs }) =
  let a = A.of_array dt s xs in
  cover_scan s axis strided;
  each_path (fun by place ->
      let x = place strided a in
      let keys = Array.map key (read x) and tb = element_bits (host_array x) in
      List.iter
        (fun (name, e, flip) ->
          let by = by ^ " " ^ name in
          let v, p = scan1 (Arg (e, 0, dt)) axis x in
          let vb = element_bits (host_array v) and p = read p in
          List.iter
            (fun ps ->
              let at = List.nth ps (List.length ps - 1) in
              let _, want =
                first_extreme (beats flip) (List.map (fun i -> keys.(i)) ps)
              in
              let first = List.nth ps (Int64.to_int want) in
              cover "a tie"
                (List.exists (fun i -> i <> first && tb.(i) = tb.(first)) ps);
              equal ~msg:(by ^ ": position") int64 want p.(at);
              equal
                ~msg:(by ^ ": the extreme's bits")
                (array int) tb.(first) vb.(at))
            (prefixes s axis))
        [ ("max", (Max : Nx_kernel.extreme), false); ("min", Min, true) ])

let test_arg_scan_cases () =
  each_path (fun by place ->
      let msg m = m ^ ", " ^ by in
      let arg (type v s) e (dt : (v, s) D.t) s (xs : v array) =
        let x = place false (A.of_array dt s xs) in
        let v, q = scan1 (Arg (e, 0, dt)) (Array.length s - 1) x in
        (read v, read q)
      in
      let f64 e xs =
        let v, q = arg e D.Float64 [| Array.length xs |] xs in
        (Array.map bits v, q)
      in
      let found = pair (array int64) (array int64) in
      let v, q = arg Max D.Int32 [| 2; 0 |] [||] in
      equal ~msg:(msg "an empty axis") (pair int int) (0, 0)
        (Array.length v, Array.length q);
      equal
        ~msg:(msg "max of signed zeros: +0, the first one")
        found
        (Array.map bits [| -0.; 0.; 0.; 0. |], [| 0L; 1L; 1L; 1L |])
        (f64 Max [| -0.; 0.; -0.; 0. |]);
      equal
        ~msg:(msg "min of signed zeros: -0, the first one")
        found
        (Array.map bits [| 0.; -0.; -0.; -0. |], [| 0L; 1L; 1L; 1L |])
        (f64 Min [| 0.; -0.; 0.; -0. |]);
      let a = nan_with 7L and b = nan_with 9L in
      let first_nan =
        (Array.map bits [| 1.; a; a; a |], [| 0L; 1L; 1L; 1L |])
      in
      equal
        ~msg:(msg "a NaN is the max: the first, its bits")
        found first_nan
        (f64 Max [| 1.; a; 5.; b |]);
      equal
        ~msg:(msg "and the min")
        found first_nan
        (f64 Min [| 1.; a; -5.; b |]);
      equal ~msg:(msg "ties: the first") found
        (Array.map bits [| 2.; 7.; 7. |], [| 0L; 1L; 1L |])
        (f64 Max [| 2.; 7.; 7. |]);
      let grid = [| 1.; 9.; 3.; 9.; 0.; 2. |] in
      let v, q =
        scan1
          (Arg (Max, 0, Nx.float64))
          0
          (place false (A.of_array D.Float64 [| 2; 3 |] grid))
      in
      equal ~msg:(msg "along axis 0")
        (pair (array float_exact) (array int64))
        ([| 1.; 9.; 3.; 9.; 9.; 3. |], [| 0L; 0L; 0L; 1L; 0L; 0L |])
        (read v, read q);
      equal ~msg:(msg "int4")
        (pair (array int) (array int64))
        ([| 3; 3; 7; 7 |], [| 0L; 0L; 2L; 2L |])
        (arg Max D.Int4 [| 4 |] [| 3; -8; 7; 7 |]);
      equal ~msg:(msg "bool")
        (pair (array bool) (array int64))
        ([| false; true; true |], [| 0L; 1L; 1L |])
        (arg Max D.Bool [| 3 |] [| false; true; true |]);
      (* Complex numbers order by real part, then imaginary part. *)
      let z re im = { Complex.re; im } in
      let parts (v, q) =
        (Array.map (fun (c : Complex.t) -> (bits c.re, bits c.im)) v, q)
      in
      let zs = pair (array (pair int64 int64)) (array int64) in
      let want v q = parts (v, q) in
      equal ~msg:(msg "complex128 max, by parts, ties the first") zs
        (want [| z 1. 2.; z 1. 3.; z 1. 3.; z 1. 3. |] [| 0L; 1L; 1L; 1L |])
        (parts
           (arg Max D.Complex128 [| 4 |]
              [| z 1. 2.; z 1. 3.; z 1. 3.; z 0. 9. |]));
      equal ~msg:(msg "complex64 min: a -0 part below +0") zs
        (want [| z 1. 0.; z 1. (-0.); z 0. 5. |] [| 0L; 1L; 2L |])
        (parts
           (arg Min D.Complex64 [| 3 |] [| z 1. 0.; z 1. (-0.); z 0. 5. |]));
      let a = nan_with 7L in
      equal ~msg:(msg "complex128: a NaN part is the extreme") zs
        (want [| z 1. 0.; z 2. a; z 2. a |] [| 0L; 1L; 1L |])
        (parts (arg Max D.Complex128 [| 3 |] [| z 1. 0.; z 2. a; z 5. 0. |])))

(* Complex products, extremes and their first positions, reduced and scanned
   through Nx on the host, against folds of their terms. A number with a NaN
   part is a NaN. A product's terms are Gaussian integers whose parts are 0 or
   ± a power of two, so every order of multiplication gives the exact product
   but for the sign of a zero part: products compare by value, NaNs by bits. *)

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
let z_bits = Testable.make ~pp:pp_z ~equal:(fun a b -> zbits a = zbits b)

let z_value =
  Testable.make ~pp:pp_z ~equal:(fun (a : Complex.t) b ->
      if znan a || znan b then zbits a = zbits b
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
    let xs = A.to_array (A.of_array dt s xs) in
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
  let x = transposed value strided (A.of_array dt s xs) in
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
  let x = transposed value strided (A.of_array dt s xs) in
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
  let quiet = Int64.float_of_bits 0x7ff8_0000_0000_0000L in
  equal ~msg:"a NaN product of numbers is the quiet NaN in each part"
    (array z_bits) [| z quiet quiet |]
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

(* The prefix law: a scan's result at an index is a function of the terms up
   to it alone, for each reduction, by the expansion and by nx.cpu, on the
   host. Axes and prefixes are 1 or 2 long, a power of two or next to one,
   or next to nx.cpu's chunks of 4096, so that a network or a path chosen
   by the axis's length would give the prefix other bits. *)

let chunk = 4096

type prefix_case =
  | Prefix_case :
      (float, 's) D.t * int array * int * int * float array
      -> prefix_case

let pow2 k = k > 0 && k land (k - 1) = 0
let next_to_pow2 k = k > 2 && (pow2 (k - 1) || pow2 (k + 1))

(* Lengths of 1 or 2, of a power of two or next to one, short ones, and ones
   next to a chunk's end. *)
let length =
  let open Gen in
  let near_pow2 =
    let* j = int_range 1 12 in
    let+ d = of_list [ -1; 0; 1 ] in
    max 1 ((1 lsl j) + d)
  in
  frequency
    [
      (1, of_list [ 1; 2 ]);
      (3, near_pow2);
      (2, int_range 1 40);
      (1, int_range (chunk - 1) (chunk + 64));
    ]

(* nx.cpu's Logsumexp and Moments scans cut a slice into tiles of 256
   terms, which several threads share where slices are few. *)
let tile = 256

(* An axis's length and the slices beside it, for a Logsumexp or Moments
   scan: a {!length} beside up to two others; one slice of 24 to 32 tiles,
   two or three of 12 to 24, and 65 to 72 of 1 to 8, each around a multiple
   of a tile, the last run in units of 16 slices side by side, one
   partial. *)
let net_shape =
  let open Gen in
  let tiles lo hi =
    let* k = int_range lo hi in
    let+ d = int_range (-2) 2 in
    (k * tile) + d
  in
  frequency
    [
      (4, pair length (int_range 1 3));
      (1, pair (int_range 0 1) (int_range 1 3));
      (1, pair (tiles 24 32) (constant 1));
      (1, pair (tiles 12 24) (int_range 2 3));
      (1, pair (tiles 1 8) (int_range 65 72));
    ]

(* The covers of a Logsumexp or Moments scan of [slices] slices of [n]
   terms. *)
let cover_tiles n slices =
  cover "an axis around a multiple of a tile"
    (n >= tile - 2 && (n + 2) mod tile <= 4);
  cover "one slice past one tile, the last partial"
    (slices = 1 && n > tile && n mod tile <> 0);
  cover "one slice of 24 tiles or more" (slices = 1 && n >= (24 * tile) - 2);
  cover "two or three slices of 12 tiles or more"
    (slices > 1 && slices <= 3 && n >= (12 * tile) - 2);
  cover "more than 64 slices side by side, past a tile"
    (slices > 64 && n > tile)

(* Finite terms of either sign. *)
let finite_term =
  Gen.(frequency [ (15, float_range 0.5 2.); (4, float_range (-2.) (-0.5)) ])

(* Terms of either sign, with NaNs of two payloads, infinities and signed
   zeros one in twenty. *)
let prefix_term =
  let open Gen in
  let special =
    of_list
      [
        nan_with 5L; nan_with 9L; Float.infinity; Float.neg_infinity; -0.; 0.;
      ]
  in
  frequency [ (19, finite_term); (1, special) ]

(* Terms along an axis of a slice or two and the length of a prefix of it. *)
let prefix_case =
  let open Gen in
  let make (type s) (dt : (float, s) D.t) =
    let* n = length in
    let* k = frequency [ (2, map (min n) length); (1, int_range 1 n) ] in
    let* outer = int_range 1 2 in
    let* axis = int_range 0 1 in
    let s = if axis = 0 then [| n; outer |] else [| outer; n |] in
    let+ xs = array ~size:(constant (n * outer)) prefix_term in
    Prefix_case (dt, s, axis, k, xs)
  in
  with_pp
    (fun ppf (Prefix_case (dt, s, axis, k, _)) ->
      Format.fprintf ppf "%a %a along %d, a prefix of %d" D.pp dt pp_ints s
        axis k)
    (one_of [ make D.Float32; make D.Float64 ])

(* The bits of each result of the scan of [x] by [r] along [axis]. *)
let scanned (type s d) (r : Nx_kernel.reduction) (dt : (float, s) D.t) axis
    (x : (float, s, d) Nx.t) =
  match r with
  | Monoid m -> [ element_bits (host_array (scan1 (Monoid (m, 0, dt)) axis x)) ]
  | Logsumexp ->
      [ element_bits (host_array (scan1 (Logsumexp (0, dt)) axis x)) ]
  | Moments ->
      let m, v = scan1 (Moments (0, dt)) axis x in
      [ element_bits (host_array m); element_bits (host_array v) ]
  | Arg e ->
      let v, p = scan1 (Arg (e, 0, dt)) axis x in
      [ element_bits (host_array v); element_bits (host_array p) ]

(* Cases the generator draws too rarely for every run: lengths 1 and 2, a
   power of two and its neighbours on either side, specials. *)
let prefix_examples =
  let specials =
    [| 1.5; nan_with 5L; Float.infinity; -0.; Float.neg_infinity; 0.75 |]
  in
  let terms n = Array.init n (fun i -> specials.(i mod 6)) in
  let along dt n k = Prefix_case (dt, [| 1; n |], 1, k, terms n) in
  [
    along D.Float64 2 1;
    along D.Float32 1 1;
    along D.Float32 9 8;
    along D.Float64 8 7;
    along D.Float32 4100 4097;
    Prefix_case (D.Float64, [| 17; 2 |], 0, 16, terms 34);
  ]

let law_scan_prefix (Prefix_case (dt, s, axis, k, xs)) =
  let n = s.(axis) in
  let outer = Array.length xs / n in
  cover "an axis of 1 or 2" (n <= 2);
  cover "a prefix of 1 or 2" (k <= 2 && k < n);
  cover "an axis next to a power of two" (next_to_pow2 n);
  cover "a prefix next to a power of two" (next_to_pow2 k && k < n);
  cover "a prefix a power of two, the axis not" (pow2 k && not (pow2 n));
  cover "an axis a power of two, the prefix not" (pow2 n && not (pow2 k));
  cover "a prefix next to a chunk's end" (abs (k - chunk) <= 1 && k < n);
  cover "a term not finite or -0"
    (Array.exists
       (fun x -> (not (Float.is_finite x)) || 1. /. x = Float.neg_infinity)
       xs);
  cover "float64" (D.equal dt D.Float64);
  cover "along axis 0 of several" (axis = 0 && outer > 1);
  let at len o j = if axis = 0 then (j * outer) + o else (o * len) + j in
  let part = Array.copy s in
  part.(axis) <- k;
  (* [from.(p)] is the position in [xs] of the prefix's term at [p]. *)
  let from = Array.make (k * outer) 0 in
  for o = 0 to outer - 1 do
    for j = 0 to k - 1 do
      from.(at k o j) <- at n o j
    done
  done;
  let pxs = Array.map (fun p -> xs.(p)) from in
  each_path (fun by place ->
      let whole = place false (A.of_array dt s xs) in
      let prefix = place false (A.of_array dt part pxs) in
      List.iter
        (fun (name, r) ->
          let msg = Printf.sprintf "%s, %s" name by in
          List.iter2
            (fun w p ->
              let kept = Array.map (fun i -> w.(i)) from in
              equal ~msg (array (array int)) kept p)
            (scanned r dt axis whole) (scanned r dt axis prefix))
        [
          ("sum", Nx_kernel.Monoid Sum);
          ("prod", Monoid Prod);
          ("max", Monoid Max);
          ("min", Monoid Min);
          ("logsumexp", Logsumexp);
          ("moments", Moments);
          ("arg max", Arg Max);
          ("arg min", Arg Min);
        ])

(* Moments scans whose bits nx.cpu and the expansion both state: float32 or
   float64 terms near an offset, half the time with specials, of a
   {!net_shape}, held transposed at times. A term not finite sends nx.cpu's
   fast pass to its exact one, so finite terms alone check the fast
   pass's results over many tiles. *)
type bits_case =
  | Bits_case :
      (float, 's) D.t * int array * int * bool * float array
      -> bits_case

let bits_case =
  let open Gen in
  let make (type s) (dt : (float, s) D.t) =
    let* n, outer = net_shape in
    let* axis = int_range 0 1 in
    let* strided = bool in
    let* c = of_list [ 0.; 1e3; -1e5 ] in
    let s = if axis = 0 then [| n; outer |] else [| outer; n |] in
    let* finite = bool in
    let shift t = if Float.is_finite t then c +. t else t in
    let term = map shift (if finite then finite_term else prefix_term) in
    let+ xs = array ~size:(constant (n * outer)) term in
    Bits_case (dt, s, axis, strided, xs)
  in
  with_pp
    (fun ppf (Bits_case (dt, s, axis, strided, _)) ->
      Format.fprintf ppf "%a %a along %d%s" D.pp dt pp_ints s axis
        (if strided then ", transposed" else ""))
    (one_of [ make D.Float32; make D.Float64 ])

(* Cases the generator draws too rarely for every run. *)
let bits_examples =
  let specials =
    [| 1.5; nan_with 5L; Float.infinity; -0.; Float.neg_infinity |]
  in
  let terms n = Array.init n (fun i -> specials.(i mod 5)) in
  let finite n =
    Array.init n (fun i -> 1e3 +. Float.of_int (i * 7919 mod 251))
  in
  [
    Bits_case (D.Float64, [| 1; 2 |], 1, false, terms 2);
    Bits_case (D.Float32, [| 9; 2 |], 0, true, terms 18);
    Bits_case (D.Float64, [| 3; 17 |], 1, true, terms 51);
    Bits_case (D.Float32, [| 1; 1 |], 1, false, [| Float.infinity |]);
    Bits_case (D.Float32, [| 6401; 1 |], 0, false, terms 6401);
    Bits_case (D.Float64, [| 3; 3074 |], 1, true, terms 9222);
    Bits_case (D.Float32, [| 70; 514 |], 1, false, terms 35980);
    Bits_case (D.Float32, [| 3070; 3 |], 0, false, finite 9210);
    Bits_case (D.Float64, [| 2; 6146 |], 1, true, finite 12292);
  ]

let law_moments_bits (Bits_case (dt, s, axis, strided, xs)) =
  let n = s.(axis) in
  cover "an axis of 1 or 2" (n <= 2);
  cover_tiles n s.(1 - axis);
  cover "one to three slices of 12 tiles or more, every term finite"
    (s.(1 - axis) <= 3
    && n >= (12 * tile) - 2
    && Array.for_all Float.is_finite xs);
  cover "an axis next to a power of two" (next_to_pow2 n);
  cover "a term not finite or -0"
    (Array.exists
       (fun x -> (not (Float.is_finite x)) || 1. /. x = Float.neg_infinity)
       xs);
  cover "float64" (D.equal dt D.Float64);
  cover "transposed" strided;
  let a = A.of_array dt s xs in
  let scan on =
    let m, v = scan1 (Moments (0, dt)) axis (transposed on strided a) in
    (element_bits (host_array m), element_bits (host_array v))
  in
  equal ~msg:"nx.cpu's bits, the expansion's"
    (pair (array (array int)) (array (array int)))
    (expanded (fun () -> scan value))
    (scan value)

(* How a Logsumexp scan's operand is read: as held, transposed, reversed
   along every axis, or broadcast along axis [b] from an extent of 1. *)
type view = Plain | Transposed | Reversed | Broadcast of int

let view_name = function
  | Plain -> "plain"
  | Transposed -> "transposed"
  | Reversed -> "reversed"
  | Broadcast _ -> "broadcast"

(* Logsumexp scans whose bits nx.cpu and the expansion both state: every
   float dtype, terms from [-w] to [w] with [-0], infinities and NaNs at a
   drawn rate, along an axis of a {!length}, of [0] or [1], or around a
   multiple of 4096, read through a {!view}. The terms are the held
   array's, of the shape the view reads but of extent 1 along a broadcast
   axis. *)
type lse_case =
  | Lse_case :
      (float, 's) D.t * int array * int * view * float array
      -> lse_case

(* The shape a case's array is held at. *)
let held s = function
  | Broadcast b -> Array.mapi (fun a e -> if a = b then 1 else e) s
  | Plain | Transposed | Reversed -> s

let lse_case =
  let open Gen in
  let special =
    of_list
      [ nan_with 5L; nan_with 9L; Float.infinity; Float.neg_infinity; -0. ]
  in
  let make (type s) (dt : (float, s) D.t) =
    let* n, outer = net_shape in
    let* axis = int_range 0 1 in
    let s = if axis = 0 then [| n; outer |] else [| outer; n |] in
    let* view =
      of_list [ Plain; Transposed; Reversed; Broadcast 0; Broadcast 1 ]
    in
    let* w = of_list [ 3.; 100.; 1000. ] in
    let* rate = of_list [ 0; 0; 400; 20 ] in
    let term =
      if rate = 0 then float_range (-.w) w
      else frequency [ (rate - 1, float_range (-.w) w); (1, special) ]
    in
    let size = Array.fold_left ( * ) 1 (held s view) in
    let+ xs = array ~size:(constant size) term in
    Lse_case (dt, s, axis, view, xs)
  in
  with_pp
    (fun ppf (Lse_case (dt, s, axis, view, _)) ->
      Format.fprintf ppf "%a %a along %d, %s" D.pp dt pp_ints s axis
        (view_name view))
    (one_of
       [
         make D.Float32;
         make D.Float64;
         make D.Float16;
         make D.Bfloat16;
         make D.Float8_e4m3fn;
         make D.Float8_e5m2;
         make D.Float4_e2m1fn;
       ])

(* Cases the generator draws too rarely for every run: specials in turn
   along short axes, long axes with a NaN, an infinity or an outlier late,
   and many slices side by side past a tile. *)
let lse_examples =
  let specials =
    [| 1.5; nan_with 5L; Float.infinity; -0.; Float.neg_infinity; -3. |]
  in
  let cycle n = Array.init n (fun i -> specials.(i mod 6)) in
  let late n v =
    Array.init n (fun i ->
        if i = n - 5 then v else 50. *. sin (Float.of_int i))
  in
  [
    Lse_case (D.Float64, [| 2; 0 |], 1, Plain, [||]);
    Lse_case (D.Float32, [| 1; 1 |], 1, Plain, [| -0. |]);
    Lse_case (D.Float32, [| 12; 2 |], 0, Transposed, cycle 24);
    Lse_case (D.Float64, [| 3; 18 |], 1, Reversed, cycle 54);
    Lse_case (D.Bfloat16, [| 6401; 1 |], 0, Plain, late 6401 Float.infinity);
    Lse_case
      (D.Float32, [| 2; 6145 |], 1, Broadcast 0, late 6145 (nan_with 7L));
    Lse_case (D.Float64, [| 3071; 3 |], 0, Broadcast 1, late 3071 1e3);
    Lse_case (D.Float32, [| 514; 70 |], 0, Plain, late (514 * 70) 1e3);
  ]

let law_logsumexp_bits (Lse_case (dt, s, axis, view, xs)) =
  let n = s.(axis) in
  cover "an axis of 0 or 1" (n <= 1);
  cover "an axis next to a power of two" (next_to_pow2 n);
  cover_tiles n s.(1 - axis);
  cover "a NaN" (Array.exists Float.is_nan xs);
  cover "an infinity" (Array.exists (fun x -> Float.abs x = Float.infinity) xs);
  cover "a narrow float" (D.bits dt < 32);
  List.iter
    (fun v -> cover (view_name v) (view_name view = view_name v))
    [ Plain; Transposed; Reversed; Broadcast 0 ];
  let a = A.of_array dt (held s view) xs in
  let x () =
    match view with
    | Plain -> value a
    | Transposed -> transposed value true a
    | Reversed ->
        let k = Array.length xs in
        let back = Array.init k (fun i -> xs.(k - 1 - i)) in
        Nx.flip (value (A.of_array dt s back))
    | Broadcast _ -> Nx.broadcast_to s (value a)
  in
  let scan () =
    let y = scan1 (Logsumexp (0, dt)) axis (x ()) in
    element_bits (host_array y)
  in
  equal ~msg:"nx.cpu's bits, the expansion's" (array (array int))
    (expanded scan) (scan ())

let test_optional_rules () =
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
           layout = L.contiguous (Nx.shape x);
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
         layout = L.contiguous (Nx.shape x);
         axis = 1;
         prog = identity (Nx.dtype x);
         reduction = r;
         loads = [| Plain x |];
         carry = None;
       })

let row_axes = [ 1 ]
let row_axis = 1

(* The operations nx.mli's reductions section promises the first NaN
   quieted of: the monoids and their scans, the positions of the extremes
   and the extremes Arg returns, the mean, the variance, the standard
   deviation, logsumexp and the running moments. *)
let nan_ops =
  [
    {
      oname = "sum";
      result = Folded;
      floats = false;
      apart = true;
      run = (fun x -> read_bytes (Nx.sum ~axes:row_axes x));
    };
    {
      oname = "prod";
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.prod ~axes:row_axes x));
    };
    {
      oname = "max";
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.max ~axes:row_axes x));
    };
    {
      oname = "min";
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.min ~axes:row_axes x));
    };
    {
      oname = "cumsum";
      result = Running;
      floats = false;
      apart = true;
      run = (fun x -> read_bytes (Nx.cumsum ~axis:row_axis x));
    };
    {
      oname = "cumprod";
      result = Running;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.cumprod ~axis:row_axis x));
    };
    {
      oname = "cummax";
      result = Running;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.cummax ~axis:row_axis x));
    };
    {
      oname = "cummin";
      result = Running;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.cummin ~axis:row_axis x));
    };
    {
      oname = "argmax";
      result = Position;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.argmax ~axis:row_axis x));
    };
    {
      oname = "argmin";
      result = Position;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (Nx.argmin ~axis:row_axis x));
    };
    {
      oname = "Arg Max's extreme";
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (fst (along_rows x (Arg (Max, 0, Nx.dtype x)))));
    };
    {
      oname = "Arg Min's extreme";
      result = Folded;
      floats = false;
      apart = false;
      run = (fun x -> read_bytes (fst (along_rows x (Arg (Min, 0, Nx.dtype x)))));
    };
    {
      oname = "running Arg Max's extreme";
      result = Running;
      floats = false;
      apart = false;
      run =
        (fun x -> read_bytes (fst (running_rows x (Arg (Max, 0, Nx.dtype x)))));
    };
    {
      oname = "running Arg Min's extreme";
      result = Running;
      floats = false;
      apart = false;
      run =
        (fun x -> read_bytes (fst (running_rows x (Arg (Min, 0, Nx.dtype x)))));
    };
    {
      oname = "mean";
      result = Folded;
      floats = true;
      apart = false;
      run = (fun x -> on_floats (fun x -> Nx.mean ~axes:row_axes x) x);
    };
    {
      oname = "var";
      result = Folded;
      floats = true;
      apart = false;
      run = (fun x -> on_floats (fun x -> Nx.var ~axes:row_axes x) x);
    };
    {
      oname = "std";
      result = Folded;
      floats = true;
      apart = false;
      run = (fun x -> on_floats (fun x -> Nx.std ~axes:row_axes x) x);
    };
    {
      oname = "logsumexp";
      result = Folded;
      floats = true;
      apart = false;
      run = (fun x -> on_floats (fun x -> Nx.logsumexp ~axes:row_axes x) x);
    };
    {
      oname = "running mean";
      result = Running;
      floats = true;
      apart = false;
      run =
        (fun x ->
          on_floats (fun x -> fst (running_rows x (Moments (0, Nx.dtype x)))) x);
    };
    {
      oname = "running variance";
      result = Running;
      floats = true;
      apart = false;
      run =
        (fun x ->
          on_floats (fun x -> snd (running_rows x (Moments (0, Nx.dtype x)))) x);
    };
  ]

(* How a value computes: by the host's nx.cpu or by nx's expansions, on the
   host or over three devices split along the rows' axis. *)
type placing = {
  pname : string;
  go : 'v 's. nan_op -> ('v, 's) A.t -> string array;
}

let placings =
  [
    { pname = "nx.cpu"; go = (fun o a -> o.run (value a)) };
    {
      pname = "an expansion";
      go = (fun o a -> expanded (fun () -> o.run (value a)));
    };
    {
      pname = "nx.cpu over three devices";
      go = (fun o a -> o.run (Nx.place (Cpu3.split ~axis:row_axis) (value a)));
    };
    {
      pname = "an expansion over three devices";
      go =
        (fun o a ->
          expanded (fun () ->
              o.run (Nx.place (Cpu3.split ~axis:row_axis) (value a))));
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

(* [want], the bytes of each result the host gives, with each part that a
   NaN term reaches replaced: by the first NaN term's, quieted, where a
   complex number with a NaN part is a NaN, by its part's first NaN part,
   quieted, where the parts fold apart, and a position by the first NaN
   term's. *)
let nan_want o rows n (nan : bool array array) (quiet : string array)
    (want : string array) =
  let parts = Array.length nan.(0) in
  let pw = if n = 0 then 0 else String.length quiet.(0) / parts in
  let want = Array.copy want in
  let first r upto test =
    let rec go j = if j > upto then None else if test (r * n + j) then Some j else go (j + 1) in
    go 0
  in
  let any i = Array.exists Fun.id nan.(i) in
  let set_part k p src =
    let b = Bytes.of_string want.(k) in
    Bytes.blit_string src (p * pw) b (p * pw) pw;
    want.(k) <- Bytes.to_string b
  in
  let settle k r upto =
    match o.result with
    | Position -> (
        match first r upto any with
        | Some j -> want.(k) <- part_bytes 8 (Int64.of_int j)
        | None -> ())
    | Folded | Running ->
        if o.apart && parts = 2 then
          List.iter
            (fun p ->
              match first r upto (fun i -> nan.(i).(p)) with
              | Some j -> set_part k p quiet.((r * n) + j)
              | None -> ())
            [ 0; 1 ]
        else (
          match first r upto any with
          | Some j -> want.(k) <- quiet.((r * n) + j)
          | None -> ())
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
  let sign i =
    let e = raw.(i) in
    Char.code e.[String.length e - 1] land 0x80 <> 0
  in
  cover "a signalling NaN first in a row"
    (List.exists (fun i -> raw.(i) <> quiet.(i)) firsts);
  cover "a quiet NaN first in a row"
    (List.exists (fun i -> raw.(i) = quiet.(i)) firsts);
  cover "a NaN of the sign bit set first in a row" (List.exists sign firsts);
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
        let host = (List.hd placings).go o a in
        let want = nan_want o rows n nan quiet host in
        List.iter
          (fun p ->
            equal ~msg:(o.oname ^ ", " ^ p.pname) (array string) want
              (p.go o a))
          placings
      end)
    nan_ops

let optional_group =
  group "optional reductions"
    [
      prop "logsumexp is m + log Σ exp (x - m), its specials exact" float64_case
        (expanding law_logsumexp);
      prop "moments are within their stated bounds" moments_case
        (expanding law_moments);
      prop "float32 moments are within their bounds where (Σ d)² overflows"
        (overflow_case ((1 lsl 23) + 2) 0x1p59)
        (expanding (law_overflow D.Float32 0x1p-24));
      prop "float64 moments are within their bounds where (Σ d)² overflows"
        (overflow_case ((1 lsl 52) + 2) 0x1p507)
        (expanding (law_overflow D.Float64 0x1p-53));
      prop "moment scans are within their stated bounds" moments_scan_case
        law_moments_scan;
      prop "arg is the extreme and its first position" int32_case
        (expanding law_arg);
      test "arg's cases" (expanding test_arg_cases);
      test "moments' cases" (expanding test_moments_cases);
      test "moment scans' cases" test_moments_scan_cases;
      test "logsumexp's cases" (expanding test_logsumexp_cases);
      prop "logsumexp scans are within their stated bound, specials exact"
        float_scan_case law_logsumexp_scan;
      test "logsumexp scans' cases" test_logsumexp_scan_cases;
      prop "arg scans are each prefix's extreme and its first position"
        arg_scan_case law_arg_scan;
      test "arg scans' cases" test_arg_scan_cases;
      prop ~examples:prefix_examples
        "a scan's prefix is the prefix of its scan, bit for bit" prefix_case
        law_scan_prefix;
      prop ~examples:bits_examples
        "nx.cpu's Moments scans give the expansion's bits" bits_case
        law_moments_bits;
      prop ~examples:lse_examples
        "nx.cpu's Logsumexp scans give the expansion's bits" lse_case
        law_logsumexp_bits;
      test "their rules raise naming the caller" test_optional_rules;
    ]

(* An extreme along an empty axis is refused only where its result has an
   element: one with none is the empty result, on one device or cut over
   three along either axis. *)
let test_empty_extremes () =
  let empty s = value (A.of_array D.Float32 s [||]) in
  let shape x = L.shape (A.layout (host_array x)) in
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

(* A donated operand that another operation consumed is dead: a scan and an
   arg reduction without an axis, which read the elements in C order, name
   themselves. *)
let test_dead_named () =
  let dead () =
    let x = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
    ignore (Nx.neg (Nx.donate x));
    x
  in
  names "cumsum" (fun () -> Nx.cumsum (dead ()));
  names "argmax" (fun () -> Nx.argmax (dead ()));
  names "argmin" (fun () -> Nx.argmin ~keepdims:true (dead ()))

(* A scan with a carry continues each slice from the carry's element. Over a
   value cut along its axis, each window's results take the terms before it,
   so it has the bits of the scan on one device. *)
let test_carried_cut () =
  let s = [| 5; 3 |] in
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
  equal (array int32) want (A.to_array (host_array (carried cut on)))

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
      test "a dead operand names the function called" test_dead_named;
      test "an extreme of no term raises only where its result has an element"
        test_empty_extremes;
      test "mean divides the sum by the count" test_mean;
      prop "an integer mean is the exact mean truncated toward zero" int_case
        law_int_mean;
      test "integer means of extremes and empty axes" test_int_mean;
      test "a complex cumsum adds the parts, a narrow float's rounds once"
        test_scan_expansions;
      test "a carried scan over a value cut along its axis is the whole's"
        test_carried_cut;
      prop
        "a NaN result is the first NaN quieted on the host, through an \
         expansion and cut along the reduced axis"
        nan_case law_nans;
    ]

let () =
  exit (run "nx reduce" [ reductions_group; complex_group; optional_group ])
