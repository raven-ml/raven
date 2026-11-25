(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's gathers and scatters (nx_cpu.mli, Gathers and scatters): each
   element against a reference that reads the operands as numbers, at every
   dtype and through strided, reversed, permuted and broadcast views; a target's
   updates in C order, a narrow float's sum in float32 rounded once, any NaN for
   a NaN; scatters of many updates, which nx.cpu splits among threads, sorts or
   sums a slice at a time, against the same reference; one thread's bits; a
   float Add's element unmoved by the other elements, their updates and the
   other slabs; no byte read or written past an operand. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module Support = Nx_kernels_support
open Elements
open Index_cases

(* Gathers *)

let gather_expected c =
  let xs = shape_of c.x and is = shape_of c.idx in
  let es = elements c.x and ps = positions c.idx in
  let z = zero (dtype_of c.x) in
  Array.init (total is) (fun k ->
      let p = ps.(k) in
      if p < 0L || p >= Int64.of_int xs.(c.axis) then z
      else
        let i = index_of is k in
        i.(c.axis) <- Int64.to_int p;
        es.(flat xs i))

let gather_agrees (b : Support.backend) c =
  match gather_on b ~axis:c.axis c.idx c.x with
  | None -> failf "%s declined a gather" b.name
  | Some y -> equal (array string) (gather_expected c) (elements y)

(* The scatter reference *)

(* An array's elements in C order, as the numbers they hold: floats, narrow ones
   decoded, complex numbers, integers by their value, booleans. *)
type numbers =
  | Floats of float array
  | Complexes of Complex.t array
  | Ints of int64 array
  | Bools of bool array

let numbers (A.Any x) =
  let ints f a = Ints (Array.map f a) in
  let u32 v = Int64.logand (Int64.of_int32 v) 0xFFFF_FFFFL in
  let x = Support.host x in
  match A.dtype x with
  | D.Float64 -> Floats (A.to_array x)
  | D.Float32 -> Floats (A.to_array x)
  | D.Float16 -> Floats (A.to_array x)
  | D.Bfloat16 -> Floats (A.to_array x)
  | D.Float8_e4m3fn -> Floats (A.to_array x)
  | D.Float8_e5m2 -> Floats (A.to_array x)
  | D.Float4_e2m1fn -> Floats (A.to_array x)
  | D.Complex128 -> Complexes (A.to_array x)
  | D.Complex64 -> Complexes (A.to_array x)
  | D.Int64 -> Ints (A.to_array x)
  | D.Uint64 -> Ints (A.to_array x)
  | D.Int32 -> ints Int64.of_int32 (A.to_array x)
  | D.Uint32 -> ints u32 (A.to_array x)
  | D.Int16 -> ints Int64.of_int (A.to_array x)
  | D.Uint16 -> ints Int64.of_int (A.to_array x)
  | D.Int8 -> ints Int64.of_int (A.to_array x)
  | D.Uint8 -> ints Int64.of_int (A.to_array x)
  | D.Int4 -> ints Int64.of_int (A.to_array x)
  | D.Uint4 -> ints Int64.of_int (A.to_array x)
  | D.Bool -> Bools (A.to_array x)
  | D.Bit -> Bools (A.to_array x)

(* The bytes of element [k], in C order, of [x]. *)
let element_at (A.Any x) k =
  let s = Iarray.to_array (L.shape (A.layout x)) in
  let i = index_of s k in
  let one =
    Iarray.of_array
      (Array.map (fun a -> { M.start = a; count = 1; step = 1 }) i)
  in
  (elements (A.Any (Option.get (A.move (M.Slice one) x)))).(0)

(* The elements of [d] that store the floats [v], as a store into [d] rounds
   each, as bytes. *)
let stored_floats (D.Any dt) v =
  let shape = Iarray.of_array [| Array.length v |] in
  elements (A.Any (A.of_array dt shape (Array.map (D.of_float dt) v)))

let stored_complexes (D.Any dt) (v : Complex.t array) =
  let shape = Iarray.of_array [| Array.length v |] in
  match dt with
  | D.Complex64 -> elements (A.Any (A.of_array dt shape v))
  | D.Complex128 -> elements (A.Any (A.of_array dt shape v))
  | _ -> invalid_arg "stored_complexes"

let add32 x y = round32 (x +. y)

(* What a scatter computes, read from its numbers (nx_cpu.mli, Gathers and
   scatters): each target's element, then the updates that land on it in C order
   of their indices, which is their order along the axis. [Set] keeps the last.
   [Add] adds left to right: float64 rounding each sum, float32 each sum to
   float32, a narrow float's sum kept in float32 and stored into its dtype once,
   complex numbers part by part, integers modulo their width. [Max] and [Min]
   keep the greater or the lesser by Prog.Maximum's and Prog.Minimum's order, -0
   below +0, a complex number by its real part then its imaginary part, the
   first NaN that meets them; booleans are Or and And of truths. Beside the
   result: whether some target's finite sum in C order, as stored, differs from
   its sum in the reverse order, as stored, so that the result shows the order,
   and whether a narrow float's sum rounded into its dtype. *)
type reference = { want : want array; order_shows : bool; rounds : bool }

let reference c =
  let (D.Any dt as d) = dtype_of c.into in
  let ts = shape_of c.into and us = shape_of c.updates in
  let axis = c.saxis in
  let inner = total (Array.sub ts (axis + 1) (Array.length ts - axis - 1)) in
  let dx = ts.(axis) and du = us.(axis) in
  let ps = positions c.sidx in
  let n = total us and t = total ts in
  (* Each update's target, -1 outside the axis. *)
  let target =
    Array.init n (fun k ->
        let p = ps.(k) in
        if p < 0L || p >= Int64.of_int dx then -1
        else
          let r = k mod inner and outer = k / (inner * du) in
          (((outer * dx) + Int64.to_int p) * inner) + r)
  in
  let xs = elements c.into in
  let reached = Array.make t false and last = Array.make t (-1) in
  Array.iteri
    (fun k q ->
      if q >= 0 then begin
        reached.(q) <- true;
        last.(q) <- k
      end)
    target;
  (* Target [q]'s element, or update [k]'s, as bytes. *)
  let moved q k = if k < 0 then xs.(q) else element_at c.updates k in
  let result f =
    Array.init t (fun q -> if reached.(q) then f q else Bits xs.(q))
  in
  let fold_order s add order =
    let step k =
      let q = target.(k) in
      if q >= 0 then s.(q) <- add s.(q) k
    in
    if order = `C then
      for k = 0 to n - 1 do
        step k
      done
    else
      for k = n - 1 downto 0 do
        step k
      done;
    s
  in
  let any f =
    Array.exists Fun.id (Array.init t (fun q -> reached.(q) && f q))
  in
  let none = { want = [||]; order_shows = false; rounds = false } in
  (* The kept operand of each target, -1 for its element, by [keeps], which
     tells whether the first of two operands is kept. *)
  let extreme get =
    let kept = Array.make t (-1) in
    Array.iteri
      (fun k q -> if q >= 0 && not (get q kept.(q) k) then kept.(q) <- k)
      target;
    { none with want = result (fun q -> Value (moved q kept.(q))) }
  in
  match (c.combine, numbers c.into, numbers c.updates) with
  | Nx_kernel.Set, _, _ ->
      { none with want = result (fun q -> Bits (moved q last.(q))) }
  | Add, Floats x, Floats u ->
      let add =
        if is_narrow d || D.Any dt = D.Any D.Float32 then add32 else ( +. )
      in
      let sum order =
        fold_order (Array.copy x) (fun s k -> add s u.(k)) order
      in
      let s = sum `C and r = sum `Reverse in
      let es = stored_floats d s and er = stored_floats d r in
      {
        want = result (fun q -> Value es.(q));
        order_shows =
          any (fun q ->
              Float.is_finite s.(q) && Float.is_finite r.(q) && es.(q) <> er.(q));
        rounds =
          is_narrow d
          && any (fun q -> Float.is_finite s.(q) && decode d es.(q) <> s.(q));
      }
  | Add, Complexes x, Complexes u ->
      let add = if D.Any dt = D.Any D.Complex64 then add32 else ( +. ) in
      let sum order =
        fold_order (Array.copy x)
          (fun (s : Complex.t) k ->
            { re = add s.re u.(k).re; im = add s.im u.(k).im })
          order
      in
      let s = sum `C and r = sum `Reverse in
      let finite (z : Complex.t) =
        Float.is_finite z.re && Float.is_finite z.im
      in
      let es = stored_complexes d s and er = stored_complexes d r in
      {
        none with
        want = result (fun q -> Value es.(q));
        order_shows =
          any (fun q -> finite s.(q) && finite r.(q) && es.(q) <> er.(q));
      }
  | Add, Ints x, Ints u ->
      let s = fold_order (Array.copy x) (fun s k -> Int64.add s u.(k)) `C in
      {
        none with
        want = result (fun q -> Bits (int_bits d (String.length xs.(q)) s.(q)));
      }
  | (Max | Min), Bools x, Bools u ->
      let op = if c.combine = Max then ( || ) else ( && ) in
      let s = fold_order (Array.copy x) (fun s k -> op s u.(k)) `C in
      {
        none with
        want = result (fun q -> Bits (if s.(q) then "\001" else "\000"));
      }
  | (Max | Min), Floats x, Floats u ->
      extreme (fun q a k ->
          let va = if a < 0 then x.(q) else u.(a) and vb = u.(k) in
          keeps c.combine ~nan_a:(Float.is_nan va) ~nan_b:(Float.is_nan vb)
            (compare_floats va vb))
  | (Max | Min), Complexes x, Complexes u ->
      extreme (fun q a k ->
          let (va : Complex.t) = if a < 0 then x.(q) else u.(a) in
          let (vb : Complex.t) = u.(k) in
          let cmp = compare_floats va.re vb.re in
          keeps c.combine
            ~nan_a:(Float.is_nan va.re || Float.is_nan va.im)
            ~nan_b:(Float.is_nan vb.re || Float.is_nan vb.im)
            (if cmp = 0 then compare_floats va.im vb.im else cmp))
  | (Max | Min), Ints x, Ints u ->
      let compare =
        if D.is D.Signed dt then Int64.compare else Int64.unsigned_compare
      in
      extreme (fun q a k ->
          let va = if a < 0 then x.(q) else u.(a) in
          keeps c.combine ~nan_a:false ~nan_b:false (compare va u.(k)))
  | _ -> invalid_arg "reference: a combine its dtype does not take"

(* [y] holds [r.want], each element's bits where the reference moves them, else
   its value, any NaN for a NaN: the first five that differ shown. *)
let holds c r y =
  let d = dtype_of c.into in
  let got = elements y in
  let hex e =
    String.concat ""
      (List.map
         (fun ch -> Printf.sprintf "%02x" (Char.code ch))
         (List.of_seq (String.to_seq e)))
  in
  let wrong = ref [] in
  Array.iteri
    (fun t w ->
      let ok =
        match w with
        | Bits b -> b = got.(t)
        | Value v -> Testable.equal (same d) v got.(t)
      in
      if (not ok) && List.length !wrong < 5 then
        let (Bits e | Value e) = w in
        wrong :=
          Printf.sprintf "%d: %s, not %s" t (hex got.(t)) (hex e) :: !wrong)
    r.want;
  equal (list string) [] (List.rev !wrong)

let scatter_agrees (b : Support.backend) c =
  match scatter_on b c with
  | None -> failf "%s declined a scatter" b.name
  | Some y -> holds c (reference c) y

(* The exponent and fraction widths of [d]'s floats, or of its parts, where a
   NaN of them can lack the quiet bit, the top of its fraction: float8_e4m3fn's
   one NaN of each sign has every fraction bit set, and float4_e2m1fn has
   none. *)
let nan_fields (D.Any dt) =
  match dt with
  | D.Float64 | D.Complex128 -> Some (11, 52)
  | D.Float32 | D.Complex64 -> Some (8, 23)
  | D.Float16 -> Some (5, 10)
  | D.Bfloat16 -> Some (8, 7)
  | D.Float8_e5m2 -> Some (5, 2)
  | _ -> None

(* Whether a part of [e], an element of [d], is a signalling NaN. *)
let signals d e =
  match nan_fields d with
  | None -> false
  | Some (exp, frac) ->
      let w = (1 + exp + frac) / 8 in
      let part p =
        let v = ref 0L in
        for j = w - 1 downto 0 do
          v :=
            Int64.(logor (shift_left !v 8) (of_int (Char.code e.[(p * w) + j])))
        done;
        let ones = Int64.(sub (shift_left 1L exp) 1L) in
        let field = Int64.(logand (shift_right_logical !v frac) ones) in
        let fraction = Int64.(logand !v (sub (shift_left 1L frac) 1L)) in
        let quiet = Int64.(logand !v (shift_left 1L (frac - 1))) in
        field = ones && fraction <> 0L && quiet = 0L
      in
      List.exists part (List.init (String.length e / w) Fun.id)

(* A signalling NaN of [d] from its bits, in one part of a complex number or
   both: the exponent's ones, the quiet bit clear, a payload below it and a sign
   drawn. float8_e4m3fn's NaNs are quiet: it draws its one NaN of each sign. *)
let signalling (D.Any dt as d) rs =
  let part, parts =
    match dt with
    | D.Complex64 -> (D.Any D.Float32, 2)
    | D.Complex128 -> (D.Any D.Float64, 2)
    | _ -> (d, 1)
  in
  let nan () =
    match nan_fields part with
    | None -> String.make 1 (if Random.State.bool rs then '\x7F' else '\xFF')
    | Some (exp, frac) ->
        let below = Int64.(sub (shift_left 1L (frac - 1)) 1L) in
        let payload = Int64.succ (Random.State.int64 rs below) in
        let ones = Int64.(shift_left (sub (shift_left 1L exp) 1L) frac) in
        let sign =
          if Random.State.bool rs then Int64.shift_left 1L (exp + frac) else 0L
        in
        let v = Int64.(logor sign (logor ones payload)) in
        String.init
          ((1 + exp + frac) / 8)
          (fun j ->
            Char.chr
              (Int64.to_int (Int64.shift_right_logical v (8 * j)) land 0xFF))
  in
  if parts = 1 then nan ()
  else
    match Random.State.int rs 3 with
    | 0 -> nan () ^ element part rs
    | 1 -> element part rs ^ nan ()
    | _ -> nan () ^ nan ()

(* An element of [d], a signalling NaN one time in three. *)
let signals_often d rs =
  if Random.State.int rs 3 = 0 then signalling d rs else element d rs

let nan_dtypes =
  D.
    [
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float32;
      Any Float64;
      Any Complex64;
      Any Complex128;
    ]

let nan_scatters =
  Gen.with_pp pp_scatter
    (scatter_gen ~dtypes:nan_dtypes
       ~combines:Nx_kernel.[ Max; Min ]
       ~draw:signals_often ())

(* A Max or Min scatter is a NaN where a target's element or an update is; a
   target no update reaches keeps its bits, a signalling NaN included. *)
let law_nan_extremes (b : Support.backend) c =
  let d = dtype_of c.into in
  let signals = signals d in
  let xs = elements c.into and ls = lands c in
  let some f = Array.exists Fun.id (Array.mapi f xs) in
  cover "a signalling NaN reaches a target"
    (some (fun t x -> ls.(t) <> [] && List.exists signals (x :: ls.(t))));
  cover "a signalling NaN no update reaches"
    (some (fun t x -> ls.(t) = [] && signals x));
  scatter_agrees b c

(* Each operand as a C-contiguous copy of its view gives the same bits, any NaN
   for a NaN. *)
let law_add_views (b : Support.backend) c =
  let plain (A.Any x) = A.Any (A.copy x) in
  let c' =
    {
      c with
      into = plain c.into;
      sidx = plain c.sidx;
      updates = plain c.updates;
    }
  in
  cover "shared targets" (fst (sharing c));
  cover "through views" (c.sviews <> []);
  match (scatter_on b c, scatter_on b c') with
  | Some y, Some y' ->
      equal (array (same (dtype_of c.into))) (elements y) (elements y')
  | None, None -> ()
  | _ -> failf "%s declined a view or its copy" b.name

(* Values that show an Add's order *)

(* A number whose sums round: a sign, a fraction in [1, 2) and a binary exponent
   within [span] of 0; one time in [rare], where [rare] is above 0, a NaN, an
   infinity or -0 in its place. *)
let mixed ~rare ~span rs =
  if rare > 0 && Random.State.int rs rare = 0 then
    [| Float.nan; infinity; neg_infinity; -0. |].(Random.State.int rs 4)
  else
    let f = 1. +. Random.State.float rs 1. in
    ldexp
      (if Random.State.bool rs then f else -.f)
      (Random.State.int rs ((2 * span) + 1) - span)

(* [n] numbers of the float or complex [dt], their exponents within 6 for a
   narrow float, whose sums then stay finite and round in float32, within 20 for
   another. *)
let mixed_array (type v s) (dt : (v, s) D.t) ~rare rs n : v array =
  let span = if is_narrow (D.Any dt) then 6 else 20 in
  let x () = mixed ~rare ~span rs in
  match dt with
  | D.Complex64 -> Array.init n (fun _ -> { Complex.re = x (); im = x () })
  | D.Complex128 -> Array.init n (fun _ -> { Complex.re = x (); im = x () })
  | _ -> Array.init n (fun _ -> D.of_float dt (x ()))

let is_float (D.Any dt) = D.is D.Float dt || D.is D.Complex dt

(* An array of the float or complex [d] and shape [s] through the view [v], of
   [mixed] numbers, made from a typed array. *)
let mixed_operand ?(rare = 16384) (D.Any dt) rs s v =
  operand_with
    (fun m ->
      A.Any
        (A.of_array dt (Iarray.of_array m) (mixed_array dt ~rare rs (total m))))
    s v

(* An operand of [d] drawn from [rs]: of [mixed] numbers at a float or complex
   dtype, else of [element]'s. *)
let drawn_operand d rs s v =
  if is_float d then mixed_operand d rs s v
  else operand d s v (fun _ -> element d rs)

(* Positions of shape [s] through the view [v], each [next ()]. *)
let typed_positions s v next =
  operand_with
    (fun m ->
      A.Any
        (A.of_array D.Int64 (Iarray.of_array m)
           (Array.init (total m) (fun _ -> next ()))))
    s v

(* 4096 elements of [mixed] numbers of each float or complex dtype, made once a
   worker. *)
let pools = Hashtbl.create 8

let pool (D.Any dt as d) =
  match Hashtbl.find_opt pools d with
  | Some p -> p
  | None ->
      let rs = Random.State.make [| 7 |] in
      let p =
        elements
          (A.Any
             (A.of_array dt
                (Iarray.of_array [| 4096 |])
                (mixed_array dt ~rare:0 rs 4096)))
      in
      Hashtbl.add pools d p;
      p

(* An element of [d] for a drawn scatter: one of [pool]'s at a float or complex
   dtype, [element]'s one time in [rare], a NaN of a drawn payload or any bits
   among them; [element]'s at another dtype. *)
let mixing ~rare d rs =
  if (not (is_float d)) || Random.State.int rs rare = 0 then element d rs
  else
    let p = pool d in
    p.(Random.State.int rs (Array.length p))

(* [scatter_example]'s case of [mixed] numbers, made from typed arrays. *)
let mixed_example d combine ~ts ~axis ~m ?(broadcast = []) ?tv ?uv ps =
  let r = Array.length ts in
  let rs = Random.State.make [| 1 |] in
  let tv = Option.value tv ~default:(plain r)
  and uv = Option.value uv ~default:(plain r) in
  {
    combine;
    saxis = axis;
    into = mixed_operand d rs ts tv;
    sidx =
      typed_positions (along ts axis m)
        { (plain r) with broadcast }
        (positions_of ps);
    updates = mixed_operand d rs (along ts axis m) uv;
    sviews =
      List.map (( ^ ) "into ") (view_names tv)
      @ List.map (( ^ ) "updates ") (view_names uv);
  }

(* An Add of a bfloat16 or float16 [dt] along axis 0 whose stored sums show
   their order, as the targets of [mixed] numbers cannot: into's elements [lo],
   half the dtype's ulp at 1, so that 1 + lo is a tie the dtype rounds to 1; the
   updates 1 one time in 512, about once a target of 600 updates, else [tiny],
   whose float32 sum with 1 + lo rounds back to it alone and past the tie two
   together. A target of one 1 stores 1 or 1 + 2 lo by the count of [tiny]s its
   sum meets before the 1, which differs between the two orders at some
   targets. *)
let tipping_example (type s) (dt : (float, s) D.t) ~lo ~tiny ~ts ~m
    ?(broadcast = []) ?tv ps =
  let r = Array.length ts in
  let rs = Random.State.make [| 1 |] in
  let tv = Option.value tv ~default:(plain r) in
  let floats s v f =
    operand_with
      (fun m ->
        A.Any (A.of_array dt (Iarray.of_array m) (Array.init (total m) f)))
      s v
  in
  {
    combine = Nx_kernel.Add;
    saxis = 0;
    into = floats ts tv (fun _ -> lo);
    sidx =
      typed_positions (along ts 0 m)
        { (plain r) with broadcast }
        (positions_of ps);
    updates =
      floats (along ts 0 m) (plain r) (fun _ ->
          if Random.State.int rs 512 = 0 then 1. else tiny);
    sviews = List.map (( ^ ) "into ") (view_names tv);
  }

let narrow_dtypes = List.filter is_narrow all_dtypes

(* The narrow floats whose float32 sums of a few hundred numbers of exponents
   within 6 are exact in any order: for them an Add checks its single rounding
   into the dtype. *)
let exact_sums = D.[ Any Float8_e4m3fn; Any Float8_e5m2; Any Float4_e2m1fn ]

(* Scatters of few updates *)

(* Scatters of any dtype and combine, their elements [mixing]'s, and narrow
   floats' Adds of 64 to 400 updates a slice, past the few that a slice's
   insertion sort takes. *)
let small_scatters =
  Gen.with_pp pp_scatter
    (Gen.frequency
       [
         (3, scatter_gen ~draw:(mixing ~rare:8) ());
         ( 1,
           scatter_gen ~dtypes:narrow_dtypes
             ~combines:Nx_kernel.[ Add ]
             ~draw:(mixing ~rare:64)
             ~shapes:(Gen.of_list ~pp:pp_ints [ [| 3 |]; [| 2; 5 |]; [| 7 |] ])
             ~ms:(Gen.int_range 64 400) () );
       ])

let law_scatter_reference (b : Support.backend) c =
  let shared, dropped = sharing c in
  let narrow_add = c.combine = Nx_kernel.Add && is_narrow (dtype_of c.into) in
  let r = reference c in
  cover "shared targets" shared;
  cover "dropped updates" dropped;
  cover "a narrow float's Add" narrow_add;
  cover "a narrow float's Add of 64 updates or more a slice"
    (narrow_add && (shape_of c.updates).(c.saxis) >= 64);
  cover
    "a target's sum in C order differs from its sum in the reverse order, as \
     stored"
    r.order_shows;
  cover "a narrow float's sum in C order differs from its reverse, as stored"
    (narrow_add && r.order_shows);
  match scatter_on b c with
  | None -> failf "%s declined a scatter" b.name
  | Some y -> holds c r y

(* Values the doc states *)

(* A float16 target that 4096 updates of one reach: in float32 the sum is exact,
   where float16 steps would stop at 2048 (nx_cpu.mli: a narrow float's Add
   keeps its float32 sums). *)
let test_float16_sum (b : Support.backend) () =
  let n = 4096 in
  let into = A.of_array D.Float16 [| 1 |] [| 0. |] in
  let u = A.of_array D.Float16 [| n |] (Array.make n 1.) in
  let idx = A.of_array D.Int64 [| n |] (Array.make n 0L) in
  let dst = A.create Rig.host D.Float16 [| 1 |] in
  let module K = (val b.kernels) in
  match K.scatter Add ~unique:false ~axis:0 ~dst ~into idx u with
  | A.Done -> equal (array float_exact) [| 4096. |] (A.to_array dst)
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* A boolean is true where its byte is not zero: Max and Min of bytes 1, 2 and 0
   into 2, 0 and 255 are Or and And of truths, stored as 0 or 1. *)
let bool_extreme (b : Support.backend) (combine, want) =
  let module K = (val b.kernels) in
  let bools s = A.v D.Bool (L.contiguous [| 3 |]) (Rig.Buffer.of_string s) in
  let idx = A.of_array D.Int64 [| 3 |] [| 0L; 1L; 2L |] in
  let dst = A.create Rig.host D.Bool [| 3 |] in
  match
    K.scatter combine ~unique:false ~axis:0 ~dst ~into:(bools "\002\000\255")
      idx (bools "\001\002\000")
  with
  | A.Done -> equal (array string) want (elements (A.Any dst))
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Gathers of each regime *)

(* Sources of 70 000 elements, past the most one unit of a gather's job takes
   (64 Ki). *)
let large_shapes =
  Gen.of_list ~pp:pp_ints
    [ [| 70_000 |]; [| 280; 250 |]; [| 3; 24_000 |]; [| 24_000; 3 |] ]

(* Constructed cases, one per covered regime, so that every seed reaches each:
   positions in order from [ps], elements drawn from seed 1. *)
let gather_example d ~xs ~axis ~n ?(broadcast = []) ps =
  lazy begin
    let r = Array.length xs in
    let rs = Random.State.make [| 1 |] in
    let is = Array.mapi (fun i e -> if i = axis then n else e) xs in
    let iv = { (plain r) with broadcast } in
    {
      axis;
      idx = int64s is iv (positions_of ps);
      x = operand d xs (plain r) (fun _ -> element d rs);
      gviews = List.map (( ^ ) "idx ") (view_names iv);
    }
  end

let gather_examples =
  [
    (* Positions outside the axis, from below and above. *)
    gather_example (D.Any D.Float32) ~xs:[| 3; 4 |] ~axis:1 ~n:5
      [| 0L; -1L; 4L; Int64.max_int; 3L |];
    (* A row take: positions broadcast along the rows. *)
    gather_example (D.Any D.Int16) ~xs:[| 5; 3 |] ~axis:0 ~n:4 ~broadcast:[ 1 ]
      [| 4L; 0L; 2L; 4L |];
    (* Row takes of rows of 8 and 16 bytes, positions outside the axis among
       them. *)
    gather_example (D.Any D.Uint8) ~xs:[| 10; 8 |] ~axis:0 ~n:6 ~broadcast:[ 1 ]
      [| 3L; -1L; 9L; 10L; Int64.max_int; 0L |];
    gather_example (D.Any D.Float64) ~xs:[| 7; 2 |] ~axis:0 ~n:5
      ~broadcast:[ 1 ] [| 6L; 7L; 0L; -3L; 2L |];
    (* Packed sub-byte rows of 16 and 12 positions, the extremes among them,
       whose offsets in the source would overflow. *)
    gather_example (D.Any D.Bit) ~xs:[| 4; 16 |] ~axis:0 ~n:2
      [| Int64.max_int; 1L; Int64.min_int; 3L |];
    gather_example (D.Any D.Int4) ~xs:[| 3; 16 |] ~axis:1 ~n:12
      [| 5L; Int64.max_int; 15L; Int64.min_int; -1L |];
  ]

(* A bfloat16 target whose stored sum shows the order of its updates: 2^-8 then
   1, 1.5 2^-25 and 1.5 2^-25. In C order the float32 sum is 1 + 2^-8, which
   neither small update moves (each under half a float32 ulp), and stores as 1,
   a tie to even; in the reverse order it is 1 + 2^-8 + 2^-23, above the tie,
   and stores as 1 + 2^-7. *)
let shows_order_bf16 () =
  let tiny = 1.5 *. ldexp 1. (-25) in
  {
    combine = Nx_kernel.Add;
    saxis = 0;
    into = A.Any (A.of_array D.Bfloat16 [| 1 |] [| ldexp 1. (-8) |]);
    sidx = A.Any (A.of_array D.Int64 [| 3 |] [| 0L; 0L; 0L |]);
    updates = A.Any (A.of_array D.Bfloat16 [| 3 |] [| 1.; tiny; tiny |]);
    sviews = [];
  }

let scatter_examples =
  [
    (* Shared targets and dropped updates. *)
    lazy
      (mixed_example (D.Any D.Float32) Nx_kernel.Add ~ts:[| 4 |] ~axis:0 ~m:8
         [| 0L; 1L; 0L; -1L; 4L; 1L; 1L; Int64.max_int |]);
    (* A narrow float's sum. *)
    lazy
      (mixed_example (D.Any D.Float16) Nx_kernel.Add ~ts:[| 3 |] ~axis:0 ~m:6
         [| 0L; 0L; 2L; 5L; 0L; 1L |]);
    lazy (shows_order_bf16 ());
  ]

(* Takes of rows a take checks a run at a time (16) or a sub-byte one packs
   (12), either side of those lengths; of 300, two runs of a take's check, with
   a position outside the table and without; from an empty table, of no
   position, and of a sub-byte row of 12 over a table past the most a sub-byte
   row copies (1024). *)
let row_examples =
  let inside k = Array.init k Int64.of_int in
  [
    gather_example (D.Any D.Int16) ~xs:[| 2; 4 |] ~axis:1 ~n:15 (inside 4);
    gather_example (D.Any D.Float64) ~xs:[| 1; 3 |] ~axis:1 ~n:16 (inside 3);
    gather_example (D.Any D.Int4) ~xs:[| 2; 5 |] ~axis:1 ~n:11 (inside 5);
    gather_example (D.Any D.Uint4) ~xs:[| 1; 3 |] ~axis:1 ~n:12 (inside 3);
    gather_example (D.Any D.Bit) ~xs:[| 1; 1100 |] ~axis:1 ~n:12 (inside 1100);
    gather_example (D.Any D.Float32) ~xs:[| 2; 16 |] ~axis:1 ~n:300 (inside 16);
    gather_example (D.Any D.Uint8) ~xs:[| 1; 5 |] ~axis:1 ~n:300
      [| 0L; 1L; 2L; 3L; 4L; 5L; -1L |];
    gather_example (D.Any D.Float32) ~xs:[| 3; 0 |] ~axis:1 ~n:4 [| 0L; -1L |];
    gather_example (D.Any D.Int8) ~xs:[| 2; 5 |] ~axis:1 ~n:0 [| 0L |];
  ]

let gathers = Gen.with_pp pp_gather (gather_gen ())

(* A position inside an axis of [d], any for an empty one. *)
let inside rs d = if d = 0 then 0L else Int64.of_int (Random.State.int rs d)
let sub_byte_dtypes = List.filter (fun (D.Any dt) -> D.bits dt < 8) all_dtypes
let byte_dtypes = List.filter (fun (D.Any dt) -> D.bits dt >= 8) all_dtypes

(* Gathers through plain views in three cases of four, their positions all
   inside the axis in half the cases: the rows a take reads unchecked. *)
let half_inside ?dtypes ?axis ~shapes ~ns () =
  let gen = gather_gen ?dtypes ~shapes ?axis ~ns in
  Gen.with_pp pp_gather
    (Gen.frequency
       [
         (1, gen ());
         (1, gen ~plain:true ());
         (2, gen ~plain:true ~position:inside ());
       ])

let table_extent =
  Gen.frequency [ (1, Gen.constant 0); (4, Gen.int_range 1 16) ]

let table_positions =
  Gen.frequency [ (1, Gen.constant 0); (4, Gen.int_range 1 5000) ]

(* Takes from tables of a few entries, as a dictionary's codes read its words:
   at every byte-wide dtype, rows of up to thousands of positions, an empty
   table or no position one time in five. *)
let table_shapes =
  let open Gen in
  let* k = table_extent in
  let* rows = int_range 1 3 in
  of_list ~pp:pp_ints [ [| k |]; [| rows; k |]; [| k; rows |] ]

let table_gathers =
  half_inside ~dtypes:byte_dtypes ~shapes:table_shapes ~ns:table_positions ()

(* The same takes into sub-byte dtypes, which pack their elements. *)
let sub_byte_table_gathers =
  half_inside ~dtypes:sub_byte_dtypes ~shapes:table_shapes ~ns:table_positions
    ()

(* Rows of hundreds of elements along which the source steps, as
   take_along_axis's off the innermost axis are. *)
let along_gathers =
  let shapes =
    let open Gen in
    let* k = int_range 1 16 in
    let+ l = int_range 257 600 in
    [| k; l |]
  in
  half_inside ~axis:0 ~shapes ~ns:(Gen.int_range 1 4) ()

(* 70 000 to 100 000 positions inside a table, past the most one unit of a
   gather's job takes (64 Ki), which the job's threads share and read unchecked:
   of [dtypes], byte-wide or sub-byte, whose blocks share edge bytes. *)
let large_table_gathers dtypes =
  Gen.with_pp pp_gather
    (gather_gen ~dtypes
       ~shapes:(Gen.map (fun k -> [| k |]) (Gen.int_range 1 16))
       ~ns:(Gen.int_range 70_000 100_000)
       ~plain:true ~position:inside ())

(* Rows of plain positions from tables of [m] rows of [k] entries: of either
   length beside the shortest row a take checks a run at a time (16 elements) or
   a sub-byte one packs (12), and of 300, two runs of a take's check; tables of
   up to 5 entries or past the most a sub-byte row copies (1024); every position
   inside half the time. *)
let edge_gathers =
  let shapes =
    let open Gen in
    let* m = int_range 1 3 in
    let+ k = frequency [ (1, int_range 1 5); (1, int_range 1025 1200) ] in
    [| m; k |]
  in
  let gens dtypes ns =
    let gen = gather_gen ~dtypes ~shapes ~axis:1 ~ns:(Gen.of_list ns) in
    [ (1, gen ~plain:true ()); (1, gen ~plain:true ~position:inside ()) ]
  in
  let byte = gens byte_dtypes [ 15; 16; 300 ] in
  Gen.with_pp pp_gather
    (Gen.frequency (byte @ gens sub_byte_dtypes [ 11; 12; 300 ]))

(* Whether [c] reads through plain views, its positions all inside the axis,
   along rows of over 256: rows a take reads with no compare. *)
let unchecked c =
  let d = Int64.of_int (shape_of c.x).(c.axis) in
  let is = shape_of c.idx in
  is.(Array.length is - 1) > 256
  && c.gviews = []
  && Array.for_all (fun p -> p >= 0L && p < d) (positions c.idx)

(* Every regime a take reads by: any gather, takes from tables of few entries,
   along a stepped source and of rows beside a take's bounds. *)
let all_gathers =
  Gen.with_pp pp_gather
    (Gen.frequency
       [
         (1, gathers);
         (1, table_gathers);
         (1, sub_byte_table_gathers);
         (1, along_gathers);
         (2, edge_gathers);
       ])

let law_gather_reference (b : Support.backend) c =
  let xs = shape_of c.x and is = shape_of c.idx in
  let (A.Any x) = c.x in
  let sub = D.bits (A.dtype x) < 8 in
  let k = xs.(c.axis) and l = is.(c.axis) in
  let last = c.axis = Array.length xs - 1 in
  let d = Int64.of_int k in
  let outside = Array.exists (fun p -> p < 0L || p >= d) (positions c.idx) in
  cover "out of range" outside;
  cover "broadcast positions" (List.mem "idx broadcast" c.gviews);
  cover "an empty table" (k = 0);
  cover "no position" (l = 0);
  cover "long plain rows inside the axis" (unchecked c);
  cover "a byte-wide row of 15" ((not sub) && last && l = 15);
  cover "a byte-wide row of 16" ((not sub) && last && l = 16);
  cover "a sub-byte row of 11" (sub && last && l = 11);
  cover "a sub-byte row of 12" (sub && last && l = 12);
  cover "a sub-byte row of 12 or more read without a copy of the table"
    (sub && last && l >= 12 && (l < 2 * k || k > 1024));
  cover "a row of 300 holding a position outside the table"
    (last && l = 300 && outside);
  cover "a row of 300 inside the table" (last && l = 300 && not outside);
  gather_agrees b c

(* Gathers of sources and tables past one unit of the job's. *)
let large_gathers =
  Gen.with_pp pp_gather
    (Gen.frequency
       [
         (1, gather_gen ~shapes:large_shapes ());
         (1, large_table_gathers byte_dtypes);
         (1, large_table_gathers sub_byte_dtypes);
       ])

let large_gather_examples =
  let table d =
    gather_example (D.Any d) ~xs:[| 16 |] ~axis:0 ~n:80_000
      (Array.init 16 Int64.of_int)
  in
  [
    gather_example (D.Any D.Float32) ~xs:[| 3; 24_000 |] ~axis:1 ~n:24_000
      (Array.init 1009 (fun i -> Int64.of_int (i * 7919 mod 24_000)));
    table D.Uint8;
    table D.Int4;
  ]

let law_large_gather (b : Support.backend) c =
  let (A.Any x) = c.x in
  cover "a byte-wide element" (D.bits (A.dtype x) >= 8);
  cover "a sub-byte element" (D.bits (A.dtype x) < 8);
  gather_agrees b c

(* Scatters of many updates *)

(* Positions past the axis on both sides, cycling over 1003 targets. *)
let cycled = Array.init 1009 (fun i -> Int64.of_int ((i * 7 mod 1003) - 1))

(* Positions that miss targets 900 to 999 over the first 2^19 updates, a sorted
   block's least, then [cycled]'s. *)
let late =
  Array.init 600_000 (fun i ->
      if i < 1 lsl 19 then Int64.of_int (i * 7 mod 900) else cycled.(i mod 1009))

(* How a call runs: on the job's threads, on the calling thread alone, or on the
   job's threads sorting a float Add's updates by target. *)
type mode = Pool | One | Sorting

let mode_name = function
  | Pool -> "the job's threads"
  | One -> "one thread"
  | Sorting -> "threads that sort"

let in_mode mode f =
  match mode with
  | Pool -> f ()
  | One -> Nx_cpu.one_thread f
  | Sorting -> Nx_cpu.with_cache 0 f

let pp_mode ppf m = Format.pp_print_string ppf (mode_name m)
let mode_gen = Gen.of_list ~pp:pp_mode [ Pool; One; Sorting ]

(* Where a scatter of many updates comes from: an example, named for the regime
   it reaches, or a draw. *)
type source = Example of string | Drawn

(* A scatter of many updates, run as [mode] says. *)
type many = { source : source; mode : mode; case : scatter_case }

let pp_many ppf m =
  let name = match m.source with Example name -> name | Drawn -> "drawn" in
  Format.fprintf ppf "%s, on %s: %a" name (mode_name m.mode) pp_scatter m.case

let on ?(mode = Pool) name case =
  lazy { source = Example name; mode; case = Lazy.force case }

(* Updates of [mixed] numbers into an axis of a thousand, each regime's least
   count past its bound in index.c: several sorted blocks past FEWEST_BLOCK
   (2^19) for a narrow float's Add, targets first reached past it, and a Set,
   whose order sorts; targets each thread folds into its own once the updates
   make two shares (past OWN_SHARE, 2^16), for Max; a float Add sorted by target
   where the cache holds no target, and on one thread; one slice summed in row
   order; an embedding gradient's rows. *)
let regime_examples =
  let ex ?tv ?uv ?broadcast ?(ts = [| 1000 |]) d combine m ps =
    lazy (mixed_example (D.Any d) combine ~ts ~axis:0 ~m ?broadcast ?tv ?uv ps)
  in
  let reversed = { (plain 1) with reversed = Some 0 } in
  let stepped = { (plain 1) with stepped = true; reversed = Some 0 } in
  [
    on "a float16 Add of several sorted blocks"
      (lazy
        (tipping_example D.Float16 ~lo:(ldexp 1. (-11)) ~tiny:(ldexp 1. (-24))
           ~ts:[| 1000 |] ~m:600_000 cycled));
    on "a bfloat16 Add into reversed targets first reached past a block"
      (lazy
        (tipping_example D.Bfloat16 ~lo:(ldexp 1. (-8))
           ~tiny:(1.5 *. ldexp 1. (-25))
           ~ts:[| 1000 |] ~m:600_000 ~tv:reversed late));
    on "a float8_e4m3fn Add of stepped and reversed updates"
      (ex ~uv:stepped D.Float8_e4m3fn Nx_kernel.Add 600_000 cycled);
    on "a float32 Set of several sorted blocks"
      (ex D.Float32 Nx_kernel.Set 600_000 cycled);
    on "a float32 Max into each thread's own targets"
      (ex D.Float32 Nx_kernel.Max 70_000 cycled);
    on ~mode:Sorting "a float32 Add sorted by target"
      (ex D.Float32 Nx_kernel.Add 100_000 cycled);
    on ~mode:Sorting "a float16 Add whose cache holds no target"
      (ex D.Float16 Nx_kernel.Add 70_000 cycled);
    on ~mode:One "a float32 Add on one thread"
      (ex D.Float32 Nx_kernel.Add 70_000 cycled);
    on "a float64 slice summed in row order"
      (ex D.Float64 Nx_kernel.Add 70_000 cycled);
    on "a complex128 slice summed in row order"
      (ex D.Complex128 Nx_kernel.Add 70_000 cycled);
    on "a float8_e5m2 slice summed in row order"
      (ex D.Float8_e5m2 Nx_kernel.Add 70_000 cycled);
    on "an embedding gradient's float32 rows"
      (ex ~ts:[| 1000; 5 |] ~broadcast:[ 1 ]
         ~tv:{ (plain 2) with perm = [| 1; 0 |] }
         ~uv:{ (plain 2) with stepped = true; reversed = Some 0 }
         D.Float32 Nx_kernel.Add 70_000 cycled);
    on "a bfloat16 Add of rows"
      (lazy
        (tipping_example D.Bfloat16 ~lo:(ldexp 1. (-8))
           ~tiny:(1.5 *. ldexp 1. (-25))
           ~ts:[| 1000; 2 |] ~m:530_000 ~broadcast:[ 1 ] cycled));
  ]

(* Sums and extremes whose result is the same in any order of the updates, which
   threads fold into targets of their own once the updates make two shares (past
   OWN_SHARE, 2^16): integers', over shared targets and positions past the axis,
   a row of three slices per position, and updates through a stepped and
   reversed view; and booleans'. *)
let owned_examples =
  let ex ?(ts = [| 1000 |]) ?broadcast ?uv d combine =
    on
      (Printf.sprintf "%s %s into each thread's own targets" (D.name d)
         (combine_name combine))
      (lazy
        (scatter_example (D.Any d) combine ~ts ~axis:0 ~m:70_000 ?broadcast ?uv
           cycled))
  in
  [
    ex D.Int64 Nx_kernel.Add;
    ex D.Uint8 Nx_kernel.Add
      ~uv:{ (plain 1) with stepped = true; reversed = Some 0 };
    ex D.Int8 Nx_kernel.Max;
    ex D.Uint32 Nx_kernel.Min;
    ex D.Bool Nx_kernel.Max;
    ex D.Bool Nx_kernel.Min;
    ex D.Int16 Nx_kernel.Add ~ts:[| 1003; 3 |] ~broadcast:[ 1 ];
  ]
  @
  (* Booleans whose true bytes are 2, into targets 900 to 999 that no update
     reaches: an unreached element keeps its bits. *)
  let short = Array.init 900 (fun i -> Int64.of_int (i * 7 mod 900)) in
  let twos _ r = if Random.State.bool r then "\002" else "\000" in
  List.map
    (fun combine ->
      on
        (Printf.sprintf "bytes of 2 %s into unreached targets"
           (combine_name combine))
        (lazy
          (scatter_example (D.Any D.Bool) combine ~ts:[| 1000 |] ~axis:0
             ~m:70_000 ~draw:twos short)))
    Nx_kernel.[ Max; Min ]

let of_float (D.Any dt as d) x =
  match dt with
  | D.Float32 -> f32 x
  | D.Float64 -> f64 x
  | D.Complex64 -> f32 x ^ f32 x
  | D.Complex128 -> f64 x ^ f64 x
  | _ -> encode d x

(* Float Max and Min, which threads fold into targets of their own, at [d], of
   two shares past OWN_SHARE: updates of the zero below, numbers past it and,
   one in a few hundred each, the zero above and a signalling NaN, so that a
   target's extreme ties at zero or is a NaN; into targets that are signalling
   NaNs and -0 in turn, 900 to 999 of which no update reaches and keep their
   bits. *)
let extremes d combine =
  let rs = Random.State.make [| 3 |] in
  let short = Array.init 900 (fun i -> Int64.of_int (i * 7 mod 900)) in
  let m = 70_000 and toward = if combine = Nx_kernel.Max then -1. else 1. in
  let below = of_float d (-.toward *. 0.)
  and above = of_float d (toward *. 0.) in
  let past = of_float d toward and minus = of_float d (-0.) in
  let update _ =
    match Random.State.int rs 1000 with
    | 0 | 1 -> signalling d rs
    | 2 | 3 | 4 -> below
    | k when k < 300 -> above
    | _ -> past
  in
  let target k = if k mod 2 = 0 then signalling d rs else minus in
  {
    combine;
    saxis = 0;
    into = operand d [| 1000 |] (plain 1) target;
    sidx = int64s [| m |] (plain 1) (positions_of short);
    updates = operand d [| m |] (plain 1) update;
    sviews = [];
  }

let float_extremes =
  List.concat_map
    (fun (D.Any dt as d) ->
      List.map
        (fun c ->
          on
            (Printf.sprintf
               "%s %s of signalling NaNs and zeros into each thread's own \
                targets"
               (D.name dt) (combine_name c))
            (lazy (extremes d c)))
        Nx_kernel.[ Max; Min ])
    nan_dtypes

let wide_rows =
  lazy
    (let d = D.Any D.Complex128 and m = 65_536 and len = 1025 in
     let rs = Random.State.make [| 2 |] in
     let row = Array.init len (fun _ -> element d rs) in
     {
       combine = Nx_kernel.Set;
       saxis = 0;
       into = operand d [| 2; len |] (plain 2) (fun _ -> element d rs);
       sidx =
         int64s [| m; len |]
           { (plain 2) with broadcast = [ 1 ] }
           (positions_of [| 0L; 1L; 2L |]);
       updates =
         operand d [| m; len |]
           { (plain 2) with broadcast = [ 0 ] }
           (fun k -> row.(k));
       sviews = [ "updates broadcast" ];
     })

(* Many updates along the axis of few slices: threads split the targets. *)
let long_scatters =
  Gen.with_pp pp_scatter
    (scatter_gen ~draw:(mixing ~rare:4096)
       ~shapes:(Gen.of_list ~pp:pp_ints [ [| 1000 |]; [| 3; 7 |]; [| 5 |] ])
       ~ms:(Gen.int_range 70_000 100_000)
       ())

let large_scatters =
  Gen.with_pp pp_scatter
    (scatter_gen ~draw:(mixing ~rare:4096) ~shapes:large_shapes ())

(* A scatter of [d] along [axis] into [ts], [m] updates along it, its positions
   broadcast along [broadcast], into and the updates through [tv] and [uv], its
   elements [drawn_operand]'s from [rs]. *)
let shaped d combine (ts, axis, m, broadcast) ~tv ~uv rs =
  let r = Array.length ts in
  let us = along ts axis m in
  {
    combine;
    saxis = axis;
    into = drawn_operand d rs ts tv;
    sidx =
      typed_positions us
        { (plain r) with broadcast }
        (fun () -> position rs ts.(axis));
    updates = drawn_operand d rs us uv;
    sviews =
      List.map (( ^ ) "into ") (view_names tv)
      @ List.map (( ^ ) "updates ") (view_names uv);
  }

(* Embedding gradients: many updates along axis 0, their positions broadcast
   along the rows, into and the updates through drawn views. *)
let row_scatters =
  let open Gen in
  let* d = of_list ~pp:pp_dtype all_dtypes in
  let* combine = combine_gen d in
  let* ts =
    of_list ~pp:pp_ints
      [ [| 300; 5 |]; [| 50; 7 |]; [| 20; 2; 3 |]; [| 40; 3 |] ]
  in
  let r = Array.length ts in
  let* m = int_range 65_536 70_000 in
  let* tv = view_of r in
  let* uv = view_of r in
  let+ seed = int in
  shaped d combine
    (ts, 0, m, List.init (r - 1) succ)
    ~tv ~uv
    (Random.State.make [| seed |])

let row_scatters = Gen.with_pp pp_scatter row_scatters

(* Scatters of many updates against the reference, run as a drawn mode: large
   shapes, long axes of few slices and embedding gradients' rows. *)
let many_scatters =
  Gen.with_pp pp_many
    (let open Gen in
     let* case =
       frequency [ (3, large_scatters); (3, long_scatters); (3, row_scatters) ]
     in
     let+ mode = mode_gen in
     { source = Drawn; mode; case })

(* A scatter of many updates combines each target's updates in C order, whatever
   its regime: each example's own shows the order where its float32 sums are
   inexact, and its single rounding where they are exact. *)
let law_many (b : Support.backend) m =
  let c = m.case in
  let d = dtype_of c.into in
  let r = reference c in
  List.iter (fun k -> cover (mode_name k) (m.mode = k)) [ Pool; One; Sorting ];
  cover "hundreds of thousands of updates into one axis"
    ((shape_of c.updates).(c.saxis) > 500_000);
  cover "a NaN reaches a target"
    (Array.exists (function Value e -> nan_of d e | Bits _ -> false) r.want);
  (match m.source with
  | Example name when c.combine = Nx_kernel.Add && is_float d ->
      if List.mem d exact_sums then
        cover (name ^ ": a sum rounds once into its dtype") r.rounds
      else
        cover
          (name
         ^ ": a target's sum in C order differs from its sum in the reverse \
            order, as stored")
          r.order_shows
  | Example _ | Drawn -> ());
  match in_mode m.mode (fun () -> scatter_on b c) with
  | None -> failf "%s declined a scatter" b.name
  | Some y -> holds c r y

(* A narrow float's Adds: an axis of more updates than targets, one of fewer, 3
   slices of positions of their own, and rows of 600 slices, the positions
   broadcast along them. *)
let narrow_shapes =
  [
    ("more updates than targets", ([| 1000 |], 0, 70_000, []));
    ("more targets than updates", ([| 200_000 |], 0, 70_000, []));
    ("positions of each slice", ([| 3; 7 |], 1, 70_000, []));
    ("rows cut into pieces", ([| 40; 600 |], 0, 600, [ 1 ]));
  ]

let narrow_examples =
  List.mapi
    (fun i (name, ((ts, _, _, _) as s)) ->
      let d = List.nth narrow_dtypes (i mod List.length narrow_dtypes) in
      let r = Array.length ts in
      on
        ("a narrow float's Add of " ^ name)
        (lazy
          (shaped d Nx_kernel.Add s ~tv:(plain r) ~uv:(plain r)
             (Random.State.make [| 1 |]))))
    narrow_shapes
  @
  (* Positions crowding the first targets, and one outside the axis, at bfloat16
     and at float4_e2m1fn, whose elements share bytes. *)
  let crowding = Array.init 1009 (fun i -> Int64.of_int ((i * i mod 38) - 1)) in
  [
    on "a bfloat16 Add crowding the first targets"
      (lazy
        (mixed_example (D.Any D.Bfloat16) Nx_kernel.Add ~ts:[| 500; 3 |] ~axis:0
           ~m:70_000 crowding));
    on "a float4_e2m1fn Add crowding the first targets"
      (lazy
        (mixed_example (D.Any D.Float4_e2m1fn) Nx_kernel.Add ~ts:[| 500; 3 |]
           ~axis:0 ~m:70_000 crowding));
    (* 3 slices of 300 000 float4_e2m1fn updates into an axis of 1000. *)
    on "a float4_e2m1fn Add of three long slices"
      (lazy
        (mixed_example (D.Any D.Float4_e2m1fn) Nx_kernel.Add ~ts:[| 1000; 3 |]
           ~axis:0 ~m:300_000 cycled));
  ]

let narrow_adds =
  let open Gen in
  let* d = of_list ~pp:pp_dtype narrow_dtypes in
  let* _, ((ts, _, _, _) as s) = of_list narrow_shapes in
  let r = Array.length ts in
  let* tv = view_of r in
  let* uv = view_of r in
  let+ seed = int in
  shaped d Nx_kernel.Add s ~tv ~uv (Random.State.make [| seed |])

let narrow_adds = Gen.with_pp pp_scatter narrow_adds

(* Sub-byte scatters *)

(* Shapes whose neighbouring elements share bytes between threads: a short axis
   of many updates, rows of an odd count of slices, and slices along a middle
   axis. Each is [(into, axis, updates along it, broadcast axes of the
   positions)]. *)
let shared_bytes =
  [
    ([| 9 |], 0, 70_000, []);
    ([| 9; 1001 |], 0, 64, [ 1 ]);
    ([| 20; 11; 3 |], 1, 2000, []);
  ]

let sub_byte_examples =
  List.concat_map
    (fun (D.Any dt as d) ->
      List.concat_map
        (fun (ts, axis, m, broadcast) ->
          let ps = Array.init ts.(axis) Int64.of_int in
          List.map
            (fun combine ->
              on
                (Printf.sprintf "%s %s sharing bytes into %s" (D.name dt)
                   (combine_name combine)
                   (Format.asprintf "%a" pp_ints ts))
                (lazy (scatter_example d combine ~ts ~axis ~m ~broadcast ps)))
            (if takes_add d then Nx_kernel.[ Set; Add; Max; Min ]
             else Nx_kernel.[ Set; Max; Min ]))
        shared_bytes)
    sub_byte_dtypes

let sub_byte_scatters =
  let open Gen in
  let* d = of_list ~pp:pp_dtype sub_byte_dtypes in
  let* combine = combine_gen d in
  let* ((ts, _, _, _) as s) = of_list shared_bytes in
  let+ seed = int in
  let r = Array.length ts in
  shaped d combine s ~tv:(plain r) ~uv:(plain r) (Random.State.make [| seed |])

let sub_byte_scatters = Gen.with_pp pp_scatter sub_byte_scatters

(* Scatters of many updates, on the job's threads or sorting: narrow floats'
   Adds, embedding gradients' rows, sub-byte elements sharing bytes, and long
   axes of few slices. *)
let threaded_scatters =
  Gen.with_pp pp_many
    (let open Gen in
     let* case =
       frequency
         [
           (3, narrow_adds);
           (3, row_scatters);
           (3, sub_byte_scatters);
           (1, long_scatters);
         ]
     in
     let+ mode = of_list ~pp:pp_mode [ Pool; Sorting ] in
     { source = Drawn; mode; case })

let threaded_examples =
  narrow_examples
  @ List.map
      (fun e -> lazy { (Lazy.force e) with mode = Sorting })
      narrow_examples
  @ List.map
      (fun e ->
        lazy
          (let m = Lazy.force e in
           if m.mode = One then { m with mode = Pool } else m))
      regime_examples
  @ sub_byte_examples @ owned_examples @ float_extremes
  @ [ on "a complex128 Set of wide broadcast rows" wide_rows ]

(* A scatter of many updates has one thread's bits, on the job's threads or
   sorting, any NaN for a NaN. *)
let law_threads (b : Support.backend) m =
  let c = m.case in
  let d = dtype_of c.into in
  cover "a narrow float's Add" (c.combine = Nx_kernel.Add && is_narrow d);
  cover "a sub-byte dtype"
    (let (D.Any dt) = d in
     D.bits dt < 8);
  cover "positions broadcast along the rows"
    (let (A.Any p) = c.sidx in
     Iarray.exists (( = ) 0) (L.strides (A.layout p)));
  cover "through views" (c.sviews <> []);
  cover "more targets than updates"
    ((shape_of c.into).(c.saxis) > (shape_of c.updates).(c.saxis));
  cover (mode_name Sorting) (m.mode = Sorting);
  match
    ( in_mode m.mode (fun () -> scatter_on b c),
      Nx_cpu.one_thread (fun () -> scatter_on b c) )
  with
  | Some y, Some y' ->
      equal (array (same (dtype_of c.into))) (elements y) (elements y')
  | _ -> failf "%s declined a scatter" b.name

(* Parted scratch *)

(* The address space a forked child gives a parted float4_e2m1fn Add past its
   own: the Add's sort takes 32 bytes for each of its 4Mi updates, and 12 KiB on
   each thread. *)
let parted_cap = 512 lsl 20

(* A float4_e2m1fn array of shape [s], each element +0. *)
let f4_zeros s =
  let n = Array.fold_left ( * ) 1 s in
  A.of_array D.Float4_e2m1fn (Iarray.of_array s) (Array.make n 0.)

(* A float4_e2m1fn Add along axis 0 of [2Mi, 2], each update at a position of
   its own, so that threads part each column's targets, its elements sharing
   bytes, fits [parted_cap]: the child that runs it exits 0 where it is done, 1
   where it raised Out_of_memory. Linux only. *)
let test_parted_memory (b : Support.backend) () =
  if not (Sys.file_exists "/proc/self/statm") then
    skip ~reason:"limits the address space through Linux's /proc" ();
  let module K = (val b.kernels) in
  let rows = 1 lsl 21 in
  let s = [| rows; 2 |] in
  let ps =
    Bigarray.Array1.init Bigarray.int64 Bigarray.c_layout (2 * rows) (fun k ->
        Int64.of_int (k * 7919 mod rows))
  in
  let idx =
    A.of_bigarray D.Int64 (Bigarray.reshape (Bigarray.genarray_of_array1 ps) s)
  in
  let into = f4_zeros s and updates = f4_zeros s in
  let dst = A.create Rig.host D.Float4_e2m1fn (Iarray.of_array s) in
  let add () =
    K.scatter Nx_kernel.Add ~unique:false ~axis:0 ~dst ~into idx updates
  in
  (* A first Add starts the child's pool before the limit: a C library may map
     its threads' stacks anew in a forked child, 8 MiB a core, which is no
     scratch of the call's. glibc 2.43 and musl do. *)
  let run () =
    ignore (add ());
    if not (Support.cap_memory parted_cap) then 3
    else match add () with A.Done -> 0 | _ -> 2 | exception Out_of_memory -> 1
  in
  match Unix.fork () with
  | 0 -> Unix._exit (try run () with _ -> 4)
  | pid -> (
      match snd (Unix.waitpid [] pid) with
      | Unix.WEXITED 3 -> skip ~reason:"no address-space limit" ()
      | Unix.WEXITED code ->
          equal string "done"
            (match code with
            | 0 -> "done"
            | 1 -> "Out_of_memory"
            | 2 -> "not done"
            | _ -> "raised")
      | Unix.WSIGNALED n | Unix.WSTOPPED n ->
          failf "the child ended by signal %d" n)

(* [n] addends of the float or complex [d] whose sums round: each part a sign, a
   fraction and a binary exponent within 20 of 0, within 6 for a narrow float,
   which is cast to it; and one in 64 any element ({!element}), a NaN or an
   infinity among them. *)
let addends (D.Any dt as d) rs n =
  let part span =
    let f = 1. +. Random.State.float rs 1. in
    let f = if Random.State.bool rs then f else -.f in
    ldexp f (Random.State.int rs ((2 * span) + 1) - span)
  in
  let narrow () =
    let shape = Iarray.of_array [| n |] in
    let src = A.of_array D.Float32 shape (Array.init n (fun _ -> part 6)) in
    let dst = A.create Rig.host dt shape in
    match Support.map1 (module Nx_cpu) Cast ~dst src with
    | A.Done -> elements (A.Any dst)
    | r -> failf "a cast answered %a" Nx_array_support.pp_answer r
  in
  let es =
    match dt with
    | _ when n = 0 -> [||]
    | D.Float32 -> Array.init n (fun _ -> f32 (part 20))
    | D.Float64 -> Array.init n (fun _ -> f64 (part 20))
    | D.Complex64 -> Array.init n (fun _ -> f32 (part 20) ^ f32 (part 20))
    | D.Complex128 -> Array.init n (fun _ -> f64 (part 20) ^ f64 (part 20))
    | _ -> narrow ()
  in
  Array.map (fun e -> if Random.State.int rs 64 = 0 then element d rs else e) es

let bits64 p =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 p;
  Bytes.to_string b

(* [m1] trues and [m2] falses in a random order. *)
let shuffled rs m1 m2 =
  let a = Array.init (m1 + m2) (fun q -> q < m1) in
  for q = m1 + m2 - 1 downto 1 do
    let k = Random.State.int rs (q + 1) in
    let t = a.(q) in
    a.(q) <- a.(k);
    a.(k) <- t
  done;
  a

(* The elements, in C order, of arrays [a1] of shape [s1] and [a2] of shape
   [s2], equal but along [axis], joined along it: index [q] along it is the next
   of [a1]'s where [order.(q)], else the next of [a2]'s. *)
let merge axis order s1 a1 s2 a2 =
  let rank = Array.make (Array.length order) 0 in
  let n1 = ref 0 and n2 = ref 0 in
  Array.iteri
    (fun q first ->
      let n = if first then n1 else n2 in
      rank.(q) <- !n;
      incr n)
    order;
  let s = along s1 axis (Array.length order) in
  Array.init (total s) (fun k ->
      let i = index_of s k in
      let q = i.(axis) in
      i.(axis) <- rank.(q);
      if order.(q) then a1.(flat s1 i) else a2.(flat s2 i))

(* A float Add's shapes: [into]'s, the axis, the updates along it a whole call
   draws, and the axes along which its positions broadcast. *)
type regime = {
  name : string;
  ts : int array;
  axis : int;
  ms : int Gen.t;
  broadcast : int list;
}

(* Few updates into shapes of rank [ranks], extents 0 to 5. *)
let few_updates ranks =
  let open Gen in
  let* ts = array ~size:ranks (int_range 0 5) in
  let r = Array.length ts in
  let* axis = int_range 0 (r - 1) in
  let+ bc = array ~size:(constant r) bool in
  let broadcast =
    List.filter (fun a -> bc.(a) && a <> axis) (List.init r Fun.id)
  in
  { name = "few updates"; ts; axis; ms = int_range 0 24; broadcast }

(* The shapes of nx.cpu's threaded Adds, at the least counts past 2^16 updates,
   which nx.cpu's split takes (nx_cpu.mli): one slice, whose parts fall below
   2^16, and in an example parts of 2^16 each; slices of positions of their own;
   more targets than updates; an embedding gradient's rows; and rows cut into
   pieces. *)
let one_slice =
  {
    name = "one slice";
    ts = [| 1000 |];
    axis = 0;
    ms = Gen.int_range 65_536 70_000;
    broadcast = [];
  }

let own_slices =
  {
    name = "slices of positions of their own";
    ts = [| 3; 7 |];
    axis = 1;
    ms = Gen.int_range 65_536 70_000;
    broadcast = [];
  }

let more_targets =
  {
    name = "more targets than updates";
    ts = [| 200_000 |];
    axis = 0;
    ms = Gen.int_range 66_000 70_000;
    broadcast = [];
  }

let row_slices =
  {
    name = "rows of slices";
    ts = [| 300; 5 |];
    axis = 0;
    ms = Gen.int_range 65_536 70_000;
    broadcast = [ 1 ];
  }

let row_pieces =
  {
    name = "rows cut into pieces";
    ts = [| 40; 600 |];
    axis = 0;
    ms = Gen.constant 600;
    broadcast = [ 1 ];
  }

(* A whole Add and its two parts: each element of [into] is [owned] or not;
   [first]'s updates land only on owned elements, or outside the axis, and
   [second]'s on the others, and each part's [into] holds the whole's elements
   where its updates may land and others elsewhere. The whole call's updates are
   the parts', interleaved along the axis, each part's in its order. *)
type local_case = {
  regime : string;
  broadcast : int list;
  whole : scatter_case;
  first : scatter_case;
  second : scatter_case;
  owned : bool array;
}

let pp_local ppf c =
  Format.fprintf ppf "%s: %a, of parts %a and %a" c.regime pp_scatter c.whole
    pp_ints (shape_of c.first.updates) pp_ints
    (shape_of c.second.updates)

(* The local case of [d] in regime [g] whose parts take [m1] and [m2] updates,
   drawn from [seed]. Positions broadcast along [g.broadcast], so an element is
   owned as its index with those axes at 0 is. *)
let local_case d g ~m1 ~m2 seed =
  let rs = Random.State.make [| seed |] in
  let ts = g.ts and axis = g.axis in
  let r = Array.length ts and n = total ts and dx = g.ts.(g.axis) in
  let broadcast a = List.mem a g.broadcast in
  let red = Array.mapi (fun a e -> if broadcast a then 1 else e) ts in
  let mask = Array.init (total red) (fun _ -> Random.State.bool rs) in
  let owned =
    Array.init n (fun e ->
        let i = index_of ts e in
        let i = Array.mapi (fun a x -> if broadcast a then 0 else x) i in
        mask.(flat red i))
  in
  (* For each line along the axis, by its index at 0 along it, the targets whose
     ownership is [own]. *)
  let targets own =
    Array.init (total red) (fun k ->
        let i = index_of red k in
        if i.(axis) <> 0 then [||]
        else
          Array.of_list
            (List.filter
               (fun p ->
                 i.(axis) <- p;
                 mask.(flat red i) = own)
               (List.init dx Fun.id)))
  in
  let dropped = Int64.[| -1L; of_int dx; of_int (dx + 3); min_int; max_int |] in
  let positions own m =
    let lines = targets own and ms = along red axis m in
    Array.init (total ms) (fun k ->
        let i = index_of ms k in
        i.(axis) <- 0;
        let c = if dx = 0 then [||] else lines.(flat red i) in
        if Array.length c = 0 || Random.State.int rs 16 = 0 then
          dropped.(Random.State.int rs (Array.length dropped))
        else Int64.of_int c.(Random.State.int rs (Array.length c)))
  in
  let p1 = positions true m1 and p2 = positions false m2 in
  let u1 = addends d rs (total (along ts axis m1)) in
  let u2 = addends d rs (total (along ts axis m2)) in
  let xs = addends d rs n and others = addends d rs n in
  let order = shuffled rs m1 m2 in
  let pw = merge axis order (along red axis m1) p1 (along red axis m2) p2 in
  let uw = merge axis order (along ts axis m1) u1 (along ts axis m2) u2 in
  let iv = { (plain r) with broadcast = g.broadcast } in
  let case into ps us m =
    {
      combine = Nx_kernel.Add;
      saxis = axis;
      into = of_elements d ts into;
      sidx =
        operand (D.Any D.Int64) (along ts axis m) iv (fun k -> bits64 ps.(k));
      updates = of_elements d (along ts axis m) us;
      sviews = (if g.broadcast = [] then [] else [ "idx broadcast" ]);
    }
  in
  let keep own =
    Array.init n (fun e -> if owned.(e) = own then xs.(e) else others.(e))
  in
  {
    regime = g.name;
    broadcast = g.broadcast;
    whole = case xs pw uw (m1 + m2);
    first = case (keep true) p1 u1 m1;
    second = case (keep false) p2 u2 m2;
    owned;
  }

let local_gen regimes =
  let open Gen in
  let* d = of_list ~pp:pp_dtype float_dtypes in
  let* g = regimes in
  let* m = g.ms in
  let* m1 = int_range 0 m in
  let+ seed = int in
  local_case d g ~m1 ~m2:(m - m1) seed

let pp_modes ppf (w, p) =
  Format.fprintf ppf "the whole on %a, the parts on %a" pp_mode w pp_mode p

let pp_local_add ppf (c, m) = Format.fprintf ppf "%a; %a" pp_local c pp_modes m

let local_adds =
  Gen.with_pp pp_local_add
    (Gen.pair
       (local_gen
          (Gen.frequency
             [
               (3, few_updates (Gen.int_range 1 3));
               ( 2,
                 Gen.of_list
                   [
                     one_slice; own_slices; more_targets; row_slices; row_pieces;
                   ] );
             ]))
       (Gen.pair mode_gen mode_gen))

(* One case of each regime and of the parts at 2^16 updates or more, at a float,
   narrow floats and complex numbers, so that every label is met on every run
   whatever the draws. *)
let local_examples =
  let few =
    {
      name = "few updates";
      ts = [| 3; 4 |];
      axis = 1;
      ms = Gen.constant 8;
      broadcast = [ 0 ];
    }
  in
  List.map
    (fun (d, g, m1, m2, modes) -> lazy (local_case d g ~m1 ~m2 1, modes))
    D.
      [
        (Any Float32, one_slice, 65_536, 65_536, (Pool, One));
        (Any Float16, one_slice, 60_000, 10_000, (One, Pool));
        (Any Complex64, own_slices, 40_000, 40_000, (Sorting, One));
        (Any Float64, more_targets, 60_000, 8_000, (Pool, Sorting));
        (Any Bfloat16, row_slices, 30_000, 40_000, (Pool, One));
        (Any Float8_e4m3fn, row_pieces, 300, 300, (Sorting, Pool));
        (Any Float32, few, 5, 3, (One, Pool));
      ]

(* A float Add's element is the Add of that element of into and of the updates
   that land on it, in their order: the other elements of into, and updates that
   land on other elements, change none of its bits, whatever the threads, any
   NaN for a NaN. *)
let law_own_updates (b : Support.backend) (c, (mw, mp)) =
  let d = dtype_of c.whole.into and axis = c.whole.saxis in
  let m1 = (shape_of c.first.updates).(axis) in
  let m2 = (shape_of c.second.updates).(axis) in
  let at_least n = n >= 1 lsl 16 in
  List.iter
    (fun g -> cover g (c.regime = g))
    [
      "few updates";
      one_slice.name;
      own_slices.name;
      more_targets.name;
      row_slices.name;
      row_pieces.name;
    ];
  cover "a narrow float" (is_narrow d);
  cover "complex numbers"
    (let (D.Any dt) = d in
     D.is D.Complex dt);
  cover "the whole of 2^16 updates or more, a part of fewer"
    (at_least (m1 + m2) && not (at_least m1 && at_least m2));
  cover "parts of 2^16 updates or more" (at_least m1 && at_least m2);
  cover "the whole and the parts on other threads" (mw <> mp);
  let on mode c = in_mode mode (fun () -> Option.get (scatter_on b c)) in
  let y = on mw c.whole in
  let y1 = elements (on mp c.first) and y2 = elements (on mp c.second) in
  let want = Array.mapi (fun e own -> if own then y1.(e) else y2.(e)) c.owned in
  matching
    (fun d -> Testable.equal (same d))
    [| of_elements d (shape_of c.whole.into) want |]
    [| y |]

(* Cuts of an axis other than a case's own: the axis, the first index and the
   count. *)
type slab = { across : int; start : int; count : int }

let slab_gen =
  let open Gen in
  let* c =
    local_gen
      (frequency
         [
           (3, few_updates (int_range 2 3));
           (2, of_list [ own_slices; row_slices; row_pieces ]);
         ])
  in
  let ts = shape_of c.whole.into in
  let axes =
    List.filter
      (fun a -> a <> c.whole.saxis)
      (List.init (Array.length ts) Fun.id)
  in
  let* across = of_list axes in
  let* start = int_range 0 ts.(across) in
  let+ count = int_range 0 (ts.(across) - start) in
  (c, { across; start; count })

let pp_slab ppf (c, s) =
  Format.fprintf ppf "%a; %d of axis %d from %d" pp_local c s.count s.across
    s.start

let pp_slab_add ppf (c, m) = Format.fprintf ppf "%a; %a" pp_slab c pp_modes m

let slabs =
  Gen.with_pp pp_slab_add (Gen.pair slab_gen (Gen.pair mode_gen mode_gen))

let slab_examples =
  List.map
    (fun (d, g, m, s, modes) -> lazy ((local_case d g ~m1:m ~m2:0 1, s), modes))
    D.
      [
        ( Any Float32,
          own_slices,
          65_536,
          { across = 0; start = 1; count = 1 },
          (Pool, One) );
        ( Any Bfloat16,
          row_slices,
          66_000,
          { across = 1; start = 2; count = 2 },
          (One, Pool) );
        ( Any Complex128,
          row_slices,
          66_000,
          { across = 1; start = 0; count = 4 },
          (Pool, Sorting) );
        ( Any Float4_e2m1fn,
          row_pieces,
          600,
          { across = 1; start = 100; count = 250 },
          (Sorting, Pool) );
      ]

(* A float Add over a slab of an axis other than its own, into, the positions
   and the updates each cut to the slab, is that slab of the whole Add, whatever
   the threads, any NaN for a NaN. *)
let law_slab (b : Support.backend) ((c, s), (mw, mp)) =
  let w = c.whole in
  let n = (shape_of w.into).(s.across) in
  let cut (A.Any x) =
    let rs =
      Iarray.mapi
        (fun a e ->
          if a = s.across then { M.start = s.start; count = s.count; step = 1 }
          else { M.start = 0; count = e; step = 1 })
        (L.shape (A.layout x))
    in
    A.Any (Option.get (A.move (M.Slice rs) x))
  in
  List.iter
    (fun g -> cover g (c.regime = g))
    [ "few updates"; own_slices.name; row_slices.name; row_pieces.name ];
  cover "a narrow float" (is_narrow (dtype_of w.into));
  cover "a slab of one" (s.count = 1);
  cover "a slab of all but one" (n > 1 && s.count = n - 1);
  cover "positions broadcast along the slab's axis"
    (List.mem s.across c.broadcast);
  cover "the whole and the slab on other threads" (mw <> mp);
  let part =
    { w with into = cut w.into; sidx = cut w.sidx; updates = cut w.updates }
  in
  let y = in_mode mw (fun () -> Option.get (scatter_on b w)) in
  let y' = in_mode mp (fun () -> Option.get (scatter_on b part)) in
  matching (fun d -> Testable.equal (same d)) [| cut y |] [| y' |]

(* Guard pages *)

let covers_views ~after views =
  cover "a stepped operand at the page before"
    ((not after) && List.exists (String.ends_with ~suffix:"stepped") views);
  cover "a reversed operand at the page after"
    (after && List.exists (String.ends_with ~suffix:"reversed") views);
  cover "broadcast positions" (List.mem "idx broadcast" views)

(* The same result with the operand and positions, through their views, and the
   destination against an inaccessible page. *)
let law_gather_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  let r = Array.length (shape_of c.x) in
  cover "along the innermost axis" (c.axis = r - 1);
  cover "along an outer axis" (c.axis < r - 1);
  covers_views ~after:(side = Support.After) c.gviews;
  let call xs ds =
    let (A.Any x) = xs.(1) in
    let dst = A.expect (A.dtype x) ds.(0) in
    K.gather ~axis:c.axis ~dst (A.expect D.Int64 xs.(0)) x
  in
  equal answer A.Done
    (same_guarded side ~call [| c.idx; c.x |]
       [| (dtype_of c.x, shape_of c.idx) |])

let law_scatter_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  let r = Array.length (shape_of c.into) in
  List.iter
    (fun k -> cover (combine_name k) (c.combine = k))
    Nx_kernel.[ Set; Add; Max; Min ];
  cover "along the innermost axis" (c.saxis = r - 1);
  cover "along an outer axis" (c.saxis < r - 1);
  covers_views ~after:(side = Support.After) c.sviews;
  let call xs ds =
    let (A.Any into) = xs.(0) in
    let dst = A.expect (A.dtype into) ds.(0) in
    let idx = A.expect D.Int64 xs.(1) in
    let updates = A.expect (A.dtype into) xs.(2) in
    K.scatter c.combine ~unique:false ~axis:c.saxis ~dst ~into idx updates
  in
  equal answer A.Done
    (same_guarded ~any_nan:true side ~call
       [| c.into; c.sidx; c.updates |]
       [| (dtype_of c.into, shape_of c.into) |])

(* The suite *)

(* Cases made when a law first runs them: the suite's declaration runs in
   windtrap's parent process too, which runs no law, and the examples took 4.5 s
   to make there. *)
let deferred pp gen =
  Gen.with_pp (fun ppf c -> pp ppf (Lazy.force c)) (Gen.map Lazy.from_val gen)

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  let forced f c = b.around (fun () -> f (Lazy.force c)) in
  group "nx.cpu"
    [
      group "gather"
        [
          prop ~count:500
            ~examples:(gather_examples @ row_examples)
            "reads each position, zero outside the axis"
            (deferred pp_gather all_gathers)
            (forced (law_gather_reference b));
          prop ~count:6 ~examples:large_gather_examples
            "on the job's threads reads each position"
            (deferred pp_gather large_gathers)
            (forced (law_large_gather b));
          prop ~count:300
            "reads and writes no byte past its operands and dst, whatever \
             pages lie beside them"
            (Gen.pair (gather_gen ~shapes:tail_shapes ()) sides)
            (run (law_gather_guards b));
        ];
      group "scatter"
        [
          prop ~examples:scatter_examples
            "combines each target's updates in C order"
            (deferred pp_scatter small_scatters)
            (forced (law_scatter_reference b));
          prop "Max and Min give a NaN where one reaches a target" nan_scatters
            (run (law_nan_extremes b));
          prop "a float Add has the bits of its operands' C-contiguous copies"
            float_adds
            (run (law_add_views b));
          cases
            ~id:(fun (c, _) -> combine_name c)
            "Max and Min of booleans are Or and And of truths, stored as 0 or 1"
            [
              (Nx_kernel.Max, [| "\001"; "\001"; "\001" |]);
              (Nx_kernel.Min, [| "\001"; "\000"; "\000" |]);
            ]
            (run (bool_extreme b));
          test "a float16 Add sums in float32 and rounds once" (fun () ->
              b.around (test_float16_sum b));
          prop ~count:300
            "reads and writes no byte past its operands and dst, whatever \
             pages lie beside them"
            (Gen.pair (scatter_gen ~shapes:tail_shapes ()) sides)
            (run (law_scatter_guards b));
        ];
      group "many updates"
        [
          prop ~count:5
            ~examples:(regime_examples @ owned_examples @ float_extremes)
            "combine each target's updates in C order, on the job's threads, \
             one thread or threads that sort"
            (deferred pp_many many_scatters)
            (forced (law_many b));
          prop ~count:8 ~examples:threaded_examples "have one thread's bits"
            (deferred pp_many threaded_scatters)
            (forced (law_threads b));
          prop ~count:10 ~examples:local_examples
            "a float Add's element is that of its element of into and the \
             updates that land on it, in their order"
            (deferred pp_local_add local_adds)
            (forced (law_own_updates b));
          prop ~count:30 ~examples:slab_examples
            "a float Add over a slab of another axis is that slab of the whole \
             Add"
            (deferred pp_slab_add slabs)
            (forced (law_slab b));
          test "a parted float4_e2m1fn Add of 4Mi updates fits 512 MiB"
            (fun () -> b.around (test_parted_memory b));
        ];
    ]

let () = exit (Windtrap.run "nx_cpu.index" [ cpu Support.cpu ])
