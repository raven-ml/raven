(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions through every kernel library the host runs, as nx_kernel.mli
   states them: within the largest bound it allows a library of the exact sum,
   outputs without products and the signed zeros of a sum, integers wrapped in
   the accumulator, scaled and segmented contractions, the same bits through a
   destination's view and into a destination identical to init, nothing written
   where declined, and misfits refused. Each library computes every case the
   laws draw: nx_cpu.mli, nx_cuda.mli and nx_amd.mli state it. *)

open Windtrap
open Contract_cases
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support

(* Cases made when a law first runs them: windtrap declares the suite in its
   parent and in every worker, which run no law of most of them. *)
let deferred gen =
  Gen.with_pp
    (fun ppf c -> pp_case ppf (Lazy.force c))
    (Gen.map Lazy.from_val gen)

let f32 = D.Any D.Float32
let f64 = D.Any D.Float64
let c64 = D.Any D.Complex64
let c128 = D.Any D.Complex128

(* The reference *)

(* An output's init and products summed: [hi + lo] their exact sum as an
   unevaluated pair, [mag] the sum of their magnitudes, [products] that of the
   products alone, and [float64] their sum in plain float64 additions, which is
   not finite where the exact one is not. *)
type sums = {
  hi : float;
  lo : float;
  mag : float;
  products : float;
  float64 : float;
}

(* [o]'s {!sums}. [hi]'s errors, by TwoSum, and the products', by a fused
   multiply-add, are exact, at most [u] times the magnitudes each with [u]
   float64's, and [lo] adds the [2n] of them with [2n] roundings, so [hi + lo]
   is within [slack n] of the exact sum of [n] terms. *)
let exact o =
  (* A loop: refs that a closure takes are boxed, a float allocated a term. *)
  let start = Option.value ~default:0. o.init in
  let hi = ref start and lo = ref 0. and products = ref 0. in
  let float64 = ref start in
  for k = 0 to Array.length o.av - 1 do
    let x = o.av.(k) and y = o.bv.(k) in
    let p = x *. y in
    let s = !hi +. p in
    let z = s -. !hi in
    lo := !lo +. (!hi -. (s -. z) +. (p -. z)) +. Float.fma x y (-.p);
    hi := s;
    products := !products +. Float.abs p;
    float64 := !float64 +. p
  done;
  {
    hi = !hi;
    lo = !lo;
    mag = Float.abs start +. !products;
    products = !products;
    float64 = !float64;
  }

(* At most [2n (n + 2) u²] times [mag], [u] float64's unit roundoff. *)
let slack n mag = Float.of_int (2 * n * (n + 2)) *. 0x1p-106 *. mag
let unit_roundoff acc = if part_of acc = f32 then 0x1p-24 else 0x1p-53
let smallest_subnormal acc = if part_of acc = f32 then 0x1p-149 else 0x1p-1074

let gamma u n =
  let n = float_of_int n in
  n *. u /. (1. -. (n *. u))

(* [o]'s sum in plain float64 additions, which is not finite where the exact one
   is not. *)
let naive o =
  let s = ref (Option.value ~default:0. o.init) in
  for k = 0 to Array.length o.av - 1 do
    s := !s +. (o.av.(k) *. o.bv.(k))
  done;
  !s

(* Σ|a||b| of an output whose parts are [parts]: a complex one's factors are its
   real part's, [ar] and [ai] at terms [2p] and [2p + 1] against [br] and
   [-bi]. *)
let magnitude parts sums =
  let o = parts.(0) in
  if Array.length parts = 1 then sums.(0).products
  else begin
    let m = ref 0. in
    for p = 0 to (Array.length o.av / 2) - 1 do
      let a = Float.hypot o.av.(2 * p) o.av.((2 * p) + 1) in
      let b = Float.hypot o.bv.(2 * p) o.bv.((2 * p) + 1) in
      m := !m +. (a *. b)
    done;
    !m
  end

let subnormal x = x <> 0. && Float.abs x < 0x1p-126

(* Where [b] flushes ({!Support.underflow}) at a float32 [acc], an output with a
   subnormal factor is barred, since a factor read as zero errs by as much as
   its product. *)
let barred ~flush c parts =
  flush
  && part_of c.acc = f32
  && Array.exists
       (fun o -> Array.exists subnormal o.av || Array.exists subnormal o.bv)
       parts

(* An output [xs] of [c] on [b], one value a part, is within the largest bound
   nx_kernel.mli allows a library that adds as [b] states, in blocks or into one
   running sum a range ({!Support.contract_beta}): [u|r| + (1 + u) e Σ|a||b|] of
   the exact [r], init and its [K] products, a complex one in modulus, [e] being
   [γ(β(K) + σ)] for a real [acc] and [√2 γ(β(2K))] for a complex one, [u] the
   unit roundoff of [acc]'s parts. [σ], [scaled], counts the scaled operands:
   nx_cpu.mli states one unit roundoff more a term for each, and nx_cuda.mli,
   nx_amd.mli and nx_metal.mli one rounding more, so [γ(β(K) + σ)] holds each.

   Where results underflow it may err by η more for each term of each part, [η]
   [acc]'s smallest subnormal, twice the most a rounding into the subnormals
   errs. Where [b] flushes at a float32 [acc], each of its at most [K + 64]
   steps, a fused multiply-add, an addition of ranges or of init, errs by
   [2{^-126}] more where it flushes its operand or its result.

   A part whose sum is not finite is matched to the bits of its naive sum.
   Beside it, the modulus bounds nothing, and a finite part, a float sum of [2K]
   terms (nx_kernel.mli), is held to a real sum's ceiling for them: within
   [u|rₚ| + (1 + u) γ(β(2K)) Σ|a||b|] of its exact [rₚ]. *)
let bounded (b : Support.backend) ~flush ~scaled c parts xs =
  let show x = Printf.sprintf "%h" x in
  let np = Array.length parts in
  let terms = Array.length parts.(0).av in
  let k = terms / np in
  let u = unit_roundoff c.acc in
  let beta = Support.contract_beta b in
  let has_init = Option.is_some parts.(0).init in
  let eta =
    if flush && part_of c.acc = f32 then 0x1p-126 *. float_of_int (k + 64)
    else smallest_subnormal c.acc *. float_of_int (terms + 1)
  in
  let sums = Array.map exact parts in
  let pad p = slack terms sums.(p).mag +. eta in
  let mag = magnitude parts sums in
  let exact p = sums.(p).hi +. sums.(p).lo in
  if Array.for_all (fun s -> Float.is_finite s.float64) sums then begin
    (* Each part subtracted apart, so that rounding the exact sum adds no
       error. *)
    let off = Array.mapi (fun p x -> x -. sums.(p).hi -. sums.(p).lo) xs in
    let err = Array.fold_left Float.hypot 0. off in
    let e =
      if np > 1 then Float.sqrt 2. *. gamma u (beta (2 * k))
      else gamma u (beta k + scaled)
    in
    let r = Array.fold_left Float.hypot 0. (Array.init np exact) in
    let pads = Array.fold_left ( +. ) 0. (Array.init np pad) in
    let bound =
      (if has_init then u *. r else 0.) +. ((1. +. u) *. e *. mag) +. pads
    in
    if err <= bound then None
    else
      let all f = String.concat ", " (List.init np f) in
      Some
        (Printf.sprintf "%s is %h from %s, past %h"
           (all (fun p -> show xs.(p)))
           err
           (all (fun p -> show (exact p)))
           bound)
  end
  else
    let part p =
      let naive = sums.(p).float64 in
      if not (Float.is_finite naive) then
        if bits xs.(p) = bits naive then None
        else Some (show xs.(p) ^ ", not " ^ show naive)
      else
        let err = Float.abs (xs.(p) -. sums.(p).hi -. sums.(p).lo) in
        let bound =
          (if has_init then u *. Float.abs (exact p) else 0.)
          +. ((1. +. u) *. gamma u (beta terms) *. mag)
          +. pad p
        in
        if err <= bound then None
        else
          Some
            (Printf.sprintf "%s beside a part not finite is %h from %s, past %h"
               (show xs.(p))
               err
               (show (exact p))
               bound)
    in
    List.find_map part (List.init np Fun.id)

let within ?(scaled = 0) (b : Support.backend) c parts xs =
  let flush = b.underflow = Flush in
  if barred ~flush c parts then None else bounded b ~flush ~scaled c parts xs

(* Whether an output has a part whose sum is not finite. *)
let special parts =
  Array.exists (fun o -> not (Float.is_finite (naive o))) parts

(* Whether an output has a part whose sum is not finite beside a finite one. *)
let mixed parts =
  Array.length parts > 1
  &&
  let finite = Array.map (fun o -> Float.is_finite (naive o)) parts in
  Array.exists Fun.id finite && Array.exists not finite

(* The first outputs of [y] past their bound, and whether one has a part not
   finite beside a finite one. Where [b] flushes, an output is barred too where
   a case of [factors], of [c]'s shapes, has a subnormal factor: a library that
   flushes reads it as a zero of its sign, whatever it multiplies. *)
let outside ?(factors = []) ?scaled (b : Support.backend) c y =
  let flush = b.underflow = Flush in
  let _, each = outputs c in
  let g = planes c.acc y in
  let marks u =
    let _, each_u = outputs u in
    let marks = ref [] in
    each_u (fun _ _ parts -> marks := barred ~flush u parts :: !marks);
    Array.of_list (List.rev !marks)
  in
  let apart =
    match if flush then List.map marks factors else [] with
    | [] -> [||]
    | m :: ms -> List.fold_left (Array.map2 ( || )) m ms
  in
  let bad = ref []
  and i = ref 0
  and seen = ref false
  and specials = ref false in
  each (fun yi k parts ->
      let barred = !i < Array.length apart && apart.(!i) in
      incr i;
      seen := !seen || mixed parts;
      specials := !specials || (k < 64 && special parts);
      if not barred then
        match within ?scaled b c parts (Array.map (fun part -> part.(k)) g) with
        | Some why when List.length !bad < 8 ->
            bad := Format.asprintf "%a: %s" Nx_array_gen.pp_ints yi why :: !bad
        | _ -> ());
  (List.rev !bad, !seen, !specials)

(* {!case_covers}, and a contracted extent past one block of 256 products. *)
let bound_covers c =
  let y, k, _ = sizes c in
  case_covers c;
  cover "past one block of 256 products" (k > 256 && total y > 0)

(* Whether [c]'s outputs sum at most 32 products: {!value}'s operands then leave
   outputs both finite and not, one element in 40 an infinity or a NaN. *)
let few c =
  let _, k, _ = sizes c in
  k <= 32

(* The bound law's cases: finite ones, and {!computed}'s of a few products, half
   each, so that every library's NaN and infinite results are held to the naive
   sum's. *)
let bound_draws =
  Gen.one_of
    [ finite_computed; Gen.with_pp pp_case (Gen.such_that few computed) ]

let law_bound ?(paths = ignore) ?(covers = bound_covers) b c =
  let placed =
    { c with a = on b c.a; b = on b c.b; init = Option.map (on b) c.init }
  in
  covers placed;
  paths c;
  match run b placed with
  | None -> fail "declined"
  | Some y ->
      let bad, seen, specials = outside b c y in
      cover "a part not finite beside a finite one" seen;
      cover "an output not finite" specials;
      equal (list string) [] bad

(* Outputs of a complex [acc] whose init is [∞ + 1i]: each real part is not
   finite, each imaginary part finite beside it. *)
let infinite_part acc =
  lazy
    (let c =
       case_in ~draw:finite_value acc (false, ([||], [| 3 |], [| 4 |], [| 5 |]))
     in
     let y, _, _ = sizes c in
     let init =
       A.of_array D.Complex128 (Iarray.of_array y)
         (Array.make (total y) { Complex.re = infinity; im = 1. })
     in
     { c with init = Some (cast_to acc (A.Any init)) })

(* Outputs without products *)

(* An array of [dt] of shape [s] from the bits [bits], [ut] an unsigned dtype of
   its width, in turn: a NaN's bits are only moved, never computed. *)
let of_bits (type u ue f fe) (dt : (f, fe) D.t) (ut : (u, ue) D.t) s
    (bits : u array) =
  let n = total s in
  let us =
    A.of_array ut (Iarray.of_array s)
      (Array.init n (fun i -> bits.(i mod Array.length bits)))
  in
  A.Any (Option.get (A.bitcast dt us))

(* Inits of each float and complex64 accumulator, by their bits: -0, +0, a
   signalling NaN, a quiet NaN with a payload and 1.5; and (-0, -0) and a
   signalling NaN beside +0 in complex64. *)
let inits =
  [
    ( "float32",
      f32,
      fun s ->
        of_bits D.Float32 D.Uint32 s
          [| 0x80000000l; 0l; 0x7F800001l; 0x7FC01234l; 0x3FC00000l |] );
    ( "float64",
      f64,
      fun s ->
        of_bits D.Float64 D.Uint64 s
          [|
            0x8000000000000000L;
            0L;
            0x7FF0000000000001L;
            0x7FF8000000001234L;
            0x3FF8000000000000L;
          |] );
    ( "complex64",
      c64,
      fun s ->
        of_bits D.Complex64 D.Uint64 s [| 0x8000000080000000L; 0x7F800001L |] );
  ]

(* [c] with [segments] on [b], into a fresh destination of [c.out]'s dtype and
   the shape [y], read back on the host; [None] if declined. *)
let contract_on (b : Support.backend) ?segments c y =
  let module K = (val b.kernels) in
  let (D.Any out) = c.out in
  let dst = A.create b.device out (Iarray.of_array y) in
  let init = Option.map (fun i -> A.expect out (on b i)) c.init in
  let ids x = A.expect D.Int64 (on b (A.Any x)) in
  let segments : _ Nx_kernel.segments option =
    match segments with
    | None -> None
    | Some (Nx_kernel.Rows r) -> Some (Rows { r with ids = ids r.ids })
    | Some (Contracting g) -> Some (Contracting { g with ids = ids g.ids })
  in
  match
    K.contract ~batch:(Iarray.of_array c.batch)
      ~contracting:(Iarray.of_array c.contracting)
      ~segments ~acc:c.acc ~dst ~init
      { array = on b c.a; scale = None }
      { array = on b c.b; scale = None }
  with
  | A.Done -> Some (host (A.Any dst))
  | A.Declined -> None
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

let ones acc s =
  cast_to acc
    (A.Any (A.of_array D.Float64 (Iarray.of_array s) (Array.make (total s) 1.)))

let int64s s xs = A.of_array D.Int64 (Iarray.of_array s) xs

(* Calls whose every output has no products, of shape [y]: a contracted extent
   of 0; a Contracting call whose ids put every index in no group; a Rows call
   whose ids put every row of a in no group. *)
let productless =
  let mm = [| (1, 0) |] in
  [
    ("a contracted extent of 0", [| 2; 0 |], [| 0; 3 |], mm, None, [| 2; 3 |]);
    ( "Contracting ids in no group",
      [| 2; 3 |],
      [| 3; 3 |],
      mm,
      Some
        (Nx_kernel.Contracting
           { ids = int64s [| 3 |] [| -1L; 2L; -1L |]; groups = 2; pair = 0 }),
      [| 2; 2; 3 |] );
    ( "Rows ids in no group",
      [| 2; 3 |],
      [| 3; 3; 2 |],
      mm,
      Some
        (Nx_kernel.Rows
           { ids = int64s [| 2 |] [| -1L; 2L |]; a_axis = 0; b_axis = 2 }),
      [| 2; 3 |] );
  ]

(* nx_kernel.mli: an output without products is its init element as the
   accumulator holds it, rounded into the destination, or [+0] without init; no
   addition touches it, so -0 stays -0 and a NaN keeps its bits. *)
let test_productless (b : Support.backend)
    ((_, sa, sb, contracting, segments, y), init) =
  let acc, init =
    match init with
    | None -> (f32, None)
    | Some (_, acc, init) -> (acc, Some (init y))
  in
  let c =
    {
      batch = [||];
      contracting;
      acc;
      a = ones acc sa;
      b = ones acc sb;
      init;
      out = acc;
      views = [];
    }
  in
  let want =
    match init with
    | Some i -> Support.elements i
    | None ->
        Support.elements
          (cast_to acc
             (A.Any
                (A.of_array D.Float64 (Iarray.of_array y)
                   (Array.make (total y) 0.))))
  in
  match contract_on b ?segments c y with
  | None -> fail "declined"
  | Some got -> equal (array string) want (Support.elements got)

let productless_rows =
  List.concat_map
    (fun call -> List.map (fun i -> (call, Some i)) inits @ [ (call, None) ])
    productless

let productless_id ((what, _, _, _, _, _), init) =
  match init with
  | None -> what ^ ", no init"
  | Some (name, _, _) -> what ^ ", " ^ name ^ " init"

(* Products sum from +0, then init adds in one addition (nx_kernel.mli), so each
   of these gives +0 under IEEE 754's round to nearest, where [x + (-x)] and [-0
   + +0] are +0: [1 · 1 + (-1) · 1] then an init of -0; [-0 · 1] without init;
   [-0 · 1] then an init of -0, which a sum started from init would give as -0;
   and in a complex accumulator [(-0 + 0i) (1 + 0i)], whose real part's terms
   [-0 · 1] and [0 · (-0)] sum from +0, where a complex product's real part is
   -0. *)
let signed_zeros =
  let real acc =
    [
      (acc, "1 · 1 + (-1) · 1, init -0", [| 1.; -1. |], [| 1.; 1. |], Some (-0.));
      (acc, "-0 · 1, no init", [| -0. |], [| 1. |], None);
      (acc, "-0 · 1, init -0", [| -0. |], [| 1. |], Some (-0.));
    ]
  in
  real f32 @ real f64
  @ List.map
      (fun acc -> (acc, "(-0 + 0i) (1 + 0i), no init", [||], [||], None))
      [ c64; c128 ]

let test_signed_zeros (b : Support.backend) (acc, _, av, bv, init) =
  let complex = is_complex acc in
  let k = if complex then 1 else Array.length av in
  let arr s xs =
    cast_to acc (A.Any (A.of_array D.Float64 (Iarray.of_array s) xs))
  in
  let a, bx =
    if complex then
      let z re im =
        A.Any (A.of_array D.Complex128 [| 1; 1 |] [| { Complex.re; im } |])
      in
      (cast_to acc (z (-0.) 0.), cast_to acc (z 1. 0.))
    else (arr [| 1; k |] av, arr [| k; 1 |] bv)
  in
  let c =
    {
      batch = [||];
      contracting = [| (1, 0) |];
      acc;
      a;
      b = bx;
      init = Option.map (fun i -> arr [| 1; 1 |] [| i |]) init;
      out = acc;
      views = [];
    }
  in
  match contract_on b c [| 1; 1 |] with
  | None -> fail "declined"
  | Some y ->
      equal (array string)
        (Support.elements (arr [| 1; 1 |] [| 0. |]))
        (Support.elements y)

(* Integers *)

(* [x] read in [acc], as Prog.Cast stores it, as int64s in C order. *)
let in_acc acc x =
  A.to_array (A.expect D.Int64 (cast_to (D.Any D.Int64) (cast_to acc x)))

(* [x] modulo [acc]'s [2^w], as [acc] holds it. *)
let wrap (D.Any d) x =
  let w = D.bits d in
  if w >= 64 then x
  else
    let low = Int64.logand x (Int64.sub (Int64.shift_left 1L w) 1L) in
    match D.kind d with
    | D.Signed when Int64.logand low (Int64.shift_left 1L (w - 1)) <> 0L ->
        Int64.sub low (Int64.shift_left 1L w)
    | _ -> low

(* An integer contraction as nx_kernel.mli states it: each element of a, b and
   init enters acc as Prog.Cast stores it (nx.cpu's Cast here), the products and
   init sum modulo [2^w] in acc, which int64's wrapping sum gives once wrapped
   to [w] bits, and the sum enters dst as Prog.Cast stores it. *)
let integer_want c =
  let y, oa, ob, walk = index c in
  let a = in_acc c.acc c.a and bx = in_acc c.acc c.b in
  let init = Option.map (in_acc c.acc) c.init in
  let sums = Array.make (total y) 0L in
  walk (fun _ k ra rb ->
      let s = ref (match init with Some i -> i.(k) | None -> 0L) in
      for t = 0 to Array.length oa - 1 do
        s := Int64.add !s (Int64.mul a.(ra + oa.(t)) bx.(rb + ob.(t)))
      done;
      sums.(k) <- wrap c.acc !s);
  cast_to c.out
    (cast_to c.acc (A.Any (A.of_array D.Int64 (Iarray.of_array y) sums)))

(* Integer accumulators, operands of [dts], and a result of the accumulator's
   dtype or of one of them. *)
let integer_case dts =
  Gen.with_pp pp_case
    (Gen.bind
       (Gen.of_list ~pp:pp_dtype (List.filter is_integer D.all))
       (fun acc ->
         Gen.bind
           (Gen.frequency
              [
                (2, Gen.constant ~pp:pp_dtype acc);
                (1, Gen.of_list ~pp:pp_dtype dts);
              ])
           (fun out -> case_of ~acc ~out ~dts ())))

let dtype_of (A.Any x) = D.Any (A.dtype x)

let law_integer (b : Support.backend) c =
  let narrow x =
    let (D.Any d) = dtype_of x in
    D.bits d <= 8 && not (D.is D.Float d)
  in
  cover "integers of 8 bits or fewer" (narrow c.a && narrow c.b);
  cover "other operands" (not (narrow c.a && narrow c.b));
  cover "init" (c.init <> None);
  cover "a result of another dtype" (c.out <> c.acc);
  match run b c with
  | None -> fail "declined"
  | Some y ->
      equal (array string)
        (Support.elements (integer_want c))
        (Support.elements y)

(* A destination through a view *)

(* A destination through any view, its axes stored apart, windowed, stepped or
   reversed, made on the backend's device, gets the bits a fresh C-contiguous
   one gets: the association reads no layout. *)
let law_dst_view ?(covers = fun _ -> ()) (b : Support.backend) (c, v) =
  match run b c with
  | None -> fail "declined"
  | Some y -> (
      let ys, _, _ = sizes c in
      let (D.Any out) = c.out in
      let make made = A.Any (A.create b.device out (Iarray.of_array made)) in
      let dst = E.operand_with make ys v in
      covers (c, dst);
      let init = Option.map (on b) c.init in
      match contract b.kernels c ~dst (on b c.a) (on b c.b) init with
      | A.Done ->
          equal (list string) []
            (differ_bits c.acc ~other:"through a view" y (host dst))
      | r -> failf "contract answered %a" Nx_array_support.pp_answer r)

(* Whether some output of [c] has other bits summed in nx.cpu's chain order than
   in its lane order (nx_cpu.mli), each part apart: values that show an
   association, which a law of an association's bits needs. *)
let shows_order (c, _) =
  let _, each = outputs c in
  let av = ref [] and bv = ref [] in
  each (fun _ _ parts ->
      Array.iter
        (fun o ->
          av := o.av :: !av;
          bv := o.bv :: !bv)
        parts);
  let a = Array.of_list !av and b = Array.of_list !bv in
  let f32 = part_of c.acc = f32 in
  let chain = Support.ordered ~chain:true ~f32 ~init:None a b in
  let lane = Support.ordered ~chain:false ~f32 ~init:None a b in
  cover "an output whose sum in chain order and in lane order differ"
    (Array.exists2 (fun x y -> bits x <> bits y) chain lane)

let view_case_of cases =
  Gen.with_pp
    (fun ppf (c, v) ->
      Format.fprintf ppf "%a; dst %s" pp_case c
        (String.concat " " (E.view_names v)))
    (Gen.bind cases (fun c ->
         let ys, _, _ = sizes c in
         (* No windows: a destination's elements are distinct. *)
         Gen.map (fun v -> (c, v)) (E.view_of ~apart:true (Array.length ys))))

(* Scales *)

(* A scale of an operand: [Exponent] codes or float32 [Value]s, a ratio along
   each axis, and the seed of its values. *)
type spec = { exponent : bool; ratios : int array; seed : int }

(* Float32 subnormals a [Value] scale draws, the largest near 2{^-126}, where a
   scaled element on the grid stays normal. *)
let subnormal_scales = [| 0x1.fffffcp-127; -0x1.8p-127; 0x1p-127 |]

(* The scale [sp] of an operand of shape [s], on the host: codes near 1, at
   times 255; values on the grid, at times a subnormal. *)
let near_one_scale s sp : Nx_kernel.scale =
  let r = Random.State.make [| sp.seed |] in
  let shape =
    Array.mapi (fun i e -> if e = 0 then 0 else e / sp.ratios.(i)) s
  in
  let n = total shape in
  let dims = Iarray.of_array shape in
  if sp.exponent then
    Exponent
      (A.of_array D.Uint8 dims
         (Array.init n (fun _ ->
              if Random.State.int r 50 = 0 then 255
              else 120 + Random.State.int r 15)))
  else
    Value
      (A.of_array D.Float32 dims
         (Array.init n (fun _ ->
              if Random.State.int r 16 = 0 then
                subnormal_scales.(Random.State.int r
                                    (Array.length subnormal_scales))
              else grid.(Random.State.int r (Array.length grid)))))

(* The scale's value of each element of [x], in C order of its indices: a code's
   power of two or NaN, a value as a cast into [acc] holds it. *)
let scale_values acc (A.Any x) (sc : Nx_kernel.scale) =
  let s = shape_of (A.Any x) in
  let values, ss =
    match sc with
    | Exponent e ->
        ( Array.map
            (fun c -> if c = 255 then Float.nan else Float.ldexp 1. (c - 127))
            (A.to_array e),
          shape_of (A.Any e) )
    | Value v ->
        let v = cast_to acc (A.Any v) in
        (A.to_array (to64 v), shape_of v)
  in
  let st = Array.make (Array.length ss) 1 in
  for i = Array.length ss - 2 downto 0 do
    st.(i) <- st.(i + 1) * ss.(i + 1)
  done;
  Array.init (total s) (fun t ->
      let at = ref 0 and rest = ref t in
      for d = Array.length s - 1 downto 0 do
        let i = !rest mod s.(d) in
        rest := !rest / s.(d);
        at := !at + (i / (s.(d) / ss.(d)) * st.(d))
      done;
      values.(!at))

(* [x] as [acc] holds it times its scale, as float64: exact for a float32 [acc],
   whose elements and scales hold 24 bits each, and for a float64 one the
   product rounded once to float64, as the library rounds it into acc. *)
let scaled acc x sc =
  let xs = A.to_array (to64 (cast_to acc x)) and ss = scale_values acc x sc in
  A.Any
    (A.of_array D.Float64
       (Iarray.of_array (shape_of x))
       (Array.mapi (fun i v -> v *. ss.(i)) xs))

(* Cases of the accumulators [accs], float32 and float64 by default, of operands
   each holds, with a scale on a, on b or on both: ratios 1 or a whole axis. *)
let scaled_case_of accs =
  let open Gen in
  let spec x =
    let s = shape_of x in
    let* exponent = bool and+ seed = int in
    let+ whole = array ~size:(constant (Array.length s)) bool in
    {
      exponent;
      ratios = Array.mapi (fun i e -> if whole.(i) && e > 0 then e else 1) s;
      seed;
    }
  in
  Gen.with_pp
    (fun ppf (c, sa, sb) ->
      Format.fprintf ppf "%a; scaled %s%s" pp_case c
        (if Option.is_some sa then "a" else "")
        (if Option.is_some sb then "b" else ""))
    (let* acc = of_list ~pp:pp_dtype accs in
     let* c = case_of ~draw:finite_value ~acc ~out:acc ~dts:(held acc) () in
     let* which = int_range 1 3 in
     let* sa = spec c.a and+ sb = spec c.b in
     constant
       ( c,
         (if which land 1 <> 0 then Some sa else None),
         if which land 2 <> 0 then Some sb else None ))

let scaled_case = scaled_case_of [ f32; f64 ]

(* A scaled contraction is within the bound of the exact sum of its terms, each
   element as the accumulator holds it times its scale, rounded once into the
   accumulator, with the roundings its scaled operands add. *)
let law_scaled (b : Support.backend) (c, sa, sb) =
  let sa = Option.map (near_one_scale (shape_of c.a)) sa in
  let sb = Option.map (near_one_scale (shape_of c.b)) sb in
  let subnormal_scale = function
    | Some (Nx_kernel.Value v) ->
        Array.exists subnormal (A.to_array (to64 (A.Any v)))
    | Some (Exponent _) | None -> false
  in
  cover "a subnormal Value scale" (subnormal_scale sa || subnormal_scale sb);
  match run ?sa ?sb b c with
  | None -> fail "declined"
  | Some y ->
      let a = Option.fold ~none:c.a ~some:(scaled c.acc c.a) sa in
      let b' = Option.fold ~none:c.b ~some:(scaled c.acc c.b) sb in
      (* Before their scales, and the scales alone, each of whose subnormal
         factors a flushing library reads as zero. *)
      let values x sc =
        let s = shape_of x in
        let vs =
          match sc with
          | None -> Array.make (total s) 1.
          | Some sc -> scale_values c.acc x sc
        in
        A.Any (A.of_array D.Float64 (Iarray.of_array s) vs)
      in
      let scales = { c with a = values c.a sa; b = values c.b sb } in
      let count = function Some _ -> 1 | None -> 0 in
      let bad, _, _ =
        outside ~factors:[ c; scales ]
          ~scaled:(count sa + count sb)
          b { c with a; b = b' } y
      in
      equal (list string) [] bad

(* Scales of a block of one element: [Exponent] codes at a format's edges, and
   float32 [Value]s, a NaN and the subnormal 2{^-127}. *)
let scale_rows =
  List.concat_map
    (fun acc ->
      List.map (fun e -> (acc, `Code e)) [ 0; 1; 127; 253; 254; 255 ]
      @ [ (acc, `Value Float.nan); (acc, `Value 0x1p-127) ])
    [ f32; f64 ]

let scale_id (acc, s) =
  Format.asprintf "%a, %s" pp_dtype acc
    (match s with
    | `Code e -> Printf.sprintf "code %d" e
    | `Value v -> Printf.sprintf "Value %h" v)

(* nx_kernel.mli: code [e] scales its block by [2^(e - 127)] and 255 makes it
   NaN; a Value is read in its dtype into the accumulator. So [1.5 · 1], a
   scaled by [s], is [1.5 s], exact in both accumulators: [1.5 · 2^-127] is a
   float32 subnormal, which a library that flushes gives as +0. *)
let test_scale (b : Support.backend) (acc, s) =
  let one v = cast_to acc (A.Any (A.of_array D.Float64 [| 1; 1 |] [| v |])) in
  let c =
    {
      batch = [||];
      contracting = [| (1, 0) |];
      acc;
      a = one 1.5;
      b = one 1.;
      init = None;
      out = acc;
      views = [];
    }
  in
  let sa : Nx_kernel.scale =
    match s with
    | `Code e -> Exponent (A.of_array D.Uint8 [| 1; 1 |] [| e |])
    | `Value v -> Value (A.of_array D.Float32 [| 1; 1 |] [| v |])
  in
  let scale =
    match s with
    | `Code 255 -> Float.nan
    | `Code e -> Float.ldexp 1. (e - 127)
    | `Value v -> v
  in
  let want = 1.5 *. scale in
  let want =
    if b.underflow = Flush && acc = f32 && subnormal want then 0. else want
  in
  match run ~sa b c with
  | None -> fail "declined"
  | Some y -> equal (array float_exact) [| want |] (planes acc y).(0)

(* Segments *)

(* A segmented contraction: its case, unsegmented, of a [z; m; k] and b [z; k;
   n] (Contracting) or [z; k; n; g] (Rows), [z] present where it has a batch;
   its segments over host ids; its result's shape. *)
type seg = {
  sc : case;
  segments : (int64, D.int64_elt) A.t Nx_kernel.segments;
  yshape : int array;
}

let pp_seg ppf s =
  let what =
    match s.segments with
    | Rows _ -> "rows"
    | Contracting g -> Printf.sprintf "contracting into %d groups" g.groups
  in
  let ids = match s.segments with Rows r -> r.ids | Contracting g -> g.ids in
  Format.fprintf ppf "%s: %a; ids %s" what pp_case s.sc
    (String.concat " "
       (List.map Int64.to_string (Array.to_list (A.to_array ids))))

(* An operand of [dt] whose made shape is [s] through the view [v], its values
   finite: {!finite_value}'s, or small integers. *)
let finite dt s v seed =
  let r = Random.State.make [| seed |] in
  let make made =
    let draw () =
      if is_integer dt then float_of_int (Random.State.int r 8)
      else finite_value dt r
    in
    cast_to dt
      (A.Any
         (A.of_array D.Float64 (Iarray.of_array made)
            (Array.init (total made) (fun _ -> draw ()))))
  in
  E.operand_with make (Array.map (fun m -> s.(m)) v.perm) v

(* The segmented case in [acc] of the extents [z], [m], [n], [k] (no batch
   without [z]), of operands of [da] and [db] through the views [va] and [vb],
   Rows of [g] columns' groups where [rows], else Contracting into [g] groups,
   [ids] each index's group along the segmented axis, an init where [with_init],
   values from [seed]. *)
let seg_of ~acc (da, va) (db, vb) ~rows (z, m, n, k) g ids ~with_init seed =
  let batch = match z with Some z -> [| z |] | None -> [||] in
  let nb = Array.length batch in
  let sa = Array.concat [ batch; [| m; k |] ] in
  let sb =
    Array.concat [ batch; [| k; n |]; (if rows then [| g |] else [||]) ]
  in
  let a = finite da sa va seed and bx = finite db sb vb (seed + 1) in
  (* Each made axis's place in the arrays the views give. *)
  let ia = E.inverse va.perm and ib = E.inverse vb.perm in
  let pairs = if nb = 1 then [| (ia.(0), ib.(0)) |] else [||] in
  let contracting = [| (ia.(nb + 1), ib.(nb)) |] in
  let ids =
    A.of_array D.Int64
      (Iarray.of_array (Array.append batch [| (if rows then m else k) |]))
      (Array.map Int64.of_int ids)
  in
  let segments : _ Nx_kernel.segments =
    if rows then Rows { ids; a_axis = ia.(nb); b_axis = ib.(nb + 2) }
    else Contracting { ids; groups = g; pair = 0 }
  in
  let y =
    match
      Nx_kernel.Shape.contract ~batch:(Iarray.of_array pairs)
        ~contracting:(Iarray.of_array contracting)
        ~segments:
          (Some
             (match segments with
             | Rows x -> Nx_kernel.Rows { x with ids = shape_of (A.Any x.ids) }
             | Contracting x ->
                 Contracting { x with ids = shape_of (A.Any x.ids) }))
        (shape_of a, None)
        (shape_of bx, None)
    with
    | Ok y -> y
    | Error e -> invalid_arg ("seg_of: " ^ e)
  in
  let init =
    if with_init then Some (finite acc y (E.plain (Array.length y)) (seed + 3))
    else None
  in
  {
    sc =
      {
        batch = pairs;
        contracting;
        acc;
        a;
        b = bx;
        init;
        out = acc;
        views = [];
      };
    segments;
    yshape = y;
  }

(* Segmented cases, Rows ones where [rows] draws true; where [metal], of the
   accumulators and operands nx.metal computes: float32 of two operands of one
   staged dtype, int32 and complex64. Each id is in [-1, g], [-1] and [g] in no
   group. *)
let seg_cases ?(metal = false) rows =
  let open Gen in
  let accs =
    if metal then [ f32; D.Any D.Int32; c64 ]
    else [ f32; f64; D.Any D.Int32; c64; c128 ]
  in
  let* acc = of_list ~pp:pp_dtype accs in
  let dts =
    if is_integer acc then D.[ Any Int32; Any Int8; Any Uint4 ]
    else if metal && acc = f32 then D.[ Any Float32; Any Float16; Any Bfloat16 ]
    else if metal then [ c64 ]
    else held acc
  in
  let* da = of_list ~pp:pp_dtype dts and+ db = of_list ~pp:pp_dtype dts in
  let db = if metal && not (is_integer acc) then da else db in
  let* rows = rows and+ nb = int_range 0 1 in
  let* z = int_range 1 3 and+ m = int_range 1 40 and+ n = int_range 1 20 in
  let* k = frequency [ (3, int_range 0 40); (1, int_range 100 700) ] in
  let* g = int_range 0 5 and+ with_init = bool and+ seed = int in
  let g = if rows then Int.max g 1 else g in
  let ra = 2 + nb and rb = 2 + nb + if rows then 1 else 0 in
  let* va = view ra nb and+ vb = view rb nb in
  let r = Random.State.make [| seed + 2 |] in
  let extent = (if nb = 1 then z else 1) * if rows then m else k in
  let ids = Array.init extent (fun _ -> Random.State.int r (g + 2) - 1) in
  constant
    (seg_of ~acc (da, va) (db, vb) ~rows
       ((if nb = 1 then Some z else None), m, n, k)
       g ids ~with_init seed)

let seg_case = Gen.with_pp pp_seg (seg_cases Gen.bool)
let k_seg_case = Gen.with_pp pp_seg (seg_cases (Gen.constant false))

(* A float32 case of plain views, of a [3; 4] and b [4; 2] or [4; 2; g], with
   init. *)
let plain_seg ~rows g ids =
  let p r = E.plain r in
  seg_of ~acc:f32
    (f32, p 2)
    (f32, p (if rows then 3 else 2))
    ~rows (None, 3, 2, 4) g ids ~with_init:true 1

(* Segments the drawn cases reach rarely: Contracting into no groups; Rows whose
   every row is in no group; a Contracting group of no index. *)
let seg_examples =
  [
    lazy (plain_seg ~rows:false 0 [| 0; -1; 0; 1 |]);
    lazy (plain_seg ~rows:true 2 [| -1; -1; 2 |]);
    lazy (plain_seg ~rows:false 3 [| 0; 0; 2; 2 |]);
  ]

let deferred_seg gen =
  Gen.with_pp
    (fun ppf s -> pp_seg ppf (Lazy.force s))
    (Gen.map Lazy.from_val gen)

(* [s] on [b], read back on the host; [None] if declined. *)
let seg_run (b : Support.backend) s =
  contract_on b ~segments:s.segments s.sc s.yshape

(* Each output of [s]'s result, in C order: its parts' init and the factors of
   its group's products alone, from the unsegmented case's outputs. Rows drops
   b's b_axis, and a row of a in no group has no products; Contracting puts the
   groups' axis first, and slice [e] takes group [e]'s indices
   (nx_kernel.mli). *)
let seg_outputs s =
  let c = s.sc in
  (* The init has the segmented result's shape. *)
  let _, each = outputs { c with init = None } in
  let init = Option.map (planes c.acc) c.init in
  let with_init j parts =
    Array.mapi
      (fun p o -> { o with init = Option.map (fun i -> i.(p).(j)) init })
      parts
  in
  let nb = Array.length c.batch in
  let ids =
    A.to_array
      (match s.segments with Rows r -> r.ids | Contracting g -> g.ids)
  in
  let extent =
    Array.length ids
    / Int.max 1 (if nb = 1 then (shape_of c.a).(fst c.batch.(0)) else 1)
  in
  let group z i = Int64.to_int ids.((z * extent) + i) in
  let ys = s.yshape in
  let stride = Array.make (Array.length ys) 1 in
  for i = Array.length ys - 2 downto 0 do
    stride.(i) <- stride.(i + 1) * ys.(i + 1)
  done;
  let at idx =
    Array.fold_left ( + ) 0 (Array.mapi (fun i x -> x * stride.(i)) idx)
  in
  let out = Array.make (total ys) [||] in
  (* The place of b's free axis [ax] among the unsegmented result's axes. *)
  let column ax =
    let sb = shape_of c.b in
    let paired ps j = Array.exists (fun (_, j') -> j' = j) ps in
    let free =
      List.filter
        (fun j -> not (paired c.batch j || paired c.contracting j))
        (List.init (Array.length sb) Fun.id)
    in
    let fa = Array.length (shape_of c.a) - nb - Array.length c.contracting in
    let rec find i = function
      | x :: _ when x = ax -> i
      | _ :: l -> find (i + 1) l
      | [] -> assert false
    in
    nb + fa + find 0 free
  in
  each (fun yi _ parts ->
      let z = if nb = 1 then yi.(0) else 0 in
      match s.segments with
      | Rows r ->
          let pos = column r.b_axis in
          let g = (shape_of c.b).(r.b_axis) in
          let id = group z yi.(nb) in
          let kept =
            Array.of_list
              (List.filteri (fun i _ -> i <> pos) (Array.to_list yi))
          in
          let j = at kept in
          if id = yi.(pos) then out.(j) <- with_init j parts
          else if (id < 0 || id >= g) && yi.(pos) = 0 then
            out.(j) <-
              with_init j
                (Array.map (fun p -> { p with av = [||]; bv = [||] }) parts)
      | Contracting g ->
          let per = if Array.length parts = 1 then 1 else 2 in
          for e = 0 to g.groups - 1 do
            let pick xs =
              let l = ref [] in
              for t = Array.length xs - 1 downto 0 do
                if group z (t / per) = e then l := xs.(t) :: !l
              done;
              Array.of_list !l
            in
            let j = at (Array.append [| e |] yi) in
            out.(j) <-
              with_init j
                (Array.map
                   (fun p -> { p with av = pick p.av; bv = pick p.bv })
                   parts)
          done);
  out

(* A segmented contraction is each group's contraction of its indices alone:
   within the bound of its group's products, or, of small integers, whose sums
   float64 holds exactly, their sum. *)
let segmented (b : Support.backend) s =
  let c = s.sc in
  match seg_run b s with
  | None -> fail "declined"
  | Some y ->
      let g = planes c.acc y in
      let bad = ref [] in
      Array.iteri
        (fun k parts ->
          let xs = Array.map (fun p -> p.(k)) g in
          let why =
            if is_integer c.acc then
              let w = naive parts.(0) in
              if bits xs.(0) = bits w then None
              else Some (Printf.sprintf "%h, expected %h" xs.(0) w)
            else within b c parts xs
          in
          match why with
          | Some why when List.length !bad < 8 ->
              bad := Printf.sprintf "output %d: %s" k why :: !bad
          | _ -> ())
        (seg_outputs s);
      equal (list string) [] (List.rev !bad)

let ids_of s =
  A.to_array (match s.segments with Rows r -> r.ids | Contracting g -> g.ids)

let groups_of s =
  match s.segments with
  | Rows r -> (shape_of s.sc.b).(r.b_axis)
  | Contracting g -> g.groups

let law_segmented (b : Support.backend) s =
  let ids = ids_of s and g = groups_of s in
  let in_group e = Array.exists (fun i -> Int64.to_int i = e) ids in
  cover "rows" (match s.segments with Rows _ -> true | _ -> false);
  cover "contracting"
    (match s.segments with Contracting _ -> true | _ -> false);
  cover "init" (Option.is_some s.sc.init);
  cover "an index in no group"
    (Array.exists (fun i -> i < 0L || i >= Int64.of_int g) ids);
  cover "no groups" (g = 0);
  cover "a group of no index"
    (List.exists (fun e -> not (in_group e)) (List.init g Fun.id));
  segmented b s

(* [s] with every index outside group 0 moved to another group or to none: [-1],
   [1], ..., [g] in turn. *)
let moved s =
  let g = groups_of s in
  let next i =
    if i = 0L then 0L
    else if i = Int64.of_int g then -1L
    else if i = -1L then 1L
    else Int64.succ i
  in
  let ids x =
    A.of_array D.Int64
      (A.Layout.shape (A.layout x))
      (Array.map next (A.to_array x))
  in
  let segments : _ Nx_kernel.segments =
    match s.segments with
    | Rows r -> Rows { r with ids = ids r.ids }
    | Contracting c -> Contracting { c with ids = ids c.ids }
  in
  { s with segments }

(* nx_kernel.mli: a segmented call's ids never enter the association, so an
   output's bits do not depend on the groups of the other indices. Group 0's
   outputs keep their bits when every index outside it moves. *)
let law_ids (b : Support.backend) s =
  let ids = ids_of s in
  let zero = Array.exists (( = ) 0L) ids
  and other = Array.exists (( <> ) 0L) ids in
  cover "rows" (match s.segments with Rows _ -> true | _ -> false);
  cover "contracting"
    (match s.segments with Contracting _ -> true | _ -> false);
  cover "an index of group 0 beside one moved" (zero && other);
  shows_order ({ s.sc with init = None }, ());
  match (seg_run b s, seg_run b (moved s)) with
  | Some y, Some y' ->
      let c = s.sc in
      let g = planes c.acc y and g' = planes c.acc y' in
      let per_slice =
        if groups_of s = 0 then 0 else total s.yshape / Int.max 1 s.yshape.(0)
      in
      (* Whether output [k] is group 0's: Contracting's first slice, or a Rows
         output whose row of a is in group 0. *)
      let ours k =
        match s.segments with
        | Contracting _ -> k < per_slice
        | Rows _ ->
            let nb = Array.length c.batch in
            let ys = s.yshape in
            let after =
              total (Array.sub ys (nb + 1) (Array.length ys - nb - 1))
            in
            let m = ys.(nb) in
            let z = k / (m * after) and r = k / after mod m in
            ids.((z * m) + r) = 0L
      in
      let bad = ref [] in
      Array.iteri
        (fun p x ->
          Array.iteri
            (fun k v ->
              if ours k && bits v <> bits g'.(p).(k) && List.length !bad < 8
              then
                bad :=
                  Printf.sprintf "output %d: %h, moved %h" k v g'.(p).(k)
                  :: !bad)
            x)
        g;
      equal (list string) [] (List.rev !bad)
  | _ -> fail "declined"

(* The contracted extent past which nx.cuda and nx.amd hold a Contracting call's
   indices in 8 bytes, without their contract_wide_k: 2^31 - 1, the most it
   takes. *)
let wide_k = (1 lsl 31) - 1

(* A Contracting call past 2^31 - 1 contracted indices holds them in 8 bytes:
   the same law, with 8 bytes from the first index, which [wide] sets. *)
let law_wide ~wide (b : Support.backend) s =
  cover "complex" (is_complex s.sc.acc);
  cover "init" (Option.is_some s.sc.init);
  Fun.protect
    ~finally:(fun () -> wide wide_k)
    (fun () ->
      wide 0;
      segmented b s)

(* In place *)

(* [x]'s values in a host array of its shape through the view [v]. *)
let through v x =
  let (A.Any t) = x in
  let make made =
    A.Any (A.create Rig.host (A.dtype t) (Iarray.of_array made))
  in
  let (A.Any y as viewed) = E.operand_with make (shape_of x) v in
  cast_into y x;
  viewed

(* [c] with an init where it has none, C-contiguous, of the result's dtype,
   drawn as its operands are. *)
let with_init (c : case) =
  match c.init with
  | Some _ -> c
  | None ->
      let y, _, _ = sizes c in
      let init =
        operand ~wide_ints:(is_integer c.acc) c.out y
          (E.plain (Array.length y))
          2
      in
      { c with init = Some init }

(* A destination identical to init, an array of the result's dtype through
   [view] (C-contiguous by default), gives the bits a fresh one does: a kernel
   reads each element of init before it writes the destination there. *)
let law_in_place ?(covers = fun _ -> ()) ?view (b : Support.backend) (c : case)
    =
  let c = with_init c in
  let i = cast_to c.out (Option.get c.init) in
  let c = { c with init = Some i } in
  (* A copy of [i]: on nx.cpu [on] keeps a host array itself. *)
  let alias =
    on b (match view with Some v -> through v i | None -> cast_to c.out i)
  in
  covers (c, alias);
  let answer =
    contract b.kernels c ~dst:alias (on b c.a) (on b c.b) (Some alias)
  in
  let computed =
    match (run b c, answer) with
    | Some y, A.Done ->
        equal (list string) []
          (differ_bits c.acc ~other:"in place" y (host alias));
        true
    | None, A.Declined ->
        collect "declined";
        false
    | y, r ->
        failf "a fresh destination %s, in place %a"
          (if Option.is_some y then "computed" else "declined")
          Nx_array_support.pp_answer r
  in
  cover "computed" computed

(* {!computed}'s cases, each with an init. *)
let in_place_case =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list ~pp:pp_dtype accs) (fun acc ->
         Gen.bind
           (Gen.frequency
              [
                (2, Gen.constant (held acc));
                (1, Gen.constant (others (held acc)));
              ])
           (fun dts ->
             case_of ~init:true ~draw:finite_value ~acc ~out:acc ~dts ())))

(* Contractions of 16 × 16 outputs, with init, either side of a block of 256
   products and over several, which a library may sum into scratch beside a
   destination identical to init. *)
let in_place_examples =
  let v = E.plain 2 in
  List.concat_map
    (fun acc ->
      List.map
        (fun k ->
          lazy
            (make ~draw:finite_value ~acc ~out:acc
               ([||], [| 16 |], [| 16 |], [| k |])
               (acc, v) (acc, v) (Some v) k))
        [ 255; 256; 257; 1025 ])
    [ f32; f64; c64 ]

(* Declines *)

(* A kernel that declines writes nothing: [dst] keeps its values. Every kernel
   library declines a scale with an integer or complex accumulator. *)
let law_declined (b : Support.backend) (c, sa, sb) =
  let scaled = Option.is_some sa || Option.is_some sb in
  let declines = scaled && (is_integer c.acc || is_complex c.acc) in
  cover "a scale with an integer acc" (declines && is_integer c.acc);
  cover "a scale with a complex acc" (declines && is_complex c.acc);
  let y, _, _ = sizes c in
  let threes =
    A.Any (A.of_array D.Float64 (Iarray.of_array y) (Array.make (total y) 3.))
  in
  let dst = on b (cast_to c.out threes) in
  let before = planes c.out (host dst) in
  let init = Option.map (on b) c.init in
  let scale s = scale_on b (scale_of s) in
  let sa = Option.map scale sa and sb = Option.map scale sb in
  match contract ?sa ?sb b.kernels c ~dst (on b c.a) (on b c.b) init with
  | A.Declined ->
      equal (array (array float_exact)) before (planes c.out (host dst))
  | A.Done when declines ->
      fail "computed a scale with an integer or complex acc"
  | A.Done -> ()
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* Misfits *)

(* A contraction whose operands Shape.contract answers Error for, or whose dst's
   shape is not the result's, does not fit: not Done, and nothing written
   outside dst. Misfits in the pairs' extents, the operands' ranks and dst, a
   result past max_numel, and one behind a group that does not merge. *)
let test_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    let ones = Array.make (total s) 1. in
    let x = A.of_array D.Float32 (Iarray.of_array s) ones in
    A.expect D.Float32 (on b (A.Any x))
  in
  let refused ~msg ?(batch : _ iarray = [||]) contracting dst a b' =
    Misfit.check ~msg ~what:"a shape" b
      [| (f32, dst) |]
      (fun g ds ->
        K.contract ~batch ~contracting ~segments:None ~acc:f32
          ~dst:(A.expect D.Float32 ds.(0))
          ~init:None
          (plain (g.g a))
          (plain (g.g b')))
  in
  let mm : _ iarray = [| (1, 0) |] in
  refused ~msg:"contracted extents" mm [| 2; 3 |]
    (arr [| 2; 4 |])
    (arr [| 5; 3 |]);
  refused ~msg:"dst's shape" mm [| 3; 2 |] (arr [| 2; 4 |]) (arr [| 4; 3 |]);
  refused ~msg:"dst's rank" mm [| 6 |] (arr [| 2; 4 |]) (arr [| 4; 3 |]);
  refused ~msg:"a's rank" mm [| 2; 3 |] (arr [| 4 |]) (arr [| 4; 3 |]);
  let k = 1 lsl 27 in
  refused ~msg:"a result past max_numel" mm [| 1; 1 |]
    (arr [| k; 0 |])
    (arr [| 0; k |]);
  (* [a]'s batch axes transposed, its contracted extent 4 against [b]'s 5. *)
  let a =
    let x = arr [| 3; 2; 2; 4 |] in
    Option.get (A.move (Nx_array.Move.Permute [| 1; 0; 2; 3 |]) x)
  in
  refused ~msg:"behind an unmerged group"
    ~batch:[| (0, 0); (1, 1) |]
    [| (3, 2) |]
    [| 2; 3; 2; 2 |] a
    (arr [| 2; 3; 5; 2 |])

(* An [acc] Nx_kernel.accumulates refuses, a narrow float's or a boolean's, does
   not fit: not Done, and nothing written outside dst. *)
let test_acc_refused (b : Support.backend) () =
  let module K = (val b.kernels) in
  let a = A.create b.device D.Float32 [| 2; 3 |] in
  let x = A.create b.device D.Float32 [| 3; 2 |] in
  List.iter
    (fun acc ->
      Misfit.check
        ~msg:(Format.asprintf "an accumulator of %a" pp_dtype acc)
        ~what:"a dtype" b
        [| (f32, [| 2; 2 |]) |]
        (fun g ds ->
          K.contract ~batch:[||]
            ~contracting:[| (1, 0) |]
            ~segments:None ~acc
            ~dst:(A.expect D.Float32 ds.(0))
            ~init:None
            (plain (g.g a))
            (plain (g.g x))))
    D.[ Any Float16; Any Bfloat16; Any Bool ]

(* The suite *)

(* Complex outputs with a part not finite beside a finite one. *)
let infinite_parts = [ infinite_part c64; infinite_part c128 ]

(* The sizes drawn cases reach rarely, of every float and complex
   accumulator. *)
let edge_examples =
  List.concat_map (edges ~draw:finite_value) accs @ infinite_parts

(* nx.cpu's regimes, its paths' sizes, which cpu/test_contract runs bit for bit
   on every table: the laws run them on every other library, where they reach a
   sum cut into ranges of k, wide columns and windowed operands. They hold a
   real accumulator's edges; a complex one's are added. *)
let regime_examples = List.concat_map (regime_cases ~draw:finite_value) accs

let gpu_examples =
  regime_examples
  @ edges ~draw:finite_value c64
  @ edges ~draw:finite_value c128
  @ infinite_parts

(* The laws of [d], the bound law with [examples] and covering [paths]. *)
let laws ?paths ~examples (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  let forced f c = run (fun b c -> f b (Lazy.force c)) c in
  let check f () = run f () in
  group d.label
    [
      prop ~examples "a contraction is within its bound of the exact sum"
        (deferred bound_draws)
        (forced (law_bound ?paths));
      cases ~id:productless_id
        "an output without products is init as the accumulator holds it, or +0"
        productless_rows (run test_productless);
      cases
        ~id:(fun (acc, what, _, _, _) ->
          Format.asprintf "%a, %s" pp_dtype acc what)
        "an output's products sum from +0, then its init adds" signed_zeros
        (run test_signed_zeros);
      prop
        "an integer contraction is its products' sum modulo the accumulator, \
         cast once into dst"
        (integer_case D.all) (run law_integer);
      prop "a destination through a view gets a fresh one's bits"
        (view_case_of finite_computed)
        (run (law_dst_view ~covers:shows_order));
      prop "a scaled contraction is within its bound of its terms' sum"
        scaled_case (run law_scaled);
      cases ~id:scale_id
        "an Exponent code e scales its block by 2^(e - 127), 255 by NaN"
        scale_rows (run test_scale);
      prop ~examples:seg_examples
        "a segmented contraction is each group's contraction"
        (deferred_seg seg_case) (forced law_segmented);
      prop "an output's bits do not depend on the groups of the other indices"
        seg_case (run law_ids);
      prop
        ~examples:
          (List.map (Lazy.map with_init) edge_examples @ in_place_examples)
        "a destination identical to init gives a fresh one's bits"
        (deferred in_place_case)
        (forced (fun b c -> law_in_place b c));
      prop "a declined contraction writes nothing" unscalable (run law_declined);
      test
        "a call of shapes that do not fit is not Done and writes nothing \
         outside dst"
        (check (fun b () -> test_misfits b ()));
      test "an acc a contraction does not compute in is not Done"
        (check (fun b () -> test_acc_refused b ()));
    ]

(* nx_cuda.mli and nx_amd.mli: an operand whose contracted axes do not form one
   run of strides is copied, here [2k] float32 elements past the 64 MiB the
   device keeps; a sum cut into ranges of k holds each range's sums of a tile of
   outputs, here within [ranges] bytes. Both lie in the call's workspace, past
   what the device keeps a buffer of the call's own, which the device counts in
   its driver's units ({!Rig.allocation}). *)
let test_loop_copy ~ranges (b : Support.backend) () =
  let module K = (val b.kernels) in
  let k = 1 lsl 24 and d = b.device in
  let base = A.create d D.Float32 [| 1; k; 2 |] in
  let a =
    A.v D.Float32
      (A.Layout.v ~strides:[| 2 * k; 1; 2 |] [| 1; 2; k |])
      (A.buffer base)
  in
  let rhs = A.create d D.Float32 [| 2; k; 1 |] in
  let dst = A.create d D.Float32 [| 1; 1 |] in
  Kernels_cases.loop_held b
    ~bound:(Rig.allocation d ((2 * k * 4) + ranges))
    (fun () ->
      equal Elements.answer A.Done
        (K.contract ~batch:[||]
           ~contracting:[| (1, 0); (2, 1) |]
           ~segments:None ~acc:f32 ~dst ~init:None (plain a) (plain rhs));
      Rig.Buffer.wait (A.buffer dst) Read_write)

(* What nx.cuda's and nx.amd's .mli state of their own: [wide] lowers the extent
   past which indices take 8 bytes, and [ranges] bounds a sum's ranges' bytes, 1
   MiB on nx.cuda, 64 ranges of a 128 x 128 float32 tile and their tickets on
   nx.amd. *)
let own ~lib ~wide ~ranges (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  group
    (lib ^ " " ^ d.label)
    [
      prop "a Contracting call with 8-byte indices is each group's contraction"
        k_seg_case
        (run (law_wide ~wide));
      test "contractions in a loop hold one call's copies"
        (run (fun b () -> test_loop_copy ~ranges b ()));
    ]

let cuda = own ~lib:"nx.cuda" ~wide:Nx_cuda.contract_wide_k ~ranges:(1 lsl 20)

let amd =
  own ~lib:"nx.amd" ~wide:Nx_amd.contract_wide_k
    ~ranges:((64 * 128 * 128 * 4) + (64 * 4))

(* nx.amd's paths *)

module K = Nx_amd.Kernels

(* The launches nx.amd plans for [c], on the host: planning runs nothing. *)
let amd_launches c =
  let y, _, _ = sizes c in
  let (D.Any out) = c.out in
  let dst = A.create Rig.host out (Iarray.of_array y) in
  match
    Nx_amd.contract_plan ~batch:(Iarray.of_array c.batch)
      ~contracting:(Iarray.of_array c.contracting)
      ~segments:None ~acc:c.acc ~dst
      ~init:(Option.map (A.expect (A.dtype dst)) c.init)
      { array = c.a; scale = None }
      { array = c.b; scale = None }
  with
  | Ok (Some p) -> Array.to_list p.launches
  | Ok None -> []
  | Error e -> failf "planned %a" Nx_array_support.pp_answer e

let word (l : Nx_amd.Plan.launch) i =
  Int64.to_int (Bytes.get_int64_le l.words (8 * i))

let gemv (l : Nx_amd.Plan.launch) =
  match l.kernel with
  | Contract_gemv_f32 | Contract_gemv_f64 | Contract_gemv_u32
  | Contract_gemv_u64 ->
      true
  | _ -> false

let load (l : Nx_amd.Plan.launch) =
  Option.map (fun (t : K.tile) -> t.load) (K.tile l.kernel)

(* Whether operand [o] of [l] (its first word) is read by rows: its free axis
   contiguous. *)
let by_rows l o = word l (o + K.operand_flags) land K.op_free <> 0

(* The paths of nx.amd's SIMT contraction that [c] takes, by the launches it
   plans: gemv, on a's few rows or on few columns with a and b exchanged; a
   tile's loads of the compute type, along k or by rows, and its converting
   loads; and a sum cut into ranges of k, whose sums combine adds. *)
let amd_paths c =
  let ls = amd_launches c in
  let any f = List.exists f ls in
  let m, _ = rows_cols c in
  let ranged l =
    (gemv l || load l <> None)
    && (word l K.contract_s1 > 1 || word l K.contract_s2 > 1)
  in
  let native l = load l = Some K.Native in
  cover "gemv" (any gemv);
  cover "gemv on few columns, a and b exchanged" (m > K.gemv_rows && any gemv);
  cover "native loads" (any native);
  cover "native loads by rows"
    (any (fun l ->
         native l && (by_rows l K.contract_a || by_rows l K.contract_b)));
  cover "converting loads" (any (fun l -> load l = Some K.Convert));
  cover "ranges of k" (any ranged)

(* Cases that reach nx.amd's converting loads in every run: an a of a dtype the
   accumulator does not hold, on a SIMT tile. The last two read an operand in
   16-byte chunks of 8 elements, and its end cuts a chunk: a float16 a along k,
   whose k = 20 leaves 4 in the last k-tile, and a bfloat16 b along its 12 rows,
   whose reads past the last row clamp to it. float16 beside bfloat16 keeps them
   off the matrix units. A chunk needs the operand's other strides to be
   multiples of 8. Placed dense, one of them is the cut extent itself, so each
   is broadcast instead, a along its rows and b along k, at stride 0. *)
let amd_examples =
  let plain = E.plain 2 in
  let broadcast axis = { plain with E.broadcast = [ axis ] } in
  List.map
    (fun (acc, a, b, (rows, cols, con)) ->
      lazy
        (make ~acc ~out:acc
           ([||], [| rows |], [| cols |], [| con |])
           a b None 1))
    D.
      [
        (Any Float32, (Any Float64, plain), (Any Float32, plain), (12, 10, 5));
        (Any Float64, (Any Int64, plain), (Any Float64, plain), (12, 10, 5));
        ( Any Float32,
          (Any Float16, broadcast 0),
          (Any Bfloat16, plain),
          (12, 16, 20) );
        ( Any Float32,
          (Any Float16, plain),
          (Any Bfloat16, broadcast 0),
          (12, 12, 20) );
      ]

(* rig.nv *)

let some = function Some y -> [| y |] | None -> [||]

(* Whether [c] sums on the matrix units, as nx_cuda.mli states: operands each
   bfloat16 or float8, or both float16, into float32. *)
let matrix_units (c : case) =
  let narrow (D.Any d) =
    match d with
    | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2 -> true
    | _ -> false
  in
  let a = dtype_of c.a and b = dtype_of c.b in
  c.acc = f32
  && ((narrow a && narrow b) || (a = D.Any D.Float16 && b = D.Any D.Float16))

(* Contractions of narrow floats into float32: half of them of two operands of
   one narrow dtype, which the matrix units take, half of float32 or a narrow
   float each. *)
let narrow_case =
  let narrow =
    D.[ Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2; Any Float16 ]
  in
  Gen.with_pp pp_case
    (Gen.one_of
       [
         case_of
           ~same:(Gen.of_list ~pp:pp_dtype narrow)
           ~acc:f32 ~out:f32 ~dts:narrow ();
         case_of ~acc:f32 ~out:f32 ~dts:narrow ();
       ])

(* rig.nv loads the cubin CUDA loads, so a contraction through it has CUDA's
   bits: what it can get wrong, its launches, memory and order, shows in them.
   Each case runs through CUDA, then through rig.nv. A NaN agrees with any NaN:
   nx_cuda.mli leaves a NaN result's sign and payload unspecified. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let agree call =
    Elements.agree (Support.opened first) (Support.opened nv) call
  in
  group nv.label
    [
      group "bits are cuda's, any NaN for a NaN"
        [
          prop ~examples:regime_examples "of float and complex contractions"
            (deferred (Gen.one_of [ finite_computed; computed ]))
            (fun c ->
              let c = Lazy.force c in
              cover "complex" (is_complex c.acc);
              cover "init" (Option.is_some c.init);
              agree (fun b -> some (run b c)));
          prop "of narrow floats' contractions into float32" narrow_case
            (fun c ->
              cover "on the matrix units" (matrix_units c);
              agree (fun b -> some (run b c)));
          prop "of contractions of any accumulator and result" any_case
            (fun c ->
              cover "an integer accumulator" (is_integer c.acc);
              agree (fun b -> some (run b c)));
          prop "of scaled contractions" scaled_case (fun (c, sa, sb) ->
              let sa = Option.map (near_one_scale (shape_of c.a)) sa in
              let sb = Option.map (near_one_scale (shape_of c.b)) sb in
              agree (fun b -> some (run ?sa ?sb b c)));
          prop "of segmented contractions" seg_case (fun s ->
              cover "rows" (match s.segments with Rows _ -> true | _ -> false);
              cover "contracting"
                (match s.segments with Contracting _ -> true | _ -> false);
              agree (fun b -> some (seg_run b s)));
          prop "of Contracting calls with 8-byte indices" k_seg_case (fun s ->
              Fun.protect
                ~finally:(fun () -> Nx_cuda.contract_wide_k wide_k)
                (fun () ->
                  Nx_cuda.contract_wide_k 0;
                  agree (fun b -> some (seg_run b s))));
        ];
    ]

(* nx.metal *)

(* The dtypes nx_metal.mli's contraction computes: operands both float32, both
   float16 or both bfloat16, a float32 accumulator, and a float32, float16 or
   bfloat16 result; and complex64 operands, accumulator and result. *)
let stage_dts = D.[ Any Float32; Any Float16; Any Bfloat16 ]

let staged c =
  c.acc = f32
  && List.mem (dtype_of c.a) stage_dts
  && dtype_of c.a = dtype_of c.b
  && List.mem c.out stage_dts
  || (c.acc = c64 && dtype_of c.a = c64 && dtype_of c.b = c64 && c.out = c64)

(* Cases of those dtypes into one of [outs], and complex64 ones. *)
let staged_into ?draw outs =
  let dts = Gen.of_list ~pp:pp_dtype stage_dts in
  let complex =
    case_of ?draw
      ~same:(Gen.constant ~pp:pp_dtype c64)
      ~acc:c64 ~out:c64 ~dts:[ c64 ] ()
  in
  Gen.with_pp pp_case
    (Gen.frequency
       [
         ( 3,
           Gen.bind (Gen.of_list ~pp:pp_dtype outs) (fun out ->
               case_of ?draw ~same:dts ~acc:f32 ~out ~dts:stage_dts ()) );
         (1, complex);
       ])

(* The bound holds before the rounding into the destination, so its cases store
   the accumulator's dtype; the bits laws take every result. *)
let staged_case = staged_into ~draw:finite_value stage_dts

(* The bound law's: finite ones, and of a few products with a NaN or an infinite
   operand at times, half each. *)
let bound_case =
  Gen.one_of
    [
      staged_into ~draw:finite_value [ f32 ];
      Gen.such_that few (staged_into [ f32 ]);
    ]

(* Whether the axes [axes] of [l] merge into one run of strides: each axis of
   more than one element strided by the next one's extent times its stride. *)
let one_run l axes =
  let runs = ref 0 and stride = ref 0 in
  List.iter
    (fun a ->
      let e = A.Layout.dim l a in
      if e > 1 then begin
        let s = A.Layout.stride l a in
        if !runs = 0 || !stride <> s * e then incr runs;
        stride := s
      end)
    axes;
  !runs <= 1

(* Whether [d]'s batch, row and column axes each do not merge into one run of
   strides: such a dst nx.metal computes in pieces. *)
let dst_apart (c, A.Any d) =
  let nb = Array.length c.batch in
  let fa = Array.length (shape_of c.a) - nb - Array.length c.contracting in
  let l = A.layout d in
  let r = A.Layout.rank l in
  let span lo hi = List.init (hi - lo) (fun i -> lo + i) in
  let off lo hi = not (one_run l (span lo hi)) in
  (off 0 nb, off nb (nb + fa), off (nb + fa) r)

let apart_covers x =
  let batch, rows, cols = dst_apart x in
  cover "batch apart" batch;
  cover "rows apart" rows;
  cover "columns apart" cols

let any_apart x =
  let batch, rows, cols = dst_apart x in
  cover "a dst apart" (batch || rows || cols)

(* The cases nx.metal's calls reach: each staged dtype, no output, no product,
   an init and each kind of view. *)
let staged_covers c =
  let y, k, _ = sizes c in
  cover "float16" (dtype_of c.a = D.Any D.Float16);
  cover "bfloat16" (dtype_of c.a = D.Any D.Bfloat16);
  cover "complex64" (c.acc = c64);
  cover "no output" (total y = 0);
  cover "no product" (k = 0 && total y > 0);
  cover "init" (c.init <> None);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [
      "stepped"; "windowed"; "reversed"; "broadcast"; "stored apart"; "permuted";
    ]

(* The dtypes nx.metal computes beside an integer accumulator: every one but
   float64 and complex128. *)
let metal_dts = List.filter (fun d -> d <> f64 && d <> c128) D.all

let metal_rows_case =
  Gen.with_pp pp_seg (seg_cases ~metal:true (Gen.constant true))

let metal_k_case =
  Gen.with_pp pp_seg (seg_cases ~metal:true (Gen.constant false))

(* A segmented contraction on nx.metal is each group's contraction of its rows,
   or of its contracted indices, alone, through every view of its operands, at
   each accumulator it computes. *)
let law_metal_segments (b : Support.backend) s =
  cover "an integer accumulator" (is_integer s.sc.acc);
  cover "complex64" (is_complex s.sc.acc);
  cover "init" (Option.is_some s.sc.init);
  segmented b s

(* Unscaled cases, of which nx.metal declines those of the dtypes it does not
   compute, and scaled ones of an integer or complex accumulator, which it
   declines. A scale with a complex acc is one case in six. *)
let metal_decline_cases =
  Gen.with_pp pp_scaled
    (Gen.one_of [ Gen.map (fun c -> (c, None, None)) any_case; unscalable ])

(* nx.metal's laws: those of the calls it computes, its bound that of its flush
   (nx_metal.mli), a dst through any view. Its examples are nx.cpu's regimes of
   the dtypes it computes, made when it is declared. *)
let metal_laws (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  let staged_examples =
    List.filter staged
      (regimes ~draw:finite_value f32 @ regimes ~draw:finite_value c64)
  in
  let bound_examples = List.filter (fun c -> c.out = c.acc) staged_examples in
  group d.label
    [
      prop ~examples:bound_examples
        "a contraction is within its bound of the exact sum" bound_case
        (run (law_bound ~covers:staged_covers));
      prop
        "an integer contraction is its products' sum modulo the accumulator, \
         cast once into dst"
        (integer_case metal_dts) (run law_integer);
      prop "a scaled contraction is within its bound of its terms' sum"
        (scaled_case_of [ f32 ]) (run law_scaled);
      prop "a Rows contraction is each group's contraction" metal_rows_case
        (run law_metal_segments);
      prop "a Contracting contraction is each group's contraction" metal_k_case
        (run law_metal_segments);
      prop "a destination through a view gets a fresh one's bits"
        (view_case_of staged_case)
        (run (law_dst_view ~covers:apart_covers));
      prop ~count:150 "a declined contraction writes nothing"
        metal_decline_cases (run law_declined);
      prop ~examples:staged_examples
        "a destination identical to init gives a fresh one's bits" staged_case
        (run (fun b c -> law_in_place b c));
      prop
        "a destination identical to init through a view gives a fresh one's \
         bits"
        (view_case_of staged_case)
        (run (fun b (c, v) -> law_in_place ~covers:any_apart ~view:v b c));
      test
        "a call of shapes that do not fit is not Done and writes nothing \
         outside dst"
        (run (fun b () -> test_misfits b ()));
      test "an acc a contraction does not compute in is not Done"
        (run (fun b () -> test_acc_refused b ()));
    ]

(* nx.cpu under each target table, then nx.cuda, then nx.amd, whose bound law
   covers its paths. *)
let () =
  let declared = List.filter_map Fun.id in
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.contract"
       (List.map
          (fun b -> laws ~examples:edge_examples (Support.declare b))
          Support.cpus
       @ List.map
           (laws ~examples:gpu_examples)
           (declared [ Support.declared_cuda ])
       @ List.map
           (laws ~paths:amd_paths ~examples:(gpu_examples @ amd_examples))
           (declared [ Support.declared_amd ])
       @ List.map cuda (declared [ Support.declared_cuda ])
       @ List.map amd (declared [ Support.declared_amd ])
       @ nv
       @ List.map metal_laws (declared [ Support.declared_metal ])))
