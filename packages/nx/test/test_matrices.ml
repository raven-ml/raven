(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Products and matrices, in nx.mli's order: dot, vdot and cross, the other
   products, then the functions of matrices. Each is checked against its
   elements computed in OCaml from its operands' by index, at every dtype it
   takes, over shapes with empty and one-element axes and over views. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let numel s = Array.fold_left ( * ) 1 s

(* The position in C order of the index [i] of shape [s]: {!index}'s inverse. *)
let ravel s i =
  let j = ref 0 in
  Array.iteri (fun a e -> j := (!j * s.(a)) + e) i;
  !j

let swap_last s =
  let r = Array.length s in
  let t = Array.copy s in
  t.(r - 2) <- s.(r - 1);
  t.(r - 1) <- s.(r - 2);
  t

(* Whether [x]'s memory holds its elements in C order. *)
let plain x =
  L.is_contiguous (A.layout (Option.get (Nx.Repr.array ~by:"test" x)))

let layout = Testable.make ~pp:L.pp ~equal:L.equal

(* Asserts that [y] reads [x]'s memory: that it is a view of [x]. *)
let reads_memory_of x y =
  let array v = Option.get (Nx.Repr.array ~by:"test" v) in
  let shared = Rig.Buffer.overlaps (A.buffer (array x)) (A.buffer (array y)) in
  satisfies ~claim:"a layout over x's memory" layout
    (fun _ -> shared)
    (A.layout (array y))

let rec sequence = function
  | [] -> Gen.constant []
  | g :: gs ->
      let open Gen in
      let* x = g in
      let+ xs = sequence gs in
      x :: xs

let cx re im = { Complex.re; im }

(* [1 + i, 2i] and [2, 1 + i]: [Σ conj(a) b] is [4 - 4i], [Σ a b] is [4i]. *)
let complex_a = lazy (Nx.create Nx.complex64 [| 2 |] [| cx 1. 1.; cx 0. 2. |])
let complex_b = lazy (Nx.create Nx.complex64 [| 2 |] [| cx 2. 0.; cx 1. 1. |])

(* Drawn operands *)

(* A host value of [dt] and shape [s] over drawn bytes: NaN payloads, [-0.],
   subnormals and integer extremes among them. *)
let drawn (type v s) (dt : (v, s) D.t) s : (v, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt (numel s))) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt
       (L.contiguous (Iarray.of_array s))
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

type batch = Batch : ('v, 's) D.t * ('v, 's, Nx.host) Nx.t -> batch

let pp_batch ppf (Batch (dt, x)) =
  Format.fprintf ppf "%a %a%s" D.pp dt pp_ints (Nx.shape x)
    (if plain x then "" else " (a view)")

let pp_any ppf (D.Any dt) = D.pp ppf dt

(* Values of every dtype and 2 or 3 axes of extents 0 to 4, or, half the time,
   {!Nx_support.bits}' matrices of a byte or more laid out every way:
   transposed, flipped, strided, broadcast and as overlapping windows. *)
let matrices =
  let open Gen in
  let drawn_batch =
    let* (D.Any dt) = of_list ~pp:pp_any D.all in
    let* s =
      array ~size:(int_range 2 3)
        (frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 4) ])
    in
    let+ x = drawn dt s in
    Batch (dt, x)
  in
  let view =
    map (fun (Bits a) -> Batch (A.dtype a, Nx.Repr.of_array Nx.Host.v a)) bits
  in
  with_pp pp_batch (one_of [ drawn_batch; view ])

(* Exact integer operands *)

(* The products' laws draw a dtype that is not boolean and small integers, exact
   in every format; an operand is those integers cast to the dtype, which keeps
   an integer modulo its width. The reference computes in OCaml's [int], and the
   result of the dtype is that integer cast: a ring map for integers, and one
   rounding of an exact value for the floats, which accumulate in a wider format
   exactly and round once. *)
type numeric = Numeric : ('v, 's) D.t -> numeric

let pp_numeric ppf (Numeric dt) = D.pp ppf dt

let numeric =
  Gen.of_list ~pp:pp_numeric
    (List.filter_map
       (fun (D.Any dt) -> if D.is D.Boolean dt then None else Some (Numeric dt))
       D.all)

let of_ints (type v s) (dt : (v, s) D.t) s xs : (v, s, Nx.host) Nx.t =
  Nx.cast dt (Nx.create Nx.int64 s (Array.map Int64.of_int xs))

(* [of_ints dt s xs] laid out with its last two axes exchanged in memory: a
   view, whose elements in C order are still [xs]. *)
let transposed dt s xs =
  if Array.length s < 2 then of_ints dt s xs
  else
    let t = swap_last s in
    let ys =
      Array.init (numel t) (fun j -> xs.(ravel s (swap_last (index t j))))
    in
    Nx.matrix_transpose (of_ints dt t ys)

let ints s = Gen.array ~size:(Gen.constant (numel s)) (Gen.int_range (-2) 2)

let extent =
  Gen.frequency
    [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 3) ]

(* Equal as numbers: a product of zero takes IEEE 754's sign, which an integer
   reference has none of. *)
let numbers (type v s) (dt : (v, s) D.t) =
  let same (a : v) (b : v) =
    match D.kind dt with
    | D.Float -> a = b
    | D.Complex -> a.Complex.re = b.Complex.re && a.im = b.im
    | D.Signed | D.Unsigned | D.Boolean -> a = b
  in
  Testable.make
    ~pp:(Format.pp_print_list (D.pp_value dt))
    ~equal:(fun a b -> List.length a = List.length b && List.for_all2 same a b)

let equal_numbers dt a b =
  equal (numbers dt) (Array.to_list a) (Array.to_list b)

(* [y] against the reference [want] of shape [s], cast to [y]'s dtype. *)
let agrees (type v s) (dt : (v, s) D.t) s want (y : (v, s, Nx.host) Nx.t) =
  equal ~msg:"shape" (array int) s (Nx.shape y);
  equal_numbers dt (Nx.to_array (of_ints dt s want)) (Nx.to_array y)

(* The sum over [pairs] of [a]'s axes against [b]'s, with no batch: the result's
   axes [a]'s others then [b]'s. *)
let reference_pairs sa xa sb xb pairs =
  let ra = Array.length sa and rb = Array.length sb in
  let la = List.map fst pairs and lb = List.map snd pairs in
  let fa = List.filter (fun i -> not (List.mem i la)) (List.init ra Fun.id) in
  let fb = List.filter (fun i -> not (List.mem i lb)) (List.init rb Fun.id) in
  let out =
    Array.of_list (List.map (Array.get sa) fa @ List.map (Array.get sb) fb)
  in
  let summed = Array.of_list (List.map (fun (i, _) -> sa.(i)) pairs) in
  let want =
    Array.init (numel out) (fun j ->
        let o = index out j in
        let acc = ref 0 in
        for t = 0 to numel summed - 1 do
          let u = index summed t in
          let ia = Array.make ra 0 and ib = Array.make rb 0 in
          List.iteri (fun k i -> ia.(i) <- o.(k)) fa;
          List.iteri (fun k i -> ib.(i) <- o.(List.length fa + k)) fb;
          List.iteri
            (fun k (i, i') ->
              ia.(i) <- u.(k);
              ib.(i') <- u.(k))
            pairs;
          acc := !acc + (xa.(ravel sa ia) * xb.(ravel sb ib))
        done;
        !acc)
  in
  (out, want)

let pp_floats ppf xs =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%h") xs)))

let transposed_mark view = if view then " (transposed)" else ""

(* Two integer operands, [a] laid out transposed or not. *)
let pp_pair ppf (dt, view, sa, xa, sb, xb) =
  Format.fprintf ppf "%a a %a%s %a, b %a %a" pp_numeric dt pp_ints sa
    (transposed_mark view) pp_ints xa pp_ints sb pp_ints xb

(* dot *)

(* [a] of 1 to 3 axes; [b] a vector of [a]'s last extent, or of 2 or 3 axes
   summed along its second to last; [a] laid out transposed one case in two. *)
let dot_case =
  let open Gen in
  let* dt = numeric in
  let* k = extent in
  let* la = array ~size:(int_range 0 2) extent in
  let* rb = int_range 1 3 in
  let* lb = array ~size:(constant (rb - 1)) extent in
  let sa = Array.append la [| k |] in
  let sb =
    if rb = 1 then [| k |]
    else Array.concat [ Array.sub lb 0 (rb - 2); [| k |]; [| lb.(rb - 2) |] ]
  in
  let* view = bool in
  let* xa = ints sa in
  let+ xb = ints sb in
  (dt, view, sa, xa, sb, xb)

let dot_case = Gen.with_pp pp_pair dot_case

let law_dot (Numeric dt, view, sa, xa, sb, xb) =
  let ra = Array.length sa and rb = Array.length sb in
  let kind =
    match D.kind dt with
    | D.Float -> "float"
    | D.Complex -> "complex"
    | D.Signed -> "signed"
    | D.Unsigned -> "unsigned"
    | D.Boolean -> "boolean"
  in
  cover "a float dtype" (kind = "float");
  cover "a complex dtype" (kind = "complex");
  cover "a signed dtype" (kind = "signed");
  cover "an unsigned dtype" (kind = "unsigned");
  cover "a vector b" (rb = 1);
  cover "an empty sum" (sa.(ra - 1) = 0);
  cover "a transposed a" (view && ra >= 2);
  let a = (if view then transposed else of_ints) dt sa xa in
  let summed = if rb = 1 then 0 else rb - 2 in
  let out, want = reference_pairs sa xa sb xb [ (ra - 1, summed) ] in
  agrees dt out want (Nx.dot a (of_ints dt sb xb))

let dot =
  let z s = Nx.zeros Nx.int32 s and b s = Nx.zeros Nx.bool s in
  group "dot"
    [
      prop ~count:200
        "sums a's last axis with b's only or second to last, a's others then \
         b's"
        dot_case law_dot;
      cases
        "raises where an operand is 0-d, the summed extents differ, or for \
         booleans"
        ~id:fst
        [
          ("a 0-d a", fun () -> ignore (Nx.dot (z [||]) (z [| 2 |])));
          ("a 0-d b", fun () -> ignore (Nx.dot (z [| 2 |]) (z [||])));
          ( "a vector b of another extent",
            fun () -> ignore (Nx.dot (z [| 2; 3 |]) (z [| 2 |])) );
          ( "a matrix b of another extent",
            fun () -> ignore (Nx.dot (z [| 2 |]) (z [| 3; 2 |])) );
          ("booleans", fun () -> ignore (Nx.dot (b [| 2 |]) (b [| 2 |])));
        ]
        (fun (_, f) -> invalid ~by:"Nx.dot" f);
    ]

(* vdot *)

(* Operands of one dtype other than bool and equal numbers of elements, of
   shapes of their own, 0-d and empty ones among them, one of [[2; m]] laid out
   transposed, and whether to pass [~sum:true]. *)
type flat_pair =
  | Flat_pair :
      ('v, 's) D.t * ('v, 's, Nx.host) Nx.t * ('v, 's, Nx.host) Nx.t * bool
      -> flat_pair

let flat_pairs =
  let pp_value ppf x =
    Format.fprintf ppf "%a%s" pp_ints (Nx.shape x)
      (if plain x then "" else " (a view)")
  in
  Gen.with_pp
    (fun ppf (Flat_pair (dt, a, b, sum)) ->
      Format.fprintf ppf "%a %a %a ~sum:%b" D.pp dt pp_value a pp_value b sum)
    (let open Gen in
     let* (Numeric dt) = numeric in
     let* n =
       frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 17) ]
     in
     let shape =
       of_list
         ((if n = 1 then [ [||] ] else [])
         @ [ [| n |]; [| 1; n |]; [| n; 1 |] ]
         @ if n mod 2 = 0 then [ [| 2; n / 2 |]; [| n / 2; 1; 2 |] ] else [])
     in
     let value s =
       if Array.length s = 2 && s.(0) = 2 && s.(1) > 1 then
         map Nx.matrix_transpose (drawn dt [| s.(1); 2 |])
       else drawn dt s
     in
     let* sa = shape in
     let* sb = shape in
     let* a = value sa in
     let* b = value sb in
     let+ sum = bool in
     Flat_pair (dt, a, b, sum))

(* The same bits, NaNs as one: vdot is the matmul of [conjugate a] and [b], each
   flattened, here from a plain copy of its elements in C order. *)
let law_vdot (Flat_pair (dt, a, b, sum)) =
  cover "a 0-d operand" (Nx.shape a = [||] || Nx.shape b = [||]);
  cover "no element" (Nx.numel a = 0);
  cover "a complex dtype" (match D.kind dt with Complex -> true | _ -> false);
  cover "a transposed operand" (not (plain a && plain b));
  let flat x = Nx.create dt [| Nx.numel x |] (Nx.to_array x) in
  let v = Nx.vdot ~sum a b in
  equal (array int) [||] (Nx.shape v);
  equal_elements dt
    (Nx.to_array (Nx.matmul ~sum (Nx.conjugate (flat a)) (flat b)))
    (Nx.to_array v)

(* What vdot raises, word for word. *)
let test_vdot_refusals () =
  let message f =
    match f () with _ -> "no exception" | exception Invalid_argument m -> m
  in
  expect
    (String.concat "\n"
       [
         message (fun () ->
             Nx.vdot (Nx.zeros Nx.int32 [| 2 |]) (Nx.zeros Nx.int32 [| 3 |]));
         message (fun () ->
             Nx.vdot (Nx.zeros Nx.bool [| 2 |]) (Nx.zeros Nx.bool [| 2 |]));
         message (fun () ->
             Nx.vdot (Nx.zeros Nx.bool [| 2 |]) (Nx.zeros Nx.bool [| 1 |]));
       ])
  @@ __POS_OF__
       {|
         Nx.vdot: int32 [2] and int32 [3] differ in their number of elements
         Nx.vdot: the result's dtype is bool
         Nx.vdot: bool [2] and bool [1] differ in their number of elements
       |}

let vdot =
  group "vdot"
    [
      prop ~count:1000 "is the matmul of conjugate a and b, both flattened"
        flat_pairs law_vdot;
      test "conjugates a" (fun () ->
          let b = Nx.reshape [| 1; 2 |] (Lazy.force complex_b) in
          equal_elements D.Complex64
            [| cx 4. (-4.) |]
            (Nx.to_array (Nx.vdot (Lazy.force complex_a) b)));
      test "of no elements is +0" (fun () ->
          let a = Nx.zeros Nx.float32 [| 0 |] in
          let b = Nx.zeros Nx.float32 [| 0; 3 |] in
          equal (array float_exact) [| 0. |] (Nx.to_array (Nx.vdot a b)));
      test "raises for operands of other numbers of elements, or for booleans"
        test_vdot_refusals;
    ]

(* cross *)

(* Operands of 3-vectors along one axis, their other axes suffixes of one shape
   with some extents 1, so that they broadcast; the axis [q] from the end,
   spelled negative or positive. *)
let cross_case =
  let open Gen in
  let* dt = numeric in
  let* lead = array ~size:(int_range 0 2) extent in
  let l = Array.length lead in
  let side =
    let* drop = int_range 0 l in
    let+ ones = array ~size:(constant l) bool in
    Array.init (l - drop) (fun i ->
        if ones.(i + drop) then 1 else lead.(i + drop))
  in
  let* la = side in
  let* lb = side in
  let* q = int_range 0 (Stdlib.min (Array.length la) (Array.length lb)) in
  let ins s =
    let p = Array.length s - q in
    Array.concat [ Array.sub s 0 p; [| 3 |]; Array.sub s p q ]
  in
  let sa = ins la and sb = ins lb in
  let r = 1 + Stdlib.max (Array.length la) (Array.length lb) in
  let* negative = bool in
  let axis = if negative then -(q + 1) else r - 1 - q in
  let* xa = ints sa in
  let+ xb = ints sb in
  (dt, axis, sa, xa, sb, xb)

let cross_case =
  Gen.with_pp
    (fun ppf (dt, axis, sa, xa, sb, xb) ->
      Format.fprintf ppf "%a ~axis:%d a %a %a, b %a %a" pp_numeric dt axis
        pp_ints sa pp_ints xa pp_ints sb pp_ints xb)
    cross_case

let law_cross (Numeric dt, axis, sa, xa, sb, xb) =
  let r = Stdlib.max (Array.length sa) (Array.length sb) in
  let pad s = Array.append (Array.make (r - Array.length s) 1) s in
  let pa = pad sa and pb = pad sb in
  let out = Array.init r (fun i -> if pa.(i) = 1 then pb.(i) else pa.(i)) in
  let ax = if axis < 0 then r + axis else axis in
  (* The element of [x], of padded shape [p], at the result's index [o] with [c]
     along the axis. *)
  let at p x o c =
    let i =
      Array.mapi (fun j e -> if j = ax then c else if p.(j) = 1 then 0 else e) o
    in
    x.(ravel p i)
  in
  let want =
    Array.init (numel out) (fun j ->
        let o = index out j in
        let a k = at pa xa o ((o.(ax) + k) mod 3)
        and b k = at pb xb o ((o.(ax) + k) mod 3) in
        (a 1 * b 2) - (a 2 * b 1))
  in
  cover "an axis other than the last" (ax < r - 1);
  cover "a negative axis" (axis < 0);
  cover "a broadcast operand" (pa <> pb);
  cover "no vector" (numel out = 0);
  cover "an unsigned component below zero, which wraps"
    ((match D.kind dt with D.Unsigned -> true | _ -> false)
    && Array.exists (fun w -> w < 0) want);
  agrees dt out want (Nx.cross ~axis (of_ints dt sa xa) (of_ints dt sb xb))

(* The position of [x] among the floats of [dt] in increasing order, [+0.] at 0
   and [-0.] at -1, from its sign and magnitude bits: nx.mli counts ulps as
   distances between positions. *)
let rank (type s) (dt : (float, s) D.t) x =
  let v = Nx.create dt [| 1 |] [| x |] in
  let bits : int64 =
    match D.bits dt with
    | 64 -> Nx.item [ 0 ] (Nx.bitcast Nx.int64 v)
    | 32 -> Nx.item [ 0 ] (Nx.cast Nx.int64 (Nx.bitcast Nx.uint32 v))
    | 16 -> Nx.item [ 0 ] (Nx.cast Nx.int64 (Nx.bitcast Nx.uint16 v))
    | 8 -> Nx.item [ 0 ] (Nx.cast Nx.int64 (Nx.bitcast Nx.uint8 v))
    | _ -> Nx.item [ 0 ] (Nx.cast Nx.int64 (Nx.bitcast Nx.uint4 v))
  in
  let sign = Int64.shift_left 1L (D.bits dt - 1) in
  if Int64.logand bits sign = 0L then bits
  else Int64.(sub (neg (logand bits (sub sign 1L))) 1L)

(* [x y - u v] as the unevaluated sum [s + t] of two floats, within about
   [2^-104 max |x y| |u v|] of it: each product split exactly by a fused
   multiply-add, their difference by Knuth's two-sum. Exact where the products
   are, as of floats of at most 26 bits of precision. *)
let difference x y u v =
  let p = x *. y and q = u *. v in
  let pe = Float.fma x y (-.p) and qe = Float.fma u v (-.q) in
  let s = p -. q in
  let z = s -. p in
  let e = p -. (s -. z) +. (-.q -. z) in
  (s, e +. (pe -. qe))

(* Component [i] of [a × b] for the 3-vectors at [a.(o)] and [b.(o)]. *)
let component a b o i =
  let at v k = v.(o + ((i + k) mod 3)) in
  difference (at a 1) (at b 2) (at a 2) (at b 1)

(* [s + e] rounded to [dt], ties to even, for an [s] that is [s + e] rounded to
   float64: rounding [s] again errs only where [s] is a tie of [dt], half way
   from [c] to the [n] beyond it, which [e] then breaks. *)
let rounded (type s) (dt : (float, s) D.t) (s, e) =
  let cast x =
    Nx.item [] (Nx.cast Nx.float64 (Nx.cast dt (Nx.scalar Nx.float64 x)))
  in
  let c = cast s in
  let n = s +. (s -. c) in
  if c = s then cast (s +. e)
  else if e = 0. || cast n <> n then c
  else if e > 0. = (n > c) then n
  else c

(* Vectors of [k] 3-vectors with entries of magnitude [2^lo] to [2^hi], half of
   them pairs within [2^-nudge] of parallel, whose components cancel in about
   [nudge] bits. *)
let vectors ~lo ~hi ~nudge =
  let open Gen in
  let* k = frequency [ (1, constant 0); (4, int_range 1 4) ] in
  let entry =
    let+ m = float_range 0.5 1. and+ e = int_range lo hi and+ neg = bool in
    Float.ldexp (if neg then -.m else m) e
  in
  let* a = array ~size:(constant (3 * k)) entry in
  let* parallel = bool and+ scale = float_range 0.5 2. in
  let+ b =
    if parallel then
      let+ wobble = array ~size:(constant (3 * k)) (float_range (-1.) 1.) in
      Array.mapi
        (fun j x -> x *. scale *. (1. +. Float.ldexp wobble.(j) (-nudge)))
        a
    else array ~size:(constant (3 * k)) entry
  in
  (k, parallel, a, b)

let vectors ~lo ~hi ~nudge =
  Gen.with_pp
    (fun ppf (_, parallel, a, b) ->
      Format.fprintf ppf "a %a, b %a%s" pp_floats a pp_floats b
        (if parallel then " (nearly parallel)" else ""))
    (vectors ~lo ~hi ~nudge)

(* [cross a b] of [dt], from [a] and [b] rounded to [dt], within [ulps] of the
   exact components rounded to [dt] wherever they fit it: the count of nonzero
   ones checked. *)
let within (type s) (dt : (float, s) D.t) ulps (k, _, a, b) =
  let on xs = Nx.cast dt (Nx.create Nx.float64 [| k; 3 |] xs) in
  let ta = on a and tb = on b in
  let a = Nx.to_array (Nx.cast Nx.float64 ta)
  and b = Nx.to_array (Nx.cast Nx.float64 tb) in
  let got = Nx.to_array (Nx.cast Nx.float64 (Nx.cross ta tb)) in
  let f = D.float_format dt in
  let checked = ref 0 in
  for o = 0 to k - 1 do
    for i = 0 to 2 do
      let s, t = component a b (3 * o) i in
      if Float.abs (s +. t) <= f.max_finite then begin
        let w = rounded dt (s, t) and v = got.((3 * o) + i) in
        if w <> 0. then incr checked;
        let apart = Int64.abs (Int64.sub (rank dt v) (rank dt w)) in
        if apart > ulps then
          failf "%a, vector %d component %d: %h, %Ld floats from %h (%h + %h)"
            D.pp dt o i v apart w s t
      end
    done
  done;
  !checked

type narrow = Narrow : (float, 's) D.t -> narrow

let pp_narrow ppf (Narrow dt) = D.pp ppf dt

let narrows =
  [
    Narrow D.Float32;
    Narrow D.Float16;
    Narrow D.Bfloat16;
    Narrow D.Float8_e4m3fn;
    Narrow D.Float8_e5m2;
    Narrow D.Float4_e2m1fn;
  ]

(* A format, then vectors with entries within the square root of its range, so
   that most components fit, its parallel pairs within half its precision. *)
let narrow_vectors =
  let open Gen in
  let* (Narrow dt) = of_list ~pp:pp_narrow narrows in
  let f = D.float_format dt in
  let _, top = Float.frexp f.max_finite in
  let _, bottom = Float.frexp f.min_normal in
  let+ v =
    vectors ~lo:(bottom / 2)
      ~hi:((top / 2) - 1)
      ~nudge:(Stdlib.max 1 (f.fraction_bits / 2))
  in
  (Narrow dt, v)

let narrow_vectors =
  Gen.with_pp
    (fun ppf (Narrow dt, (_, _, a, b)) ->
      Format.fprintf ppf "%a a %a, b %a (before rounding)" D.pp dt pp_floats a
        pp_floats b)
    narrow_vectors

let law_cross_narrow (Narrow dt, ((k, _, _, _) as v)) =
  List.iter
    (fun (Narrow f) -> cover ("at " ^ D.name f) (D.name f = D.name dt))
    narrows;
  let checked = within dt 1L v in
  cover "a nonzero component checked" (k > 0 && checked > 0)

let law_cross_float64 ((k, parallel, _, _) as v) =
  cover "nearly parallel vectors" (k > 0 && parallel);
  cover "no vector" (k = 0);
  ignore (within D.Float64 2L v)

(* IEEE 754's difference where a product is infinite or the difference is zero:
   [1 - inf], [inf 0], and [-0 - +0]. *)
let specials (type s) (dt : (float, s) D.t) =
  let at xs = Nx.cast dt (Nx.create Nx.float64 [| 3 |] xs) in
  let got a b = Nx.to_array (Nx.cast Nx.float64 (Nx.cross (at a) (at b))) in
  equal_elements D.Float64
    [| Float.neg_infinity; Float.nan; 0. |]
    (got [| 0.; 1.; Float.infinity |] [| 0.; 1.; 1. |]);
  equal_elements D.Float64 [| -0.; 0.; 0. |]
    (got [| 0.; -1.; 0. |] [| 0.; 1.; 0. |])

let cross =
  let z s = Nx.zeros Nx.float32 s in
  group "cross"
    [
      prop ~count:200
        ~examples:
          [ (Numeric D.Int8, -1, [| 0; 3 |], [||], [| 3 |], [| 1; 2; 3 |]) ]
        "component i is a(i+1) b(i+2) - a(i+2) b(i+1) along axis of the \
         broadcast shape"
        cross_case law_cross;
      prop ~count:200
        "at a float format of at most 32 bits is within one ulp of its exact \
         components, finite wherever they fit"
        narrow_vectors law_cross_narrow;
      prop "at float64 is within 2 ulps of its exact components"
        (vectors ~lo:(-20) ~hi:20 ~nudge:20)
        law_cross_float64;
      cases "a product past the format's range gives the difference that fits"
        ~id:(fun (Narrow dt, _) -> D.name dt)
        [
          (Narrow D.Float16, 0x1p8);
          (Narrow D.Bfloat16, 0x1p64);
          (Narrow D.Float32, 0x1p64);
        ]
        (fun (Narrow dt, big) ->
          (* For [next] the float after the power of two [big], component 0 is
             [big next - big big = big² ε], both products past the format's
             largest float. *)
          let eps = (D.float_format dt).epsilon in
          let v xs = Nx.create dt [| 3 |] xs in
          let a = v [| 0.; big; big |]
          and b = v [| 0.; big; big *. (1. +. eps) |] in
          equal_elements dt
            [| big *. big *. eps; 0.; 0. |]
            (Nx.to_array (Nx.cross a b)));
      test
        "at float64, products past the range give their difference's infinity"
        (fun () ->
          let v xs = Nx.create Nx.float64 [| 3 |] xs in
          equal_elements D.Float64
            [| Float.neg_infinity; 0.; 0. |]
            (Nx.to_array
               (Nx.cross (v [| 0.; 1.; 1e200 |]) (v [| 0.; 1e200; 1. |]))));
      cases
        "gives IEEE 754's difference of an infinite product, and of a zero \
         difference its sign"
        ~id:fst
        [
          ("float32", fun () -> specials D.Float32);
          ("float64", fun () -> specials D.Float64);
        ]
        (fun (_, f) -> f ());
      cases
        "raises where axis is not one of the broadcast shape's, an operand has \
         no 3 elements along it, or for booleans"
        ~id:fst
        [
          ("2 elements", fun () -> ignore (Nx.cross (z [| 2 |]) (z [| 2 |])));
          ( "one operand broadcast along the axis",
            fun () -> ignore (Nx.cross (z [| 2; 3 |]) (z [| 2; 1 |])) );
          ( "an operand without the axis",
            fun () -> ignore (Nx.cross ~axis:0 (z [| 3; 3 |]) (z [| 3 |])) );
          ( "an axis past the shape",
            fun () -> ignore (Nx.cross ~axis:1 (z [| 3 |]) (z [| 3 |])) );
          ( "an axis before the shape",
            fun () -> ignore (Nx.cross ~axis:(-2) (z [| 3 |]) (z [| 3 |])) );
          ( "bool",
            fun () ->
              ignore
                (Nx.cross (Nx.zeros Nx.bool [| 3 |]) (Nx.zeros Nx.bool [| 3 |]))
          );
          ( "bit",
            fun () ->
              ignore
                (Nx.cross (Nx.zeros Nx.bit [| 3 |]) (Nx.zeros Nx.bit [| 3 |]))
          );
        ]
        (fun (_, f) -> invalid ~by:"Nx.cross" f);
    ]

(* Other products *)

(* A row per function for the booleans of [dt]. *)
let refusals_of_booleans (type s) name (dt : (bool, s) D.t) =
  let z s = Nx.zeros dt s in
  [
    ( name ^ ", tensordot",
      "Nx.tensordot",
      fun () ->
        ignore (Nx.tensordot ~axes:([ 0 ], [ 0 ]) (z [| 2 |]) (z [| 2 |])) );
    ( name ^ ", inner",
      "Nx.inner",
      fun () -> ignore (Nx.inner (z [| 2 |]) (z [| 2 |])) );
    ( name ^ ", inner of a 0-d operand",
      "Nx.inner",
      fun () -> ignore (Nx.inner (z [||]) (z [| 2 |])) );
    ( name ^ ", outer",
      "Nx.outer",
      fun () -> ignore (Nx.outer (z [| 2 |]) (z [| 2 |])) );
    ( name ^ ", vecdot",
      "Nx.vecdot",
      fun () -> ignore (Nx.vecdot (z [| 2 |]) (z [| 2 |])) );
    ( name ^ ", kron",
      "Nx.kron",
      fun () -> ignore (Nx.kron (z [| 2 |]) (z [| 2 |])) );
    ( name ^ ", multi_dot",
      "Nx.multi_dot",
      fun () -> ignore (Nx.multi_dot [ z [| 2; 2 |]; z [| 2; 2 |] ]) );
    ( name ^ ", matrix_power",
      "Nx.matrix_power",
      fun () -> ignore (Nx.matrix_power (z [| 2; 2 |]) 0) );
  ]

let other_products =
  group "Other products"
    [
      cases "each raises for booleans"
        ~id:(fun (id, _, _) -> id)
        (refusals_of_booleans "bool" D.Bool @ refusals_of_booleans "bit" D.Bit)
        (fun (_, by, f) -> invalid ~by f);
    ]

(* tensordot *)

type tensordot_case = {
  td_dt : numeric;
  sa : int array;
  sb : int array;
  pairs : (int * int) list;
  axes : int list * int list;
  default : bool;
  view : bool;
  xa : int array;
  xb : int array;
}

(* Axes paired in a drawn order, [b]'s placed among its free axes, each spelled
   from either end; one case in three takes the default, [a]'s last two against
   [b]'s first two. [a] is laid out transposed one case in two. *)
let tensordot_case =
  let open Gen in
  with_pp
    (fun ppf c ->
      Format.fprintf ppf "%a %a%s . %a over %s%s" pp_numeric c.td_dt pp_ints
        c.sa
        (if c.view then " (transposed)" else "")
        pp_ints c.sb
        (String.concat ", "
           (List.map2 (Printf.sprintf "%d-%d") (fst c.axes) (snd c.axes)))
        (if c.default then " (default)" else ""))
    (let* td_dt = numeric in
     let* default = frequency [ (1, constant true); (2, constant false) ] in
     let* sa = array ~size:(int_range (if default then 2 else 0) 3) extent in
     let ra = Array.length sa in
     let* c = if default then constant 2 else int_range 0 (Stdlib.min ra 2) in
     let* order = permutation (List.init ra Fun.id) in
     let la =
       if default then [ ra - 2; ra - 1 ]
       else List.filteri (fun k _ -> k < c) order
     in
     let* free = array ~size:(int_range 0 (3 - c)) extent in
     let rb = c + Array.length free in
     let* border = permutation (List.init rb Fun.id) in
     let lb =
       if default then [ 0; 1 ] else List.filteri (fun k _ -> k < c) border
     in
     let sb = Array.make rb 0 in
     List.iter2 (fun i j -> sb.(j) <- sa.(i)) la lb;
     let rest =
       List.filter (fun j -> not (List.mem j lb)) (List.init rb Fun.id)
     in
     List.iteri (fun k j -> sb.(j) <- free.(k)) rest;
     let spell r l =
       let+ negative = array ~size:(constant (List.length l)) bool in
       List.mapi (fun k i -> if negative.(k) then i - r else i) l
     in
     let* na = spell ra la in
     let* nb = spell rb lb in
     let* view = bool in
     let* xa = ints sa in
     let+ xb = ints sb in
     {
       td_dt;
       sa;
       sb;
       pairs = List.combine la lb;
       axes = (na, nb);
       default;
       view;
       xa;
       xb;
     })

let law_tensordot c =
  let (Numeric dt) = c.td_dt in
  cover "the default axes" c.default;
  cover "no axis paired" (c.pairs = []);
  cover "an empty sum" (List.exists (fun (i, _) -> c.sa.(i) = 0) c.pairs);
  cover "a negative axis"
    ((not c.default) && List.exists (fun i -> i < 0) (fst c.axes @ snd c.axes));
  cover "a transposed a" (c.view && Array.length c.sa >= 2);
  let a = (if c.view then transposed else of_ints) dt c.sa c.xa in
  let b = of_ints dt c.sb c.xb in
  let y =
    if c.default then Nx.tensordot a b else Nx.tensordot ~axes:c.axes a b
  in
  let out, want = reference_pairs c.sa c.xa c.sb c.xb c.pairs in
  agrees dt out want y

let tensordot =
  let z s = Nx.zeros Nx.int32 s in
  group "tensordot"
    [
      prop ~count:200
        "sums the products over the axes paired in order, a's others then b's"
        tensordot_case law_tensordot;
      test "takes a as it is" (fun () ->
          equal_elements D.Complex64
            [| cx 0. 4. |]
            (Nx.to_array
               (Nx.tensordot ~axes:([ 0 ], [ 0 ]) (Lazy.force complex_a)
                  (Lazy.force complex_b))));
      cases
        "raises where the lists differ in length, an axis is not its operand's \
         or repeats, or paired extents differ"
        ~id:fst
        [
          ( "default axes of a vector",
            fun () -> ignore (Nx.tensordot (z [| 2 |]) (z [| 2; 2 |])) );
          ( "lists of two lengths",
            fun () ->
              ignore (Nx.tensordot ~axes:([ 0 ], []) (z [| 2 |]) (z [| 2 |])) );
          ( "an axis past a",
            fun () ->
              ignore (Nx.tensordot ~axes:([ 1 ], [ 0 ]) (z [| 2 |]) (z [| 2 |]))
          );
          ( "a repeated axis of a",
            fun () ->
              ignore
                (Nx.tensordot
                   ~axes:([ 0; 0 ], [ 0; 1 ])
                   (z [| 2; 2 |])
                   (z [| 2; 2 |])) );
          ( "a repeated axis of b, spelled from both ends",
            fun () ->
              ignore
                (Nx.tensordot
                   ~axes:([ 0; 1 ], [ 1; -1 ])
                   (z [| 2; 2 |])
                   (z [| 2; 2 |])) );
          ( "extents differ",
            fun () ->
              ignore (Nx.tensordot ~axes:([ 0 ], [ 0 ]) (z [| 2 |]) (z [| 3 |]))
          );
        ]
        (fun (_, f) -> invalid ~by:"Nx.tensordot" f);
    ]

(* inner *)

(* Two operands sharing a last extent, either of them possibly 0-d; [a] laid out
   transposed one case in two. *)
let inner_case =
  let open Gen in
  let* dt = numeric in
  let* k = extent in
  let* la = array ~size:(int_range 0 2) extent in
  let* lb = array ~size:(int_range 0 2) extent in
  let* za = frequency [ (1, constant true); (5, constant false) ] in
  let* zb = frequency [ (1, constant true); (5, constant false) ] in
  let sa = if za then [||] else Array.append la [| k |] in
  let sb = if zb then [||] else Array.append lb [| k |] in
  let* view = bool in
  let* xa = ints sa in
  let+ xb = ints sb in
  (dt, view, sa, xa, sb, xb)

let inner_case = Gen.with_pp pp_pair inner_case

let law_inner (Numeric dt, view, sa, xa, sb, xb) =
  let ra = Array.length sa and rb = Array.length sb in
  cover "a 0-d operand" (ra = 0 || rb = 0);
  cover "an empty last axis" (ra > 0 && rb > 0 && sa.(ra - 1) = 0);
  cover "a transposed a" (view && ra >= 2);
  let a = (if view then transposed else of_ints) dt sa xa in
  let y = Nx.inner a (of_ints dt sb xb) in
  if ra = 0 || rb = 0 then
    let s = if ra = 0 then sb else sa in
    let c = if ra = 0 then xa.(0) else xb.(0) in
    agrees dt s (Array.map (( * ) c) (if ra = 0 then xb else xa)) y
  else
    let out, want = reference_pairs sa xa sb xb [ (ra - 1, rb - 1) ] in
    agrees dt out want y

let inner =
  group "inner"
    [
      prop ~count:200
        ~examples:[ (Numeric D.Int32, false, [| 2; 0 |], [||], [| 0 |], [||]) ]
        "sums a's last axis with b's, a's others then b's; a 0-d operand gives \
         mul"
        inner_case law_inner;
      test "takes a as it is" (fun () ->
          equal_elements D.Complex64
            [| cx 0. 4. |]
            (Nx.to_array
               (Nx.inner (Lazy.force complex_a) (Lazy.force complex_b))));
      test "raises where the last extents differ" (fun () ->
          invalid ~by:"Nx.inner" (fun () ->
              Nx.inner (Nx.zeros Nx.int32 [| 2 |]) (Nx.zeros Nx.int32 [| 3 |])));
    ]

(* outer *)

(* Two operands of 0 to 3 axes; [a] laid out transposed one case in two. *)
let pair_case =
  let open Gen in
  let* dt = numeric in
  let* sa = array ~size:(int_range 0 3) extent in
  let* sb = array ~size:(int_range 0 3) extent in
  let* view = bool in
  let* xa = ints sa in
  let+ xb = ints sb in
  (dt, view, sa, xa, sb, xb)

let pair_case = Gen.with_pp pp_pair pair_case

let law_outer (Numeric dt, view, sa, xa, sb, xb) =
  let n = numel sa and m = numel sb in
  cover "a 0-d operand" (sa = [||] || sb = [||]);
  cover "an empty operand" (n = 0 || m = 0);
  cover "a transposed a" (view && Array.length sa >= 2);
  let a = (if view then transposed else of_ints) dt sa xa in
  let y = Nx.outer a (of_ints dt sb xb) in
  agrees dt [| n; m |]
    (Array.init (n * m) (fun j -> xa.(j / m) * xb.(j mod m)))
    y

let outer =
  group "outer"
    [
      prop
        "is of shape [numel a; numel b], (i, j) a's ith element times b's jth"
        pair_case law_outer;
      test "takes a as it is" (fun () ->
          equal_elements D.Complex64
            [| cx 2. 2.; cx 0. 2.; cx 0. 4.; cx (-2.) 2. |]
            (Nx.to_array
               (Nx.outer (Lazy.force complex_a) (Lazy.force complex_b))));
    ]

(* vecdot *)

(* Operands with one axis [k] each at one place, [p] from the end or from the
   start in both, their other axes suffixes of one shape with some extents [1]:
   they broadcast. [a] is laid out transposed one case in two. *)
let vecdot_case =
  let open Gen in
  let* dt = numeric in
  let* k = extent in
  let* lead = array ~size:(int_range 0 2) extent in
  let l = Array.length lead in
  let side =
    let* drop = int_range 0 l in
    let+ ones = array ~size:(constant l) bool in
    Array.init (l - drop) (fun i ->
        if ones.(i + drop) then 1 else lead.(i + drop))
  in
  let* la = side in
  let* lb = side in
  let* from_end = bool in
  let* p = int_range 0 (Stdlib.min (Array.length la) (Array.length lb)) in
  let ins s i =
    Array.concat
      [ Array.sub s 0 i; [| k |]; Array.sub s i (Array.length s - i) ]
  in
  let axis, sa, sb =
    if from_end then
      (-(p + 1), ins la (Array.length la - p), ins lb (Array.length lb - p))
    else (p, ins la p, ins lb p)
  in
  let* view = bool in
  let* xa = ints sa in
  let+ xb = ints sb in
  (dt, axis, view, sa, xa, sb, xb)

let vecdot_case =
  Gen.with_pp
    (fun ppf (dt, axis, view, sa, xa, sb, xb) ->
      Format.fprintf ppf "~axis:%d %a" axis pp_pair (dt, view, sa, xa, sb, xb))
    vecdot_case

(* [Σ_t a(..t..) b(..t..)] over each operand's own axis, the others
   broadcast. *)
let law_vecdot (Numeric dt, axis, view, sa, xa, sb, xb) =
  let ia = if axis < 0 then Array.length sa + axis else axis in
  let ib = if axis < 0 then Array.length sb + axis else axis in
  let others s i =
    Array.of_list (List.filteri (fun j _ -> j <> i) (Array.to_list s))
  in
  let oa = others sa ia and ob = others sb ib in
  let l = Stdlib.max (Array.length oa) (Array.length ob) in
  let pad s = Array.append (Array.make (l - Array.length s) 1) s in
  let pa = pad oa and pb = pad ob in
  let lead = Array.init l (fun i -> if pa.(i) = 1 then pb.(i) else pa.(i)) in
  cover "an operand broadcast" (pa <> pb);
  cover "an axis other than the last" (ia < Array.length sa - 1);
  cover "a negative axis other than -1" (axis < -1);
  cover "a transposed a" (view && Array.length sa >= 2);
  let k = sa.(ia) in
  let read s i pad o t =
    let lo = Array.mapi (fun j e -> if pad.(j) = 1 then 0 else e) o in
    let lo = Array.sub lo (l - (Array.length s - 1)) (Array.length s - 1) in
    let idx =
      Array.concat
        [ Array.sub lo 0 i; [| t |]; Array.sub lo i (Array.length lo - i) ]
    in
    ravel s idx
  in
  let want =
    Array.init (numel lead) (fun j ->
        let o = index lead j in
        let acc = ref 0 in
        for t = 0 to k - 1 do
          acc := !acc + (xa.(read sa ia pa o t) * xb.(read sb ib pb o t))
        done;
        !acc)
  in
  let a = (if view then transposed else of_ints) dt sa xa in
  agrees dt lead want (Nx.vecdot ~axis a (of_ints dt sb xb))

let vecdot =
  let z s = Nx.zeros Nx.int32 s in
  group "vecdot"
    [
      prop ~count:200
        ~examples:
          [
            ( Numeric D.Int32,
              -2,
              false,
              [| 3; 2 |],
              [| 1; 2; -1; 0; 2; -2 |],
              [| 3; 2 |],
              [| 2; 1; 0; -1; 1; 1 |] );
          ]
        "sums conjugate a times b along each operand's axis, the others \
         broadcast"
        vecdot_case law_vecdot;
      test "conjugates a" (fun () ->
          equal_elements D.Complex64
            [| cx 4. (-4.) |]
            (Nx.to_array
               (Nx.vecdot (Lazy.force complex_a) (Lazy.force complex_b))));
      cases
        "raises where axis is not both operands', their extents along it \
         differ, or the others do not broadcast"
        ~id:fst
        [
          ( "an axis outside",
            fun () -> ignore (Nx.vecdot ~axis:1 (z [| 3 |]) (z [| 3 |])) );
          ( "extents differ along the axis",
            fun () -> ignore (Nx.vecdot (z [| 3 |]) (z [| 1 |])) );
          ( "others do not broadcast",
            fun () -> ignore (Nx.vecdot (z [| 2; 3 |]) (z [| 4; 3 |])) );
        ]
        (fun (_, f) -> invalid ~by:"Nx.vecdot" f);
    ]

(* kron *)

(* Kronecker's element at [i sb + j], the shorter shape led by ones. *)
let law_kron (Numeric dt, view, sa, xa, sb, xb) =
  let r = Stdlib.max (Array.length sa) (Array.length sb) in
  let pad s = Array.append (Array.make (r - Array.length s) 1) s in
  let pa = pad sa and pb = pad sb in
  cover "ranks differ" (Array.length sa <> Array.length sb);
  cover "an empty axis" (Array.exists (( = ) 0) (Array.append pa pb));
  cover "a transposed a" (view && Array.length sa >= 2);
  let out = Array.init r (fun k -> pa.(k) * pb.(k)) in
  let want =
    Array.init (numel out) (fun j ->
        let o = index out j in
        let i = Array.mapi (fun k e -> e / pb.(k)) o in
        let i' = Array.mapi (fun k e -> e mod pb.(k)) o in
        xa.(ravel pa i) * xb.(ravel pb i'))
  in
  let a = (if view then transposed else of_ints) dt sa xa in
  agrees dt out want (Nx.kron a (of_ints dt sb xb))

let kron =
  let z s = Nx.zeros Nx.int32 s in
  group "kron"
    [
      prop "is a's element at the i's times b's at the j's, at i sb + j"
        pair_case law_kron;
      test "of two 2 × 2 matrices is the textbook product" (fun () ->
          let v xs = Nx.create Nx.int32 [| 2; 2 |] xs in
          equal (array int32)
            [|
              0l;
              5l;
              0l;
              10l;
              6l;
              7l;
              12l;
              14l;
              0l;
              15l;
              0l;
              20l;
              18l;
              21l;
              24l;
              28l;
            |]
            (Nx.to_array
               (Nx.kron (v [| 1l; 2l; 3l; 4l |]) (v [| 0l; 5l; 6l; 7l |]))));
      test "of max_rank axes of extent other than 1 together multiplies"
        (fun () ->
          (* 16 axes of [a] of extent 0 and 16 of [b] of extent 2: 32. *)
          equal (array int) (Array.make 16 0)
            (Nx.shape (Nx.kron (z (Array.make 16 0)) (z (Array.make 16 2)))));
      test "raises past max_rank axes of extent other than 1 together"
        (fun () ->
          (* 16 axes of extent 0 and 17 of extent 2: 33. *)
          let b = Nx.broadcast_to (Array.make 17 2) (Nx.scalar Nx.int32 0l) in
          invalid ~by:"Nx.kron" (fun () -> Nx.kron (z (Array.make 16 0)) b));
    ]

(* multi_dot *)

(* A chain of 2 to 4 operands of extents [e0 … en], an end 1-d at times, the
   first laid out transposed one case in two. *)
let chain_case =
  let open Gen in
  let* dt = numeric in
  let* n = int_range 2 4 in
  let* e = array ~size:(constant (n + 1)) extent in
  let* row = bool in
  let* column = bool in
  let* view = bool in
  let shape i =
    if i = 0 && row then [| e.(1) |]
    else if i = n - 1 && column then [| e.(n - 1) |]
    else [| e.(i); e.(i + 1) |]
  in
  let+ xs = sequence (List.init n (fun i -> ints (shape i))) in
  (dt, e, row, column, view, List.mapi (fun i x -> (shape i, x)) xs)

let chain_case =
  Gen.with_pp
    (fun ppf (dt, _, _, _, view, xs) ->
      Format.fprintf ppf "%a%s %s" pp_numeric dt (transposed_mark view)
        (String.concat " . "
           (List.map
              (fun (s, x) -> Format.asprintf "%a %a" pp_ints s pp_ints x)
              xs)))
    chain_case

(* The product in [int], left to right, each end's unit axis kept, then
   dropped. *)
let law_multi_dot (Numeric dt, e, row, column, view, xs) =
  let n = Array.length e - 1 in
  cover "a row" row;
  cover "a column" column;
  cover "an empty inner extent" (Array.exists (( = ) 0) (Array.sub e 1 (n - 1)));
  cover "a transposed first operand" (view && not row);
  let m0 = if row then 1 else e.(0) and mn = if column then 1 else e.(n) in
  let mat i x =
    let r = if i = 0 then m0 else e.(i) in
    let c = if i = n - 1 then mn else e.(i + 1) in
    (r, c, x)
  in
  let mul (r, k, x) (_, c, y) =
    ( r,
      c,
      Array.init (r * c) (fun j ->
          let i = j / c and l = j mod c in
          let acc = ref 0 in
          for t = 0 to k - 1 do
            acc := !acc + (x.((i * k) + t) * y.((t * c) + l))
          done;
          !acc) )
  in
  let _, _, want =
    List.fold_left mul
      (mat 0 (snd (List.hd xs)))
      (List.mapi (fun i (_, x) -> mat (i + 1) x) (List.tl xs))
  in
  let out =
    Array.of_list
      ((if row then [] else [ e.(0) ]) @ if column then [] else [ e.(n) ])
  in
  let s0, x0 = List.hd xs in
  let first = (if view then transposed else of_ints) dt s0 x0 in
  let rest = List.map (fun (s, x) -> of_ints dt s x) (List.tl xs) in
  agrees dt out want (Nx.multi_dot (first :: rest))

let chain_formats = [ Narrow D.Float32; Narrow D.Float16; Narrow D.Bfloat16 ]

(* Chains of 2 to 4 matrices of extents 0 to 4, entries in [-1, 1] of a float
   format of at most 32 bits. *)
let float_chain =
  let open Gen in
  let* (Narrow dt) = of_list ~pp:pp_narrow chain_formats in
  let* n = int_range 2 4 in
  let* e = array ~size:(constant (n + 1)) (int_range 0 4) in
  let+ xs =
    sequence
      (List.init n (fun i ->
           array ~size:(constant (e.(i) * e.(i + 1))) (float_range (-1.) 1.)))
  in
  (Narrow dt, e, xs)

let float_chain =
  Gen.with_pp
    (fun ppf (Narrow dt, e, xs) ->
      Format.fprintf ppf "%a extents %a: %s" D.pp dt pp_ints e
        (String.concat " . " (List.map (Format.asprintf "%a" pp_floats) xs)))
    float_chain

(* The [r × k] by [k × c] product of row-major floats in float64, each element a
   sum of [k] products from the first. *)
let product r k c x y =
  Array.init (r * c) (fun j ->
      let i = j / c and l = j mod c in
      let acc = ref 0. in
      for t = 0 to k - 1 do
        acc := !acc +. (x.((i * k) + t) *. y.((t * c) + l))
      done;
      !acc)

let gamma n u =
  let n = Float.of_int n in
  n *. u /. (1. -. (n *. u))

(* nx.mli's einsum bound: before rounding, the result is within [γ(N, 2u) M] of
   the exact product, [M] the product of the operands' magnitudes, [N] the sum
   of [K + 1] over the inner extents [K] (one contraction each, whatever the
   order), [u] float32's unit roundoff, the carrier's. Rounding once into the
   dtype adds [u_dt] of the value, which is at most [M + γ(N, 2u) M], or half a
   subnormal step. The float64 reference and its [M] err by [γ(N, u64) M]
   each. *)
let law_multi_dot_bound (Narrow dt, e, xs) =
  let n = Array.length e - 1 in
  List.iter
    (fun (Narrow f) -> cover ("at " ^ D.name f) (D.name f = D.name dt))
    chain_formats;
  cover "a chain of three or more" (n >= 3);
  let ops =
    List.mapi
      (fun i x -> Nx.cast dt (Nx.create Nx.float64 [| e.(i); e.(i + 1) |] x))
      xs
  in
  let wide = List.map (fun x -> Nx.to_array (Nx.cast Nx.float64 x)) ops in
  let chain f =
    let _, m =
      List.fold_left
        (fun (i, p) x -> (i + 1, product e.(0) e.(i) e.(i + 1) p (f x)))
        (1, f (List.hd wide))
        (List.tl wide)
    in
    m
  in
  let exact = chain Fun.id and size = chain (Array.map Float.abs) in
  let y = Nx.to_array (Nx.cast Nx.float64 (Nx.multi_dot ops)) in
  let nn = Array.fold_left (fun s k -> s + k + 1) 0 (Array.sub e 1 (n - 1)) in
  let f = D.float_format dt in
  let ud = f.epsilon /. 2. and g64 = gamma nn 0x1p-53 in
  let rel =
    (((1. +. ud) *. gamma nn 0x1p-23) +. ud +. g64) *. (1. +. (3. *. g64))
  in
  let step = f.min_normal *. f.epsilon in
  Array.iteri
    (fun j r ->
      let bound = (rel *. size.(j)) +. (step /. 2.) in
      let err = Float.abs (y.(j) -. r) in
      if not (err <= bound) then
        failf "element %d: %h, %h from the exact %h, past the bound %h" j y.(j)
          err r bound)
    exact

(* A chain whose every order's work passes an [int]: traced operands, so that
   nothing is computed. *)
type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let mark i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

let multi_dot =
  let z s = Nx.zeros Nx.int32 s in
  let f32 s xs = Nx.create Nx.float32 s xs in
  group "multi_dot"
    [
      prop
        ~examples:
          [
            ( Numeric D.Int8,
              [| 1; 2; 1 |],
              true,
              true,
              false,
              [ ([| 2 |], [| 1; 2 |]); ([| 2 |], [| -1; 2 |]) ] );
          ]
        "is the chain's product, a 1-d first operand a row and a 1-d last a \
         column"
        chain_case law_multi_dot;
      prop "is within einsum's bound of the exact product, rounded once"
        float_chain law_multi_dot_bound;
      test "keeps each product in the carrier and rounds once" (fun () ->
          (* (h ones) 3 is the cheaper order, 2 + 1 against 2 + 2. float16 keeps
             [h ones = 2049] in float32 and rounds [6147] once, to 6148; rounded
             to float16 first, 2049 would be 2048 and the result 6144. *)
          let v s xs = Nx.create Nx.float16 s xs in
          let h = v [| 1; 2 |] [| 2048.; 1. |]
          and ones = v [| 2; 1 |] [| 1.; 1. |] in
          equal_elements D.Float16 [| 6148. |]
            (Nx.to_array (Nx.multi_dot [ h; ones; v [| 1; 1 |] [| 3. |] ])));
      cases "multiplies in the order of fewest multiply-adds" ~id:fst
        [
          (* 2 × 1, 1 × 2, 2 × 1 is cheaper as a (b c), 4 against 8, where b c =
             1 - 1 cancels and a b = 1e60 is infinite. *)
          ( "a (b c)",
            fun () ->
              [
                f32 [| 2; 1 |] [| 1e30; 1e30 |];
                f32 [| 1; 2 |] [| 1e30; 1e30 |];
                f32 [| 2; 1 |] [| 1e-30; -1e-30 |];
              ] );
          (* 1 × 2, 2 × 1, 1 × 2 is cheaper as (a b) c, the mirror. *)
          ( "(a b) c",
            fun () ->
              [
                f32 [| 1; 2 |] [| 1e-30; -1e-30 |];
                f32 [| 2; 1 |] [| 1e30; 1e30 |];
                f32 [| 1; 2 |] [| 1e30; 1e30 |];
              ] );
        ]
        (fun (_, xs) ->
          equal_numbers D.Float32 [| 0.; 0. |]
            (Nx.to_array (Nx.multi_dot (xs ()))));
      test "multiplies a chain whose every order passes max_int multiply-adds"
        (fun () ->
          (* Three [2^21 × 2^21] operands: every order does 2^64 or more. *)
          let side = 1 lsl 21 in
          let b = Nx.broadcast_to [| side; side |] (Nx.scalar Nx.float32 1.) in
          Nx.Prim.interpret ~name:"test.chain" Values mark (fun i ->
              let t = Nx.Prim.traced i (Nx.Prim.form b) Mark in
              equal (array int) [| side; side |]
                (Nx.shape (Nx.multi_dot [ t; t; t ]))));
      cases
        "raises where n < 2, an operand has another rank, or adjacent inner \
         extents differ"
        ~id:fst
        [
          ( "no operand",
            fun () -> ignore (Nx.multi_dot ([] : (int32, _, Nx.host) Nx.t list))
          );
          ("one operand", fun () -> ignore (Nx.multi_dot [ z [| 2; 2 |] ]));
          ( "a vector in the middle",
            fun () ->
              ignore (Nx.multi_dot [ z [| 2; 2 |]; z [| 2 |]; z [| 2; 2 |] ]) );
          ( "a 3-d end",
            fun () -> ignore (Nx.multi_dot [ z [| 1; 2; 2 |]; z [| 2; 2 |] ]) );
          ( "inner extents differ",
            fun () -> ignore (Nx.multi_dot [ z [| 2; 3 |]; z [| 2; 2 |] ]) );
        ]
        (fun (_, f) -> invalid ~by:"Nx.multi_dot" f);
    ]

(* matrix_power *)

(* Powers of matrices of [-1], [0] and [1] up to the 6th: at most 3^5 in
   magnitude, exact in every float format of 8 bits of precision or more; [a]
   laid out transposed one case in two. *)
let power_case =
  let open Gen in
  let exact =
    List.filter
      (fun (Numeric dt) ->
        match dt with
        | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn -> false
        | _ -> true)
      (List.filter_map
         (fun (D.Any dt) ->
           if D.is D.Boolean dt then None else Some (Numeric dt))
         D.all)
  in
  let* dt = of_list ~pp:pp_numeric exact in
  let* m = int_range 0 3 in
  let* lead = array ~size:(int_range 0 1) extent in
  let s = Array.append lead [| m; m |] in
  let* n = int_range 0 6 in
  let* view = bool in
  let+ x = array ~size:(constant (numel s)) (int_range (-1) 1) in
  (dt, s, n, view, x)

let power_case =
  Gen.with_pp
    (fun ppf (dt, s, n, view, x) ->
      Format.fprintf ppf "%a %a%s %a to the %d" pp_numeric dt pp_ints s
        (transposed_mark view) pp_ints x n)
    power_case

let law_matrix_power (Numeric dt, s, n, view, x) =
  let m = s.(Array.length s - 1) in
  cover "n = 0" (n = 0);
  cover "an odd n of 3 or more" (n >= 3 && n land 1 = 1);
  cover "an empty matrix" (m = 0);
  cover "a transposed a" (view && m >= 2);
  let mm = m * m in
  let product p q =
    Array.init mm (fun j ->
        let i = j / m and l = j mod m in
        let acc = ref 0 in
        for t = 0 to m - 1 do
          acc := !acc + (p.((i * m) + t) * q.((t * m) + l))
        done;
        !acc)
  in
  let identity = Array.init mm (fun j -> if j / m = j mod m then 1 else 0) in
  let want =
    Array.concat
      (List.init
         (numel s / Stdlib.max 1 mm)
         (fun b ->
           let a = Array.sub x (b * mm) mm in
           let rec pow k =
             if k = 0 then identity else product (pow (k - 1)) a
           in
           pow n))
  in
  let want = if mm = 0 then [||] else want in
  let a = (if view then transposed else of_ints) dt s x in
  agrees dt s want (Nx.matrix_power a n)

(* Matrices of a dtype {!Nx.inv} takes, of entries in [-1, 1]. *)
let inverse_case =
  let open Gen in
  let* dt =
    of_list ~pp:pp_numeric
      [
        Numeric D.Float32;
        Numeric D.Float64;
        Numeric D.Complex64;
        Numeric D.Complex128;
      ]
  in
  let* m = int_range 1 3 in
  let* lead = array ~size:(int_range 0 1) (int_range 1 2) in
  let s = Array.append lead [| m; m |] in
  let* n = int_range 1 6 in
  let* re = array ~size:(constant (numel s)) (float_range (-1.) 1.) in
  let+ im = array ~size:(constant (numel s)) (float_range (-1.) 1.) in
  (dt, s, n, re, im)

let inverse_case =
  Gen.with_pp
    (fun ppf (dt, s, n, re, im) ->
      Format.fprintf ppf "%a %a re %a im %a to the -%d" pp_numeric dt pp_ints s
        pp_floats re pp_floats im n)
    inverse_case

(* The same bits, NaNs as one. *)
let law_negative_power (Numeric dt, s, n, re, im) =
  cover "a batch" (Array.length s > 2);
  cover "a power of 3 or more" (n >= 3);
  let part xs = Nx.create Nx.float64 s xs in
  let a = Nx.cast dt (Nx.complex Nx.complex128 ~re:(part re) ~im:(part im)) in
  equal_elements dt
    (Nx.to_array (Nx.matrix_power (Nx.inv a) n))
    (Nx.to_array (Nx.matrix_power a (-n)))

let matrix_power =
  let z s = Nx.zeros Nx.int32 s in
  group "matrix_power"
    [
      prop ~count:200 "is each matrix to the power n, the identity for n = 0"
        power_case law_matrix_power;
      prop "at a negative n is inv a to the power -n" inverse_case
        law_negative_power;
      test "squares repeatedly, each product rounded to the dtype" (fun () ->
          (* [x = 1 + 17/1024]: [x²] rounds to [1 + 34/1024] and its square to
             [1 + 69/1024], where [x⁴] rounds to [1 + 70/1024], as do products
             from the left. *)
          let x = Nx.create Nx.float16 [| 1; 1 |] [| 1. +. (17. /. 1024.) |] in
          equal_elements D.Float16
            [| 1. +. (69. /. 1024.) |]
            (Nx.to_array (Nx.matrix_power x 4)));
      cases "at the values its definition gives"
        ~id:(fun (id, _, _, _, _) -> id)
        [
          ( "a quarter turn to the 4th is the identity",
            [| 2; 2 |],
            [| 0.; -1.; 1.; 0. |],
            4,
            [| 1.; 0.; 0.; 1. |] );
          ( "a quarter turn to the -1st is its transpose",
            [| 2; 2 |],
            [| 0.; -1.; 1.; 0. |],
            -1,
            [| 0.; 1.; -1.; 0. |] );
          ( "diag (2, 4) to the -2nd is diag (1/4, 1/16)",
            [| 2; 2 |],
            [| 2.; 0.; 0.; 4. |],
            -2,
            [| 0.25; 0.; 0.; 0.0625 |] );
          ( "the swap to max_int is the swap",
            [| 2; 2 |],
            [| 0.; 1.; 1.; 0. |],
            max_int,
            [| 0.; 1.; 1.; 0. |] );
          ( "the swap to min_int is the identity",
            [| 2; 2 |],
            [| 0.; 1.; 1.; 0. |],
            min_int,
            [| 1.; 0.; 0.; 1. |] );
          ("an empty batch is empty", [| 3; 0; 0 |], [||], 5, [||]);
        ]
        (fun (_, s, xs, n, want) ->
          let y = Nx.matrix_power (Nx.create Nx.float64 s xs) n in
          equal ~msg:"shape" (array int) s (Nx.shape y);
          equal_numbers D.Float64 want (Nx.to_array y));
      cases
        "raises below two axes, for a matrix not square, or at a negative n \
         for a dtype inv does not take"
        ~id:fst
        [
          ("a vector", fun () -> ignore (Nx.matrix_power (z [| 2 |]) 2));
          ("not square", fun () -> ignore (Nx.matrix_power (z [| 2; 3 |]) 2));
          ( "an integer's inverse",
            fun () -> ignore (Nx.matrix_power (z [| 2; 2 |]) (-1)) );
          ( "a float16's inverse",
            fun () ->
              ignore (Nx.matrix_power (Nx.zeros Nx.float16 [| 2; 2 |]) (-1)) );
        ]
        (fun (_, f) -> invalid ~by:"Nx.matrix_power" f);
    ]

(* matrix_transpose *)

let law_matrix_transpose (Batch (dt, x)) =
  cover "a view" (not (plain x));
  cover "a batch" (Nx.ndim x > 2);
  let s = Nx.shape x in
  let t = swap_last s in
  let xs = Nx.to_array x in
  let want =
    Array.init (numel t) (fun j -> xs.(ravel s (swap_last (index t j))))
  in
  let y = Nx.matrix_transpose x in
  equal (array int) t (Nx.shape y);
  equal_elements dt want (Nx.to_array y)

let matrix_transpose =
  let grid s = Nx.reshape s (Nx.arange Nx.int8 0 (numel s) 1) in
  group "matrix_transpose"
    [
      prop "exchanges the last two axes" matrices law_matrix_transpose;
      cases "is x itself below two axes" ~id:fst
        [ ("0-d", fun () -> grid [||]); ("1-d", fun () -> grid [| 3 |]) ]
        (fun (_, x) ->
          let x = x () in
          satisfies ~claim:"x itself" (array int)
            (fun _ -> Nx.matrix_transpose x == x)
            (Nx.shape x));
      test "is a view" (fun () ->
          let x = grid [| 2; 3 |] in
          reads_memory_of x (Nx.matrix_transpose x));
    ]

(* tril, triu *)

(* [x]'s elements where [inside] holds of column - row, zero elsewhere. *)
let region (type v s) (dt : (v, s) D.t) (x : (v, s, Nx.host) Nx.t) inside =
  let s = Nx.shape x in
  let r = Array.length s in
  Array.mapi
    (fun j v ->
      let i = index s j in
      if inside (i.(r - 1) - i.(r - 2)) then v else D.zero dt)
    (Nx.to_array x)

let triangle_case = Gen.pair matrices (Gen.int_range (-4) 4)

(* [k] at the bounds of [int]: every element kept, or none. *)
let extreme_ks =
  let x = Batch (D.Int8, Nx.create Nx.int8 [| 2; 3 |] [| 1; 2; 3; 4; 5; 6 |]) in
  [ (x, max_int); (x, min_int) ]

let law_tril (Batch (dt, x), k) =
  cover "a view" (not (plain x));
  cover "an empty matrix" (Nx.numel x = 0);
  equal_elements dt (region dt x (fun d -> d <= k)) (Nx.to_array (Nx.tril ~k x))

let law_triu (Batch (dt, x), k) =
  cover "a view" (not (plain x));
  cover "an empty matrix" (Nx.numel x = 0);
  equal_elements dt (region dt x (fun d -> d >= k)) (Nx.to_array (Nx.triu ~k x))

let tril =
  group "tril"
    [
      prop ~examples:extreme_ks "is x where column - row <= k, zero above"
        triangle_case law_tril;
      cases "raises below two axes" ~id:fst
        [ ("1-d", [| 3 |]); ("0-d", [||]) ]
        (fun (_, s) ->
          invalid ~by:"Nx.tril" (fun () -> Nx.tril (Nx.zeros Nx.int8 s)));
    ]

let triu =
  group "triu"
    [
      prop ~examples:extreme_ks "is x where column - row >= k, zero below"
        triangle_case law_triu;
      cases "raises below two axes" ~id:fst
        [ ("1-d", [| 3 |]); ("0-d", [||]) ]
        (fun (_, s) ->
          invalid ~by:"Nx.triu" (fun () -> Nx.triu (Nx.zeros Nx.int8 s)));
    ]

(* diagonal *)

(* A value, an offset and two distinct axes of the value, each spelled negative
   one time in two, or, one case in four, left to the default. *)
let diagonal_case =
  let open Gen in
  let* b = matrices in
  let (Batch (_, x)) = b in
  let r = Nx.ndim x in
  let* order = permutation (List.init r Fun.id) in
  let* default = frequency [ (1, constant true); (3, constant false) ] in
  let* n1 = bool in
  let* n2 = bool in
  let+ offset = int_range (-4) 4 in
  let spell negative a = if negative then a - r else a in
  let axes =
    if default then None
    else Some (spell n1 (List.nth order 0), spell n2 (List.nth order 1))
  in
  (b, offset, axes)

let diagonal_case =
  Gen.with_pp
    (fun ppf (b, offset, axes) ->
      Format.fprintf ppf "%a ~offset:%d%s" pp_batch b offset
        (match axes with
        | None -> ""
        | Some (a1, a2) -> Printf.sprintf " ~axis1:%d ~axis2:%d" a1 a2))
    diagonal_case

let law_diagonal (Batch (dt, x), offset, axes) =
  let s = Nx.shape x in
  let r = Array.length s in
  let a1, a2 =
    match axes with
    | None -> (r - 2, r - 1)
    | Some (a1, a2) -> ((a1 + r) mod r, (a2 + r) mod r)
  in
  let m = s.(a1) and n = s.(a2) in
  let count =
    Stdlib.max 0
      (if offset >= 0 then Stdlib.min m (n - offset)
       else Stdlib.min (m + offset) n)
  in
  cover "an empty diagonal" (count = 0);
  cover "the default axes" (axes = None);
  cover "a negative axis"
    (match axes with Some (a1, a2) -> a1 < 0 || a2 < 0 | None -> false);
  cover "a view" (not (plain x));
  let rest = List.filter (fun a -> a <> a1 && a <> a2) (List.init r Fun.id) in
  let lead = Array.of_list (List.map (fun a -> s.(a)) rest) in
  let out = Array.append lead [| count |] in
  let xs = Nx.to_array x in
  let want =
    Array.init (numel out) (fun j ->
        let o = index out j in
        let d = o.(Array.length o - 1) in
        let i = Array.make r 0 in
        List.iteri (fun k a -> i.(a) <- o.(k)) rest;
        i.(a1) <- (if offset >= 0 then d else d - offset);
        i.(a2) <- (if offset >= 0 then d + offset else d);
        xs.(ravel s i))
  in
  let y =
    match axes with
    | None -> Nx.diagonal ~offset x
    | Some (axis1, axis2) -> Nx.diagonal ~offset ~axis1 ~axis2 x
  in
  equal (array int) out (Nx.shape y);
  equal_elements dt want (Nx.to_array y)

let diagonal =
  let z s = Nx.zeros Nx.int8 s in
  let grid s = Nx.reshape s (Nx.arange Nx.int8 0 (numel s) 1) in
  group "diagonal"
    [
      prop
        "is x at i along axis1 and i + offset along axis2, that axis last, the \
         others in order"
        diagonal_case law_diagonal;
      test "is a view where the two axes merge" (fun () ->
          let x = grid [| 3; 4 |] in
          reads_memory_of x (Nx.diagonal x);
          let t = grid [| 2; 3; 4 |] in
          reads_memory_of t (Nx.diagonal ~axis1:0 ~axis2:1 t));
      cases
        "raises below two axes, or where axis1 and axis2 are not two distinct \
         axes of x"
        ~id:fst
        [
          ("a vector", fun () -> ignore (Nx.diagonal (z [| 3 |])));
          ( "one axis twice",
            fun () -> ignore (Nx.diagonal ~axis1:0 ~axis2:0 (z [| 2; 2 |])) );
          ( "one axis spelled from both ends",
            fun () -> ignore (Nx.diagonal ~axis1:1 ~axis2:(-1) (z [| 2; 2 |]))
          );
          ( "an axis past x",
            fun () -> ignore (Nx.diagonal ~axis1:3 (z [| 2; 2; 2 |])) );
          ( "an axis before x",
            fun () -> ignore (Nx.diagonal ~axis2:(-4) (z [| 2; 2; 2 |])) );
        ]
        (fun (_, f) -> invalid ~by:"Nx.diagonal" f);
    ]

(* trace *)

let numeric_matrices =
  Gen.such_that (fun (Batch (dt, _)) -> not (D.is D.Boolean dt)) matrices

(* The same bits, NaNs as one, as nx.mli defines trace. *)
let law_trace (Batch (dt, x), offset) =
  let d = Nx.diagonal ~offset x in
  cover "an empty diagonal" (Nx.dim (-1) d = 0);
  cover "a view" (not (plain x));
  equal_elements dt
    (Nx.to_array (Nx.sum ~axes:[ -1 ] d))
    (Nx.to_array (Nx.trace ~offset x))

let trace =
  group "trace"
    [
      prop "is the sum of diagonal ~offset x along its last axis"
        (Gen.pair numeric_matrices (Gen.int_range (-4) 4))
        law_trace;
      cases "sums each matrix's diagonal"
        ~id:(fun (offset, _) -> Printf.sprintf "~offset:%d" offset)
        (* The matrices [[0 1 2] [3 4 5] [6 7 8]] and that plus 9. *)
        [ (0, [| 12l; 39l |]); (1, [| 6l; 24l |]); (3, [| 0l; 0l |]) ]
        (fun (offset, want) ->
          let x = Nx.reshape [| 2; 3; 3 |] (Nx.arange Nx.int32 0 18 1) in
          equal (array int32) want (Nx.to_array (Nx.trace ~offset x)));
      test "of an empty or a -0 diagonal is +0" (fun () ->
          let trace x = Nx.to_array (Nx.flatten (Nx.trace x)) in
          equal ~msg:"empty" (array float_exact) [| 0. |]
            (trace (Nx.zeros Nx.float32 [| 1; 0; 4 |]));
          equal ~msg:"-0" (array float_exact) [| 0. |]
            (trace (Nx.create Nx.float32 [| 1; 1 |] [| -0. |])));
      cases "raises as diagonal and sum do" ~id:fst
        [
          ("a vector", fun () -> ignore (Nx.trace (Nx.zeros Nx.int8 [| 3 |])));
          ("booleans", fun () -> ignore (Nx.trace (Nx.zeros Nx.bool [| 2; 2 |])));
        ]
        (fun (_, f) -> invalid ~by:"Nx.trace" f);
    ]

(* diag *)

(* A vector of every dtype, of 0 to 4 elements, and a [k]. *)
let vector_case =
  let open Gen in
  let* (D.Any dt) = of_list ~pp:pp_any D.all in
  let* n = int_range 0 4 in
  let* k = int_range (-4) 4 in
  let+ v = drawn dt [| n |] in
  (Batch (dt, v), k)

let vector_case =
  Gen.with_pp
    (fun ppf (b, k) -> Format.fprintf ppf "%a ~k:%d" pp_batch b k)
    vector_case

let law_diag (Batch (dt, v), k) =
  let n = Nx.numel v in
  let side = n + abs k in
  cover "an empty vector" (n = 0);
  cover "a diagonal below the main one" (k < 0);
  let vs = Nx.to_array v in
  let want =
    Array.init (side * side) (fun j ->
        let i = j / side and c = j mod side in
        if c - i = k then vs.(Stdlib.min i c) else D.zero dt)
  in
  let y = Nx.diag ~k v in
  equal (array int) [| side; side |] (Nx.shape y);
  equal_elements dt want (Nx.to_array y)

let diag =
  group "diag"
    [
      prop
        "of a vector of n elements is the square of n + |k| with the vector on \
         its kth diagonal, zero elsewhere"
        vector_case law_diag;
      cases "of a matrix is its diagonal ~offset:k"
        ~id:(fun (k, _) -> Printf.sprintf "~k:%d" k)
        (* The matrix [[0 1 2] [3 4 5] [6 7 8]]. *)
        [ (0, [| 0; 4; 8 |]); (1, [| 1; 5 |]); (-1, [| 3; 7 |]) ]
        (fun (k, want) ->
          let m = Nx.reshape [| 3; 3 |] (Nx.arange Nx.int16 0 9 1) in
          equal (array int) want (Nx.to_array (Nx.diag ~k m)));
      cases "raises for neither one nor two axes" ~id:fst
        [ ("0-d", [||]); ("3-d", [| 1; 1; 1 |]) ]
        (fun (_, s) ->
          invalid ~by:"Nx.diag" (fun () -> Nx.diag (Nx.zeros Nx.int16 s)));
    ]

let () =
  exit
    (run "nx matrices"
       [
         dot;
         vdot;
         cross;
         other_products;
         tensordot;
         inner;
         outer;
         vecdot;
         kron;
         multi_dot;
         matrix_power;
         matrix_transpose;
         tril;
         triu;
         diagonal;
         trace;
         diag;
       ])
