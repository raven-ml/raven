(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reductions and scans through every kernel library the host runs, with
   operands and results on its device: a float sum within its bound of the
   exact sum, the same bits under every layout of the same values, a scan of
   a prefix the prefix of the scan, and nothing written where declined.
   Every library states nx.cpu's cases and order, which a group checks for
   each: the cases it computes, and its order, bit for bit, against a
   reference built here. nx.cpu's own group checks the same bits on one
   thread as on the job's, and a sum as a contraction with ones. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

type monoid = Nx_kernel.monoid = Sum | Prod | Max | Min | Logsumexp

let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let total s = Array.fold_left ( * ) 1 s
let shape_of (A.Any x) = L.shape (A.layout x)

let monoid_name = function
  | Sum -> "Sum"
  | Prod -> "Prod"
  | Max -> "Max"
  | Min -> "Min"
  | Logsumexp -> "Logsumexp"

(* Cases *)

(* A reduction or scan of [x] by [monoid] along [axes], a scan's one, [x]
   drawn through the views [views]. *)
type case = {
  scan : bool;
  monoid : monoid;
  axes : int array;
  x : A.any;
  views : string list;
}

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "%s %s along %a of %a %a (%s)"
    (if c.scan then "scan" else "reduce")
    (monoid_name c.monoid) pp_ints c.axes D.pp (A.dtype x) L.pp (A.layout x)
    (String.concat ", " c.views)

(* [In 0] of one operand of [dt]. *)
let identity dt = P.v ~ins:[| dt |] [| P.In 0 |] ~outs:[| 0 |]

let plain_of (A.Any x) = Nx_kernel.Plain x

(* The loop's shape: the loads' once loaded, else, for a padding that fits
   none, the first load's operand's, which the kernels refuse with the
   padding. *)
let loop_shape (loads : Nx_kernel.load array) =
  match Nx_kernel.Shape.loaded loads with
  | Ok s -> s
  | Error _ -> (
      match loads.(0) with
      | Plain x -> L.shape (A.layout x)
      | Padded { x; _ } -> L.shape (A.layout x))
let dtype_of (A.Any x) = D.Any (A.dtype x)
let accepts m (D.Any dt) = Nx_kernel.accepts (Monoid m) dt

(* By [k], the reduction by [m] of [p]'s output along [axes], or with [scan]
   its scan along [axes.(0)], of [loads] into [dst]. *)
let compute (module K : Nx_kernel.S) ~scan p m axes dst loads =
  if scan then K.scan p ~axis:axes.(0) (Monoid m, 0) ~dsts:[| dst |] loads
  else
    K.reduce p ~shape:(loop_shape loads) ~axes [| (Monoid m, 0) |]
      ~dsts:[| dst |] loads

(* The result's shape. *)
let result_shape c =
  let s = shape_of c.x in
  if c.scan then s
  else
    Array.of_list
      (List.filteri (fun i _ -> not (Array.mem i c.axes)) (Array.to_list s))

(* [x] where the host reads it: [x] itself on the host, a copy elsewhere. *)
let host (A.Any x as a) =
  if Rig.equal (A.device x) Rig.host then a
  else A.Any (A.to_device Rig.host x)

(* [x] on [b]'s device, its layout kept. *)
let on (b : Support.backend) (A.Any x as a) =
  if Rig.equal (A.device x) b.device then a
  else A.Any (A.to_device b.device x)

(* [c] by [b]'s kernels into [dst], on [b]'s device. *)
let call (b : Support.backend) c dst =
  let x = on b c.x in
  compute b.kernels ~scan:c.scan (identity (dtype_of x)) c.monoid c.axes dst
    [| plain_of x |]

(* The kernels' result into a fresh destination, read where the host reads
   it; [None] if they declined. *)
let run (b : Support.backend) c =
  let (A.Any x) = c.x in
  let dst = A.Any (A.create b.device (A.dtype x) (result_shape c)) in
  match call b c dst with
  | A.Done -> Some (host dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Values *)

(* Bits of float32 and float64 NaNs, quiet and signalling, of both signs. *)
let nans32 = [| 0x7FC00000l; 0xFFC12345l; 0x7F800001l; 0xFFA00002l |]

let nans64 =
  [| 0x7FF8000000000000L; 0x7FF0000000000001L; 0xFFF4000000000002L |]

(* A float of [p] significant bits in (-4, 4). *)
let full r p =
  let m = Random.State.int64 r (Int64.shift_left 1L (p + 1)) in
  Float.ldexp (Int64.to_float (Int64.sub m (Int64.shift_left 1L p))) (2 - p)

(* Integers at and about the edges of every width. *)
let edges =
  [|
    0L; 1L; -1L; 0x7FL; 0x80L; 0xFFL; 0x7FFFL; 0x8000L; 0x7FFF_FFFFL;
    0x8000_0000L; Int64.max_int; Int64.min_int;
  |]

(* [n] elements of [dt] as the bytes of a C-contiguous array, from [seed]:
   floats, and complex numbers' parts, of full precision in (-4, 4), with
   [-0.], infinities and NaNs of drawn payloads at the rate [specials] (one
   in that many, none for 0); integers small, or at the edges of the widths
   at that rate. *)
let elements (type v s) (dt : (v, s) D.t) n ~specials seed =
  let r = Random.State.make [| seed |] in
  let special () = specials > 0 && Random.State.int r specials = 0 in
  let pick a = a.(Random.State.int r (Array.length a)) in
  let b = Bytes.create (max 1 (D.bytes dt n)) in
  let f32 at =
    let bits =
      if special () then
        pick (Array.append nans32 [| 0x80000000l; 0x7F800000l; 0xFF800000l |])
      else Int32.bits_of_float (full r 23)
    in
    Bytes.set_int32_le b at bits
  in
  let f64 at =
    let bits =
      if special () then
        pick
          (Array.append nans64
             [|
               Int64.bits_of_float (-0.);
               Int64.bits_of_float infinity;
               Int64.bits_of_float neg_infinity;
             |])
      else Int64.bits_of_float (full r 52)
    in
    Bytes.set_int64_le b at bits
  in
  for i = 0 to n - 1 do
    match dt with
    | D.Float32 -> f32 (4 * i)
    | D.Float64 -> f64 (8 * i)
    | D.Complex64 ->
        f32 (8 * i);
        f32 ((8 * i) + 4)
    | D.Complex128 ->
        f64 (16 * i);
        f64 ((16 * i) + 8)
    | _ when D.bits dt >= 8 ->
        let v =
          if special () then pick edges
          else Int64.of_int (Random.State.int r 16 - 8)
        in
        let w = D.bits dt / 8 in
        for j = 0 to w - 1 do
          Bytes.set b ((w * i) + j)
            (Char.chr
               (Int64.to_int
                  (Int64.logand (Int64.shift_right_logical v (8 * j)) 0xFFL)))
        done;
        if D.equal dt D.Bool then Bytes.set b i (Char.chr (Random.State.int r 2))
    | _ ->
        (* Sub-byte: random codes. *)
        Bytes.set b (i * D.bits dt / 8) (Char.chr (Random.State.int r 256))
  done;
  A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string (Bytes.to_string b))

(* Views *)

(* How an array of shape [s] is viewed: made with its axes in another order,
   stepped by two along its last made axis, reversed along a made axis, or
   broadcast along its first axis from one element. *)
type view = {
  perm : int array;
  stepped : bool;
  reversed : int option;
  broadcast : bool;
}

let view_names v =
  (if v.perm <> Array.init (Array.length v.perm) Fun.id then [ "permuted" ]
   else [])
  @ (if v.stepped then [ "stepped" ] else [])
  @ (if v.reversed <> None then [ "reversed" ] else [])
  @ if v.broadcast then [ "broadcast" ] else []

let view_of r =
  let open Gen in
  let* p = permutation ~pp:Format.pp_print_int (List.init r Fun.id) in
  let* stepped = bool in
  let* reversed = option (int_range 0 (max 0 (r - 1))) in
  let+ broadcast = frequency [ (4, constant false); (1, constant true) ] in
  {
    perm = Array.of_list p;
    stepped = stepped && r > 0;
    reversed = (if r > 0 then reversed else None);
    broadcast = broadcast && r > 0;
  }

(* An array of shape [s] through the view [v]: its elements, [make n] of
   them, are made C-contiguous in the shape before the view, then moved into
   it. *)
let operand_with make s v =
  let r = Array.length s in
  let s0 = Array.mapi (fun i e -> if v.broadcast && i = 0 then 1 else e) s in
  let inv = Array.make r 0 in
  Array.iteri (fun i m -> inv.(m) <- i) v.perm;
  (* Made: [s0] with its axes in made order, the last doubled if stepped. *)
  let made = Array.init r (fun m -> s0.(inv.(m))) in
  let last = r - 1 in
  let made' = Array.mapi (fun i e -> if v.stepped && i = last then 2 * e else e) made in
  let whole e = { M.start = 0; count = e; step = 1 } in
  let move m x = Option.get (A.move m x) in
  let (A.Any x) = make (total made') in
  let x = move (M.Reshape made') x in
  let x =
    if not v.stepped then x
    else
      move
        (M.Slice
           (Array.mapi
              (fun i e ->
                if i = last then { M.start = 1; count = made.(i); step = 2 }
                else whole e)
              made'))
        x
  in
  let x =
    match v.reversed with
    | Some a when made.(a) > 0 ->
        move
          (M.Slice
             (Array.mapi
                (fun i e ->
                  if i = a then { M.start = e - 1; count = e; step = -1 }
                  else whole e)
                made))
          x
    | _ -> x
  in
  let x = move (M.Permute v.perm) x in
  A.Any (if v.broadcast then move (M.Broadcast s) x else x)

(* An array of [dt] of shape [s] through the view [v], of [elements]. *)
let operand (D.Any dt) s v ~specials seed =
  operand_with (fun n -> A.Any (elements dt n ~specials seed)) s v

(* One long axis past several blocks or chunks, or at a block's or a
   chunk's edge, between up to two short ones. *)
let long =
  let open Gen in
  let edge =
    of_list ~pp:Format.pp_print_int
      [ 15; 16; 17; 1023; 1024; 1025; 4095; 4096; 4097 ]
  in
  let* n = frequency [ (2, edge); (1, int_range 5000 70000) ] in
  let* before = array ~size:(int_range 0 1) (int_range 1 3) in
  let+ after = array ~size:(int_range 0 1) (int_range 1 3) in
  Array.concat [ before; [| n |]; after ]

(* Shapes: small extents of any rank; a long axis; rows of a few to a few
   hundred terms; an empty axis beside others. *)
let shape_gen =
  let open Gen in
  let small = array ~size:(int_range 0 4) (int_range 0 5) in
  let rows =
    let* r = int_range 1 300 in
    let+ k = of_list ~pp:Format.pp_print_int [ 2; 4; 9; 64; 300 ] in
    [| r; k |]
  in
  let empty =
    of_list [ [| 0 |]; [| 3; 0 |]; [| 0; 4 |]; [| 2; 0; 3 |] ]
  in
  frequency [ (5, small); (3, long); (2, rows); (4, empty) ]

let monoids = [ Sum; Prod; Max; Min ]

(* A case of a dtype of [dts]: a reduction over any subset of the axes, or a
   scan along one. A scan of rank 0 takes a shape of one axis; a maximum or
   minimum takes an element along each empty reduced axis, since one of no
   term is refused. *)
let case_of ?(scan = Gen.bool) ?(monoids = monoids) ?(shapes = shape_gen) dts =
  let open Gen in
  let* scan = scan in
  let* d = of_list ~pp:pp_dtype dts in
  let* monoid =
    of_list (List.filter (fun m -> accepts m d) monoids)
  in
  let* s = shapes in
  let s = if scan && Array.length s = 0 then [| 3 |] else s in
  let r = Array.length s in
  let* axes =
    if scan then map (fun a -> [| a |]) (int_range 0 (r - 1))
    else
      let+ keep = array ~size:(constant r) bool in
      Array.of_list (List.filter (fun i -> keep.(i)) (List.init r Fun.id))
  in
  let s =
    if scan || not (monoid = Max || monoid = Min) then s
    else Array.mapi (fun i e -> if Array.mem i axes then max 1 e else e) s
  in
  let* v = view_of r in
  let* specials =
    frequency
      [ (2, constant 0); (2, constant 500); (1, constant 3000); (3, constant 20) ]
  in
  let+ seed = int in
  { scan; monoid; axes; x = operand d s v ~specials seed; views = view_names v }

let base =
  D.
    [
      Any Float32;
      Any Float64;
      Any Int8;
      Any Uint8;
      Any Int16;
      Any Uint16;
      Any Int32;
      Any Uint32;
      Any Int64;
      Any Uint64;
      Any Bool;
    ]

let floats = D.[ Any Float32; Any Float64 ]
(* Floats, whose order and NaNs the laws check, come twice as often. *)
let computed = Gen.with_pp pp_case (case_of (floats @ floats @ base))
let any_case = Gen.with_pp pp_case (case_of D.all)
(* Scans draw long axes as often as the others, so that prefixes pass a
   chunk. *)
let scans =
  Gen.with_pp pp_case
    (case_of ~scan:(Gen.constant true)
       ~shapes:(Gen.frequency [ (1, shape_gen); (1, long) ])
       base)

let sums =
  Gen.with_pp pp_case
    (case_of ~scan:(Gen.constant false) ~monoids:[ Sum ] floats)

(* Sums of one axis, of any length. *)
let dots =
  Gen.with_pp pp_case
    (case_of ~scan:(Gen.constant false) ~monoids:[ Sum ]
       ~shapes:(Gen.map (fun n -> [| n |]) (Gen.int_range 0 6000))
       floats)

(* Operands of a few hundred thousand elements, which the job gives several
   threads. *)
let large =
  Gen.with_pp pp_case
    (case_of
       ~shapes:
         (Gen.of_list
            [
              [| 400_000 |];
              [| 640; 640 |];
              [| 3; 150_000 |];
              [| 150_000; 3 |];
              [| 64; 8_000 |];
            ])
       (base @ D.[ Any Bfloat16; Any Int4 ]))

(* Examples: one case for each regime a law covers, so that every seed
   reaches every label whatever the generator draws. *)
let example ?(scan = false) ?view ?(specials = 0) monoid dt s axes =
  let r = Array.length s in
  let view =
    Option.value view
      ~default:
        {
          perm = Array.init r Fun.id;
          stepped = false;
          reversed = None;
          broadcast = false;
        }
  in
  {
    scan;
    monoid;
    axes;
    x = operand dt s view ~specials 1;
    views = view_names view;
  }

let f32 = D.Any D.Float32
let f64 = D.Any D.Float64

let computed_examples =
  [
    example Sum f32 [| 0; 3 |] [| 1 |];
    example Sum f32 [| 3; 0 |] [| 1 |];
    example Sum f32 [| 5 |] [||];
    example ~specials:20 Sum f32 [| 5000 |] [| 0 |];
    example Max f32 [| 100; 3 |] [| 1 |];
    (* Short rows of few terms, which units take in bands of rows. *)
    (* CR: These two examples and the matching large_examples case have
       contiguous kept axes, which coalesce into one: no multirow bands.
       Use the existing view with perm=[|0;2;1|], no step, reversal or
       broadcast. Its strides [26;1;13] retain row width 13 and separate
       outer rows, covering full and partial bands with the same laws. *)
    example ~specials:20 Sum f32 [| 46; 13; 2 |] [| 2 |];
    example ~specials:20 Max f64 [| 46; 13; 2 |] [| 2 |];
    example
      ~view:
        {
          perm = [| 2; 0; 1 |];
          stepped = true;
          reversed = Some 0;
          broadcast = true;
        }
      Sum f64 [| 3; 4; 5 |] [| 1 |];
  ]

(* A float vector of [xs] at [dt], read with a step of two where
   [stepped]. *)
let vector_of (type s) (dt : (float, s) D.t) ?(stepped = false) xs =
  let n = Array.length xs in
  if not stepped then A.Any (A.of_array dt [| n |] xs)
  else
    let spread i = if i mod 2 = 1 then xs.(i / 2) else 7. in
    let x = A.of_array dt [| 2 * n |] (Array.init (2 * n) spread) in
    let odd = { M.start = 1; count = n; step = 2 } in
    A.Any (Option.get (A.move (M.Slice [| odd |]) x))

(* NaNs made from numbers, before a NaN term and with none: opposite
   infinities summed, zero times an infinity, along one block, past one,
   past a chunk, in a few-term reduction's rows and through a stepped view. *)
let made_examples =
  let inf = Float.infinity and ninf = Float.neg_infinity in
  let nan = Int64.float_of_bits 0x7FF0_0000_0000_0042L in
  let case ?(views = []) ?(axes = [| 0 |]) scan monoid x =
    { scan; monoid; axes; x; views }
  in
  let f32 = vector_of D.Float32 and f64 = vector_of D.Float64 in
  let stepped = [ "stepped" ] in
  let at n pairs =
    Array.init n (fun i -> Option.value ~default:1. (List.assoc_opt i pairs))
  in
  let rows = at 128 [ (10, inf); (11, ninf) ] in
  let rows = A.Any (A.of_array D.Float32 [| 64; 2 |] rows) in
  [
    case false Sum (f32 [| inf; ninf |]);
    case false Prod (f32 [| 0.; inf |]);
    case true Sum (f32 [| inf; ninf; 1. |]);
    case ~views:stepped false Sum
      (f64 ~stepped:true [| 1.; inf; 2.; ninf; 3. |]);
    case ~views:stepped true Prod
      (f64 ~stepped:true [| 2.; 0.; inf; 1.; nan |]);
    case false Sum (f32 (at 1025 [ (0, inf); (1024, ninf) ]));
    case true Sum (f32 (at 4097 [ (3, inf); (4096, ninf) ]));
    case ~axes:[| 1 |] false Sum rows;
  ]

let any_examples =
  [
    example Sum f32 [| 4 |] [| 0 |];
    example Sum (D.Any D.Float16) [| 4 |] [| 0 |];
  ]

let scan_examples = [ example ~scan:true Sum f32 [| 9000 |] [| 0 |] ]

let dot_examples =
  [ example Sum f32 [| 3000 |] [| 0 |]; example Sum f32 [| 3 |] [||] ]

let large_examples =
  [
    example ~scan:true Sum f32 [| 400_000 |] [| 0 |];
    example Sum f32 [| 400_000 |] [| 0 |];
    example Sum f32 [| 4000; 13; 2 |] [| 2 |];
    example ~scan:true Sum (D.Any D.Bfloat16) [| 400_000 |] [| 0 |];
    (* Rows of more than 512 bytes of outputs and more than a chunk of
       terms, which one thread totals in pieces. *)
    example ~scan:true Sum f32 [| 4100; 300 |] [| 0 |];
    example ~scan:true Sum (D.Any D.Bfloat16) [| 4100; 300 |] [| 0 |];
  ]

(* The reference *)

(* A term: a float's value with its bits, or an integer's value as an
   [int64] of its bits, sign- or zero-extended. *)
type term = F of float * int64 | I of int64

(* [x]'s elements in C order of indices. *)
let terms (A.Any x) =
  let c = A.copy x in
  match A.dtype x with
  | D.Float32 ->
      Array.map
        (fun b -> F (Int32.float_of_bits b, Int64.of_int32 b))
        (A.to_array (Option.get (A.bitcast D.Int32 c)))
  | D.Float64 ->
      Array.map
        (fun b -> F (Int64.float_of_bits b, b))
        (A.to_array (Option.get (A.bitcast D.Int64 c)))
  | _ ->
      let d = A.create Rig.host D.Int64 (L.shape (A.layout c)) in
      (match Nx_cpu.apply1 P.Cast ~dst:d c with
      | A.Done -> ()
      | r -> failf "a cast answered %a" Nx_array_support.pp_answer r);
      Array.map (fun v -> I v) (A.to_array d)

(* An expected result: its bits, or some NaN. *)
type want = Bits of int64 | Some_nan

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* [v] in the integer dtype [dt]: its low bits, sign- or zero-extended. *)
let wrap (D.Any dt) v =
  let bits = D.bits dt in
  if bits >= 64 then v
  else
    let s = 64 - bits in
    if D.is D.Signed dt then Int64.shift_right (Int64.shift_left v s) s
    else Int64.shift_right_logical (Int64.shift_left v s) s

(* The monoid [m] at [dt] over two terms, and its identity. Floats round
   once to [dt], through a double, which holds float32's sums and products
   exactly or rounds them innocuously; a NaN operand gives the first. An
   extreme's identity is the least (greatest) element. *)
let op (D.Any dt as d) m =
  let float x =
    if D.equal dt D.Float32 then
      let x = round32 x in
      F (x, Int64.of_int32 (Int32.bits_of_float x))
    else F (x, Int64.bits_of_float x)
  in
  let unsigned = not (D.is D.Signed dt) in
  let cmp x y =
    if unsigned then Int64.unsigned_compare x y else Int64.compare x y
  in
  let least = if unsigned then 0L else wrap d (Int64.shift_left 1L (D.bits dt - 1)) in
  let greatest = Int64.lognot least |> wrap d in
  (* The greater of two floats, -0 below +0. *)
  let fmax x y =
    if x > y then x else if y > x then y else if Float.sign_bit x then y else x
  in
  let fmin x y =
    if x < y then x else if y < x then y else if Float.sign_bit x then x else y
  in
  let both fl it a b =
    match (a, b) with
    | F (x, _), _ when Float.is_nan x -> a
    | _, F (y, _) when Float.is_nan y -> b
    | F (x, _), F (y, _) -> float (fl x y)
    | I x, I y -> I (it x y)
    | _ -> invalid_arg "op: a float and an integer"
  in
  let is_float = D.is D.Float dt in
  match m with
  | Sum ->
      (both ( +. ) (fun x y -> wrap d (Int64.add x y)), if is_float then float 0. else I 0L)
  | Prod ->
      (both ( *. ) (fun x y -> wrap d (Int64.mul x y)), if is_float then float 1. else I 1L)
  | Max ->
      ( both fmax (fun x y -> if cmp x y >= 0 then x else y),
        if is_float then float neg_infinity else I least )
  | Min ->
      ( both fmin (fun x y -> if cmp x y <= 0 then x else y),
        if is_float then float infinity else I greatest )
  | Logsumexp -> invalid_arg "op: Logsumexp"

(* [ts] reduced in nx.cpu's order: blocks of 1024 terms in 16 lanes from
   the identity, or from their first term where [first], term t into lane
   t mod 16, the lanes' tree, in which a lane holding no term takes no
   part, then the blocks' tree whose left part holds the largest power of
   two of blocks below their count. *)
let ordered ?(first = false) (f, id) ts =
  let k = Array.length ts in
  let block x =
    let l = Array.make 16 id in
    let hi = min k ((x + 1) * 1024) in
    for t = x * 1024 to hi - 1 do
      l.(t mod 16) <-
        (if first && t - (x * 1024) < 16 then ts.(t) else f l.(t mod 16) ts.(t))
    done;
    let used = min 16 (hi - (x * 1024)) in
    List.iter
      (fun w ->
        for i = 0 to w - 1 do
          if i + w < used then l.(i) <- f l.(i) l.(i + w)
        done)
      [ 8; 4; 2; 1 ];
    l.(0)
  in
  let rec tree s n =
    if n = 1 then s.(0)
    else
      let h = ref 1 in
      while 2 * !h < n do
        h := 2 * !h
      done;
      f (tree s !h) (tree (Array.sub s !h (n - !h)) (n - !h))
  in
  let n = (k + 1023) / 1024 in
  if n = 0 then id else tree (Array.init n block) n

let is_nan = function F (x, _) -> Float.is_nan x | I _ -> false
let bits_of = function F (_, b) -> b | I v -> v

(* The NaN the kinds make from numbers at the float dtype [dt]: the quiet
   NaN of a clear sign bit. *)
let made (D.Any dt) =
  if D.bits dt = 32 then 0x7FC00000L else 0x7FF8_0000_0000_0000L

(* The bits of [t], a NaN of [made]'s dtype, quieted: its quiet bit, the
   top of its fraction, set, its sign and the rest of its payload kept. *)
let quieted made t =
  Int64.logor (bits_of t)
    (if made = 0x7FC00000L then 0x40_0000L else 0x8_0000_0000_0000L)

(* [t]'s bits, quieted where a NaN, [u] its dtype's roundoff. *)
let quiet_u u t =
  if not (is_nan t) then bits_of t
  else quieted (if u > 0x1p-30 then 0x7FC00000L else 0x7FF8000000000000L) t

(* A float result's NaN holds its first NaN term, quieted, else it is
   [made]'s. *)
let settle made ts r =
  if not (is_nan r) then Bits (bits_of r)
  else
    match Array.find_opt is_nan ts with
    | Some t -> Bits (quieted made t)
    | None -> Bits made

(* The terms of each output of [c], in C order of outputs, each in C order
   of the reduced axes' indices; a scan's are its slices, in order. *)
let outputs c ts =
  let s = shape_of c.x in
  let r = Array.length s in
  let strides = Array.make r 1 in
  for i = r - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * s.(i + 1)
  done;
  let kept = List.filter (fun i -> not (Array.mem i c.axes)) (List.init r Fun.id) in
  let ks = Array.of_list kept in
  let at idx = Array.fold_left ( + ) 0 (Array.mapi (fun i j -> j * strides.(i)) idx) in
  let outer = indices (Array.map (fun i -> s.(i)) ks) in
  let inner = indices (Array.map (fun a -> s.(a)) c.axes) in
  List.map
    (fun o ->
      let idx = Array.make r 0 in
      Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
      Array.of_list
        (List.map
           (fun t ->
             Array.iteri (fun q a -> idx.(a) <- t.(q)) c.axes;
             ts.(at idx))
           inner))
    outer

(* [ts] scanned in nx.cpu's order: chunks of 4096 terms from the start, each
   result the carry combined with the chunk's terms up to it, left to right,
   the carry into a chunk the carry into the one before combined with that
   chunk's total in [ordered]'s order. From the first NaN term on, a float
   result is that term, quieted; a NaN result before it is [made]'s. *)
let scanned made ((f, id) as m) ts =
  let n = Array.length ts in
  let out = Array.make n id in
  let carry = ref id in
  for c = 0 to ((n + 4095) / 4096) - 1 do
    let lo = 4096 * c and hi = min n (4096 * (c + 1)) in
    let acc = ref !carry in
    for i = lo to hi - 1 do
      acc := f !acc ts.(i);
      out.(i) <- !acc
    done;
    carry := f !carry (ordered m (Array.sub ts lo (hi - lo)))
  done;
  let first = Array.find_index is_nan ts in
  Array.mapi
    (fun i v ->
      match first with
      | Some k when i >= k -> Bits (quieted made ts.(k))
      | _ -> if is_nan v then Bits made else Bits (bits_of v))
    out

(* The position of each index of shape [s] in C order. *)
let position s idx =
  let p = ref 0 in
  Array.iteri (fun i j -> p := (!p * s.(i)) + j) idx;
  !p

(* The results of [c] in C order of the result's indices, from its terms
   [ts]: [reduce] of each output's terms, or [scan] of each slice's. [fill]
   fills the array before. *)
let results c ts ~reduce ~scan ~fill =
  if not c.scan then Array.of_list (List.map reduce (outputs c ts))
  else
    let s = shape_of c.x and a = c.axes.(0) in
    let r = Array.length s in
    let ks = Array.of_list (List.filter (( <> ) a) (List.init r Fun.id)) in
    let y = Array.make (total s) fill in
    List.iter
      (fun o ->
        let idx = Array.make r 0 in
        Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
        let at j =
          idx.(a) <- j;
          position s idx
        in
        let slice = Array.init s.(a) (fun j -> ts.(at j)) in
        Array.iteri (fun j w -> y.(at j) <- w) (scan slice))
      (indices (Array.map (fun i -> s.(i)) ks));
    y

(* The expected results of [c] in C order of the result's indices. *)
let expected c =
  let (A.Any x) = c.x in
  let m = op (D.Any (A.dtype x)) c.monoid in
  let made = made (D.Any (A.dtype x)) in
  results c (terms c.x) ~fill:Some_nan
    ~reduce:(fun o -> settle made o (ordered m o))
    ~scan:(scanned made m)

(* The kernels' result's bits in C order. *)
let got y =
  Array.map
    (function F (x, b) -> if Float.is_nan x then Some_nan else Bits b | I v -> Bits v)
    (terms y)

let show = function Bits b -> Printf.sprintf "%Lx" b | Some_nan -> "nan"

(* The first results that differ from [want]: a [Some_nan] wanted matches
   any NaN; [Bits] of a NaN only those bits. *)
let differ want y =
  let g = terms y in
  let bad = ref [] in
  Array.iteri
    (fun i w ->
      let ok =
        match (w, g.(i)) with
        | Some_nan, t -> is_nan t
        | Bits b, t -> bits_of t = b
      in
      if (not ok) && List.length !bad < 8 then
        bad := Printf.sprintf "result %d: %Lx, expected %s" i (bits_of g.(i)) (show w) :: !bad)
    want;
  List.rev !bad

(* Complex numbers *)

(* A complex term: its parts' values, which a double holds exactly, and
   their bits as stored, a float32's sign-extended. A number with a NaN part
   is a NaN. *)
type z = { re : float; im : float; bits : int64 * int64 }

let znan z = Float.is_nan z.re || Float.is_nan z.im

(* The number of parts [re] and [im], of float32 where [narrow]. *)
let zof narrow re im =
  let b v =
    if narrow then Int64.of_int32 (Int32.bits_of_float v)
    else Int64.bits_of_float v
  in
  { re; im; bits = (b re, b im) }

let narrow (A.Any x) = D.equal (A.dtype x) D.Complex64

(* [x]'s complex elements in C order of indices. *)
let zterms (A.Any x as a) =
  let c = A.copy x in
  let bits =
    match A.dtype c with
    | D.Complex64 ->
        let p = Option.get (A.bitcast D.Float32 c) in
        Array.map Int64.of_int32 (A.to_array (Option.get (A.bitcast D.Int32 p)))
    | D.Complex128 ->
        let p = Option.get (A.bitcast D.Float64 c) in
        A.to_array (Option.get (A.bitcast D.Int64 p))
    | _ -> invalid_arg "zterms: not a complex array"
  in
  let v b =
    if narrow a then Int32.float_of_bits (Int64.to_int32 b)
    else Int64.float_of_bits b
  in
  Array.init
    (Array.length bits / 2)
    (fun i ->
      let rb = bits.(2 * i) and ib = bits.((2 * i) + 1) in
      { re = v rb; im = v ib; bits = (rb, ib) })

(* The float32 next to [x] toward [y]. *)
let step32 x y =
  if x = 0. then Float.copy_sign 0x1p-149 y
  else
    let b = Int32.bits_of_float x in
    Int32.float_of_bits
      (if y > x = (x > 0.) then Int32.succ b else Int32.pred b)

(* [a b + c] of float32s rounded once to float32. The exact product is a
   double; their sum [s] rounds as the exact value does, since every
   float32 midpoint is a double, but at a midpoint, where the sum's error
   [e] (Knuth's TwoSum) decides. *)
let fma32 a b c =
  let p = Sys.opaque_identity (a *. b) in
  let s = p +. c in
  let r = round32 s in
  if r = s || not (Float.is_finite s) then r
  else
    let t = s -. p in
    let e = (p -. (s -. t)) +. (c -. t) in
    let o = step32 r s in
    if (r +. o) /. 2. <> s || e = 0. then r
    else if e > 0. = (o > r) then o
    else r

(* nx_kinds.h's complex Mul: each part one fma over the other product. *)
let zmul narrow a b =
  let rd = if narrow then round32 else Fun.id in
  let fma = if narrow then fma32 else Float.fma in
  let ii = rd (a.im *. b.im) and ir = rd (a.im *. b.re) in
  zof narrow (fma a.re b.re (-.ii)) (fma a.re b.im ir)

(* The order of real parts, then imaginary parts, [-0] below [+0]. *)
let zorder a b =
  let part x y =
    if x = y then Bool.compare (Float.sign_bit y) (Float.sign_bit x)
    else Float.compare x y
  in
  let c = part a.re b.re in
  if c <> 0 then c else part a.im b.im

(* [m] at the complex dtype, float32's where [narrow], and its identity: a
   NaN operand of an extreme gives the first. *)
let zop narrow m =
  let extreme keep a b =
    if znan a then a
    else if znan b then b
    else if keep (zorder a b) then a
    else b
  in
  match m with
  | Prod -> (zmul narrow, zof narrow 1. 0.)
  | Max -> (extreme (fun c -> c >= 0), zof narrow neg_infinity neg_infinity)
  | Min -> (extreme (fun c -> c <= 0), zof narrow infinity infinity)
  | Sum | Logsumexp -> invalid_arg "zop: Sum or Logsumexp"

(* The quiet NaN of a clear sign bit in each part. *)
let zmade narrow =
  let q = Int64.float_of_bits 0x7FF8_0000_0000_0000L in
  (zof narrow q q).bits

(* [t]'s bits with each NaN part quieted, of complex64's parts where
   [narrow]. *)
let zquieted narrow t =
  let q = if narrow then 0x40_0000L else 0x8_0000_0000_0000L in
  let part x b = if Float.is_nan x then Int64.logor b q else b in
  (part t.re (fst t.bits), part t.im (snd t.bits))

(* A NaN result's bits are its first NaN term's, quieted, else
   [zmade]'s. *)
let zsettle narrow ts r =
  if not (znan r) then r.bits
  else
    match Array.find_opt znan ts with
    | Some t -> zquieted narrow t
    | None -> zmade narrow

(* [ts] scanned as [scanned] does, in bits, but from the first term where
   [first]: the first chunk then has no carry. *)
let zscanned narrow ~first ((f, id) as m) ts =
  let n = Array.length ts in
  let out = Array.make n id in
  let step a t = match a with None -> t | Some a -> f a t in
  let carry = ref (if first then None else Some id) in
  for c = 0 to ((n + 4095) / 4096) - 1 do
    let lo = 4096 * c and hi = min n (4096 * (c + 1)) in
    let acc = ref !carry in
    for i = lo to hi - 1 do
      acc := Some (step !acc ts.(i));
      out.(i) <- Option.get !acc
    done;
    carry := Some (step !carry (ordered ~first m (Array.sub ts lo (hi - lo))))
  done;
  let first = Array.find_index znan ts in
  Array.mapi
    (fun i v ->
      match first with
      | Some k when i >= k -> zquieted narrow ts.(k)
      | _ -> if znan v then zmade narrow else v.bits)
    out

(* A complex sum's results: each part's, a float sum of the terms' parts in
   nx.cpu's order. *)
let zsums c ts =
  let part = if narrow c.x then D.Any D.Float32 else D.Any D.Float64 in
  let m = op part Sum and made = made part in
  let sums f =
    results c (Array.map f ts) ~fill:Some_nan
      ~reduce:(fun o -> settle made o (ordered m o))
      ~scan:(scanned made m)
  in
  let bits = function Bits b -> b | Some_nan -> invalid_arg "zsums: a NaN" in
  Array.map2
    (fun r i -> (bits r, bits i))
    (sums (fun t -> F (t.re, fst t.bits)))
    (sums (fun t -> F (t.im, snd t.bits)))

(* A product starts from its first term: 1 + 0i times a term would change
   its signed zeros and make NaN parts from its infinite ones. *)
let zexpected c =
  let narrow = narrow c.x and first = c.monoid = Prod in
  let ts = zterms c.x in
  if c.monoid = Sum then zsums c ts
  else
    let m = zop narrow c.monoid in
    results c ts ~fill:(0L, 0L)
      ~reduce:(fun o -> zsettle narrow o (ordered ~first m o))
      ~scan:(zscanned narrow ~first m)

(* Complex sums, products and extremes, reduced and scanned, in nx.cpu's
   order, bit for bit. *)
let law_complex (b : Support.backend) c =
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  cover "no term" (per = 0 && total (result_shape c) > 0);
  cover "several blocks" (per > 1024);
  let ts = zterms c.x in
  cover "a NaN term" (Array.exists znan ts);
  (* A product whose last block leaves lanes without a term, which its tree
     leaves out, of a zero part, whose sign 1 + 0i times would change. *)
  let zero t = t.re = 0. || t.im = 0. in
  cover "a product of a lane of no term and a zero part"
    (c.monoid = Prod && per mod 1024 > 0 && per mod 1024 < 16
    && Array.exists zero ts);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  match run b c with
  | None -> failf "%s declined a complex %s" b.name (monoid_name c.monoid)
  | Some y ->
      let want = zexpected c and got = zterms y in
      let bad = ref [] in
      Array.iteri
        (fun i (wr, wi) ->
          let gr, gi = got.(i).bits in
          if (gr, gi) <> (wr, wi) && List.length !bad < 8 then
            bad :=
              Printf.sprintf "result %d: %Lx %Lx, expected %Lx %Lx" i gr gi wr
                wi
              :: !bad)
        want;
      equal (list string) [] (List.rev !bad)

let complex_case =
  Gen.with_pp pp_case
    (case_of D.[ Any Complex64; Any Complex128 ])

(* [n] terms, -0 - i then 1 - 0i, whose products in every order are
   -0 - i; 1 + 0i times that would be +0 - i. *)
let units n =
  let z re im = { Complex.re; im } in
  A.Any
    (A.of_array D.Complex64 [| n |]
       (Array.init n (fun i -> if i = 0 then z (-0.) (-1.) else z 1. (-0.))))

(* Products of one term, which are that term: -0 - 0i, which 1 + 0i times
   would make (+0, -0), 3 - 0i, (3, +0), and inf + 0i, (inf, NaN); the
   scans of two, whose first results are so; and scans past a chunk, of
   one slice, of slices side by side and of rows. *)
let complex_examples =
  let one ~scan dt ts =
    {
      scan;
      monoid = Prod;
      axes = [| 0 |];
      x = A.Any (A.of_array dt [| Array.length ts |] ts);
      views = [];
    }
  in
  let z re im = Complex.{ re; im } in
  [
    one ~scan:false D.Complex128 [| z (-0.) (-0.) |];
    one ~scan:false D.Complex64 [| z 3. (-0.) |];
    one ~scan:false D.Complex128 [| z infinity 0. |];
    one ~scan:true D.Complex64 [| z infinity 0.; z 2. 0. |];
    one ~scan:true D.Complex128 [| z (-0.) (-0.); z 1. 1. |];
    (* Outputs of no term, which the generator draws now and then. *)
    example Sum (D.Any D.Complex64) [| 3; 0 |] [| 1 |];
  ]
  @ List.map
      (fun (s, axis) ->
        let plain =
          {
            perm = Array.init (Array.length s) Fun.id;
            stepped = false;
            reversed = None;
            broadcast = false;
          }
        in
        {
          scan = true;
          monoid = Prod;
          axes = [| axis |];
          x = operand_with units s plain;
          views = [];
        })
      [ ([| 9000 |], 0); ([| 3; 5000 |], 1); ([| 4100; 3 |], 0) ]

(* Covers *)

let covers c =
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  let outs = total (result_shape c) in
  cover "no output" (outs = 0);
  cover "no term" (per = 0 && outs > 0);
  cover "one term" (per = 1);
  cover "two terms" (per = 2);
  let pow2 k = k > 0 && k land (k - 1) = 0 in
  cover "2^k ± 1 terms" (per > 2 && (pow2 (per - 1) || pow2 (per + 1)));
  cover "several blocks" (per > 1024);
  cover "many outputs" (outs >= 64);
  let nans =
    List.filter_map (Array.find_index is_nan) (outputs c (terms c.x))
  in
  cover "a NaN term" (nans <> []);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ]

(* The laws of S *)

(* A float sum of [n] terms is within γ(n - 1) Σ|x| of the exact sum; a NaN
   or infinite sum is one where the plain sum is. *)
let law_bound (b : Support.backend) c =
  let (A.Any x) = c.x in
  match (A.dtype x, c.monoid, c.scan) with
  | (D.Float32 | D.Float64), Sum, false -> (
      match run b c with
      | None -> cover "declined" true
      | Some y ->
          cover "computed" true;
          let u = if D.bits (A.dtype x) = 32 then 0x1p-24 else 0x1p-53 in
          let per = outputs c (terms c.x) in
          let g = terms y in
          let bad =
            List.concat
              (List.mapi
                 (fun i o ->
                   let xs = Array.map (function F (v, _) -> v | I _ -> assert false) o in
                   let plain = Array.fold_left ( +. ) 0. xs in
                   let v = match g.(i) with F (v, _) -> v | I _ -> assert false in
                   if not (Float.is_finite plain) then
                     if Float.is_nan plain = Float.is_nan v && (Float.is_nan v || v = plain) then []
                     else [ Printf.sprintf "result %d: %h, the plain sum %h" i v plain ]
                   else
                     let n = float_of_int (max 0 (Array.length xs - 1)) in
                     let mag = Array.fold_left (fun a x -> a +. Float.abs x) 0. xs in
                     (* The exact sum: TwoSum's errors carried in a second part. *)
                     let hi = ref 0. and lo = ref 0. in
                     Array.iter
                       (fun x ->
                         let s = !hi +. x in
                         let z = s -. !hi in
                         lo := !lo +. (!hi -. (s -. z)) +. (x -. z);
                         hi := s)
                       xs;
                     let exact = !hi +. !lo in
                     let g = n *. u /. (1. -. (n *. u)) in
                     let bound = (g *. mag) +. (0x1p-100 *. mag) in
                     if Float.abs (v -. exact) <= bound then []
                     else [ Printf.sprintf "result %d: %h is %h from %h, past %h" i v (v -. exact) exact bound ])
                 per)
          in
          equal (list string) [] (List.filteri (fun i _ -> i < 8) bad))
  | _ -> cover "not a float sum" true

(* Each operand as a C-contiguous copy of its view gives the same bits. *)
let law_layouts b c =
  let (A.Any x) = c.x in
  let c' = { c with x = A.Any (A.copy x); views = [] } in
  match (run b c, run b c') with
  | Some y, Some y' ->
      cover "computed from views" (c.views <> []);
      equal (list string) [] (differ (got y') y)
  | None, None -> cover "declined" true
  | _ -> failf "a view and its copy were not both computed"

(* The scan of a prefix of the axis is the prefix of the scan. *)
let law_prefix (b : Support.backend) c =
  let s = shape_of c.x and a = c.axes.(0) in
  let m = s.(a) / 2 in
  let (A.Any x) = c.x in
  let cut =
    A.Any
      (Option.get
         (A.move
            (M.Slice
               (Array.mapi
                  (fun i e ->
                    if i = a then { M.start = 0; count = m; step = 1 }
                    else { M.start = 0; count = e; step = 1 })
                  s))
            x))
  in
  match (run b c, run b { c with x = cut }) with
  | Some (A.Any y), Some y' ->
      cover "a prefix past a chunk" (m > 4096);
      let prefix =
        A.Any
          (Option.get
             (A.move
                (M.Slice
                   (Array.mapi
                      (fun i e ->
                        if i = a then { M.start = 0; count = m; step = 1 }
                        else { M.start = 0; count = e; step = 1 })
                      s))
                y))
      in
      equal (list string) [] (differ (got prefix) y')
  | None, None -> cover "declined" true
  | _ -> failf "a scan and its prefix were not both computed"

(* A kernel that declines writes nothing. *)
(* CR: terms casts narrow floats and complex values to Int64, so writes
   can disappear (Float16 0.5 and 0.75 both become 0). Initialize dst
   with deterministic bytes, keeping spare packed bits zero, and compare
   its full buffer via Rig.Buffer.copy to host and blit_to_bytes. Use the
   same byte check in test_contract's refusal law. Keep the Float16 reduce
   example and add its nonempty scan counterpart. *)
let law_declined (b : Support.backend) c =
  let (A.Any x) = c.x in
  let dst = on b (A.Any (A.create Rig.host (A.dtype x) (result_shape c))) in
  let bits () = Array.map bits_of (terms (host dst)) in
  let before = bits () in
  match call b c dst with
  | A.Declined ->
      cover "declined" true;
      equal (array int64) before (bits ())
  | A.Done -> cover "computed" true
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* nx.cpu's cases and order *)

(* Whether a float result of [c] is a NaN made from numbers: one whose terms,
   a scan's up to it, hold no NaN. *)
let makes_nan c =
  let (A.Any x) = c.x in
  let dt = D.Any (A.dtype x) in
  match A.dtype x with
  | D.Float32 | D.Float64 ->
      let m = op dt c.monoid and made = made dt in
      let one o =
        let n = Array.length o in
        let k = Option.value ~default:n (Array.find_index is_nan o) in
        if c.scan then Array.mem (Bits made) (scanned made m (Array.sub o 0 k))
        else k = n && is_nan (ordered m o)
      in
      List.exists one (outputs c (terms c.x))
  | _ -> false

(* The dtype nx.cpu reduces terms of [dt] in: float32 for a narrow float,
   int8 for int4, uint8 for uint4, bool for bit, else [dt]. *)
let carrier (D.Any dt as d) =
  match dt with
  | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
  | D.Float4_e2m1fn ->
      D.Any D.Float32
  | D.Int4 -> D.Any D.Int8
  | D.Uint4 -> D.Any D.Uint8
  | D.Bit -> D.Any D.Bool
  | _ -> d

let same (D.Any a) (D.Any b) = D.equal a b
let staged d = not (same (carrier d) d)

let law_order (b : Support.backend) c =
  covers c;
  cover "a NaN made from numbers" (makes_nan c);
  match run b c with
  | None -> failf "%s declined a case nx_cpu.mli says it computes" b.name
  | Some y -> equal (list string) [] (differ (expected c) y)

let law_computes b c =
  cover "narrow" (staged (dtype_of c.x));
  equal ~msg:"computed" bool true (Option.is_some (run b c))

(* Narrow dtypes and destinations of another dtype *)

(* [x] cast into [d] by nx.cpu, C-contiguous. *)
let cast_into (D.Any d) (A.Any x) =
  let dst = A.create Rig.host d (L.shape (A.layout x)) in
  match Nx_cpu.apply1 P.Cast ~dst x with
  | A.Done -> A.Any dst
  | r -> failf "a cast answered %a" Nx_array_support.pp_answer r

(* [c] by [k] into a fresh destination of [d]. *)
let run_into k (D.Any d) c x =
  let dst = A.Any (A.create Rig.host d (result_shape c)) in
  match compute k ~scan:c.scan (identity (dtype_of x)) c.monoid c.axes dst
          [| plain_of x |] with
  | A.Done -> dst
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let narrow_dtypes =
  D.
    [
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
      Any Int4;
      Any Uint4;
      Any Bit;
    ]

(* A case of a narrow dtype, or of a byte-wide one, and a destination's
   dtype: mostly the operand's, else another. A vector past a scan's chunk
   is drawn often, so that a scan runs its chunks side by side. *)
let into_case =
  Gen.with_pp
    (fun ppf (c, d) -> Format.fprintf ppf "%a into %a" pp_case c pp_dtype d)
    (let open Gen in
     let shapes =
       frequency
         [ (3, shape_gen); (1, map (fun n -> [| n |]) (int_range 4097 20000)) ]
     in
     let* c =
       case_of ~shapes (narrow_dtypes @ narrow_dtypes @ floats @ D.[ Any Int32 ])
     in
     let+ d =
       frequency
         [
           (3, constant (dtype_of c.x));
           ( 1,
             of_list ~pp:pp_dtype
               D.[ Any Float32; Any Float64; Any Float16; Any Int64; Any Int8 ] );
         ]
     in
     (c, d))

(* For each result of [c] in C order, its first NaN term's bits, quieted, a
   scan's up to it, if one is a NaN. *)
let first_nans c =
  let raw = Array.map (Elements.quiet (dtype_of c.x)) (Elements.elements c.x) in
  let nan = Array.map is_nan (terms (cast_into f32 c.x)) in
  let first o = Option.map (Array.get raw) (Array.find_opt (Array.get nan) o) in
  let scan o =
    let k = Array.find_index (Array.get nan) o in
    Array.mapi
      (fun j _ -> match k with Some k when k <= j -> first o | _ -> None)
      o
  in
  let all = Array.init (Array.length raw) Fun.id in
  results c all ~reduce:first ~scan ~fill:None

(* NaNs made from numbers at narrow floats, reduced and scanned, into their
   own dtype and float32. *)
let into_examples =
  let inf = Float.infinity and ninf = Float.neg_infinity in
  let plain2 =
    { perm = [| 0; 1 |]; stepped = false; reversed = None; broadcast = false }
  in
  let case scan monoid x = { scan; monoid; axes = [| 0 |]; x; views = [] } in
  let bf16 = vector_of D.Bfloat16 [| inf; ninf; 1. |] in
  [
    (case false Sum bf16, D.Any D.Bfloat16);
    (case true Sum bf16, D.Any D.Bfloat16);
    (case false Prod (vector_of D.Float8_e5m2 [| 0.; inf |]), f32);
    (* A scan down columns, a unit a whole row of 4096 staged outputs. *)
    ( {
        scan = true;
        monoid = Sum;
        axes = [| 0 |];
        x = operand (D.Any D.Bfloat16) [| 5; 4100 |] plain2 ~specials:0 3;
        views = [];
      },
      D.Any D.Bfloat16 );
  ]

(* A narrow operand's terms reduce in its carrier, and each result rounds
   once into its destination: the bits of the carrier's reduction of the
   operand cast into it, cast into the destination. A NaN result into the
   operand's own dtype is its first NaN term quieted, as nx_kernel.mli
   states. *)
let law_into (b : Support.backend) (c, d) =
  let x = dtype_of c.x in
  cover "narrow" (staged x);
  cover "into another dtype" (not (same d x));
  cover "scan" c.scan;
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  cover "several blocks" (per > 1024);
  cover "a scan of several chunks" (c.scan && per > 4096);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  let got = run_into b.kernels d c c.x in
  let wide = run_into b.kernels (carrier x) c (cast_into (carrier x) c.x) in
  let firsts = first_nans c in
  let made =
    Array.exists2 (fun t f -> is_nan t && f = None) (terms wide) firsts
  in
  cover "a NaN made from numbers" made;
  let want = Elements.elements (cast_into d wide) in
  let want =
    if not (same d x) then want
    else Array.map2 (fun w f -> Option.value f ~default:w) want firsts
  in
  equal (array string) want (Elements.elements got)

(* A NaN result of a narrow float into its own dtype is its first NaN
   term quieted, whether signalling or quiet: the reduction of zeros with
   two NaNs, a few terms and past a block and a scan's chunk, by each
   monoid, Logsumexp and Moments, the running results from the first NaN
   on, and Arg's extreme, with that NaN's position, reduced and scanned. *)
let test_nan_bits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let code w c =
    String.init w (fun j -> Char.chr ((c lsr (8 * j)) land 0xFF))
  in
  let check (D.Any dt as d) first other (n, i, j) =
    let w = D.bits dt / 8 in
    let es =
      Array.init n (fun k ->
          code w (if k = i then first else if k = j then other else 0))
    in
    let x = Elements.of_elements d [| n |] es in
    let nan = Elements.quiet d es.(i) in
    (* Zeros' running results, then the first NaN's. *)
    let running = Array.init n (fun k -> if k < i then es.(0) else nan) in
    let run scan m =
      let c = { scan; monoid = m; axes = [| 0 |]; x; views = [] } in
      Elements.elements (run_into b.kernels d c x)
    in
    let msg r =
      Format.asprintf "%s of %d %a terms, %x first" r n D.pp dt first
    in
    List.iter
      (fun m ->
        let msg = msg (monoid_name m) in
        equal ~msg (array string) [| nan |] (run false m);
        equal ~msg:(msg ^ ", scanned") (array string) running (run true m))
      monoids;
    let stat ~scan r k =
      let s = if scan then [| n |] else [||] in
      let dsts = Array.init k (fun _ -> A.Any (A.create Rig.host dt s)) in
      let loads = [| plain_of x |] in
      let answer =
        if scan then K.scan (identity d) ~axis:0 (r, 0) ~dsts loads
        else
          K.reduce (identity d) ~shape:[| n |] ~axes:[| 0 |] [| (r, 0) |] ~dsts
            loads
      in
      equal ~msg:"its answer" bool true (answer = A.Done);
      Array.map Elements.elements dsts
    in
    equal ~msg:(msg "Logsumexp") (array (array string)) [| [| nan |] |]
      (stat ~scan:false (Monoid Logsumexp) 1);
    equal ~msg:(msg "Moments") (array (array string))
      [| [| nan |]; [| nan |] |]
      (stat ~scan:false Moments 2);
    equal ~msg:(msg "Moments, scanned") (array (array string))
      [| running; running |]
      (stat ~scan:true Moments 2);
    let lse = (stat ~scan:true (Monoid Logsumexp) 1).(0) in
    equal ~msg:(msg "Logsumexp, scanned, from the first NaN") (array string)
      (Array.sub running i (n - i))
      (Array.sub lse i (n - i));
    List.iter
      (fun (e : Nx_kernel.extreme) ->
        let v = A.create Rig.host dt [||]
        and at = A.create Rig.host D.Int64 [||] in
        let answer =
          K.reduce (identity d) ~shape:[| n |] ~axes:[| 0 |] [| (Arg e, 0) |]
            ~dsts:[| A.Any v; A.Any at |] [| plain_of x |]
        in
        equal ~msg:"its answer" bool true (answer = A.Done);
        equal
          ~msg:(msg (if e = Max then "Arg Max" else "Arg Min"))
          (pair (array string) int64)
          ([| nan |], Int64.of_int i)
          (Elements.elements (A.Any v), (A.to_array at).(0));
        let v = A.create Rig.host dt [| n |]
        and at = A.create Rig.host D.Int64 [| n |] in
        let answer =
          K.scan (identity d) ~axis:0 (Arg e, 0) ~dsts:[| A.Any v; A.Any at |]
            [| plain_of x |]
        in
        equal ~msg:"its answer" bool true (answer = A.Done);
        equal
          ~msg:(msg (if e = Max then "Arg Max scanned" else "Arg Min scanned"))
          (pair (array string) (array int64))
          ( running,
            Array.init n (fun k -> if k < i then 0L else Int64.of_int i) )
          (Elements.elements (A.Any v), A.to_array at))
      [ Max; Min ]
  in
  List.iter
    (fun (d, first, other) ->
      List.iter (check d first other) [ (4, 1, 2); (5000, 4500, 4700) ])
    D.
      [
        (Any Float16, 0x7C01, 0x7E00);
        (Any Bfloat16, 0x7F81, 0x7FC0);
        (Any Float8_e5m2, 0x7D, 0x7F);
        (Any Float8_e5m2, 0x7E, 0x7F);
        (Any Float8_e5m2, 0xFD, 0x7E);
        (Any Float8_e4m3fn, 0xFF, 0x7F);
      ]

(* A sum of one axis is a contraction of it with ones, in lane order: the
   two share their order. *)
let law_contract (b : Support.backend) c =
  let (A.Any x) = c.x in
  let s = shape_of c.x in
  let ones n : A.any option =
    match A.dtype x with
    | D.Float32 -> Some (A.Any (A.of_array D.Float32 [| n |] (Array.make n 1.)))
    | D.Float64 -> Some (A.Any (A.of_array D.Float64 [| n |] (Array.make n 1.)))
    | _ -> None
  in
  let sum_of_axis = (not c.scan) && c.monoid = Sum && Array.length s = 1 && c.axes = [| 0 |] in
  match if sum_of_axis then ones s.(0) else None with
  | Some (A.Any ones) -> (
      let module K = (val b.kernels) in
      let dst = A.create Rig.host (A.dtype x) [||] in
      let summed =
        K.contract ~order:Lane ~batch:[||] ~contracting:[| (0, 0) |]
          ~acc:(dtype_of c.x) ~dst ~init:None x ones
      in
      match (run b c, summed) with
      | Some y, A.Done ->
          cover "computed" true;
          equal (list string) [] (differ (got (A.Any dst)) y)
      | _ -> cover "declined" true)
  | _ -> cover "not a sum of one float axis" true

(* nx.cpu with its reductions and scans on one thread. *)
module One = struct
  include Nx_cpu

  let reduce = Nx_cpu.reduce_on ~threads:1
  let scan = Nx_cpu.scan_on ~threads:1
end

(* One thread gives the bits of the job's threads. *)
let law_threads (b : Support.backend) c =
  let (A.Any x) = c.x in
  let into k =
    let dst = A.create Rig.host (A.dtype x) (result_shape c) in
    let p = identity (dtype_of c.x) in
    match compute k ~scan:c.scan p c.monoid c.axes (A.Any dst) [| plain_of c.x |] with
    | A.Done -> A.Any dst
    | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r
  in
  let one = into (module One) and many = into b.kernels in
  let d = dtype_of c.x in
  cover "scan" c.scan;
  cover "few outputs" (total (result_shape c) < 64);
  cover "narrow" (staged d);
  (* A narrow result read through its carrier, which holds its value. *)
  let wide y = if staged d then cast_into (carrier d) y else y in
  let exact y = Array.map (fun t -> Bits (bits_of t)) (terms (wide y)) in
  equal (list string) [] (differ (exact many) (wide one))

(* A NaN made from numbers before two NaN terms past the first block and
   chunk: the sum, the maximum and the running sum hold the first NaN term,
   bit for bit, and agree with the reference along the way. *)
let test_far_nans (b : Support.backend) () =
  let n = 9000 in
  let r = Random.State.make [| 7 |] in
  let xs = Array.init n (fun _ -> Int32.bits_of_float (full r 23)) in
  xs.(100) <- Int32.bits_of_float infinity;
  xs.(101) <- Int32.bits_of_float neg_infinity;
  xs.(5000) <- 0x7FC12345l;
  xs.(6000) <- 0xFFC00042l;
  let bytes = Bytes.create (4 * n) in
  Array.iteri (fun i v -> Bytes.set_int32_le bytes (4 * i) v) xs;
  let x =
    A.Any
      (A.v D.Float32 (L.contiguous [| n |])
         (Rig.Buffer.of_string (Bytes.to_string bytes)))
  in
  List.iter
    (fun (scan, monoid) ->
      let c = { scan; monoid; axes = [| 0 |]; x; views = [] } in
      match run b c with
      | None -> failf "%s declined a float32 %s" b.name (monoid_name monoid)
      | Some y ->
          equal (list string) [] (differ (expected c) y);
          let last = (terms y).(if scan then n - 1 else 0) in
          equal ~msg:(monoid_name monoid) int64 0x7FC12345L
            (Int64.logand (bits_of last) 0xFFFFFFFFL))
    [ (false, Sum); (false, Max); (true, Sum); (true, Prod) ]

(* Products and sums of 8- and 16-bit integers at their extremes wrap in
   their width: 0xFFFF · 0xFFFF is 1 at uint16. *)
let test_wrap (b : Support.backend) () =
  let fold (type s) (dt : (int, s) D.t) m xs want =
    let x = on b (A.Any (A.of_array dt [| Array.length xs |] xs)) in
    let dst = on b (A.Any (A.create Rig.host dt [||])) in
    let p = identity (D.Any dt) in
    (match compute b.kernels ~scan:false p m [| 0 |] dst [| plain_of x |] with
    | A.Done -> ()
    | r -> failf "reduce answered %a" Nx_array_support.pp_answer r);
    equal
      ~msg:(Printf.sprintf "%s of %s" (monoid_name m) (D.name dt))
      int want
      (A.get (A.expect dt (host dst)) [||])
  in
  fold D.Uint16 Prod [| 0xFFFF; 0xFFFF; 3 |] 3;
  fold D.Int16 Prod [| -32768; -1 |] (-32768);
  fold D.Int16 Prod [| 32767; 32767 |] 1;
  fold D.Uint8 Prod [| 255; 255 |] 1;
  fold D.Int8 Prod [| -128; -1 |] (-128);
  fold D.Uint16 Sum [| 0xFFFF; 0xFFFF |] 0xFFFE;
  fold D.Int16 Sum [| 32767; 1 |] (-32768)

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let on x = on b (A.Any x) in
  let x = on (A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]) in
  let p = identity (D.Any D.Float32) in
  let reduce m dst x = compute b.kernels ~scan:false p m [| 1 |] dst [| plain_of x |] in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let wrong = on (A.create Rig.host D.Float32 [| 3 |]) in
  equal ~msg:"a destination of another shape" answer A.Shape_mismatch
    (reduce Sum wrong x);
  let empty = on (A.create Rig.host D.Float32 [| 2; 0 |]) in
  let dst = on (A.create Rig.host D.Float32 [| 2 |]) in
  equal ~msg:"a maximum of no term" answer A.Shape_mismatch (reduce Max dst empty);
  equal ~msg:"a sum of no term" answer A.Done (reduce Sum dst empty);
  equal ~msg:"is +0" (array float_exact) [| 0.; 0. |]
    (A.to_array (A.expect D.Float32 (host dst)));
  let sum = [| (Nx_kernel.Monoid Sum, 0) |] in
  equal ~msg:"a load of another shape than the loop's" answer A.Shape_mismatch
    (K.reduce p ~shape:[| 2; 4 |] ~axes:[| 1 |] sum
       ~dsts:[| on (A.create Rig.host D.Float32 [| 2 |]) |]
       [| plain_of x |]);
  let ints = on (A.of_array D.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]) in
  let mean = on (A.create Rig.host D.Int32 [| 2; 3 |]) in
  let var = on (A.create Rig.host D.Int32 [| 2; 3 |]) in
  equal ~msg:"a Moments scan of int32" answer A.Wrong_dtype
    (K.scan (identity (D.Any D.Int32)) ~axis:1 (Moments, 0)
       ~dsts:[| mean; var |] [| plain_of ints |]);
  let bools = on (A.create Rig.host D.Bool [| 2; 3 |]) in
  equal ~msg:"a Sum of bool" answer A.Wrong_dtype
    (K.reduce (identity (D.Any D.Bool)) ~shape:[| 2; 3 |] ~axes:[| 1 |]
       [| (Monoid Sum, 0) |]
       ~dsts:[| on (A.create Rig.host D.Bool [| 2 |]) |]
       [| plain_of bools |])

(* A padded load of more elements than a layout admits, or whose loop does
   not fit its destinations, is refused before the kernel copies it: padded
   to 2^50 elements, more than any host's memory, it answers Shape_mismatch
   and writes nothing. *)
let test_padded_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let on x = on b (A.Any x) in
  let x = A.expect D.Float32 (on (A.of_array D.Float32 [| 2 |] [| 1.; 2. |])) in
  let padded lo : Nx_kernel.load =
    Padded
      {
        x;
        fill = 0.;
        pad = { lo = [| lo |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] };
      }
  in
  let big = 1 lsl 50 in
  let huge = padded big and past = padded (L.max_numel - 1) in
  let f32 = D.Any D.Float32 in
  let id = identity f32 in
  let twice = P.v ~ins:[| f32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |] in
  let dst s = on (A.of_array D.Float32 s (Array.make (total s) 9.)) in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let refused msg call dsts =
    let before = Array.map (fun d -> A.to_array (A.expect D.Float32 (host d))) dsts in
    equal ~msg answer A.Shape_mismatch (call dsts);
    Array.iteri
      (fun i d ->
        equal ~msg:(msg ^ ": untouched") (array float_exact) before.(i)
          (A.to_array (A.expect D.Float32 (host d))))
      dsts
  in
  let sum = [| (Nx_kernel.Monoid Sum, 0) |] in
  let reduce p load dsts =
    K.reduce p ~shape:[| big + 2 |] ~axes:[| 0 |] sum ~dsts [| load |]
  in
  let scan p r load dsts = K.scan p ~axis:0 (r, 0) ~dsts [| load |] in
  List.iter
    (fun (name, p) ->
      refused (name ^ ": a reduction into another shape") (reduce p huge)
        [| dst [| 2 |] |];
      refused (name ^ ": a scan into another shape") (scan p (Monoid Sum) huge)
        [| dst [| 3 |] |];
      refused (name ^ ": a scan past max_numel") (scan p (Monoid Sum) past)
        [| dst [| 3 |] |];
      refused (name ^ ": a Moments scan into another shape")
        (scan p Moments huge)
        [| dst [| 3 |]; dst [| 3 |] |])
    [ ("identity", id); ("a program", twice) ]

(* Padded loads *)

(* What a padded load reads: the case of the array, built here, where
   Nx_kernel.Shape takes the padding and the destination; else the shape a
   reading of the record that checked nothing would give, its windows
   counted by truncating division, with the reduction's or scan's axes in
   it. *)
type loaded =
  | Reads of case
  | Refused of { scan : bool; monoid : monoid; axes : int array; read : int array }

(* A reduction or scan of a padded load: the operand, its padding and its
   fill's bits, and what it reads. *)
type padded = { x : A.any; fill : string; pad : Nx_kernel.pad; loaded : loaded }

let pp_padded ppf p =
  let pp_window ppf (w : M.window) =
    Format.fprintf ppf "(%d %d %d %d)" w.axis w.size w.step w.dilation
  in
  Format.fprintf ppf "lo %a hi %a interior %a windows %a of %a: " pp_ints
    p.pad.lo pp_ints p.pad.hi pp_ints p.pad.interior
    (Format.pp_print_list pp_window)
    (Array.to_list p.pad.windows) pp_ints (shape_of p.x);
  match p.loaded with
  | Reads c -> pp_case ppf c
  | Refused r ->
      Format.fprintf ppf "refused, %s %s along %a of %a"
        (if r.scan then "scan" else "reduce")
        (monoid_name r.monoid) pp_ints r.axes pp_ints r.read

(* The array [x] padded with the element [fill] (its bits) as [pad] says,
   C-contiguous, built element by element: along each axis, padded index
   [i] holds element [(i - lo) / (interior + 1)] where that divides and lies
   in the operand, the fill elsewhere; then the windows' view. *)
let pad_array (A.Any x) fill (pad : Nx_kernel.pad) =
  let dt = A.dtype x in
  let w = D.bits dt / 8 in
  let s = L.shape (A.layout x) in
  let r = Array.length s in
  let padded =
    Array.init r (fun i ->
        let d = s.(i) in
        pad.lo.(i) + pad.hi.(i) + d + if d > 0 then pad.interior.(i) * (d - 1) else 0)
  in
  let bytes = A.to_array (Option.get (A.bitcast D.Uint8 (A.copy x))) in
  let b = Bytes.create (max 1 (w * total padded)) in
  List.iteri
    (fun k idx ->
      let src = ref 0 and inside = ref true in
      Array.iteri
        (fun i j ->
          let c = j - pad.lo.(i) and step = pad.interior.(i) + 1 in
          if c < 0 || c mod step <> 0 || c / step >= s.(i) then inside := false
          else src := (!src * s.(i)) + (c / step))
        idx;
      for q = 0 to w - 1 do
        Bytes.set b ((k * w) + q)
          (if !inside then Char.chr bytes.((!src * w) + q) else fill.[q])
      done)
    (indices padded);
  let a = A.v dt (L.contiguous padded) (Rig.Buffer.of_string (Bytes.to_string b)) in
  if pad.windows = [||] then A.Any a
  else A.Any (Option.get (A.move (M.Window pad.windows) a))

(* The fills a law draws: zero, one, the extremes and a NaN of floats, the
   least integer. *)
let fills (D.Any dt) =
  let le w v =
    String.init w (fun i ->
        Char.chr (Int64.to_int (Int64.logand (Int64.shift_right_logical v (8 * i)) 0xFFL)))
  in
  let w = D.bits dt / 8 in
  match dt with
  | D.Float32 ->
      List.map (fun v -> le 4 (Int64.of_int32 v))
        [ 0l; 0x3F800000l; 0xFF800000l; 0x7FC01234l ]
  | D.Float64 ->
      List.map (le 8)
        [ 0L; Int64.bits_of_float 1.; Int64.bits_of_float neg_infinity; 0x7FF8000000000123L ]
  | D.Bool -> [ "\000"; "\001" ]
  | _ -> [ le w 0L; le w 7L; le w (Int64.shift_left 1L ((8 * w) - 1)) ]

(* A padding of an operand of shape [s]: low and high padding from -2 to 3
   (negative crops), interior padding up to 2 and at most one window of size
   up to 3, as drawn, so that some crop an axis below zero or hold a window
   wider than its axis; one in eight is of rank one more than [s]'s. *)
let pad_of s =
  let open Gen in
  let* extra = frequency [ (7, constant 0); (1, constant 1) ] in
  let r = Array.length s + extra in
  let ints lo hi = array ~size:(constant r) (int_range lo hi) in
  let* lo = ints (-2) 3 in
  let* hi = ints (-2) 3 in
  let* interior = ints 0 2 in
  let* axis = option (int_range 0 (r - 1)) in
  let* size = int_range 1 3 in
  let* step = int_range 1 2 in
  let+ dilation = int_range 1 2 in
  let windows =
    match axis with
    | Some axis -> [| { M.axis; size; step; dilation } |]
    | None -> [||]
  in
  { Nx_kernel.lo; hi; interior; windows }

(* The padded extents of an operand of shape [s], below zero where a crop
   passes the axis. *)
let extents s (pad : Nx_kernel.pad) =
  Array.mapi
    (fun i d ->
      pad.lo.(i) + pad.hi.(i) + d + if d > 0 then pad.interior.(i) * (d - 1) else 0)
    s

(* The shape a reading of [pad] that checked nothing would give: [s] for a
   padding of another rank, else the padded extents, each window's axis by
   its count from truncating division, then the windows' sizes. *)
let unchecked s (pad : Nx_kernel.pad) =
  if Array.length pad.lo <> Array.length s then s
  else
    let e = extents s pad in
    Array.iter
      (fun (w : M.window) ->
        e.(w.axis) <- ((e.(w.axis) - 1 - (w.dilation * (w.size - 1))) / w.step) + 1)
      pad.windows;
    Array.append e (Array.map (fun (w : M.window) -> w.size) pad.windows)

(* [x] loaded padded with the element of bits [fill] as [pad] says. *)
let padded_load (A.Any x) fill pad =
  let dt = A.dtype x in
  let fill = A.get (A.v dt (L.contiguous [||]) (Rig.Buffer.of_string fill)) [||] in
  Nx_kernel.Padded { x; fill; pad }

(* Whether Nx_kernel.Shape refuses the reduction or scan of [load]. *)
let refuses ~scan monoid axes load =
  let module Shape = Nx_kernel.Shape in
  let y = Shape.loaded [| load |] in
  Result.is_error
    (if scan then Result.bind y (Shape.scan ~axis:axes.(0))
     else Result.bind y (Shape.reduce ~axes [| Monoid monoid |]))

(* The reduction or scan by [monoid] along [axes] of an operand of dtype
   [d] and shape [s] drawn from [seed], loaded as [pad] with [fill]. *)
let padded ~scan d monoid s pad fill axes seed =
  let r = Array.length s in
  let plain =
    { perm = Array.init r Fun.id; stepped = false; reversed = None; broadcast = false }
  in
  let x = operand d s plain ~specials:20 seed in
  let loaded =
    if refuses ~scan monoid axes (padded_load x fill pad) then
      Refused { scan; monoid; axes; read = unchecked s pad }
    else Reads { scan; monoid; axes; x = pad_array x fill pad; views = [] }
  in
  { x; fill; pad; loaded }

let padded_case =
  Gen.with_pp pp_padded
    (let open Gen in
     let* scan = bool in
     let* d = of_list ~pp:pp_dtype base in
     let* monoid = of_list (List.filter (fun m -> accepts m d) monoids)
     in
     let* s = array ~size:(int_range 1 3) (int_range 0 6) in
     let* pad = pad_of s in
     let* fill = of_list (fills d) in
     let* seed = int in
     let lr = Array.length (unchecked s pad) in
     let+ axes =
       if scan then map (fun a -> [| a |]) (int_range 0 (lr - 1))
       else
         let+ keep = array ~size:(constant lr) bool in
         Array.of_list (List.filter (fun i -> keep.(i)) (List.init lr Fun.id))
     in
     padded ~scan d monoid s pad fill axes seed)

(* One padded load of each regime the law covers, float32 filled with
   one. *)
let padded_examples =
  let f32 = D.Any D.Float32 and one = "\000\000\128\063" in
  let pad ?(windows = [||]) lo hi interior = { Nx_kernel.lo; hi; interior; windows } in
  let w axis size step = { M.axis; size; step; dilation = 1 } in
  let reduce m s p axes = padded ~scan:false f32 m s p one axes 7 in
  [
    reduce Sum [| 4 |]
      (pad [| -1 |] [| 2 |] [| 1 |] ~windows:[| w 0 2 1 |])
      [| 1 |];
    (* A window of 4 over an axis of 3. *)
    reduce Sum [| 3 |] (pad [| 0 |] [| 0 |] [| 0 |] ~windows:[| w 0 4 2 |]) [| 1 |];
    (* Both axes cropped to -1. *)
    reduce Sum [| 2; 2 |] (pad [| -3; -3 |] [| 0; 0 |] [| 0; 0 |]) [| 0; 1 |];
    (* A padding of rank 2 over an operand of rank 1. *)
    reduce Sum [| 3 |] (pad [| 0; 5 |] [| 0; 0 |] [| 0; 0 |]) [| 0 |];
    (* A padding of more axes than an array has, each with a window. *)
    (let r = L.max_rank + 1 in
     reduce Sum [| 3 |]
       (pad (Array.make r 0) (Array.make r 0) (Array.make r 0)
          ~windows:(Array.init r (fun a -> w a 1 1)))
       [| 0 |]);
    reduce Max [| 0; 2 |] (pad [| 0; 0 |] [| 0; 0 |] [| 0; 0 |]) [| 0 |];
  ]

(* nx.cpu reduces and scans a padded load as the array it reads: the
   reference's bits over that array, built here. It refuses with
   Shape_mismatch a padding Nx_kernel.Shape refuses and an extreme of no term,
   here into a destination of the shape a reading that checked nothing
   would give; the sanitize profile shows that it reads nothing then. *)
let law_padded (b : Support.backend) p =
  let loads = [| padded_load p.x p.fill p.pad |] in
  let (A.Any x) = p.x in
  let s = shape_of p.x in
  let pad = p.pad in
  cover "windows" (pad.windows <> [||]);
  cover "cropped" (Array.exists (fun l -> l < 0) pad.lo);
  cover "interior" (Array.exists (fun i -> i > 0) pad.interior);
  match p.loaded with
  | Reads c ->
      let dst = A.Any (A.create Rig.host (A.dtype x) (result_shape c)) in
      cover "computed" true;
      (* It gives no window along that axis. *)
      let e = extents s pad in
      cover "a window wider than its axis"
        (Array.exists
           (fun (w : M.window) -> e.(w.axis) < (w.dilation * (w.size - 1)) + 1)
           pad.windows);
      let p = identity (dtype_of p.x) in
      (match compute b.kernels ~scan:c.scan p c.monoid c.axes dst loads with
      | A.Done -> equal (list string) [] (differ (expected c) dst)
      | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r)
  | Refused r ->
      let same = Array.length pad.lo = Array.length s in
      let e = if same then extents s pad else s in
      cover "a padding of another rank" (not same);
      cover "an axis cropped below zero" (Array.exists (fun d -> d < 0) e);
      let extreme = r.monoid = Max || r.monoid = Min in
      let terms = Array.fold_left (fun n a -> n * r.read.(a)) 1 r.axes in
      cover "an extreme of no term" (same && (not r.scan) && extreme && terms = 0);
      let kept =
        if r.scan then r.read
        else
          Array.of_list
            (List.filteri (fun i _ -> not (Array.mem i r.axes)) (Array.to_list r.read))
      in
      let dst = A.Any (A.create Rig.host (A.dtype x) (Array.map (max 0) kept)) in
      let p = identity (dtype_of p.x) in
      (match compute b.kernels ~scan:r.scan p r.monoid r.axes dst loads with
      | A.Shape_mismatch -> ()
      | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a)

(* Moments scans *)

(* A Moments scan of [x] along [axis], [x] drawn through the views
   [views]. *)
type moments = { axis : int; x : A.any; views : string list }

let pp_moments ppf (c : moments) =
  let (A.Any x) = c.x in
  Format.fprintf ppf "scan Moments along %d of %a %a (%s)" c.axis D.pp
    (A.dtype x) L.pp (A.layout x)
    (String.concat ", " c.views)

(* A tenth of the operands are one axis of more than 4096 terms, the law's
   "long" case, and a tenth slices of one term, its "one term" case, which
   [shape_gen] alone draws too rarely for every run to cover. Four tenths
   are tiles: 2 to 130 slices of 32, 33 or more terms, scanned along the
   last axis, whose lanes the results hold a row apart, or along the first
   through a transposed operand, whose lanes it holds a row apart. *)
let moments_case =
  Gen.with_pp pp_moments
    (let open Gen in
     let* d = of_list ~pp:pp_dtype floats in
     let any =
       let* s = shape_gen in
       let s = if Array.length s = 0 then [| 3 |] else s in
       let+ axis = int_range 0 (Array.length s - 1) in
       (s, axis, false)
     in
     let long = map (fun n -> ([| n |], 0, false)) (int_range 4097 6000) in
     let one = map (fun n -> ([| n; 1 |], 1, false)) (int_range 1 40) in
     let tiles =
       let* len =
         frequency
           [
             (1, constant 32);
             (1, constant 33);
             ( 3,
               let* k = int_range 1 9 in
               let+ r = int_range 1 31 in
               (32 * k) + r );
           ]
       in
       let* slices = int_range 2 130 in
       let+ last = bool in
       if last then ([| slices; len |], 1, false) else ([| len; slices |], 0, true)
     in
     let* s, axis, transposed =
       frequency [ (4, any); (1, long); (1, one); (4, tiles) ]
     in
     let* v = view_of (Array.length s) in
     let v = if transposed then { v with perm = [| 1; 0 |] } else v in
     let* specials =
       frequency [ (3, constant 0); (1, constant 300); (1, constant 4) ]
     in
     let+ seed = int in
     { axis; x = operand d s v ~specials seed; views = view_names v })

(* Slices of more than 4096 terms, about 2^-20 to 2^20 in magnitude, of
   both signs, whose joins of the tiles' states round differently in each
   order. *)
let moments_examples =
  let mixed d n seed =
    let r = Random.State.make [| seed |] in
    let term _ =
      let sign = if Random.State.bool r then 1. else -1. in
      let e = Random.State.int r 41 - 20 in
      sign *. Float.ldexp (1. +. Random.State.float r 1.) e
    in
    let xs = Array.init n term in
    match d with
    | D.Any D.Float32 -> A.Any (A.of_array D.Float32 [| n |] xs)
    | D.Any D.Float64 -> A.Any (A.of_array D.Float64 [| n |] xs)
    | _ -> invalid_arg "moments_examples"
  in
  [
    { axis = 0; x = mixed f32 4500 1; views = [] };
    { axis = 0; x = mixed f64 5000 2; views = [] };
  ]

(* [c] by [k] into fresh destinations, the mean's and the variance's terms;
   [None] if declined. *)
let run_moments k (c : moments) =
  let module K = (val k : Nx_kernel.S) in
  let (A.Any x) = c.x in
  let s = shape_of c.x in
  let mean = A.create Rig.host (A.dtype x) s in
  let var = A.create Rig.host (A.dtype x) s in
  let p = identity (dtype_of c.x) in
  match
    K.scan p ~axis:c.axis (Moments, 0)
      ~dsts:[| A.Any mean; A.Any var |]
      [| plain_of c.x |]
  with
  | A.Done -> Some (terms (A.Any mean), terms (A.Any var))
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* A result: some bits, or a value within a tolerance. *)
type expect = Is of want | Near of float * float

(* The most combinations a Moments scan's result [i] passes through:
   [max 0 (2 ⌈log₂ (i + 1)⌉ - 1)]. *)
let depth i =
  let rec log2 k p = if p >= k then 0 else 1 + log2 k (2 * p) in
  Float.of_int (max 0 ((2 * log2 (i + 1) 1) - 1))

(* The expected mean and variance at each index of [c], in C order: past a
   NaN term, its bits quieted; past an infinity, the infinities' sum and the NaN
   the kinds make, the mean that NaN too past infinities of both signs;
   else the running mean and variance by float64 sums of the terms less the
   first and of their squares, within nx.cpu's bound at [x]'s unit roundoff
   [u], the depth [D] of the index and the range [R] and variance [V] of the
   terms so far, plus the sums' own bound in float64. *)
let expected_moments (c : moments) =
  let (A.Any x) = c.x in
  let u = if D.bits (A.dtype x) = 32 then 0x1p-24 else 0x1p-53 in
  let made = Is (Bits (made (D.Any (A.dtype x)))) in
  let gamma k =
    Float.of_int k *. 0x1p-53 /. (1. -. (Float.of_int k *. 0x1p-53))
  in
  let s = shape_of c.x and a = c.axis in
  let ts = terms c.x in
  let r = Array.length s in
  let ks = Array.of_list (List.filter (( <> ) a) (List.init r Fun.id)) in
  let mean = Array.make (total s) (Is Some_nan) in
  let var = Array.make (total s) (Is Some_nan) in
  List.iter
    (fun o ->
      let idx = Array.make r 0 in
      Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
      let nan = ref None and pos = ref false and neg = ref false in
      let s1 = ref 0. and s2 = ref 0. and c = ref None in
      let lo = ref Float.infinity and hi = ref Float.neg_infinity in
      for j = 0 to s.(a) - 1 do
        idx.(a) <- j;
        let at = position s idx in
        let v = match ts.(at) with F (v, _) -> v | I _ -> assert false in
        if Float.is_nan v then begin
          if !nan = None then
            nan := Some (quiet_u u ts.(at))
        end
        else if v = Float.infinity then pos := true
        else if v = Float.neg_infinity then neg := true
        else begin
          let c = match !c with Some c -> c | None -> (c := Some v; v) in
          s1 := !s1 +. (v -. c);
          s2 := !s2 +. ((v -. c) *. (v -. c));
          lo := Float.min !lo v;
          hi := Float.max !hi v
        end;
        let n = Float.of_int (j + 1) in
        match !nan with
        | Some b ->
            mean.(at) <- Is (Bits b);
            var.(at) <- Is (Bits b)
        | None when !pos && !neg ->
            mean.(at) <- made;
            var.(at) <- made
        | None when !pos || !neg ->
            let inf = if !pos then Float.infinity else Float.neg_infinity in
            mean.(at) <- Near (inf, 0.);
            var.(at) <- made
        | None ->
            let c = Option.get !c and d = depth j and k = j + 1 in
            let m = !s1 /. n and r = !hi -. !lo in
            let v = Float.max 0. ((!s2 /. n) -. (m *. m)) in
            let slack = gamma (3 * k + 4) *. ((r *. r) +. r +. Float.abs c) in
            let tm =
              (((4. *. d) +. 1.) *. u *. r)
              +. (u *. Float.abs (c +. m))
              +. slack
            in
            let tv =
              (((7. *. d) +. 1.) *. u *. v)
              +. (((4. *. d *. d) -. (2. *. d)) *. u *. r *. Float.sqrt v)
              +. slack
            in
            mean.(at) <- Near (c +. m, tm);
            var.(at) <- Near (v, tv)
      done)
    (indices (Array.map (fun i -> s.(i)) ks));
  (mean, var)

(* The bits of each slice's mean and variance by nx_kernel.mli's network,
   replayed term by term as a stream: term t's state joins its level's
   waiting state while t's low bits are 1, and the result at the level it
   reaches is that state, or the next level's last result joined with it.
   Each operation rounds to [x]'s dtype. A slice with a term not finite
   takes nx_kinds.h's NaN rules, which this does not replay: its results
   are [None]. *)
let replayed_moments (c : moments) =
  let (A.Any x) = c.x in
  let f32 = D.bits (A.dtype x) = 32 in
  let rnd v = if f32 then round32 v else v in
  let bits v =
    if f32 then Int64.of_int32 (Int32.bits_of_float v) else Int64.bits_of_float v
  in
  (* A state: its mean less the slice's first term, S and count. *)
  let join (ma, sa, na) (mb, sb, nb) =
    let k = rnd (na +. nb) in
    let f = rnd (nb /. k) in
    let g = rnd (na *. f) and d = rnd (mb -. ma) in
    (rnd (ma +. rnd (d *. f)), rnd (rnd (sa +. sb) +. rnd (rnd (d *. d) *. g)), k)
  in
  let s = shape_of c.x and a = c.axis in
  let ts = terms c.x in
  let r = Array.length s in
  let ks = Array.of_list (List.filter (( <> ) a) (List.init r Fun.id)) in
  let mean = Array.make (total s) None in
  let var = Array.make (total s) None in
  List.iter
    (fun o ->
      let idx = Array.make r 0 in
      Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
      let at j = idx.(a) <- j; position s idx in
      let xs =
        Array.init s.(a) (fun j ->
            match ts.(at j) with F (v, _) -> v | I _ -> assert false)
      in
      if Array.for_all Float.is_finite xs && Array.length xs > 0 then begin
        let c = xs.(0) in
        let held = Array.make 64 (0., 0., 0.) and got = Array.make 65 (0., 0., 0.) in
        Array.iteri
          (fun t v ->
            let st = ref (rnd (v -. c), 0., 1.) and k = ref 0 and j = ref t in
            while !j land 1 = 1 do
              st := join held.(!k) !st;
              incr k;
              j := !j lsr 1
            done;
            let res = if !j = 0 then !st else join got.(!k + 1) !st in
            held.(!k) <- !st;
            for h = 0 to !k do got.(h) <- res done;
            let mu, sv, _ = res in
            let n = Float.of_int (t + 1) in
            mean.(at t) <- Some (Is (Bits (bits (rnd (c +. mu)))));
            var.(at t) <- Some (Is (Bits (bits (rnd (sv /. n))))))
          xs
      end)
    (indices (Array.map (fun i -> s.(i)) ks));
  (mean, var)

(* The first results that miss [want]. *)
let misses want got =
  let bad = ref [] in
  Array.iteri
    (fun i w ->
      let t = got.(i) in
      let ok =
        match (w, t) with
        | Is Some_nan, t -> is_nan t
        | Is (Bits b), t -> bits_of t = b
        | Near (v, _), F (g, _) when not (Float.is_finite v) -> g = v
        | Near (v, tol), F (g, _) -> Float.abs (g -. v) <= tol
        | Near _, I _ -> false
      in
      if (not ok) && List.length !bad < 8 then
        let w =
          match w with
          | Is w -> show w
          | Near (v, tol) -> Printf.sprintf "%h within %g" v tol
        in
        bad :=
          Printf.sprintf "result %d: %Lx, expected %s" i (bits_of t) w :: !bad)
    want;
  List.rev !bad

(* A Moments scan has the bits of nx_kernel.mli's network, within nx.cpu's
   bound of the running mean and variance, its NaNs and infinities as
   stated, and the same bits on one thread and from a C-contiguous copy of
   its operand. *)
let law_moments (b : Support.backend) (c : moments) =
  let s = shape_of c.x in
  let len = s.(c.axis) in
  cover "no result" (total s = 0);
  cover "one term" (len = 1);
  cover "slices past a unit's" (len > 0 && total s / len > 64);
  cover "long" (len > 4096);
  cover "a tile's terms, or one more" (len = 32 || len = 33);
  (* Lanes a row apart go through rows of the scratch in tiles of 32 terms. *)
  let partial = len > 32 && len mod 32 <> 0 && total s / len > 1 in
  let rank = Array.length s in
  cover "a partial tile written through rows"
    (partial && rank >= 2 && c.axis = rank - 1);
  cover "a partial tile read through rows"
    (partial && rank = 2 && c.axis = 0
    && List.mem "permuted" c.views
    && not (List.mem "broadcast" c.views));
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  match run_moments b.kernels c with
  | None -> failf "nx.cpu declined a Moments scan"
  | Some (mean, var) ->
      let want_mean, want_var = expected_moments c in
      cover "a NaN"
        (Array.exists (function Is (Bits _) -> true | _ -> false) want_mean);
      cover "an infinity"
        (Array.exists
           (function Near (v, _) -> not (Float.is_finite v) | Is _ -> false)
           want_mean);
      equal ~msg:"mean" (list string) [] (misses want_mean mean);
      equal ~msg:"variance" (list string) [] (misses want_var var);
      let net_mean, net_var = replayed_moments c in
      let net want got =
        Array.mapi
          (fun i w -> Option.value w ~default:(Is (Bits (bits_of got.(i)))))
          want
      in
      cover "replayed" (Array.exists Option.is_some net_mean);
      equal ~msg:"the network's mean" (list string) []
        (misses (net net_mean mean) mean);
      equal ~msg:"the network's variance" (list string) []
        (misses (net net_var var) var);
      let exact ts = Array.map (fun t -> Is (Bits (bits_of t))) ts in
      let (A.Any x) = c.x in
      let copy = { c with x = A.Any (A.copy x) } in
      List.iter
        (fun (msg, k, c) ->
          match run_moments k c with
          | None -> failf "%s declined" msg
          | Some (m, v) ->
              equal ~msg (list string) [] (misses (exact mean) m);
              equal ~msg (list string) [] (misses (exact var) v))
        [
          ("one thread", (module One : Nx_kernel.S), c);
          ("a copy", b.kernels, copy);
        ]

(* Stateful reductions *)

(* Values that break Arg, Logsumexp and Moments: about [center] within
   [scale]; on [ties], multiples of half [scale], so that extremes repeat
   and -0 meets +0. *)
type values = { name : string; center : float; scale : float; ties : bool }

let value_kinds =
  [
    { name = "small"; center = 0.; scale = 1.; ties = false };
    (* A mean that dwarfs the spread. *)
    { name = "offset"; center = 1e4; scale = 1e-2; ties = false };
    (* exp of the terms less their maximum spans past both ends of float32. *)
    { name = "wide"; center = 0.; scale = 40.; ties = false };
    { name = "ties"; center = 0.; scale = 1.; ties = true };
  ]

(* [n] elements of the float [dt] of the kind [v]: one in [outliers] (none
   for 0) is a power of two up to 2^20 times larger, one in [specials] a
   NaN, an infinity or -0, as [elements] draws them. *)
let float_elements (type a b) (dt : (a, b) D.t) n v ~outliers ~specials seed
    =
  let r = Random.State.make [| seed |] in
  let one_in k = k > 0 && Random.State.int r k = 0 in
  let pick a = a.(Random.State.int r (Array.length a)) in
  let b = Bytes.create (max 1 (D.bytes dt n)) in
  for i = 0 to n - 1 do
    let x = v.center +. (v.scale *. full r 52) in
    let x =
      if v.ties then Float.round (2. *. x /. v.scale) *. v.scale /. 2. else x
    in
    let x =
      if one_in outliers then x *. Float.ldexp 1. (Random.State.int r 21)
      else x
    in
    let special = one_in specials in
    match dt with
    | D.Float32 ->
        Bytes.set_int32_le b (4 * i)
          (if special then
             pick
               (Array.append nans32
                  [| 0x80000000l; 0x7F800000l; 0xFF800000l |])
           else Int32.bits_of_float x)
    | D.Float64 ->
        Bytes.set_int64_le b (8 * i)
          (if special then
             pick
               (Array.append nans64
                  [|
                    Int64.bits_of_float (-0.);
                    Int64.bits_of_float infinity;
                    Int64.bits_of_float neg_infinity;
                  |])
           else Int64.bits_of_float x)
    | _ -> invalid_arg "float_elements: not float32 or float64"
  done;
  A.Any (A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string (Bytes.to_string b)))

(* An [Arg], [Logsumexp] or [Moments] reduction of [x] along [axes], [x]
   drawn through the views [views] from values of the kind [values]. *)
type stat = {
  r : Nx_kernel.reduction;
  axes : int array;
  x : A.any;
  views : string list;
  values : string;
}

let reduction_name : Nx_kernel.reduction -> string = function
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"
  | Monoid m -> monoid_name m
  | Moments -> "Moments"

let pp_stat ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "%s along %a of %a %a (%s, %s)" (reduction_name c.r)
    pp_ints c.axes D.pp (A.dtype x) L.pp (A.layout x) c.values
    (String.concat ", " c.views)

let is_float (D.Any dt) = D.is D.Float dt

(* The reduction [r] of [x] along [axes] drawn [make]: an Arg takes an
   element along each empty reduced axis, since one of no term is
   refused. *)
let stat_of r s axes v values make =
  let s =
    match r with
    | Nx_kernel.Arg _ ->
        Array.mapi (fun i e -> if Array.mem i axes then max 1 e else e) s
    | _ -> s
  in
  { r; axes; x = operand_with make s v; views = view_names v; values }

let stat_case =
  Gen.with_pp pp_stat
    (let open Gen in
     let ints =
       List.filter (fun (D.Any dt) -> not (D.is D.Float dt || D.equal dt D.Bool)) base
     in
     let* d = of_list ~pp:pp_dtype (floats @ floats @ ints) in
     let* r =
       of_list
         (if is_float d then Nx_kernel.[ Arg Max; Arg Min; Monoid Logsumexp; Moments ]
          else Nx_kernel.[ Arg Max; Arg Min ])
     in
     let* s = shape_gen in
     let rank = Array.length s in
     let* keep = array ~size:(constant rank) bool in
     let axes =
       Array.of_list (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let* v = view_of rank in
     let* values = of_list value_kinds in
     let* outliers = frequency [ (3, constant 0); (1, constant 50) ] in
     let* specials =
       frequency [ (3, constant 0); (1, constant 300); (1, constant 20) ]
     in
     let+ seed = int in
     let (D.Any dt) = d in
     let make n =
       if is_float d then float_elements dt n values ~outliers ~specials seed
       else A.Any (elements dt n ~specials seed)
     in
     stat_of r s axes v values.name make)

(* [c] as a reduction case of Sum, whose result's shape and outputs it
   shares. *)
let as_case c = { scan = false; monoid = Sum; axes = c.axes; x = c.x; views = [] }
let stat_shape c = result_shape (as_case c)

(* The destinations of [c]: the extreme and its int64 position for Arg, the
   mean and the variance for Moments, one for Logsumexp. *)
let stat_dsts c =
  let (A.Any x) = c.x in
  let s = stat_shape c in
  let like () = A.Any (A.create Rig.host (A.dtype x) s) in
  match c.r with
  | Arg _ -> [| like (); A.Any (A.create Rig.host D.Int64 s) |]
  | Moments -> [| like (); like () |]
  | Monoid _ -> [| like () |]

(* [c] by [k] into fresh destinations, their terms; [None] if declined. *)
let run_stat k c =
  let module K = (val k : Nx_kernel.S) in
  let dsts = stat_dsts c in
  match
    K.reduce (identity (dtype_of c.x)) ~shape:(shape_of c.x) ~axes:c.axes
      [| (c.r, 0) |] ~dsts [| plain_of c.x |]
  with
  | A.Done -> Some (Array.map terms dsts)
  | A.Declined -> None
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* The quiet NaN of a clear sign bit, made from numbers. *)
let made_nan u = Is (Bits (if u > 0x1p-30 then 0x7FC00000L else 0x7FF8000000000000L))

(* [t]'s bits, quieted where a NaN of [d]. *)
let quiet_term d t = if is_nan t then quieted (made d) t else bits_of t

(* A float's value. *)
let value = function F (x, _) -> x | I _ -> invalid_arg "value: an integer"

(* [xs] summed by Neumaier's compensated sum, within 2 ε |Σ| of the exact
   sum, ε the unit roundoff of float64, barring terms of n² ε. *)
let neumaier xs =
  let s = ref 0. and c = ref 0. in
  Array.iter
    (fun x ->
      let t = !s +. x in
      (c := !c +. if Float.abs !s >= Float.abs x then !s -. t +. x else x -. t +. !s);
      s := t)
    xs;
  !s +. !c

(* nx_cpu.mli's terms of the bounds for [n] terms: [l] the most roundings
   of a block's sum in lane order and [d] the depth of the blocks' tree. *)
let depths n =
  let blocks = (n + 1023) / 1024 in
  let rec log2 k p = if p >= k then 0 else 1 + log2 k (2 * p) in
  (Float.of_int (((min n 1024 + 15) / 16) + 3), Float.of_int (log2 blocks 1))

let eps = 0x1p-53

(* The extreme by [e] of the terms [ts] at [d], then its first position. *)
let arg_expect d (e : Nx_kernel.extreme) ts =
  let f, _ = op d (match e with Max -> Max | Min -> Min) in
  let best = Array.fold_left f ts.(0) ts in
  let same t = if is_nan best then is_nan t else bits_of t = bits_of best in
  let i = Option.get (Array.find_index same ts) in
  [| Is (Bits (quiet_term d ts.(i))); Is (Bits (Int64.of_int i)) |]

(* Logsumexp of [ts] within nx_cpu.mli's bound at the unit roundoff [u],
   from a float64 reference within ε (ln n + 4 + |r|). *)
let lse_expect u ts =
  match Array.find_opt is_nan ts with
  | Some t -> [| Is (Bits (quiet_u u t)) |]
  | None ->
      let xs = Array.map value ts in
      let n = Array.length xs in
      let m = Array.fold_left Float.max neg_infinity xs in
      if not (Float.is_finite m) then [| Near (m, 0.) |]
      else
        let r = m +. log (neumaier (Array.map (fun x -> exp (x -. m)) xs)) in
        let l, d = depths n and ln = log (Float.of_int n) in
        let bound = ((l +. d +. 5. +. (6. *. ln)) *. u) +. (u *. Float.abs r) in
        [| Near (r, bound +. (eps *. (ln +. 4. +. Float.abs r))) |]

(* Moments of [ts] within nx_cpu.mli's bounds, [u] the dtype's roundoff,
   from float64 references of the terms less the first, by compensated
   sums: within ε (4 R + 2 |μ|) and ε (5 V + 8 R √V). *)
let moments_expect u ts =
  let nan = made_nan u in
  match Array.find_opt is_nan ts with
  | Some t -> [| Is (Bits (quiet_u u t)); Is (Bits (quiet_u u t)) |]
  | None when ts = [||] -> [| nan; nan |]
  | None -> (
      let xs = Array.map value ts in
      let pos = Array.mem infinity xs and neg = Array.mem neg_infinity xs in
      match (pos, neg) with
      | true, true -> [| nan; nan |]
      | true, false -> [| Near (infinity, 0.); nan |]
      | false, true -> [| Near (neg_infinity, 0.); nan |]
      | false, false ->
          let n = Array.length xs and c = xs.(0) in
          let nf = Float.of_int n in
          let ys = Array.map (fun x -> x -. c) xs in
          let my = neumaier ys /. nf in
          let mean = c +. my in
          let v = neumaier (Array.map (fun y -> (y -. my) *. (y -. my)) ys) /. nf in
          let hi = Array.fold_left Float.max neg_infinity xs in
          let r = hi -. Array.fold_left Float.min infinity xs in
          let l, d = depths n and sv = Float.sqrt v in
          let m = Array.fold_left (fun a x -> a +. Float.abs x) 0. xs /. nf in
          let tm = ((l +. d) *. u *. m) +. (u *. Float.abs mean) in
          let k = l +. d in
          let tv =
            ((k +. 5.) *. u *. v)
            +. (2. *. (k +. 1.) *. u *. tm *. sv)
            +. (((3. *. k) +. 7.) *. u *. tm *. tm)
          in
          [|
            Near (mean, tm +. (eps *. ((4. *. r) +. (2. *. Float.abs mean))));
            Near (v, tv +. (eps *. ((5. *. v) +. (8. *. r *. sv))));
          |])

(* The expected results of each destination of [c], in C order. *)
let stat_expected c =
  let (A.Any x) = c.x in
  let d = D.Any (A.dtype x) in
  let u = if D.bits (A.dtype x) = 32 then 0x1p-24 else 0x1p-53 in
  let per = outputs (as_case c) (terms c.x) in
  let one ts =
    match c.r with
    | Arg e -> arg_expect d e ts
    | Monoid _ -> lse_expect u ts
    | Moments -> moments_expect u ts
  in
  let all = List.map one per in
  Array.init (Array.length (stat_dsts c)) (fun k ->
      Array.of_list (List.map (fun e -> e.(k)) all))

(* Each result exact (Arg) or within nx_cpu.mli's bound of the exact one,
   NaNs and infinities as Nx_kernel.reduction states them, and the same
   bits on one thread and from a C-contiguous copy of the operand. *)
let law_stateful (b : Support.backend) c =
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  let outs = total (stat_shape c) in
  cover "no output" (outs = 0);
  cover "no term" (per = 0 && outs > 0);
  cover "one term" (per = 1);
  cover "several blocks" (per > 1024);
  cover "many outputs" (outs >= 64);
  List.iter (fun (k : values) -> cover k.name (c.values = k.name)) value_kinds;
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  match run_stat b.kernels c with
  | None -> failf "nx.cpu declined %s" (reduction_name c.r)
  | Some got ->
      let want = stat_expected c in
      let wanted p = Array.exists (Array.exists p) want in
      cover "a NaN" (wanted (function Is (Bits _) -> c.r <> Arg Max && c.r <> Arg Min | _ -> false));
      cover "an infinity"
        (wanted (function Near (v, _) -> not (Float.is_finite v) | Is _ -> false));
      Array.iteri
        (fun k w -> equal ~msg:(Printf.sprintf "destination %d" k) (list string) [] (misses w got.(k)))
        want;
      let exact = Array.map (Array.map (fun t -> Is (Bits (bits_of t)))) got in
      let (A.Any x) = c.x in
      List.iter
        (fun (msg, k, c) ->
          match run_stat k c with
          | None -> failf "%s declined" msg
          | Some g ->
              Array.iteri (fun i w -> equal ~msg (list string) [] (misses w g.(i))) exact)
        [
          ("one thread", (module One : Nx_kernel.S), c);
          ("a copy", b.kernels, { c with x = A.Any (A.copy x) });
        ]

(* Arg and Logsumexp scans *)

(* A scan by Arg of a float, an integer or a boolean, or by Logsumexp of a
   float, along one axis: a vector past a scan's chunk now and then, so
   that the chunks' totals and carries run. *)
let stat_scan_case =
  Gen.with_pp pp_stat
    (let open Gen in
     let ints = List.filter (fun (D.Any dt) -> not (D.is D.Float dt)) base in
     let* d = of_list ~pp:pp_dtype (floats @ floats @ ints) in
     let* r =
       of_list
         (if is_float d then Nx_kernel.[ Arg Max; Arg Min; Monoid Logsumexp ]
          else Nx_kernel.[ Arg Max; Arg Min ])
     in
     let* s =
       frequency
         [ (4, shape_gen); (1, map (fun n -> [| n |]) (int_range 4097 9000)) ]
     in
     let s = if Array.length s = 0 then [| 3 |] else s in
     let rank = Array.length s in
     let* a = int_range 0 (rank - 1) in
     let* v = view_of rank in
     let* values = of_list value_kinds in
     let* outliers = frequency [ (3, constant 0); (1, constant 50) ] in
     let* specials =
       frequency [ (3, constant 0); (1, constant 300); (1, constant 20) ]
     in
     let+ seed = int in
     let (D.Any dt) = d in
     let make n =
       if is_float d then float_elements dt n values ~outliers ~specials seed
       else A.Any (elements dt n ~specials seed)
     in
     stat_of r s [| a |] v values.name make)

(* [c]'s scan by [k] into fresh destinations of [c.x]'s shape, their
   terms; [None] if declined. *)
let run_stat_scan k (c : stat) =
  let module K = (val k : Nx_kernel.S) in
  let (A.Any x) = c.x in
  let like dt = A.Any (A.create Rig.host dt (shape_of c.x)) in
  let dsts =
    match c.r with
    | Arg _ -> [| like (A.dtype x); like D.Int64 |]
    | Moments -> [| like (A.dtype x); like (A.dtype x) |]
    | Monoid _ -> [| like (A.dtype x) |]
  in
  match
    K.scan (identity (dtype_of c.x)) ~axis:c.axes.(0) (c.r, 0) ~dsts
      [| plain_of c.x |]
  with
  | A.Done -> Some (Array.map terms dsts)
  | A.Declined -> None
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* The running Arg of a slice [ts] at [d]: each prefix's extreme and its
   first position. *)
let arg_scan d (e : Nx_kernel.extreme) ts =
  let f, _ = op d (match e with Max -> Max | Min -> Min) in
  let best = ref ts.(0) and at = ref 0 in
  Array.mapi
    (fun i t ->
      let g = f !best t in
      let same = if is_nan !best then is_nan g else bits_of g = bits_of !best in
      if not same then begin
        best := t;
        at := i
      end;
      [| Is (Bits (quiet_term d !best)); Is (Bits (Int64.of_int !at)) |])
    ts

(* nx_cpu.mli's bound of a Logsumexp scan's result [r] after [n] terms in
   [c] chunks, [k] terms past every earlier one of their chunk, at the unit
   roundoff [u]. *)
let lse_scan_bound ~n ~c ~k u r =
  let ln = log (Float.of_int n) and c = Float.of_int c in
  let n = Float.of_int (min n 8192) and k = Float.of_int (min k 8192) in
  ((n +. c +. (6. *. (k +. c)) +. 5. +. (6. *. ln)) *. u) +. (u *. Float.abs r)

(* The running Logsumexp of a slice [ts] within nx_cpu.mli's bound at [u]:
   past a NaN, its bits; past +∞, +∞; -∞ while every term is; else from a
   float64 running maximum and a compensated sum rescaled as it rises,
   within the bound at float64's roundoff besides. *)
let lse_scan u ts =
  let nan = ref None and m = ref Float.neg_infinity in
  let s = ref 0. and comp = ref 0. and rises = ref 0 in
  let top = ref Float.neg_infinity in
  let add x =
    let t = !s +. x in
    let lost =
      if Float.abs !s >= Float.abs x then !s -. t +. x else x -. t +. !s
    in
    comp := !comp +. lost;
    s := t
  in
  Array.mapi
    (fun i t ->
      if i mod 4096 = 0 then top := Float.neg_infinity;
      (match !nan with
      | Some _ -> ()
      | None when is_nan t -> nan := Some (quiet_u u t)
      | None ->
          let x = value t in
          if x > !top then begin
            if !top > Float.neg_infinity then incr rises;
            top := x
          end;
          if x > !m then begin
            let f = if Float.is_finite !m then exp (!m -. x) else 0. in
            s := !s *. f;
            comp := !comp *. f;
            m := x
          end;
          if Float.is_finite !m then add (exp (x -. !m)));
      match !nan with
      | Some b -> [| Is (Bits b) |]
      | None when not (Float.is_finite !m) -> [| Near (!m, 0.) |]
      | None ->
          let r = !m +. log (!s +. !comp) in
          let c = (i / 4096) + 1 in
          let b u = lse_scan_bound ~n:(i + 1) ~c ~k:!rises u r in
          [| Near (r, b u +. b eps) |])
    ts

(* The bits of a Logsumexp scan of a slice [ts], float32 where [f32] else
   float64, replayed as nx_cpu.mli states it: chunks of 4096 terms; a
   chunk's total from none, the carry into the next chunk the carry merged
   with it, each result the carry taken on through the chunk's terms; a
   term x makes the maximum n and s × exp (m - n) + exp (x - n), s itself
   where n is m and 0 for exp (x - n) where n is -∞, the result n + log s or
   n where n is not finite. exp and log are nx_kinds.h's, exp is 0 below the
   reduction's cut, and each operation rounds to the dtype. From the first
   NaN term on, the result is its bits, quieted. *)
let lse_replay ~f32 ts =
  let rnd x = if f32 then round32 x else x in
  (* A kind over a vector, in one call. *)
  let kind name xs =
    let n = Array.length xs in
    if f32 then begin
      let ba k = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout k in
      let x = ba n and y = ba n in
      Array.iteri (fun i v -> x.{i} <- v) xs;
      Nx_kinds_support.run name x x y;
      Array.init n (fun i -> y.{i})
    end
    else begin
      let ba k = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout k in
      let x = ba n and y = ba n in
      Array.iteri (fun i v -> x.{i} <- v) xs;
      Nx_kinds_support.run name x x y;
      Array.init n (fun i -> y.{i})
    end
  in
  let cut = if f32 then -87. else -708. in
  let past m x =
    if Float.is_nan m then m else if Float.is_nan x || x > m then x else m
  in
  (* The recurrence, its exps from [ex] in the order it takes them: no
     branch and no exp's argument depends on s, so a first run records the
     arguments, which run through exp at once, and a second takes them. *)
  let replay ex =
    let rescaled s m n = if m = n then s else rnd (s *. ex (rnd (m -. n))) in
    (* A state is a float array, its maximum then its s. *)
    let step st x =
      let m = st.(0) in
      let n = past m x in
      let e = if n = Float.neg_infinity then 0. else ex (rnd (x -. n)) in
      st.(1) <- rnd (rescaled st.(1) m n +. e);
      st.(0) <- n
    in
    let k = Array.length ts in
    let ms = Array.make k 0. and ss = Array.make k 0. in
    let carry = [| Float.neg_infinity; 0. |] in
    for c = 0 to ((k + 4095) / 4096) - 1 do
      let lo = c * 4096 and hi = min k ((c + 1) * 4096) in
      let st = Array.copy carry and total = [| Float.neg_infinity; 0. |] in
      for i = lo to hi - 1 do
        let x = value ts.(i) in
        step st x;
        step total x;
        ms.(i) <- st.(0);
        ss.(i) <- st.(1)
      done;
      if c = 0 then Array.blit total 0 carry 0 2
      else begin
        let ma = carry.(0) and mb = total.(0) in
        let n = past ma mb in
        carry.(1) <- rnd (rescaled carry.(1) ma n +. rescaled total.(1) mb n);
        carry.(0) <- n
      end
    done;
    (ms, ss)
  in
  let args = ref [] in
  ignore (replay (fun y -> args := y :: !args; 0.));
  let ys = Array.of_list (List.rev !args) in
  let es = kind "exp" (Array.map (fun y -> if y < cut then cut else y) ys) in
  let next = ref 0 in
  let ms, ss =
    replay (fun y ->
        let e = es.(!next) in
        incr next;
        if y < cut then 0. else e)
  in
  let logs = kind "log" ss in
  let first = Array.find_index is_nan ts in
  Array.mapi
    (fun i m ->
      match first with
      | Some j when j <= i ->
          Is (Bits (quiet_u (if f32 then 0x1p-24 else 0x1p-53) ts.(j)))
      | _ ->
          let r = if Float.is_finite m then rnd (m +. logs.(i)) else m in
          Is (Bits (if f32 then Int64.of_int32 (Int32.bits_of_float r) else Int64.bits_of_float r)))
    ms

(* The expected results of [c]'s scan in C order, by destination. *)
let stat_scan_expected (c : stat) =
  let (A.Any x) = c.x in
  let d = D.Any (A.dtype x) in
  let u = if D.bits (A.dtype x) = 32 then 0x1p-24 else 0x1p-53 in
  let slice ts =
    match c.r with
    | Arg e -> arg_scan d e ts
    | _ -> lse_scan u ts
  in
  let scanned = { (as_case c) with scan = true } in
  let k = match c.r with Arg _ -> 2 | _ -> 1 in
  Array.init k (fun j ->
      results scanned (terms c.x) ~fill:(Is Some_nan)
        ~reduce:(fun _ -> assert false)
        ~scan:(fun ts -> Array.map (fun e -> e.(j)) (slice ts)))

(* Arg scans are exact, Logsumexp scans within nx_cpu.mli's bound and the
   bits of its recurrence replayed, NaNs and infinities as
   Nx_kernel.reduction states them, and both give the same bits on one
   thread and from a C-contiguous copy of the operand. *)
let law_stat_scan (b : Support.backend) (c : stat) =
  let s = shape_of c.x in
  let a = c.axes.(0) in
  cover "several chunks" (s.(a) > 4096);
  let last = Array.length s - 1 in
  cover "slices side by side" (a < last && s.(last) > 1);
  List.iter (fun (k : values) -> cover k.name (c.values = k.name)) value_kinds;
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  match run_stat_scan b.kernels c with
  | None -> failf "nx.cpu declined %s" (reduction_name c.r)
  | Some got ->
      let want = stat_scan_expected c in
      let wanted p = Array.exists (Array.exists p) want in
      let lse = c.r = Monoid Logsumexp in
      cover "a NaN" (lse && wanted (function Is (Bits _) -> true | _ -> false));
      cover "a -∞ term"
        (lse
        && Array.exists
             (fun t -> (not (is_nan t)) && value t = Float.neg_infinity)
             (terms c.x));
      cover "an infinity"
        (wanted (function Near (v, _) -> not (Float.is_finite v) | Is _ -> false));
      Array.iteri
        (fun k w ->
          equal ~msg:(Printf.sprintf "destination %d" k) (list string) []
            (misses w got.(k)))
        want;
      (if lse then
         let (A.Any x) = c.x in
         let f32 = D.bits (A.dtype x) = 32 in
         let replay =
           results { (as_case c) with scan = true } (terms c.x)
             ~fill:(Is Some_nan) ~reduce:(fun _ -> assert false)
             ~scan:(lse_replay ~f32)
         in
         equal ~msg:"the replayed recurrence" (list string) []
           (misses replay got.(0)));
      let exact = Array.map (Array.map (fun t -> Is (Bits (bits_of t)))) got in
      let (A.Any x) = c.x in
      List.iter
        (fun (msg, k, c) ->
          match run_stat_scan k c with
          | None -> failf "%s declined" msg
          | Some g ->
              Array.iteri (fun i w -> equal ~msg (list string) [] (misses w g.(i))) exact)
        [
          ("one thread", (module One : Nx_kernel.S), c);
          ("a copy", b.kernels, { c with x = A.Any (A.copy x) });
        ]

let stat_example r d s axes ?view ?(outliers = 0) ?(specials = 0) values =
  let rank = Array.length s in
  let view =
    Option.value view
      ~default:
        { perm = Array.init rank Fun.id; stepped = false; reversed = None; broadcast = false }
  in
  let v = List.find (fun v -> v.name = values) value_kinds in
  let (D.Any dt) = d in
  let make n =
    if is_float d then float_elements dt n v ~outliers ~specials 3
    else A.Any (elements dt n ~specials 3)
  in
  stat_of r s axes view values make

(* The arithmetic of Nx's expansions. Logsumexp and Moments reduce in two
   passes, as Nx expands them: drawn from these spreads of terms, they give
   the bits of that arithmetic run on the kernels' own Max, Sum and kinds. *)

let normal r =
  let u1 = Random.State.float r 1. and u2 = Random.State.float r 1. in
  Float.sqrt (-2. *. log (u1 +. 1e-300)) *. Float.cos (2. *. Float.pi *. u2)

let unit_ r = Random.State.float r 2. -. 1.

(* Spreads of terms: term [i] of [n] from the state [r]. *)
let spreads =
  [
    ("normal", fun r _ _ -> normal r);
    ("offset", fun r _ _ -> 1e4 +. (1e-2 *. unit_ r));
    ("wide", fun r _ _ -> 40. *. unit_ r);
    ("first-out", fun r i _ -> if i = 0 then 1e3 else unit_ r);
    ( "outliers",
      fun r _ _ ->
        let x = unit_ r in
        if Random.State.int r 50 = 0 then x *. Float.ldexp 1. (Random.State.int r 21)
        else x );
    ("ramp", fun r i n -> (-60. *. Float.of_int i /. Float.of_int n) +. (1e-3 *. unit_ r));
    ("lognormal", fun r _ _ -> Float.exp (3. *. normal r));
  ]


type spread_case = {
  r : Nx_kernel.reduction;
  dt : D.any;
  spread : string;
  n : int;
  seed : int;
}

(* Long ramps, whose blocks' maxima differ: a sum rescaled block by block
   loses their bits. *)
let spread_examples =
  List.concat_map
    (fun dt ->
      List.map
        (fun seed -> { r = Nx_kernel.Monoid Logsumexp; dt; spread = "ramp"; n = 70_000; seed })
        [ 0; 1; 2; 3 ])
    floats

let spread_case =
  Gen.with_pp
    (fun ppf c ->
      Format.fprintf ppf "%s of %d %a terms, %s (seed %d)" (reduction_name c.r)
        c.n pp_dtype c.dt c.spread c.seed)
    (let open Gen in
     let* r = of_list Nx_kernel.[ Monoid Logsumexp; Moments ] in
     let* dt = of_list floats in
     let* spread = of_list (List.map fst spreads) in
     let* n = of_list [ 1; 2; 17; 1023; 1025; 4097; 70_000 ] in
     let+ seed = int in
     { r; dt; spread; n; seed })

(* The float32 or float64 [dt], of [bits], array of [xs]. *)
let vector_bits (type a b) (dt : (a, b) D.t) bits xs =
  let b = Bytes.create (D.bytes dt (Array.length xs)) in
  Array.iteri
    (fun i x ->
      if bits = 32 then Bytes.set_int32_le b (4 * i) (Int32.bits_of_float x)
      else Bytes.set_int64_le b (8 * i) (Int64.bits_of_float x))
    xs;
  A.Any
    (A.v dt (L.contiguous [| Array.length xs |]) (Rig.Buffer.of_string (Bytes.to_string b)))

(* [r] of the terms [xs] by [k], each destination's terms. *)
let reduce_terms k (D.Any dt) r xs =
  let x = vector_bits dt (D.bits dt) xs in
  match run_stat k { r; axes = [| 0 |]; x; views = []; values = "" } with
  | None -> failf "nx.cpu declined %s" (reduction_name r)
  | Some got -> got

(* The kind [k] of each of [xs] at the float [dt], by the kernels. *)
let kind (D.Any dt) k xs =
  let (A.Any x) = vector_bits dt (D.bits dt) xs in
  let d = A.create Rig.host (A.dtype x) [| Array.length xs |] in
  match Nx_cpu.apply1 (P.Unary k) ~dst:d x with
  | A.Done -> Array.map value (terms (A.Any d))
  | a -> failf "a kind answered %a" Nx_array_support.pp_answer a

(* Logsumexp: the maximum [m], [c] where it is finite or else [0], then
   [c + log] of the sum of [exp (x - c)], an exp below the cut taken as
   [0]. Moments: the sum over the count, the mean, then the sums [s1] of
   the deviations [d] from it and [s2] of their squares, and [(s2 - s1 (s1
   / n)) / n], +0 where that is negative, or [s2] where that is +inf. Each
   operation rounds to the dtype. *)
let law_record (b : Support.backend) c =
  let (D.Any dt) = c.dt in
  let bits = D.bits dt in
  let gen = List.assoc c.spread spreads in
  let r = Random.State.make [| c.seed |] in
  let round x = if bits = 32 then Int32.float_of_bits (Int32.bits_of_float x) else x in
  let xs = Array.init c.n (fun i -> round (gen r i c.n)) in
  cover "several blocks" (c.n > 1024);
  cover "groups of blocks" (c.n >= 70_000);
  let got = reduce_terms b.kernels c.dt c.r xs in
  let sum ys = value (reduce_terms b.kernels c.dt (Monoid Sum) ys).(0).(0) in
  let bits_at x =
    if bits = 32 then Int64.of_int32 (Int32.bits_of_float x) else Int64.bits_of_float x
  in
  match c.r with
  | Monoid _ ->
      let m = value (reduce_terms b.kernels c.dt (Monoid Max) xs).(0).(0) in
      let c0 = if Float.is_finite m then m else 0. in
      let cut = if bits = 32 then -87. else -708. in
      let ys = Array.map (fun x -> round (x -. c0)) xs in
      let es = kind c.dt Exp (Array.map (Float.max cut) ys) in
      let es = Array.mapi (fun i e -> if ys.(i) < cut then 0. else e) es in
      let r = round (c0 +. (kind c.dt Log [| sum es |]).(0)) in
      equal ~msg:"logsumexp" int64 (bits_at r) (bits_of got.(0).(0))
  | _ ->
      let n = round (Float.of_int c.n) in
      let mean = round (sum xs /. n) in
      let d x = round (x -. mean) in
      let s1 = sum (Array.map d xs) in
      let s2 = sum (Array.map (fun x -> round (d x *. d x)) xs) in
      let q = round (round (s2 -. round (s1 *. round (s1 /. n))) /. n) in
      let var = if s2 = Float.infinity then s2 else if q < 0. then 0. else q in
      equal ~msg:"mean" int64 (bits_at mean) (bits_of got.(0).(0));
      equal ~msg:"variance" int64 (bits_at var) (bits_of got.(1).(0))

let stat_examples =
  Nx_kernel.
    [
      (* Groups of blocks on several threads, an outlier among offset terms. *)
      stat_example Moments f32 [| 70_000 |] [| 0 |] ~outliers:5000 "offset";
      stat_example (Monoid Logsumexp) f32 [| 3000 |] [| 0 |] ~specials:20 "wide";
      stat_example (Monoid Logsumexp) f64 [| 300; 9 |] [| 1 |] ~specials:20 "small";
      stat_example Moments f64 [| 300; 9 |] [| 1 |] ~specials:20 "small";
      stat_example (Arg Max) f32 [| 46; 13; 2 |] [| 2 |] ~specials:300 "ties";
      stat_example (Arg Min) (D.Any D.Int64) [| 5; 2000 |] [| 0 |] "small";
      stat_example (Arg Max) f64 [| 3; 4; 5 |] [| 1 |] "ties"
        ~view:{ perm = [| 2; 0; 1 |]; stepped = true; reversed = Some 0; broadcast = true };
      stat_example (Arg Min) f32 [| 3 |] [||] "small";
      stat_example Moments f32 [| 3; 0 |] [| 1 |] "small";
      stat_example Moments f32 [| 0; 3 |] [| 0 |] "small";
      stat_example (Monoid Logsumexp) f64 [| 2; 0 |] [| 1 |] "small";
      stat_example (Arg Max) (D.Any D.Uint8) [| 200; 64 |] [| 1 |] "small";
      (* Every block's extreme ties with the others'. *)
      stat_example (Arg Max) (D.Any D.Int16) [| 70_000 |] [| 0 |] "small";
      stat_example (Arg Min) f32 [| 2; 5000 |] [| 1 |] "ties";
    ]

(* A reduction or a scan into its own operand gives the bits of one into
   fresh destinations, as nx_cpu.mli states, though it overwrites the terms
   it reads: a Moments scan whose terms take its exact pass, its mean or its
   variance in place; running sums, Logsumexps and Arg's extremes of slices
   past a chunk, more units than a job splits into chunks; and a float16
   sum along no axis of signalling NaNs, whose bits come from the operand,
   quieted. *)
let test_in_place (b : Support.backend) () =
  let module K = (val b.kernels) in
  let copy (A.Any x) = A.Any (A.copy x) in
  let fresh (A.Any x as a) = A.Any (A.create Rig.host (A.dtype x) (shape_of a)) in
  let scan r axis x dsts =
    match
      K.scan (identity (dtype_of x)) ~axis (r, 0) ~dsts [| plain_of x |]
    with
    | A.Done -> Array.map Elements.elements dsts
    | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a
  in
  (* Destination [k] the operand; an Arg's positions int64. *)
  let check what (r : Nx_kernel.reduction) axis x k =
    let dst j =
      match r with
      | Arg _ when j = 1 ->
          A.Any (A.create Rig.host D.Int64 (shape_of x))
      | _ -> fresh x
    in
    let n = match r with Moments | Arg _ -> 2 | Monoid _ -> 1 in
    let want = scan r axis x (Array.init n dst) in
    let x' = copy x in
    let dsts = Array.init n (fun j -> if j = k then x' else dst j) in
    equal ~msg:what (array (array string)) want (scan r axis x' dsts)
  in
  let terms = A.Any (A.of_array D.Float32 [| 5 |] [| 1.; 2.; infinity; 4.; 5. |]) in
  let long =
    let l = elements D.Float32 (300 * 4100) ~specials:0 7 in
    A.Any (Option.get (A.move (M.Reshape [| 300; 4100 |]) l))
  in
  check "a Moments scan's mean in place" Moments 0 terms 0;
  check "a Moments scan's variance in place" Moments 0 terms 1;
  check "running sums in place" (Monoid Sum) 1 long 0;
  check "a Logsumexp scan in place" (Monoid Logsumexp) 1 long 0;
  check "an Arg scan's extreme in place" (Arg Max) 1 long 0;
  let codes = [| "\001\124"; "\000\060"; "\001\252" |] in
  let x = Elements.of_elements (D.Any D.Float16) [| 3 |] codes in
  let x' = copy x in
  let answer =
    K.reduce (identity (D.Any D.Float16)) ~shape:[| 3 |] ~axes:[||]
      [| (Monoid Sum, 0) |] ~dsts:[| x' |] [| plain_of x' |]
  in
  equal ~msg:"its answer" bool true (answer = A.Done);
  equal ~msg:"a float16 sum along no axis in place" (array string)
    (Array.map (Elements.quiet (D.Any D.Float16)) codes)
    (Elements.elements x')

(* A scan by Arg, Logsumexp or Moments along the rows of [k; n] terms: a
   few rows of up to a chunk's terms, past one now and then. *)
let lanes_case =
  Gen.with_pp pp_stat
    (let open Gen in
     let* r =
       of_list Nx_kernel.[ Arg Max; Arg Min; Monoid Logsumexp; Moments ]
     in
     let* d =
       of_list ~pp:pp_dtype
         (match r with Arg _ -> D.Any D.Int32 :: floats | _ -> floats)
     in
     let* k = int_range 1 20 in
     let* n = frequency [ (4, int_range 1 600); (1, int_range 4097 9000) ] in
     let* values = of_list value_kinds in
     let* specials = frequency [ (2, constant 0); (1, constant 100) ] in
     let+ seed = int in
     let (D.Any dt) = d in
     let make m =
       if is_float d then float_elements dt m values ~outliers:0 ~specials seed
       else A.Any (elements dt m ~specials seed)
     in
     let plain = { perm = [| 0; 1 |]; stepped = false; reversed = None; broadcast = false } in
     stat_of r [| k; n |] [| 1 |] plain values.name make)

(* Each row of a scan by Arg, Logsumexp or Moments has the bits of the row
   scanned alone, whichever way its lanes run: the rows side by side,
   staged into lanes; their transpose scanned down its columns, whose lanes
   lie side by side; and one row alone, whose chunks run as lanes past a
   chunk, or a Moments scan's tiles. *)
let law_lanes (b : Support.backend) (c : stat) =
  let s = shape_of c.x in
  let k = s.(0) and n = s.(1) in
  cover "past a chunk" (n > 4096);
  cover "sixteen lanes" (k >= 16);
  let (A.Any x) = c.x in
  let move m a = Option.get (A.move m a) in
  let scan a axis =
    let c = { c with x = A.Any a; axes = [| axis |] } in
    match run_stat_scan b.kernels c with
    | Some g -> Array.map (Array.map (fun t -> bits_of t)) g
    | None -> failf "nx.cpu declined %s" (reduction_name c.r)
  in
  let rows = scan x 1 in
  let cols = scan (A.copy (move (M.Permute [| 1; 0 |]) x)) 0 in
  for i = 0 to k - 1 do
    let whole = { M.start = 0; count = n; step = 1 } in
    let row = move (M.Slice [| { M.start = i; count = 1; step = 1 }; whole |]) x in
    let alone = scan (move (M.Reshape [| n |]) (A.copy row)) 0 in
    Array.iteri
      (fun d a ->
        let msg = Printf.sprintf "row %d, destination %d" i d in
        equal ~msg:(msg ^ ", side by side") (array int64) a
          (Array.sub rows.(d) (i * n) n);
        equal ~msg:(msg ^ ", transposed") (array int64) a
          (Array.init n (fun j -> cols.(d).((j * k) + i))))
      alone
  done

let lanes_examples =
  Nx_kernel.
    [
      stat_example (Monoid Logsumexp) f32 [| 17; 5000 |] [| 1 |] "wide";
      stat_example (Arg Max) f32 [| 17; 5000 |] [| 1 |] ~specials:100 "ties";
      stat_example Moments f32 [| 17; 5000 |] [| 1 |] "offset";
      stat_example Moments f64 [| 3; 1000 |] [| 1 |] ~specials:100 "small";
    ]

(* Scans past several chunks, one slice's several at a time and slices
   side by side, and a NaN, an infinity and -∞ among finite terms in a
   Logsumexp's terms. *)
let stat_scan_examples =
  let vector xs =
    {
      r = Monoid Logsumexp;
      axes = [| 0 |];
      x = A.Any (A.of_array D.Float32 [| Array.length xs |] xs);
      views = [];
      values = "small";
    }
  in
  Nx_kernel.
    [
      stat_example (Monoid Logsumexp) f32 [| 20_000 |] [| 0 |] ~specials:3000
        "wide";
      stat_example (Monoid Logsumexp) f64 [| 5000; 3 |] [| 0 |] "small";
      (* Sixteen chunks side by side, and sixteen slices, contiguous and
         staged. *)
      stat_example (Monoid Logsumexp) f32 [| 70_000 |] [| 0 |] ~specials:3000
        "small";
      stat_example (Monoid Logsumexp) f32 [| 300; 40 |] [| 0 |] "wide";
      stat_example (Monoid Logsumexp) f64 [| 40; 300 |] [| 1 |] "offset";
      stat_example (Arg Min) f64 [| 70_000 |] [| 0 |] ~specials:3000 "ties";
      stat_example (Arg Max) f32 [| 9000 |] [| 0 |] ~specials:300 "ties";
      stat_example (Arg Min) (D.Any D.Int32) [| 300; 5 |] [| 0 |] "small";
      vector [| 1.; Float.infinity; 2. |];
      vector [| 1.; Float.nan; 2. |];
      vector [| Float.neg_infinity; 1.; Float.neg_infinity; 3. |];
    ]

(* A complex Arg scan orders real parts, then imaginary ones; a number with
   a NaN part holds from there on. *)
let test_complex_arg_scan (b : Support.backend) () =
  let module K = (val b.kernels) in
  let z re im = { Complex.re; im } in
  let xs = [| z 1. 0.; z 1. 1.; z 0. 5.; z Float.nan 0.; z 3. 0. |] in
  let x = A.of_array D.Complex64 [| 5 |] xs in
  let scan e =
    let v = A.create Rig.host D.Complex64 [| 5 |]
    and at = A.create Rig.host D.Int64 [| 5 |] in
    let answer =
      K.scan (identity (D.Any D.Complex64)) ~axis:0 (Arg e, 0)
        ~dsts:[| A.Any v; A.Any at |] [| Nx_kernel.Plain x |]
    in
    equal ~msg:"its answer" bool true (answer = A.Done);
    (Elements.elements (A.Any v), A.to_array at)
  in
  let pick ks =
    Array.map
      (fun k -> Elements.quiet (D.Any D.Complex64) (Elements.elements (A.Any x)).(k))
      ks
  in
  equal ~msg:"Arg Max" (pair (array string) (array int64))
    (pick [| 0; 1; 1; 3; 3 |], [| 0L; 1L; 1L; 3L; 3L |])
    (scan Max);
  equal ~msg:"Arg Min" (pair (array string) (array int64))
    (pick [| 0; 0; 2; 3; 3 |], [| 0L; 0L; 2L; 3L; 3L |])
    (scan Min)

(* nx.cpu refuses an Arg of no term, and a Moments scan into destinations
   of other shapes, float32 or narrow. *)
let test_stateful_cases (b : Support.backend) () =
  let module K = (val b.kernels) in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let reduce r x dsts =
    K.reduce (identity (dtype_of x)) ~shape:(shape_of x) ~axes:[| 1 |]
      [| (r, 0) |] ~dsts [| plain_of x |]
  in
  let make dt s = A.Any (A.create Rig.host dt s) in
  let pos = make D.Int64 [| 2 |] in
  equal ~msg:"an Arg of no term" answer A.Shape_mismatch
    (reduce (Arg Min) (make D.Float32 [| 2; 0 |]) [| make D.Float32 [| 2 |]; pos |]);
  let scan dt mean var =
    let x = make dt [| 2; 3 |] in
    K.scan (identity (dtype_of x)) ~axis:1 (Moments, 0)
      ~dsts:[| make dt mean; make dt var |] [| plain_of x |]
  in
  equal ~msg:"a float32 Moments scan into a mean of another shape" answer
    A.Shape_mismatch
    (scan D.Float32 [| 3; 2 |] [| 2; 3 |]);
  equal ~msg:"a bfloat16 Moments scan into a variance of another shape" answer
    A.Shape_mismatch
    (scan D.Bfloat16 [| 2; 3 |] [| 3; 2 |])

(* Guard pages *)

(* A reduction by [r] along [axes], or a scan along [axes.(0)], of [x]
   through the views [views]: each that nx.cpu computes. *)
type guard = {
  scan : bool;
  r : Nx_kernel.reduction;
  axes : int array;
  x : A.any;
  views : string list;
}

let pp_guard ppf (g : guard) =
  let (A.Any x) = g.x in
  Format.fprintf ppf "%s %s along %a of %a %a (%s)"
    (if g.scan then "scan" else "reduce")
    (reduction_name g.r) pp_ints g.axes D.pp (A.dtype x) L.pp (A.layout x)
    (String.concat ", " g.views)

(* Innermost extents of one element to a few vectors of the widest dtype
   past the widest vector, and at the edges of lane blocks and scan
   chunks. *)
let inner =
  let open Gen in
  frequency
    [
      (6, int_range 1 72);
      (2, of_list ~pp:Format.pp_print_int [ 127; 128; 129; 255; 256; 257 ]);
      (1, of_list ~pp:Format.pp_print_int [ 1023; 1025; 4095; 4097 ]);
    ]

let guard_case =
  Gen.with_pp pp_guard
    (let open Gen in
     let* d =
       of_list ~pp:pp_dtype
         (floats @ floats @ floats @ floats @ base @ narrow_dtypes)
     in
     let* scan = bool in
     let float = is_float d
     and boolean = same d (D.Any D.Bool) || same d (D.Any D.Bit) in
     (* Moments and Logsumexp, which floats alone take, come twice as
        often. *)
     let* r =
       of_list
         (Nx_kernel.[ Monoid Max; Monoid Min ]
         @ (if boolean then [] else Nx_kernel.[ Monoid Sum; Monoid Prod ])
         @ (if float then Nx_kernel.[ Moments; Moments ] else [])
         @ Nx_kernel.[ Arg Max; Arg Min ]
         @ if float then Nx_kernel.[ Monoid Logsumexp; Monoid Logsumexp ]
           else [])
     in
     let* n = inner in
     let* outer = array ~size:(int_range 0 2) (int_range 1 5) in
     let s = Array.append outer [| n |] in
     let rank = Array.length s in
     let* axes =
       if scan then map (fun a -> [| a |]) (int_range 0 (rank - 1))
       else
         let+ keep = array ~size:(constant rank) bool in
         Array.of_list (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let* v = view_of rank in
     let* specials = frequency [ (3, constant 0); (1, constant 20) ] in
     let+ seed = int in
     { scan; r; axes; x = operand d s v ~specials seed; views = view_names v })

(* A running sum of bfloat16 [+inf; -inf] at the page after: its last
   result is a NaN made from numbers, and no term is a NaN for the results
   to take. *)
let guard_examples =
  let x = A.Any (A.of_array D.Bfloat16 [| 2 |] [| infinity; neg_infinity |]) in
  [
    ( ({ scan = true; r = Monoid Sum; axes = [| 0 |]; x; views = [] } : guard),
      Support.After );
  ]

(* The same bits with the operand, through its view, and each destination
   against an inaccessible page. *)
let law_guards (b : Support.backend) ((g : guard), side) =
  let (module K : Nx_kernel.S) = b.kernels in
  let after = side = Support.After in
  let rank = Array.length (shape_of g.x) in
  let n = (shape_of g.x).(rank - 1) in
  collect (Printf.sprintf "%s %s" (if g.scan then "scan" else "reduce")
       (reduction_name g.r));
  let is r = g.r = r in
  cover "a monoid reduced" ((not g.scan) && is (Monoid Sum));
  cover "a monoid scanned" (g.scan && is (Monoid Max));
  cover "Logsumexp" (is (Monoid Logsumexp));
  cover "an Arg" (is (Arg Max) || is (Arg Min));
  cover "an Arg or Logsumexp scanned"
    (g.scan && (is (Arg Max) || is (Arg Min) || is (Monoid Logsumexp)));
  cover "Moments reduced" ((not g.scan) && is Moments);
  cover "Moments scanned" (g.scan && is Moments);
  cover "the innermost axis reduced" (Array.mem (rank - 1) g.axes);
  cover "outer axes alone reduced"
    (g.axes <> [||] && not (Array.mem (rank - 1) g.axes));
  cover "an innermost extent past a vector, not a multiple of one"
    (n > 32 && n mod 16 <> 0);
  cover "a stepped operand at the page before"
    ((not after) && List.mem "stepped" g.views);
  cover "a reversed operand at the page after"
    (after && List.mem "reversed" g.views);
  cover "a broadcast operand" (List.mem "broadcast" g.views);
  let (A.Any x) = g.x in
  let dt = D.Any (A.dtype x) in
  let s =
    let s = shape_of g.x in
    if g.scan then s
    else
      Array.of_list
        (List.filteri (fun i _ -> not (Array.mem i g.axes)) (Array.to_list s))
  in
  let dsts =
    match g.r with
    | Arg _ -> [| (dt, s); (D.Any D.Int64, s) |]
    | Moments -> [| (dt, s); (dt, s) |]
    | Monoid _ -> [| (dt, s) |]
  in
  let call xs ds =
    let p = identity dt and loads = [| plain_of xs.(0) |] in
    if g.scan then K.scan p ~axis:g.axes.(0) (g.r, 0) ~dsts:ds loads
    else K.reduce p ~shape:(loop_shape loads) ~axes:g.axes [| (g.r, 0) |]
      ~dsts:ds loads
  in
  equal Elements.answer A.Done
    (Elements.same_guarded side ~call [| g.x |] dsts)

(* A padded load's operand, which nx.cpu copies padded first, against the
   page, and the destination likewise. *)
let law_padded_guards (b : Support.backend) (p, side) =
  match p.loaded with
  | Refused _ -> reject ()
  | Reads c ->
      cover "windows" (p.pad.windows <> [||]);
      cover "cropped" (Array.exists (fun l -> l < 0) p.pad.lo);
      cover "at the page after" (side = Support.After);
      cover "at the page before" (side = Support.Before);
      let dt = dtype_of p.x in
      let call xs ds =
        compute b.kernels ~scan:c.scan (identity dt) c.monoid c.axes ds.(0)
          [| padded_load xs.(0) p.fill p.pad |]
      in
      equal Elements.answer A.Done
        (Elements.same_guarded side ~call [| p.x |] [| (dt, result_shape c) |])

(* Fused programs *)

(* A reduction of output [k] of [prog], a program over [loads] of one shape,
   by [red] along [axes], or with [scan] its scan along [axes.(0)]; [guard]
   puts each load against an inaccessible page on its side. *)
type fused = {
  name : string;
  shape : int array;
  scan : bool;
  prog : P.t;
  k : int;
  red : Nx_kernel.reduction;
  axes : int array;
  loads : A.any array;
  lanes : bool;
  guard : Support.side option;
  fviews : string list;
}

let reduction_label : Nx_kernel.reduction -> string = function
  | Monoid m -> monoid_name m
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"
  | Moments -> "Moments"

let pp_fused ppf c =
  let shape = c.shape in
  Format.fprintf ppf "%s, %s %s of output %d along %a of %a%s (%s)" c.name
    (if c.scan then "scan" else "reduction")
    (reduction_label c.red) c.k pp_ints c.axes pp_ints shape
    (match c.guard with
    | None -> ""
    | Some Support.Before -> ", guarded before"
    | Some Support.After -> ", guarded after")
    (String.concat ", " c.fviews)

(* The programs, over loads of the float dtype [d]: each one's name, its
   loads, whether its second load is a row of lanes (broadcast along the
   reduced axes), its nodes, its outputs, and the output the reduction
   reads. [rank] bounds its coordinates. *)
let templates (D.Any dt as d) rank =
  let num x =
    match dt with
    | D.Float32 -> P.Const (d, P.bits D.Float32 x)
    | D.Float64 -> P.Const (d, P.bits D.Float64 x)
    | D.Complex64 -> P.Const (d, P.bits D.Complex64 { Complex.re = x; im = 0. })
    | D.Complex128 ->
        P.Const (d, P.bits D.Complex128 { Complex.re = x; im = 0. })
    | _ -> invalid_arg "templates: a float or complex dtype"
  in
  let i64 = D.Any D.Int64 in
  if D.is D.Complex dt then
    [
      ("a difference with lanes", [| d; d |], true,
       P.[| In 0; In 1; Op2 (Binary Sub, 0, 1) |], [| 2 |], 0);
      ("a scaled operand", [| d |], false,
       P.[| In 0; num 0.5; Op2 (Binary Mul, 0, 1) |], [| 2 |], 0);
      ("a product of two operands", [| d; d |], false,
       P.[| In 0; In 1; Op2 (Binary Mul, 0, 1) |], [| 2 |], 0);
    ]
  else
  [
    ("exp of a difference with lanes", [| d; d |], true,
     P.[| In 0; In 1; Op2 (Binary Sub, 0, 1); Op1 (Unary Exp, d, 2) |],
     [| 3 |], 0);
    ("a scaled operand", [| d |], false,
     P.[| In 0; num 0.5; Op2 (Binary Mul, 0, 1) |], [| 2 |], 0);
    ("a product of two operands", [| d; d |], false,
     P.[| In 0; In 1; Op2 (Binary Mul, 0, 1) |], [| 2 |], 0);
    ("a log, NaN below 0", [| d |], false,
     P.[| In 0; Op1 (Unary Log, d, 0) |], [| 1 |], 0);
    ("the second of two outputs", [| d |], false,
     P.[| In 0; Op1 (Unary Neg, d, 0); Op1 (Unary Abs, d, 0) |], [| 1; 2 |], 1);
    ("a non-zero test", [| d |], false,
     P.[| In 0; num 0.; Op2 (Compare Not_equal, 0, 1) |], [| 2 |], 0);
    ("a count of positives", [| d |], false,
     P.[| In 0; num 0.; Op2 (Compare Less, 1, 0); Op1 (Cast, i64, 2) |],
     [| 3 |], 0);
    ("a constant alone, of no load", [||], false, [| num 0.5 |], [| 0 |], 0);
    ("an operand, beside its negation", [| d |], false,
     P.[| In 0; Op1 (Unary Neg, d, 0) |], [| 0; 1 |], 0);
    ("an operand cast into bfloat16", [| d |], false,
     P.[| In 0; Op1 (Cast, D.Any D.Bfloat16, 0) |], [| 1 |], 0);
    ("an operand cast into int4", [| d |], false,
     P.[| In 0; Op1 (Cast, D.Any D.Int4, 0) |], [| 1 |], 0);
  ]
  @
  if rank = 0 then []
  else
    [
      ("an operand plus its last coordinate", [| d |], false,
       P.[| In 0; Coord (rank - 1); Op1 (Cast, d, 1); Op2 (Binary Add, 0, 2) |],
       [| 3 |], 0);
      ("its last coordinate alone, of no load", [||], false,
       P.[| Coord (rank - 1); Op1 (Cast, d, 0) |], [| 1 |], 0);
    ]

(* The reductions an output of [dt] takes, or with [scan] its scans. *)
let reductions_of ~scan (D.Any dt) : Nx_kernel.reduction list =
  match D.kind dt with
  | D.Float when scan ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Monoid Logsumexp;
        Arg Max; Arg Min; Moments ]
  | D.Float ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Monoid Logsumexp;
        Arg Max; Arg Min; Moments ]
  | D.Complex when scan ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Arg Max; Arg Min ]
  | D.Complex ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Arg Max; Arg Min ]
  | D.Boolean -> [ Monoid Max; Monoid Min ]
  | _ when scan ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Arg Max ]
  | _ -> [ Monoid Sum; Monoid Max; Monoid Min; Arg Max ]

let fused_shapes =
  let open Gen in
  let small = array ~size:(int_range 1 3) (int_range 1 5) in
  let rows =
    let* r = int_range 1 40 in
    let+ k = of_list ~pp:Format.pp_print_int [ 9; 17; 64; 1100; 5000 ] in
    [| r; k |]
  in
  let columns =
    let* k = of_list ~pp:Format.pp_print_int [ 9; 17; 300; 1100 ] in
    let+ r = int_range 1 200 in
    [| k; r |]
  in
  let apart =
    let+ a = int_range 2 20
    and+ b = int_range 2 5
    and+ c = int_range 2 20
    and+ d = int_range 2 9 in
    [| a; b; c; d |]
  in
  let empty =
    of_list ~pp:pp_ints [ [| 0 |]; [| 0; 5 |]; [| 3; 0 |]; [| 2; 0; 4 |] ]
  in
  frequency
    [ (3, small); (3, rows); (2, columns); (1, long); (2, apart);
      (1, constant ~pp:pp_ints [||]); (1, empty) ]

let fused_case =
  let open Gen in
  with_pp pp_fused
  @@ let* d =
       frequency
         [
           (3, constant ~pp:pp_dtype (D.Any D.Float32));
           (3, constant ~pp:pp_dtype (D.Any D.Float64));
           (1, of_list ~pp:pp_dtype D.[ Any Complex64; Any Complex128 ]);
         ]
     in
  let* s = fused_shapes in
  let r = Array.length s in
  (* A scan runs along an axis: a shape of rank 0 has none. *)
  let* scan =
    if r = 0 then constant false
    else frequency [ (2, constant false); (1, constant true) ]
  in
  (* Any subset of the axes, or for a shape of four, the first and third
     at times: two reduced axes split by a kept one, the last kept. *)
  let* axes =
    let any =
      let+ keep = array ~size:(constant r) bool in
      let axes = List.filter (fun i -> keep.(i)) (List.init r Fun.id) in
      Array.of_list (if axes = [] && r > 0 then [ r - 1 ] else axes)
    in
    if scan then map (fun a -> [| a |]) (int_range 0 (r - 1))
    else if r = 4 then frequency [ (1, any); (2, constant [| 0; 2 |]) ]
    else any
  in
  let ts = templates d r in
  let* t = int_range 0 (List.length ts - 1) in
  let name, ins, lanes, nodes, outs, k = List.nth ts t in
  let prog = P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
      ~outs:(Iarray.of_array outs) in
  let o = P.dtype prog outs.(k) in
  let* red = of_list (reductions_of ~scan o) in
  let* v = view_of r in
  let* lv = view_of r in
  let* specials = frequency [ (3, constant 0); (1, constant 20) ] in
  let* guard =
    frequency
      [ (3, constant None); (1, constant (Some Support.Before));
        (1, constant (Some Support.After)) ]
  in
  let+ seed = int in
  let load i =
    if i = 1 && lanes then
      let row = Array.mapi (fun j e -> if Array.mem j axes then 1 else e) s in
      let (A.Any x) = operand d row lv ~specials (seed + i) in
      A.Any (Option.get (A.move (M.Broadcast s) x))
    else operand d s (if i = 0 then v else lv) ~specials (seed + i)
  in
  {
    name; shape = s; scan; prog; k; red; axes; lanes; guard;
    loads = Array.init (Array.length ins) load;
    fviews = view_names v;
  }

(* The case of template [i] over [d] of shape [s], plain or with
   [broadcast] its first load broadcast along its first axis, by [red]
   along [axes], or with [scan] scanned. *)
let fused_example ?(d = D.Any D.Float32) ?(scan = false) ?(broadcast = false)
    i red s axes =
  let r = Array.length s in
  let name, ins, lanes, nodes, outs, k = List.nth (templates d r) i in
  let plain =
    { perm = Array.init r Fun.id; stepped = false; reversed = None;
      broadcast = false }
  in
  let load j =
    if j = 1 && lanes then
      let row = Array.mapi (fun a e -> if Array.mem a axes then 1 else e) s in
      let (A.Any x) = operand d row plain ~specials:0 (7 + j) in
      A.Any (Option.get (A.move (M.Broadcast s) x))
    else operand d s { plain with broadcast = broadcast && j = 0 } ~specials:0
        (7 + j)
  in
  {
    name; shape = s; scan; k; red; axes; lanes; guard = None; fviews = [];
    prog =
      P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
        ~outs:(Iarray.of_array outs);
    loads = Array.init (Array.length ins) load;
  }

(* A constant's sum over a long axis and a broadcast load's own sum, whose
   terms map.c holds once for a whole row; complex products, whose lanes
   and scans start from their first terms:
   one output's block, rows of outputs side by side, two reduced axes split
   by a kept one, several blocks, and scans past a chunk; float32 scans
   past a chunk: of one slice, in quads of chunks, as Moments, and of a row
   of slices side by side; a reduction of several blocks; Logsumexp down
   columns, several outputs' blocks computed at once; and outputs
   narrower than their carrier: bfloat16's Arg, Moments scan and scan in
   quads of chunks, and int4's scan so; Logsumexp scans of one slice past
   a chunk and of slices side by side down rows, an Arg scan past a chunk,
   and bfloat16's Arg scan, its extreme the program's term. *)
let fused_examples =
  let c = D.Any D.Complex64 in
  [
    fused_example ~d:c 2 (Monoid Prod) [| 3; 40 |] [| 1 |];
    fused_example ~d:c 2 (Monoid Prod) [| 40; 3 |] [| 0 |];
    fused_example ~d:c 2 (Monoid Prod) [| 3; 4; 5 |] [| 0; 2 |];
    fused_example ~d:c 0 (Monoid Prod) [| 2; 2000 |] [| 1 |];
    fused_example ~d:c ~scan:true 2 (Monoid Prod) [| 9000 |] [| 0 |];
    fused_example ~d:c ~scan:true 2 (Monoid Prod) [| 3; 5000 |] [| 1 |];
    fused_example ~d:c ~scan:true 2 (Monoid Prod) [| 5000; 3 |] [| 0 |];
    fused_example 7 (Monoid Sum) [| 100000 |] [| 0 |];
    fused_example ~broadcast:true 8 (Monoid Sum) [| 100000 |] [| 0 |];
    fused_example ~scan:true 0 (Monoid Sum) [| 9000 |] [| 0 |];
    fused_example ~scan:true 1 Moments [| 3; 9000 |] [| 1 |];
    fused_example ~scan:true 1 Moments [| 9000 |] [| 0 |];
    fused_example ~scan:true 3 (Monoid Max) [| 9000; 3 |] [| 0 |];
    fused_example ~scan:true 2 (Monoid Prod) [| 20000 |] [| 0 |];
    fused_example 2 (Monoid Sum) [| 5000 |] [| 0 |];
    fused_example 0 (Monoid Logsumexp) [| 300; 20 |] [| 0 |];
    fused_example 9 (Arg Max) [| 3; 40 |] [| 1 |];
    fused_example ~scan:true 9 Moments [| 3; 100 |] [| 1 |];
    fused_example ~scan:true 9 (Monoid Sum) [| 20000 |] [| 0 |];
    fused_example ~scan:true 10 (Monoid Max) [| 20000 |] [| 0 |];
    fused_example ~scan:true 0 (Monoid Logsumexp) [| 20000 |] [| 0 |];
    fused_example ~scan:true 1 (Monoid Logsumexp) [| 9000; 3 |] [| 0 |];
    fused_example ~scan:true 3 (Arg Min) [| 20000 |] [| 0 |];
    fused_example ~scan:true 9 (Arg Max) [| 3; 40 |] [| 1 |];
  ]

(* The destinations of [c]'s reduction, fresh, of output [c.k]'s dtype. *)
let fused_dsts (b : Support.backend) c =
  let s = c.shape in
  let y =
    if c.scan then s
    else
      Array.of_list
        (List.filteri (fun i _ -> not (Array.mem i c.axes)) (Array.to_list s))
  in
  let (D.Any dt) = P.dtype c.prog (Iarray.get c.prog.outs c.k) in
  let make () = A.Any (A.create b.device dt y) in
  match c.red with
  | Monoid _ -> [| make () |]
  | Moments -> [| make (); make () |]
  | Arg _ -> [| make (); A.Any (A.create b.device D.Int64 y) |]

(* Each destination's bytes. *)
let bytes_of (A.Any x) =
  let n = total (L.shape (A.layout x)) in
  let w = max 1 (D.bits (A.dtype x) / 8) in
  let u = A.copy x in
  match A.bitcast D.Uint8 u with
  | Some b -> Array.to_list (A.to_array b)
  | None -> List.init (n * w) (fun _ -> 0)

(* [c] fused, its loads guarded where it says, against [c]'s program mapped
   into fresh arrays and reduced as their outputs' identity. *)
let law_fused (b : Support.backend) c =
  let module K = (val b.kernels) in
  let s = c.shape in
  let terms = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  classify c.name true;
  cover "a loop of no load" (c.loads = [||]);
  let complex =
    let (D.Any dt) = P.dtype c.prog (Iarray.get c.prog.outs c.k) in
    D.is D.Complex dt
  in
  cover "a complex output" complex;
  cover "a complex product" (complex && c.red = Monoid Prod);
  cover "a narrow output" (List.mem c.name
    [ "an operand cast into bfloat16"; "an operand cast into int4" ]);
  cover (reduction_label c.red) true;
  cover "a scan" c.scan;
  cover "a Moments scan" (c.scan && c.red = Moments);
  cover "a stateful scan"
    (c.scan && (c.red = Monoid Logsumexp || c.red = Arg Max || c.red = Arg Min));
  cover "a scan past a chunk" (c.scan && s.(c.axes.(0)) > 4096);
  cover "a row of lanes" c.lanes;
  cover "a lane or fewer terms an output" (terms <= 16);
  cover "several blocks of terms" (terms > 1024);
  let r = Array.length s in
  let side = r > 1 && not (Array.mem (r - 1) c.axes) in
  let split =
    Array.exists (fun a -> a + 1 < r && not (Array.mem (a + 1) c.axes)
                           && Array.exists (fun b -> b > a + 1) c.axes) c.axes
  in
  cover "outputs side by side" side;
  cover "reduced axes split by a kept one" split;
  cover "reduced axes split, outputs side by side, more than a lane of terms"
    (split && side && terms > 16);
  cover "guarded" (c.guard <> None);
  cover "rank 0" (r = 0);
  cover "no element" (total s = 0);
  let plain = Array.map (fun (A.Any x) -> Nx_kernel.Plain x) in
  let outs =
    Array.map
      (fun o ->
        let (D.Any dt) = P.dtype c.prog o in
        A.Any (A.create b.device dt s))
      (Iarray.to_array c.prog.outs)
  in
  (match K.map c.prog ~dsts:outs (plain c.loads) with
  | A.Done -> ()
  | r -> failf "the map answered %a" Nx_array_support.pp_answer r);
  let (A.Any y) = outs.(c.k) in
  let want = fused_dsts b c in
  let compute_in (module M : Nx_kernel.S) p k ~dsts loads =
    if c.scan then M.scan p ~axis:c.axes.(0) (c.red, k) ~dsts loads
    else M.reduce p ~shape:c.shape ~axes:c.axes [| (c.red, k) |] ~dsts loads
  in
  let compute = compute_in b.kernels in
  (* An extreme of no term refuses alike. *)
  let expected =
    compute (identity (D.Any (A.dtype y))) 0 ~dsts:want [| Nx_kernel.Plain y |]
  in
  let guarded =
    Array.map
      (fun (A.Any x as a) ->
        match c.guard with
        | None -> (a, Fun.id)
        | Some side ->
            let g, free = Support.guarded side (A.dtype x) (A.layout x) in
            (* Each element once: one index along an axis stepped 0. *)
            let once y =
              let l = A.layout y in
              let st = L.strides l in
              Option.get
                (A.move
                   (M.Slice
                      (Array.mapi
                         (fun i e ->
                           { M.start = 0; count = (if st.(i) = 0 then min e 1 else e);
                             step = 1 })
                         (L.shape l)))
                   y)
            in
            A.blit ~src:(once x) ~dst:(once g);
            (A.Any g, free))
      c.loads
  in
  let got = fused_dsts b c in
  Fun.protect
    ~finally:(fun () -> Array.iter (fun (_, free) -> free ()) guarded)
    (fun () ->
      let answer = compute c.prog c.k ~dsts:got (plain (Array.map fst guarded)) in
      equal ~msg:"answer" Elements.answer expected answer;
      if answer = A.Done then begin
        (* One thread gives the job's bits. *)
        let one = fused_dsts b c in
        equal ~msg:"one thread's answer" Elements.answer A.Done
          (compute_in (module One) c.prog c.k ~dsts:one (plain c.loads));
        Array.iteri
          (fun i w ->
            let bytes = bytes_of (host w) in
            equal ~msg:(Printf.sprintf "destination %d" i) (list int) bytes
              (bytes_of (host got.(i)));
            equal ~msg:(Printf.sprintf "destination %d, one thread" i)
              (list int) bytes (bytes_of (host one.(i))))
          want
      end)

(* A program computed into its own load gives the bits of one computed
   into fresh destinations, as nx_cpu.mli states: a float16 negation of
   signalling NaNs summed along no axis, whose bits come from the load,
   quieted; scanned, its NaN results settled from the load;
   summed beside its maximum, each written after the other reads; and its
   Arg, reduced along no axis and scanned over numbers, each extreme
   computed from the load at its position after earlier results are
   written. *)
let test_program_in_place (b : Support.backend) () =
  let module K = (val b.kernels) in
  let h = D.Any D.Float16 in
  let p =
    P.v ~ins:(Iarray.of_array [| h |])
      (Iarray.of_array P.[| In 0; Op1 (Unary Neg, h, 0) |])
      ~outs:(Iarray.of_array [| 1 |])
  in
  let nans = [| "\001\124"; "\000\060"; "\001\252" |] in
  (* 3, 1 and 2: their negations' running maximum is at 0, 1, then 1. *)
  let numbers = [| "\000\066"; "\000\060"; "\000\064" |] in
  let check what codes dts call =
    let load () = Elements.of_elements h [| 3 |] codes in
    let fresh j =
      let (D.Any dt) = dts.(j) in
      A.Any (A.create Rig.host dt [| 3 |])
    in
    let want = Array.init (Array.length dts) fresh in
    equal ~msg:(what ^ ", fresh") Elements.answer A.Done
      (call (load ()) want);
    let x = load () in
    let got = Array.init (Array.length dts) (fun j -> if j = 0 then x else fresh j) in
    equal ~msg:(what ^ ", into its load") Elements.answer A.Done (call x got);
    equal ~msg:what (array (array string))
      (Array.map Elements.elements want)
      (Array.map Elements.elements got)
  in
  let reduce rs x dsts =
    K.reduce p ~shape:[| 3 |] ~axes:[||] rs ~dsts [| plain_of x |]
  in
  let scan r x dsts = K.scan p ~axis:0 (r, 0) ~dsts [| plain_of x |] in
  let i64 = D.Any D.Int64 in
  check "a sum along no axis" nans [| h |] (reduce [| (Monoid Sum, 0) |]);
  check "a running sum" nans [| h |] (scan (Monoid Sum));
  check "a sum and a maximum" nans [| h; h |]
    (reduce [| (Monoid Sum, 0); (Monoid Max, 0) |]);
  check "an Arg along no axis" numbers [| h; i64 |]
    (reduce [| (Arg Max, 0) |]);
  check "an Arg scan" numbers [| h; i64 |] (scan (Arg Max))

(* Several reductions of one loop: [c]'s and [red] of output [k], with
   [c]'s axes, one call. *)
type several = { first : fused; red2 : Nx_kernel.reduction; k2 : int }

let pp_several ppf s =
  Format.fprintf ppf "%a, then %s of output %d" pp_fused s.first
    (reduction_label s.red2) s.k2

let several_case =
  let open Gen in
  with_pp pp_several
  @@ let* c = fused_case in
     let c = { c with scan = false; guard = None } in
     let outs = Iarray.length c.prog.outs in
     (* The other output where there are two, mostly. *)
     let* k2 =
       if outs = 1 then constant 0
       else frequency [ (3, constant (1 - c.k)); (1, constant c.k) ]
     in
     let o = P.dtype c.prog (Iarray.get c.prog.outs k2) in
     let+ red2 = of_list (reductions_of ~scan:false o) in
     { first = c; red2; k2 }

(* The destinations of [red] of output [k] of [c], fresh. *)
let dsts_of (b : Support.backend) c red k =
  fused_dsts b { c with red; k }

(* Several reductions in one call give each one's bits alone. *)
let law_several (b : Support.backend) s =
  let module K = (val b.kernels) in
  let c = s.first in
  cover "two outputs" (s.k2 <> c.k);
  cover "a state and a monoid"
    ((match c.red with Monoid _ -> false | _ -> true)
    <> (match s.red2 with Monoid _ -> false | _ -> true));
  let loads = Array.map (fun (A.Any x) -> Nx_kernel.Plain x) c.loads in
  let alone red k =
    let d = dsts_of b c red k in
    (K.reduce c.prog ~shape:c.shape ~axes:c.axes [| (red, k) |] ~dsts:d loads, d)
  in
  let a1, w1 = alone c.red c.k and a2, w2 = alone s.red2 s.k2 in
  let d1 = dsts_of b c c.red c.k and d2 = dsts_of b c s.red2 s.k2 in
  let answer =
    K.reduce c.prog ~shape:c.shape ~axes:c.axes
      [| (c.red, c.k); (s.red2, s.k2) |]
      ~dsts:(Array.append d1 d2) loads
  in
  (* The pair refuses as the first of them that refuses alone. *)
  let expected = if a1 <> A.Done then a1 else a2 in
  equal ~msg:"answer" Elements.answer expected answer;
  if answer = A.Done then begin
    let same w g =
      Array.iteri
        (fun i x -> equal (list int) (bytes_of (host x)) (bytes_of (host g.(i))))
        w
    in
    same w1 d1;
    same w2 d2
  end

(* Two reductions whose second destination has another shape refuse,
   writing neither. *)
let test_several_refuse (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let p = identity (D.Any D.Float32) in
  let y1 = A.of_array D.Float32 [| 2 |] [| 7.; 7. |] in
  let y2 = A.create b.device D.Float32 [| 3 |] in
  let answer =
    K.reduce p ~shape:[| 2; 3 |] ~axes:[| 1 |]
      [| (Monoid Sum, 0); (Monoid Max, 0) |]
      ~dsts:[| A.Any y1; A.Any y2 |]
      [| Nx_kernel.Plain x |]
  in
  equal ~msg:"answer" bool true (answer = A.Shape_mismatch);
  equal ~msg:"the first destination" (array float_exact) [| 7.; 7. |]
    (A.to_array y1)

(* Carries *)

(* A dtype a carried scan draws: its finite elements, near one for a
   product, its NaNs, its other specials, and two elements whose sum, or
   product, is a NaN made from numbers. *)
type 'v drawn = {
  finite : monoid -> Random.State.t -> 'v;
  nans : 'v list;
  others : 'v list;
  made : monoid -> ('v * 'v) option;
}

type carried = Carried : ('v, 's) D.t * 'v drawn -> carried

let carried_kinds =
  let float_values =
    {
      finite =
        (fun m r ->
          match m with
          | Prod ->
              (if Random.State.bool r then 1. else -1.)
              *. (0.999 +. Random.State.float r 0.002)
          | _ ->
              Float.ldexp
                (Random.State.float r 2. -. 1.)
                (Random.State.int r 9 - 4));
      nans =
        [
          nan; -.nan; Int64.float_of_bits 0x7FF4_0000_0000_0000L;
          Int64.float_of_bits 0xFFF9_2340_0000_0000L;
        ];
      others = [ -0.; 0.; infinity; neg_infinity ];
      made =
        (function
        | Sum -> Some (infinity, neg_infinity)
        | Prod -> Some (0., infinity)
        | Max | Min | Logsumexp -> None);
    }
  in
  let float dt = Carried (dt, float_values) in
  let complex dt =
    let c re im = { Complex.re; im } in
    Carried
      ( dt,
        {
          finite =
            (fun m r ->
              c (float_values.finite m r) (0.01 *. float_values.finite Sum r));
          nans = [ c nan 1.; c 1. (-.nan); c (-.nan) nan ];
          others = [ c (-0.) (-0.); c infinity 0.; c 0. neg_infinity ];
          made =
            (function
            | Sum -> Some (c infinity 0., c neg_infinity 1.)
            | Prod -> Some (c 0. 0., c infinity 0.)
            | Max | Min | Logsumexp -> None);
        } )
  in
  let ints dt ~small ~edges =
    Carried
      ( dt,
        {
          finite = (fun _ r -> small (Random.State.int r 16 - 8));
          nans = [];
          others = edges;
          made = (fun _ -> None);
        } )
  in
  [
    float D.Float32; float D.Float64; float D.Float16; float D.Bfloat16;
    complex D.Complex64; complex D.Complex128;
    ints D.Int32 ~small:Int32.of_int ~edges:[ Int32.max_int; Int32.min_int ];
    ints D.Int8 ~small:Fun.id ~edges:[ 127; -128 ];
    ints D.Int4 ~small:Fun.id ~edges:[ 7; -8 ];
    Carried
      ( D.Bool,
        {
          finite = (fun _ r -> Random.State.bool r);
          nans = [];
          others = [ true; false ];
          made = (fun _ -> None);
        } );
  ]

(* A scan along axis 1 of [[outer; p + m; inner]] whose last [m] terms of
   each slice are scanned from the carry of its first [p]: [p] whole chunks,
   or [p + m] terms within one. The first [p] hold no NaN, but a NaN made
   from numbers where [made]; the last hold a NaN where [nan_after]. The
   terms are a program's copy of the operand where [through]. *)
type carry_case = {
  kind : carried;
  cm : monoid;
  through : bool;
  outer : int;
  inner : int;
  p : int;
  m : int;
  made : bool;
  nan_after : bool;
  cseed : int;
}

let pp_carry_case ppf c =
  let (Carried (dt, _)) = c.kind in
  Format.fprintf ppf
    "scan %s of %s%a [%d; %d + %d; %d]%s%s, seed %d" (monoid_name c.cm)
    (if c.through then "a copy of " else "")
    D.pp dt c.outer c.p c.m c.inner
    (if c.made then ", a made NaN before" else "")
    (if c.nan_after then ", a NaN after" else "")
    c.cseed

let carry_case =
  let open Gen in
  with_pp pp_carry_case
    (* One case in five carries a NaN made from numbers and holds a NaN
       term after it: few draws of independent choices meet all of them. *)
    (let* both = frequency [ (4, constant false); (1, constant true) ] in
     let* (Carried (dt, vs) as kind) =
       of_list
         (List.filter
            (fun (Carried (_, vs)) -> (not both) || vs.made Sum <> None)
            carried_kinds)
     in
     let* cm =
       of_list
         (List.filter
            (fun m -> accepts m (D.Any dt) && ((not both) || vs.made m <> None))
            [ Sum; Prod; Max; Min ])
     in
     let* outer = of_list [ 1; 2 ] in
     let* inner = of_list [ 1; 3 ] in
     let* aligned = bool in
     (* Such a case needs two terms before the carry and one after it. *)
     let least k l = if both then List.filter (fun n -> n >= k) l else l in
     let* p, m =
       if aligned then
         let* k = of_list [ 1; 2 ] in
         let+ m = of_list (least 1 [ 0; 1; 5; 4095; 4096; 4097; 9000 ]) in
         (4096 * k, m)
       else
         let* p = of_list (least 2 [ 1; 2; 17; 1024; 2049; 4095 ]) in
         let+ m = of_list (least 1 [ 0; 1; 4096 - p; (4096 - p) / 2 ]) in
         (p, m)
     in
     let* made = if both then constant true else bool in
     let* nan_after = if both then constant true else bool in
     let* through = bool in
     let+ cseed = int in
     { kind; cm; through; outer; inner; p; m; made; nan_after; cseed })

(* The case's terms in C order of [[outer; p + m; inner]]. *)
let carry_terms (type v) (vs : v drawn) c : v array =
  let r = Random.State.make [| c.cseed |] in
  let n = c.p + c.m in
  let pick l = List.nth l (Random.State.int r (List.length l)) in
  let term t =
    let special = Random.State.int r 64 = 0 in
    if special && t >= c.p && c.nan_after && vs.nans <> [] then pick vs.nans
    else if special then pick vs.others
    else vs.finite c.cm r
  in
  let a =
    Array.init (c.outer * n * c.inner) (fun i -> term (i / c.inner mod n))
  in
  let set t v =
    for o = 0 to c.outer - 1 do
      for i = 0 to c.inner - 1 do
        a.((((o * n) + t) * c.inner) + i) <- v
      done
    done
  in
  (match vs.made c.cm with
  | Some (u, v) when c.made && c.p > 1 ->
      set (Random.State.int r (c.p - 1)) u;
      set (c.p - 1) v
  | _ -> ());
  (if c.nan_after && c.m > 0 && vs.nans <> [] then
     let t = c.p + Random.State.int r c.m in
     set t (pick vs.nans));
  a

(* Elements [lo, lo + count) along axis 1 of [x]. *)
let along1 (A.Any x) lo count =
  let s = L.shape (A.layout x) in
  let r =
    Array.mapi
      (fun a e ->
        if a = 1 then { M.start = lo; count; step = 1 }
        else { M.start = 0; count = e; step = 1 })
      s
  in
  A.Any (Option.get (A.move (M.Slice r) x))

let law_carry (b : Support.backend) c =
  let (Carried (dt, vs)) = c.kind in
  let (module K : Nx_kernel.S) = b.kernels in
  let n = c.p + c.m in
  let aligned = c.p mod 4096 = 0 in
  cover "a carry into a chunk's start" aligned;
  cover "a carry within a chunk" (not aligned);
  cover "several chunks after a carry" (aligned && c.m > 4096);
  cover "no term after a carry" (c.m = 0);
  cover "several slices" (c.outer * c.inner > 1);
  let floating = D.is D.Float dt || D.is D.Complex dt in
  let made = c.made && c.p > 1 && vs.made c.cm <> None in
  cover "a NaN carry made from numbers" made;
  cover "a NaN carry and a NaN term after it" (made && c.nan_after && c.m > 0);
  cover "a NaN term after a carry" (floating && c.nan_after && c.m > 0);
  cover "a narrow float's carry" (D.bits dt = 16 && D.is D.Float dt);
  cover "a program's terms" c.through;
  let z =
    on b
      (A.Any (A.of_array dt [| c.outer; n; c.inner |] (carry_terms vs c)))
  in
  let (D.Any ct) = carrier (D.Any dt) in
  let ok what = function
    | A.Done -> ()
    | r -> failf "%s: the kernels answered %a" what Nx_array_support.pp_answer r
  in
  let p0 =
    if c.through then
      P.v ~ins:[| D.Any dt |]
        [| P.In 0; P.Op1 (Copy, D.Any dt, 0) |]
        ~outs:[| 1 |]
    else identity (D.Any dt)
  in
  (* The scan of [x] into a fresh destination of [into]. *)
  let scan (type v s) ?carry (into : (v, s) D.t) (A.Any x) =
    let y = A.Any (A.create b.device into (L.shape (A.layout x))) in
    ok "scan"
      (K.scan ?carry p0 ~axis:1 (Monoid c.cm, 0) ~dsts:[| y |] [| Plain x |]);
    y
  in
  let carry =
    if aligned then begin
      (* The carry chain over the chunks' totals, each a reduction in the
         carrier, combined by the monoid's Prog kind from the first. *)
      let total k =
        let (A.Any x) = along1 z (4096 * k) 4096 in
        let y = A.Any (A.create b.device ct [| c.outer; c.inner |]) in
        ok "reduce"
          (K.reduce p0 ~shape:(L.shape (A.layout x))
             ~axes:[| 1 |] [| (Monoid c.cm, 0) |] ~dsts:[| y |] [| Plain x |]);
        y
      in
      let kind : P.binary =
        match c.cm with
        | Sum -> Add
        | Prod -> Mul
        | Max -> Maximum
        | Min | Logsumexp -> Minimum
      in
      let combine =
        P.v
          ~ins:[| D.Any ct; D.Any ct |]
          [| P.In 0; P.In 1; P.Op2 (Binary kind, 0, 1) |]
          ~outs:[| 2 |]
      in
      let rec chain acc k =
        if 4096 * k >= c.p then acc
        else
          let (A.Any a) = acc in
          let (A.Any t) = total k in
          let y = A.Any (A.create b.device ct [| c.outer; c.inner |]) in
          ok "map" (K.map combine ~dsts:[| y |] [| Plain a; Plain t |]);
          chain y (k + 1)
      in
      let (A.Any v) = chain (total 0) 1 in
      A.Any (Option.get (A.move (M.Reshape [| c.outer; 1; c.inner |]) v))
    end
    else along1 (scan ct (along1 z 0 c.p)) (c.p - 1) 1
  in
  let got = scan ~carry dt (along1 z c.p c.m) in
  let want = along1 (scan dt z) c.p c.m in
  equal (array string)
    (Elements.elements (host want))
    (Elements.elements (host got))

(* A carry of a reduction that takes none, or of another shape or dtype than
   the carrier's, is refused. *)
let test_carry_refusals (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let make dt s = A.Any (A.create b.device dt s) in
  let scan (type v s) (dt : (v, s) D.t) ?carry r dsts =
    let (A.Any x) = make dt [| 2; 3 |] in
    K.scan ?carry (identity (D.Any dt)) ~axis:1 (r, 0) ~dsts [| Plain x |]
  in
  let f32 s = make D.Float32 s in
  let refuses msg want got = equal ~msg Elements.answer want got in
  let carry = f32 [| 2; 1 |] in
  refuses "a Logsumexp's carry" A.Bad_arity
    (scan D.Float32 ~carry (Monoid Logsumexp) [| f32 [| 2; 3 |] |]);
  refuses "a Moments' carry" A.Bad_arity
    (scan D.Float32 ~carry Moments [| f32 [| 2; 3 |]; f32 [| 2; 3 |] |]);
  refuses "an Arg's carry" A.Bad_arity
    (scan D.Float32 ~carry (Arg Max)
       [| f32 [| 2; 3 |]; make D.Int64 [| 2; 3 |] |]);
  refuses "a carry of the destination's shape" A.Shape_mismatch
    (scan D.Float32 ~carry:(f32 [| 2; 3 |]) (Monoid Sum) [| f32 [| 2; 3 |] |]);
  refuses "a carry of one element" A.Shape_mismatch
    (scan D.Float32 ~carry:(f32 [| 1; 1 |]) (Monoid Sum) [| f32 [| 2; 3 |] |]);
  refuses "a carry of another dtype" A.Wrong_dtype
    (scan D.Float32 ~carry:(make D.Float64 [| 2; 1 |]) (Monoid Max)
       [| f32 [| 2; 3 |] |]);
  refuses "a float16 carry of float16 terms" A.Wrong_dtype
    (scan D.Float16 ~carry:(make D.Float16 [| 2; 1 |]) (Monoid Sum)
       [| make D.Float16 [| 2; 3 |] |]);
  refuses "a float32 carry of float16 terms" A.Done
    (scan D.Float16 ~carry (Monoid Sum) [| make D.Float16 [| 2; 3 |] |])

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "a float sum is within its bound of the exact sum" sums
        (run (law_bound b));
      prop ~examples:computed_examples "bits do not depend on layouts" computed
        (run (law_layouts b));
      prop ~examples:scan_examples "a scan of a prefix is the prefix of the scan"
        scans (run (law_prefix b));
      prop ~examples:any_examples "a declined case writes nothing" any_case
        (run (law_declined b));
      prop "a scan from a carry continues the scan whose terms it carries"
        carry_case (run (law_carry b));
      test "a carry the scan cannot take is refused" (fun () ->
          b.around (test_carry_refusals b));
    ]

(* A plain sum and scan whose program is Prog.of_node's [In 0], which keeps
   its operand as two nodes, compute as [In 0]'s. *)
let test_of_node (b : Support.backend) () =
  let x = A.Any (A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]) in
  let dt = D.Any D.Float32 in
  let of_node = P.of_node ~ins:[| dt |] (P.In 0) in
  let run scan prog shape =
    let dst = A.Any (A.create b.device D.Float32 shape) in
    let answer = compute b.kernels ~scan prog Sum [| 1 |] dst [| plain_of (on b x) |] in
    equal ~msg:"its answer" bool true (answer = A.Done);
    let (A.Any h) = host dst in
    Array.map Int32.bits_of_float (A.to_array (A.expect D.Float32 (A.Any h)))
  in
  equal (array int32)
    (run false (identity dt) [| 2 |])
    (run false of_node [| 2 |]);
  equal (array int32)
    (run true (identity dt) [| 2; 3 |])
    (run true of_node [| 2; 3 |])

(* Bools stored as bytes other than 0 and 1, through a Uint8 bitcast: a
   bool is true where its byte is not 0 (dtype.mli), so a Max or a Min of
   them, or an Arg's extreme, stores 0 or 1, and nx.cpu computes each. Rows
   of [0; 2], [255; 0], [3; 7] and [0; 0]. *)
let test_bool_bytes (b : Support.backend) () =
  let bytes = [| 0; 2; 255; 0; 3; 7; 0; 0 |] in
  let x =
    A.Any
      (Option.get (A.bitcast D.Bool (A.of_array D.Uint8 [| 4; 2 |] bytes)))
  in
  let p = identity (D.Any D.Bool) in
  let answered what = function
    | A.Done -> ()
    | a -> failf "%s: %s answered %a" what b.name Nx_array_support.pp_answer a
  in
  let bits d =
    let (A.Any h) = host d in
    A.to_array (Option.get (A.bitcast D.Uint8 (A.expect D.Bool (A.Any h))))
  in
  let check what scan m shape want =
    let dst = A.create b.device D.Bool shape in
    let answer =
      compute b.kernels ~scan p m [| 1 |] (A.Any dst) [| plain_of (on b x) |]
    in
    answered what answer;
    equal ~msg:what (array int) want (bits (A.Any dst))
  in
  check "max" false Max [| 4 |] [| 1; 1; 1; 0 |];
  check "min" false Min [| 4 |] [| 0; 0; 1; 0 |];
  check "max scan" true Max [| 4; 2 |] [| 0; 1; 1; 1; 1; 1; 0; 0 |];
  let arg what e want =
    let v = A.create b.device D.Bool [| 4 |]
    and i = A.create b.device D.Int64 [| 4 |] in
    let module K = (val b.kernels) in
    let answer =
      K.reduce p ~shape:[| 4; 2 |] ~axes:[| 1 |] [| (Arg e, 0) |]
        ~dsts:[| A.Any v; A.Any i |]
        [| plain_of (on b x) |]
    in
    answered what answer;
    equal ~msg:what (pair (array int) (array int64)) want
      (bits (A.Any v), A.to_array (A.expect D.Int64 (host (A.Any i))))
  in
  arg "arg max: the first true" Max ([| 1; 1; 1; 0 |], [| 1L; 0L; 0L; 0L |]);
  arg "arg min: the first false" Min ([| 0; 0; 1; 0 |], [| 0L; 1L; 0L; 0L |])

(* Arg, Logsumexp and Moments of narrow dtypes and booleans, and into
   destinations of other dtypes *)

(* The dtype a reference reads terms of [d] through: their carrier, a byte
   integer for booleans, which keeps their order and tells their bits
   apart. *)
let arg_carrier = function
  | D.Any (D.Bool | D.Bit) -> D.Any D.Uint8
  | d -> carrier d

(* A stateful reduction, a Moments scan where [scan], and its destinations'
   dtype, an Arg's positions aside. *)
type stat_into = { st : stat; scan : bool; into : D.any }

let pp_stat_into ppf { st; scan; into } =
  Format.fprintf ppf "%s%a into %a"
    (if scan then "scan " else "")
    pp_stat st pp_dtype into

let narrow_floats =
  D.
    [
      Any Float16; Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2;
      Any Float4_e2m1fn;
    ]

(* An Arg of a narrow dtype or a boolean, or a Logsumexp or Moments of a
   float, reduced or scanned, into its dtype or another float's. Narrow
   floats are float32's values of [value_kinds] rounded into them. *)
let stat_into_case =
  Gen.with_pp pp_stat_into
    (let open Gen in
     let* r =
       of_list Nx_kernel.[ Arg Max; Arg Min; Monoid Logsumexp; Moments ]
     in
     let arg = match r with Nx_kernel.Arg _ -> true | _ -> false in
     let* (D.Any dt as d) =
       of_list ~pp:pp_dtype
         (if arg then D.Any D.Bool :: narrow_dtypes
          else narrow_floats @ narrow_floats @ floats)
     in
     let* scan = bool in
     let* s =
       frequency
         [ (3, shape_gen); (1, map (fun n -> [| n |]) (int_range 1025 5000)) ]
     in
     let s = if scan && Array.length s = 0 then [| 3 |] else s in
     let rank = Array.length s in
     let* axes =
       if scan then map (fun a -> [| a |]) (int_range 0 (rank - 1))
       else
         let+ keep = array ~size:(constant rank) bool in
         Array.of_list
           (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let* v = view_of rank in
     let* values = of_list value_kinds in
     let* specials = frequency [ (1, constant 0); (1, constant 20) ] in
     let* into =
       frequency
         [
           (2, constant d);
           ( 1,
             of_list ~pp:pp_dtype D.[ Any Float32; Any Float64; Any Float16 ] );
         ]
     in
     let+ seed = int in
     let make n =
       if not (is_float d) then A.Any (elements dt n ~specials seed)
       else if D.bits dt >= 32 then
         float_elements dt n values ~outliers:0 ~specials seed
       else
         cast_into d
           (float_elements D.Float32 n values ~outliers:0 ~specials seed)
     in
     { st = stat_of r s axes v values.name make; scan; into })

(* A boolean's Arg, which the generator draws once in 18 cases. *)
let stat_into_examples =
  let v = { perm = [| 0 |]; stepped = false; reversed = None; broadcast = false } in
  let make n = A.Any (elements D.Bool n ~specials:0 1) in
  [
    {
      st = stat_of (Arg Max) [| 9 |] [| 0 |] v "small" make;
      scan = false;
      into = D.Any D.Bool;
    };
  ]

(* [c]'s reduction, or its scan where [scan], of [x] by [k], into fresh
   destinations of [d], an Arg's positions of int64. *)
let stat_run k ~scan (c : stat) (D.Any d) x =
  let module K = (val k : Nx_kernel.S) in
  let s = if scan then shape_of x else stat_shape c in
  let make : type a b. (a, b) D.t -> A.any =
   fun dt -> A.Any (A.create Rig.host dt s)
  in
  let dsts =
    match c.r with
    | Arg _ -> [| make d; make D.Int64 |]
    | Moments -> [| make d; make d |]
    | Monoid _ -> [| make d |]
  in
  let p = identity (dtype_of x) and loads = [| plain_of x |] in
  let answer =
    if scan then K.scan p ~axis:c.axes.(0) (c.r, 0) ~dsts loads
    else K.reduce p ~shape:(shape_of x) ~axes:c.axes [| (c.r, 0) |] ~dsts loads
  in
  match answer with
  | A.Done -> dsts
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* A narrow operand's terms reduce in their carrier, and each result rounds
   once into its destination: the bits of the reduction of the terms cast
   into [arg_carrier], cast into the destination. *)
let law_stat_into (b : Support.backend) { st = c; scan; into } =
  let x = dtype_of c.x in
  let s = shape_of c.x in
  cover "narrow" (staged x);
  cover "bool" (same x (D.Any D.Bool));
  cover "into another dtype" (not (same into x));
  cover "a scan" scan;
  cover "several blocks" (Array.fold_left (fun n a -> n * s.(a)) 1 c.axes > 1024);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  let wide = arg_carrier x in
  let got = stat_run b.kernels ~scan c into c.x in
  let want = stat_run b.kernels ~scan c wide (cast_into wide c.x) in
  (* Into the operand's own dtype, a staged Arg's extreme is the term at
     its position, and a NaN result the first NaN term, a scan's up to it,
     each as the operand holds it, quieted. *)
  let own = staged x && same into x in
  let case = { (as_case c) with scan } in
  (* Each result's terms, a scan's its slice's. *)
  let terms () =
    results case (Elements.elements c.x) ~fill:[||] ~reduce:Fun.id
      ~scan:(fun ts -> Array.map (fun _ -> ts) ts)
  in
  let firsts () = first_nans case in
  Array.iteri
    (fun k w ->
      let want =
        match (c.r, k) with
        | Arg _, 1 -> Elements.elements w
        | Arg _, _ when own ->
            let at = A.to_array (A.expect D.Int64 want.(1)) in
            Array.mapi
              (fun o ts -> Elements.quiet x ts.(Int64.to_int at.(o)))
              (terms ())
        | _ when own ->
            Array.map2
              (fun w f -> Option.value f ~default:w)
              (Elements.elements (cast_into into w))
              (firsts ())
        | _ -> Elements.elements (cast_into into w)
      in
      equal
        ~msg:(Printf.sprintf "destination %d" k)
        (array string) want
        (Elements.elements got.(k)))
    want

let order (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu's cases and order, " ^ b.name)
    [
      prop ~examples:(computed_examples @ made_examples)
        "each result folds its terms in nx.cpu's order" computed
        (run (law_order b));
      prop ~examples:any_examples
        "computes every Sum, Prod, Max and Min"
        any_case (run (law_computes b));
      test "a bool of any non-zero byte is true" (fun () ->
          b.around (test_bool_bytes b));
      test "a program of Prog.of_node's operand computes" (fun () ->
          b.around (test_of_node b));
      test "a NaN past the first block and chunk is the result's" (fun () ->
          b.around (test_far_nans b));
      test "8- and 16-bit products and sums wrap at their extremes" (fun () ->
          b.around (test_wrap b));
      test
        "refuses a destination or a reduction's load of another shape and an \
         extreme of nothing, answers Wrong_dtype for dtypes a reduction does \
         not take"
        (fun () -> b.around (test_refusals b));
      test
        "refuses a padded load past max_numel or one that does not fit its \
         destinations before copying it"
        (fun () -> b.around (test_padded_refusals b));
    ]

(* A reduction and a scan of a program of as many loads as it holds, the sum
   of fifteen loads of [x]: fifteen times [x]'s sums along the last axis, and
   its running sums. *)
let test_most_loads (b : Support.backend) () =
  let module K = (val b.kernels) in
  let nl = P.max_operands - 1 in
  let nodes =
    List.init nl (fun k -> P.In k)
    @ List.init (nl - 1) (fun i ->
          P.Op2 (Binary Add, (if i = 0 then 0 else nl + i - 1), i + 1))
  in
  let p =
    P.v
      ~ins:(Iarray.init nl (fun _ -> D.Any D.Float32))
      (Iarray.of_list nodes)
      ~outs:[| List.length nodes - 1 |]
  in
  let x = A.of_array D.Float32 [| 4; 3 |] (Array.init 12 Float.of_int) in
  let loads = Array.make nl (Nx_kernel.Plain x) in
  let sums = A.create Rig.host D.Float32 [| 4 |] in
  equal ~msg:"reduce" Elements.answer A.Done
    (K.reduce p ~shape:[| 4; 3 |] ~axes:[| 1 |] [| (Monoid Sum, 0) |]
       ~dsts:[| A.Any sums |] loads);
  equal ~msg:"sums" (array float_exact) [| 45.; 180.; 315.; 450. |]
    (A.to_array sums);
  let running = A.create Rig.host D.Float32 [| 4; 3 |] in
  equal ~msg:"scan" Elements.answer A.Done
    (K.scan p ~axis:1 (Monoid Sum, 0) ~dsts:[| A.Any running |] loads);
  equal ~msg:"running sums" (array float_exact)
    (Array.init 12 (fun i ->
         let r = i / 3 and c = i mod 3 in
         Float.of_int (15 * (((c + 1) * 3 * r) + (c * (c + 1) / 2)))))
    (A.to_array running)

(* Nine Moments of one loop take 18 destinations, and twenty sums 20 tables
   of reductions, more than one loop holds beside its load: each result has
   the bits of a call of few reductions. *)
let test_many_reductions (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p = identity (D.Any D.Float32) and shape = [| 4; 3 |] in
  let x =
    A.of_array D.Float32 shape
      [| 1.; 5.; 2.; 0.5; 7.; 3.; 9.; 4.; 4.; 2.; 8.; 6. |]
  in
  let width : Nx_kernel.reduction -> int = function
    | Monoid _ -> 1
    | Arg _ | Moments -> 2
  in
  let reduce r n =
    let dsts =
      Array.init (n * width r) (fun _ ->
          A.Any (A.create Rig.host D.Float32 [| 4 |]))
    in
    equal
      ~msg:(Printf.sprintf "%d reductions" n)
      Elements.answer A.Done
      (K.reduce p ~shape ~axes:[| 1 |] (Array.make n (r, 0)) ~dsts
         [| plain_of (A.Any x) |]);
    Array.map
      (fun (A.Any d) ->
        Array.map Int32.bits_of_float
          (A.to_array (A.expect D.Float32 (A.Any d))))
      dsts
  in
  let agree r few many =
    let want = reduce r few and got = reduce r many in
    Array.iteri
      (fun i g ->
        equal
          ~msg:(Printf.sprintf "destination %d" i)
          (array int32)
          want.(i mod Array.length want)
          g)
      got
  in
  agree Nx_kernel.Moments 1 9;
  agree (Nx_kernel.Monoid Sum) 1 20

(* Nine Moments over no axis, the mean each element and the variance 0, take
   every destination and the load in one claim: a broadcast destination or two
   identical ones are refused before any is written, and a destination
   identical to the load reads it as it was. *)
let test_many_claims (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p = identity (D.Any D.Float32) in
  let x = A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let fresh () = A.of_array D.Float32 [| 4 |] [| 7.; 7.; 7.; 7. |] in
  let dsts () = Array.init 18 (fun _ -> A.Any (fresh ())) in
  let call dsts =
    K.reduce p ~shape:[| 4 |] ~axes:[||]
      (Array.make 9 (Nx_kernel.Moments, 0))
      ~dsts [| plain_of (A.Any x) |]
  in
  let values (A.Any d) = A.to_array (A.expect D.Float32 (A.Any d)) in
  let untouched msg ds =
    equal ~msg (array float_exact) [| 7.; 7.; 7.; 7. |] (values ds.(0))
  in
  let ds = dsts () in
  let one = A.of_array D.Float32 [| 1 |] [| 7. |] in
  ds.(17) <- A.Any (Option.get (A.move (Nx_array.Move.Broadcast [| 4 |]) one));
  equal ~msg:"a broadcast last destination" Elements.answer A.Repeated_elements
    (call ds);
  untouched "first destination after the broadcast" ds;
  let ds = dsts () in
  ds.(17) <- ds.(0);
  equal ~msg:"the first destination as the last" Elements.answer A.Overlapping
    (call ds);
  untouched "first destination after the overlap" ds;
  let ds = dsts () in
  let x' = A.copy x in
  ds.(0) <- A.Any x';
  equal ~msg:"the first destination as the load" Elements.answer A.Done
    (K.reduce p ~shape:[| 4 |] ~axes:[||]
       (Array.make 9 (Nx_kernel.Moments, 0))
       ~dsts:ds [| plain_of (A.Any x') |]);
  Array.iteri
    (fun i d ->
      equal ~msg:(Printf.sprintf "destination %d" i) (array float_exact)
        (if i mod 2 = 0 then [| 1.; 2.; 3.; 4. |] else [| 0.; 0.; 0.; 0. |])
        (values d))
    ds

(* A reduction or scan given destinations other than its reduction's is
   refused, never declined: nx.cpu computes every case its contract offers. *)
let test_malformed (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let f32 s = A.Any (A.create Rig.host D.Float32 s) in
  let p = identity (D.Any D.Float32) and shape = [| 2; 3 |] in
  let reduce r dsts =
    K.reduce p ~shape ~axes:[| 1 |] [| (r, 0) |] ~dsts [| plain_of (A.Any x) |]
  in
  equal ~msg:"a sum into two destinations" Elements.answer A.Bad_arity
    (reduce (Monoid Sum) [| f32 [| 2 |]; f32 [| 2 |] |]);
  equal ~msg:"Moments into one destination" Elements.answer A.Bad_arity
    (reduce Moments [| f32 [| 2 |] |]);
  equal ~msg:"Arg's positions in int32" Elements.answer A.Wrong_dtype
    (reduce (Arg Max) [| f32 [| 2 |]; A.Any (A.create Rig.host D.Int32 [| 2 |]) |]);
  equal ~msg:"a scan into no destination" Elements.answer A.Bad_arity
    (K.scan p ~axis:1 (Monoid Sum, 0) ~dsts:[||] [| plain_of (A.Any x) |])

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      test "reduces many reductions in one claim of their operands" (fun () ->
          b.around (test_many_claims b));
      test "reduces more reductions than one loop holds" (fun () ->
          b.around (test_many_reductions b));
      test "reduces and scans a program of as many loads as it holds"
        (fun () -> b.around (test_most_loads b));
      test "refuses destinations other than a reduction's" (fun () ->
          b.around (test_malformed b));
      prop ~examples:dot_examples "a sum of one axis is a contraction with ones"
        dots (run (law_contract b));
      prop ~examples:into_examples
        "narrow terms reduce in their carrier, rounded once" into_case
        (run (law_into b));
      test "a narrow NaN result is its first NaN term quieted" (fun () ->
          b.around (test_nan_bits b));
      prop ~examples:stat_into_examples
        "narrow Arg, Logsumexp and Moments reduce in their carrier"
        stat_into_case (run (law_stat_into b));
      prop ~examples:complex_examples
        "complex sums, products and extremes fold in nx.cpu's order"
        complex_case (run (law_complex b));
      prop ~count:20 ~examples:large_examples "one thread gives the job's bits"
        large (run (law_threads b));
      prop ~examples:padded_examples "a padded load reduces as the array it reads"
        padded_case (run (law_padded b));
      prop ~examples:fused_examples
        "a reduction or scan of a program's output gives the bits of its \
         map's"
        fused_case (run (law_fused b));
      test "a destination aliasing a load computes as a fresh one" (fun () ->
          b.around (test_program_in_place b));
      prop
        ~examples:
          [
            { first = fused_example 4 Moments [| 5; 40 |] [| 1 |];
              red2 = Monoid Max; k2 = 0 };
          ]
        "several reductions in one call give each one's bits" several_case
        (run (law_several b));
      test "several reductions refuse a misfit, writing nothing" (fun () ->
          b.around (test_several_refuse b));
      prop ~examples:moments_examples
        "a Moments scan is within its bound, NaNs and infinities as stated"
        moments_case (run (law_moments b));
      prop ~examples:stat_examples
        "Arg is exact, Logsumexp and Moments within their bounds, each the \
         same bits on one thread and from a copy"
        stat_case (run (law_stateful b));
      prop ~examples:spread_examples
        "Logsumexp and Moments give the bits of Nx's expansions" spread_case
        (run (law_record b));
      test "a reduction or scan into its own operand gives fresh bits"
        (fun () -> b.around (test_in_place b));

      prop ~examples:stat_scan_examples
        "Arg scans are exact, Logsumexp scans their recurrence's bits within \
         their bound, each the same on one thread and from a copy"
        stat_scan_case (run (law_stat_scan b));
      prop ~examples:lanes_examples
        "a scan by Arg, Logsumexp or Moments gives each slice its bits alone"
        lanes_case (run (law_lanes b));
      test "a complex Arg scan holds its first extreme and first NaN"
        (fun () -> b.around (test_complex_arg_scan b));
      test "refuses an Arg of nothing and a Moments scan's misshapen \
            destinations"
        (fun () -> b.around (test_stateful_cases b));
      prop ~count:500 ~examples:guard_examples
        "reads and writes no byte past the operand and destinations, \
         whatever pages lie beside them"
        (Gen.pair guard_case Elements.sides)
        (run (law_guards b));
      prop ~count:200
        "reads no byte past a padded load's operand, whatever pages lie \
         beside it"
        (Gen.pair padded_case Elements.sides)
        (run (law_padded_guards b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.reduce"
       (List.map laws Support.backends
       @ List.map order Support.cpus
       @ List.map cpu Support.cpus))
