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
let dtype_of (A.Any x) = D.Any (A.dtype x)
let accepts m (D.Any dt) = Nx_kernel.accepts (Monoid m) dt

(* By [k], the reduction by [m] of [p]'s output along [axes], or with [scan]
   its scan along [axes.(0)], of [loads] into [dst]. *)
let compute (module K : Nx_kernel.S) ~scan p m axes dst loads =
  if scan then K.scan p ~axis:axes.(0) (Monoid m, 0) ~dsts:[| dst |] loads
  else K.reduce p ~axes [| (Monoid m, 0) |] ~dsts:[| dst |] loads

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
   floats of full precision in (-4, 4), with [-0.], infinities and NaNs of
   drawn payloads at the rate [specials] (one in that many, none for 0);
   integers small, or at the edges of the widths at that rate. *)
let elements (type v s) (dt : (v, s) D.t) n ~specials seed =
  let r = Random.State.make [| seed |] in
  let special () = specials > 0 && Random.State.int r specials = 0 in
  let pick a = a.(Random.State.int r (Array.length a)) in
  let b = Bytes.create (max 1 (D.bytes dt n)) in
  for i = 0 to n - 1 do
    match dt with
    | D.Float32 ->
        let bits =
          if special () then
            pick
              (Array.append nans32
                 [| 0x80000000l; 0x7F800000l; 0xFF800000l |])
          else Int32.bits_of_float (full r 23)
        in
        Bytes.set_int32_le b (4 * i) bits
    | D.Float64 ->
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
        Bytes.set_int64_le b (8 * i) bits
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
       base)

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
   the identity, term t into lane t mod 16, the lanes' tree, then the
   blocks' tree whose left part holds the largest power of two of blocks
   below their count. An extreme's lanes start from its first term, which
   is no order's choice: the extremes are exact. *)
let ordered (f, id) ts =
  let k = Array.length ts in
  let block x =
    let l = Array.make 16 id in
    for t = x * 1024 to min k ((x + 1) * 1024) - 1 do
      l.(t mod 16) <- f l.(t mod 16) ts.(t)
    done;
    List.iter
      (fun w ->
        for i = 0 to w - 1 do
          l.(i) <- f l.(i) l.(i + w)
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

(* A float result's NaN holds its first NaN term, else it is [made]'s. *)
let settle made ts r =
  if not (is_nan r) then Bits (bits_of r)
  else
    match Array.find_opt is_nan ts with
    | Some t -> Bits (bits_of t)
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
   result is that term; a NaN result before it is [made]'s. *)
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
      | Some k when i >= k -> Bits (bits_of ts.(k))
      | _ -> if is_nan v then Bits made else Bits (bits_of v))
    out

(* The position of each index of shape [s] in C order. *)
let position s idx =
  let p = ref 0 in
  Array.iteri (fun i j -> p := (!p * s.(i)) + j) idx;
  !p

(* The expected results of [c] in C order of the result's indices. *)
let expected c =
  let (A.Any x) = c.x in
  let m = op (D.Any (A.dtype x)) c.monoid in
  let made = made (D.Any (A.dtype x)) in
  let ts = terms c.x in
  if not c.scan then
    Array.of_list
      (List.map (fun o -> settle made o (ordered m o)) (outputs c ts))
  else
    let s = shape_of c.x and a = c.axes.(0) in
    let r = Array.length s in
    let ks = Array.of_list (List.filter (( <> ) a) (List.init r Fun.id)) in
    let y = Array.make (total s) Some_nan in
    List.iter
      (fun o ->
        let idx = Array.make r 0 in
        Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
        let at j =
          idx.(a) <- j;
          position s idx
        in
        let slice = Array.init s.(a) (fun j -> ts.(at j)) in
        Array.iteri (fun j w -> y.(at j) <- w) (scanned made m slice))
      (indices (Array.map (fun i -> s.(i)) ks));
    y

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

(* Whether nx_cpu.mli says nx.cpu computes [c]: the base dtypes, booleans
   for the extremes alone. *)
let cpu_computes c =
  let (A.Any x) = c.x in
  List.mem (D.Any (A.dtype x)) base

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

let law_order (b : Support.backend) c =
  covers c;
  cover "a NaN made from numbers" (makes_nan c);
  match run b c with
  | None when cpu_computes c -> failf "%s declined a case it states it computes" b.name
  | None -> cover "declined" true
  | Some y -> equal (list string) [] (differ (expected c) y)

let law_computes b c =
  let computes = cpu_computes c in
  cover "computed" computes;
  cover "declined" (not computes);
  equal ~msg:"computed" bool computes (Option.is_some (run b c))

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
        K.contract ~batch:[||] ~contracting:[| (0, 0) |] ~acc:(dtype_of c.x) ~dst
          ~init:None x ones
      in
      match (run b c, summed) with
      | Some y, A.Done ->
          cover "computed" true;
          equal (list string) [] (differ (got (A.Any dst)) y)
      | _ -> cover "declined" true)
  | _ -> cover "not a sum of one float axis" true

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
  let one = into Support.serial and many = into b.kernels in
  cover "scan" c.scan;
  cover "few outputs" (total (result_shape c) < 64);
  let exact y = Array.map (fun t -> Bits (bits_of t)) (terms y) in
  equal (list string) [] (differ (exact many) one)

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
  let ints = on (A.of_array D.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]) in
  let mean = on (A.create Rig.host D.Int32 [| 2; 3 |]) in
  let var = on (A.create Rig.host D.Int32 [| 2; 3 |]) in
  equal ~msg:"a Moments scan of int32" answer A.Wrong_dtype
    (K.scan (identity (D.Any D.Int32)) ~axis:1 (Moments, 0)
       ~dsts:[| mean; var |] [| plain_of ints |]);
  let bools = on (A.create Rig.host D.Bool [| 2; 3 |]) in
  equal ~msg:"a Sum of bool" answer A.Wrong_dtype
    (K.reduce (identity (D.Any D.Bool)) ~axes:[| 1 |] [| (Monoid Sum, 0) |]
       ~dsts:[| on (A.create Rig.host D.Bool [| 2 |]) |]
       [| plain_of bools |])

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
      let p = identity (dtype_of p.x) in
      (match compute b.kernels ~scan:c.scan p c.monoid c.axes dst loads with
      | A.Done -> equal (list string) [] (differ (expected c) dst)
      | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r)
  | Refused r ->
      let same = Array.length pad.lo = Array.length s in
      let e = if same then extents s pad else s in
      cover "a padding of another rank" (not same);
      cover "an axis cropped below zero" (Array.exists (fun d -> d < 0) e);
      cover "a window wider than its axis"
        (same
        && Array.for_all (fun d -> d >= 0) e
        && Array.exists
             (fun (w : M.window) -> e.(w.axis) < (w.dilation * (w.size - 1)) + 1)
             pad.windows);
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

(* A fifth of the operands are one axis of more than 4096 terms, the law's
   "long" case, which [shape_gen] alone draws too rarely for every run to
   cover. *)
let moments_case =
  Gen.with_pp pp_moments
    (let open Gen in
     let* d = of_list ~pp:pp_dtype floats in
     let long = map (fun n -> [| n |]) (int_range 4097 6000) in
     let* s = frequency [ (4, shape_gen); (1, long) ] in
     let s = if Array.length s = 0 then [| 3 |] else s in
     let* axis = int_range 0 (Array.length s - 1) in
     let* v = view_of (Array.length s) in
     let* specials =
       frequency [ (3, constant 0); (1, constant 300); (1, constant 4) ]
     in
     let+ seed = int in
     { axis; x = operand d s v ~specials seed; views = view_names v })

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
   NaN term, its bits; past an infinity, the infinities' sum and the NaN
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
          if !nan = None then nan := Some (bits_of ts.(at))
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

(* A Moments scan is within nx.cpu's bound of the running mean and variance,
   its NaNs and infinities as stated, and the same bits on one thread and
   from a C-contiguous copy of its operand. *)
let law_moments (b : Support.backend) (c : moments) =
  let s = shape_of c.x in
  let len = s.(c.axis) in
  cover "no result" (total s = 0);
  cover "one term" (len = 1);
  cover "slices past a unit's" (len > 0 && total s / len > 64);
  cover "long" (len > 4096);
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
        [ ("one thread", Support.serial, c); ("a copy", b.kernels, copy) ]

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
    K.reduce (identity (dtype_of c.x)) ~axes:c.axes [| (c.r, 0) |] ~dsts
      [| plain_of c.x |]
  with
  | A.Done -> Some (Array.map terms dsts)
  | A.Declined -> None
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* The quiet NaN of a clear sign bit, made from numbers. *)
let made_nan u = Is (Bits (if u > 0x1p-30 then 0x7FC00000L else 0x7FF8000000000000L))

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
  [| Is (Bits (bits_of ts.(i))); Is (Bits (Int64.of_int i)) |]

(* Logsumexp of [ts] within nx_cpu.mli's bound at the unit roundoff [u],
   from a float64 reference within ε (ln n + 4 + |r|). *)
let lse_expect u ts =
  match Array.find_opt is_nan ts with
  | Some t -> [| Is (Bits (bits_of t)) |]
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
  | Some t -> [| Is (Bits (bits_of t)); Is (Bits (bits_of t)) |]
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
          let tv = ((l +. d +. 4.) *. u *. v) +. (tm *. tm) in
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
        [ ("one thread", Support.serial, c); ("a copy", b.kernels, { c with x = A.Any (A.copy x) }) ]

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
   [0]. Moments: the sum over the count, then the sum of squared deviations
   from it over the count. Each operation rounds to the dtype. *)
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
      let square x = round (round (x -. mean) *. round (x -. mean)) in
      let var = round (sum (Array.map square xs) /. n) in
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

(* nx.cpu declines Arg at bool and the reductions at the narrow floats, and
   refuses an Arg of no term. *)
let test_stateful_cases (b : Support.backend) () =
  let module K = (val b.kernels) in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let reduce r x dsts =
    K.reduce (identity (dtype_of x)) ~axes:[| 1 |] [| (r, 0) |] ~dsts [| plain_of x |]
  in
  let make dt s = A.Any (A.create Rig.host dt s) in
  let pos = make D.Int64 [| 2 |] in
  equal ~msg:"an Arg of bool" answer A.Declined
    (reduce (Arg Max) (make D.Bool [| 2; 3 |]) [| make D.Bool [| 2 |]; pos |]);
  equal ~msg:"Moments of float16" answer A.Declined
    (reduce Moments (make D.Float16 [| 2; 3 |])
       [| make D.Float16 [| 2 |]; make D.Float16 [| 2 |] |]);
  equal ~msg:"an Arg of no term" answer A.Shape_mismatch
    (reduce (Arg Min) (make D.Float32 [| 2; 0 |]) [| make D.Float32 [| 2 |]; pos |])

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
   them stores 0 or 1, as nx.cpu's. Rows of [0; 2], [255; 0], [3; 7] and
   [0; 0]. *)
let test_bool_bytes (b : Support.backend) () =
  let bytes = [| 0; 2; 255; 0; 3; 7; 0; 0 |] in
  let x =
    A.Any
      (Option.get (A.bitcast D.Bool (A.of_array D.Uint8 [| 4; 2 |] bytes)))
  in
  let p = identity (D.Any D.Bool) in
  let run scan m shape =
    let dst = A.create b.device D.Bool shape in
    let answer =
      compute b.kernels ~scan p m [| 1 |] (A.Any dst) [| plain_of (on b x) |]
    in
    equal ~msg:"its answer" bool true (answer = A.Done);
    let (A.Any h) = host (A.Any dst) in
    A.to_array (Option.get (A.bitcast D.Uint8 (A.expect D.Bool (A.Any h))))
  in
  equal ~msg:"max" (array int) [| 1; 1; 1; 0 |] (run false Max [| 4 |]);
  equal ~msg:"min" (array int) [| 0; 0; 1; 0 |] (run false Min [| 4 |]);
  equal ~msg:"max scan" (array int)
    [| 0; 1; 1; 1; 1; 1; 0; 0 |]
    (run true Max [| 4; 2 |])

let order (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu's cases and order, " ^ b.name)
    [
      prop ~examples:(computed_examples @ made_examples)
        "each result folds its terms in nx.cpu's order" computed
        (run (law_order b));
      prop ~examples:any_examples
        "computes the cases nx_cpu.mli lists, declines others" any_case
        (run (law_computes b));
      test "a bool of any non-zero byte is true" (fun () ->
          b.around (test_bool_bytes b));
      test "a program of Prog.of_node's operand computes" (fun () ->
          b.around (test_of_node b));
      test "a NaN past the first block and chunk is the result's" (fun () ->
          b.around (test_far_nans b));
      test "8- and 16-bit products and sums wrap at their extremes" (fun () ->
          b.around (test_wrap b));
      test
        "refuses a destination of another shape and an extreme of nothing, \
         answers Wrong_dtype for dtypes a reduction does not take"
        (fun () -> b.around (test_refusals b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~examples:dot_examples "a sum of one axis is a contraction with ones"
        dots (run (law_contract b));
      prop ~count:20 ~examples:large_examples "one thread gives the job's bits"
        large (run (law_threads b));
      prop ~examples:padded_examples "a padded load reduces as the array it reads"
        padded_case (run (law_padded b));
      prop "a Moments scan is within its bound, NaNs and infinities as stated"
        moments_case (run (law_moments b));
      prop ~examples:stat_examples
        "Arg is exact, Logsumexp and Moments within their bounds, each the \
         same bits on one thread and from a copy"
        stat_case (run (law_stateful b));
      prop "Logsumexp and Moments give the bits of Nx's expansions"
        spread_case (run (law_record b));
      test "declines Arg at bool and narrow floats, refuses an Arg of nothing"
        (fun () -> b.around (test_stateful_cases b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.reduce"
       (List.map laws Support.backends
       @ List.map order Support.backends
       @ List.map cpu Support.cpus))
