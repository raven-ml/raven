(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the suites of Nx_kernel.S's reductions and scans and of nx.cpu's
   share: the cases, their values and views, their terms in C order and the
   results expected of them. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

type monoid = Nx_kernel.monoid = Sum | Prod | Max | Min

let pp_dtype ppf (D.Any dt) = D.pp ppf dt

let total s = Array.fold_left ( * ) 1 s

let shape_of (A.Any x) = Iarray.to_array (L.shape (A.layout x))

let monoid_name = function
  | Sum -> "Sum"
  | Prod -> "Prod"
  | Max -> "Max"
  | Min -> "Min"

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

(* The shape [l] has once loaded: its array's, padded as its padding says. *)
let load_shape : Nx_kernel.load -> _ = function
  | Plain x -> Ok (Iarray.to_array (L.shape (A.layout x)))
  | Padded { x; pad; _ } ->
      Nx_kernel.Shape.padded pad (Iarray.to_array (L.shape (A.layout x)))

(* The loop's shape: the loads' once loaded, else, for loads of several
   shapes or a padding that fits none, the first load's operand's, which
   the kernels refuse. *)
let loop_shape (loads : Nx_kernel.load array) =
  let shapes = Array.map load_shape loads in
  match shapes.(0) with
  | Ok s when Array.for_all (( = ) (Ok s)) shapes -> s
  | _ -> (
      match loads.(0) with
      | Plain x -> Iarray.to_array (L.shape (A.layout x))
      | Padded { x; _ } -> Iarray.to_array (L.shape (A.layout x)))

let dtype_of (A.Any x) = D.Any (A.dtype x)

let same (D.Any a) (D.Any b) = D.equal a b

let accepts m (D.Any dt) = Nx_kernel.accepts (Monoid m) dt

(* By [k], the reduction by [m] of [p]'s output along [axes], or with [scan]
   its scan along [axes.(0)], of [loads] into [dst]. *)
let compute (module K : Nx_kernel.S) ~scan p m axes dst loads =
  if scan then K.scan p ~axis:axes.(0) (Monoid m, 0) ~dsts:[| dst |] loads
  else
    K.reduce p ~shape:(loop_shape loads) ~axes:(Iarray.of_array axes)
      [| (Monoid m, 0) |] ~dsts:[| dst |] loads

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
  let ys = Iarray.of_array (result_shape c) in
  let dst = A.Any (A.create b.device (A.dtype x) ys) in
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
  let x = move (M.Reshape (Iarray.of_array made')) x in
  let x =
    if not v.stepped then x
    else
      move
        (M.Slice
           (Iarray.of_array
              (Array.mapi
              (fun i e ->
                if i = last then { M.start = 1; count = made.(i); step = 2 }
                else whole e)
              made')))
        x
  in
  let x =
    match v.reversed with
    | Some a when made.(a) > 0 ->
        move
          (M.Slice
             (Iarray.of_array
                (Array.mapi
                (fun i e ->
                  if i = a then { M.start = e - 1; count = e; step = -1 }
                  else whole e)
                made)))
          x
    | _ -> x
  in
  let x = move (M.Permute (Iarray.of_array v.perm)) x in
  A.Any (if v.broadcast then move (M.Broadcast (Iarray.of_array s)) x else x)

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

let any_case = Gen.with_pp pp_case (case_of D.all)

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

let any_examples =
  [
    example Sum f32 [| 4 |] [| 0 |];
    example Sum (D.Any D.Float16) [| 4 |] [| 0 |];
    example ~scan:true Sum (D.Any D.Float16) [| 4 |] [| 0 |];
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
      (match Support.map1 (module Nx_cpu) Cast ~dst:d c with
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
   exactly or rounds them innocuously; a NaN operand gives a NaN. An
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

let is_nan = function F (x, _) -> Float.is_nan x | I _ -> false

let bits_of = function F (_, b) -> b | I v -> v

(* [t]'s bits, or some NaN where it is one: a NaN result promises a NaN,
   never which one. *)
let want_of t = if is_nan t then Some_nan else Bits (bits_of t)

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

(* The kernels' result's bits in C order. *)
let got y =
  Array.map
    (function F (x, b) -> if Float.is_nan x then Some_nan else Bits b | I v -> Bits v)
    (terms y)

let show = function Bits b -> Printf.sprintf "%Lx" b | Some_nan -> "nan"

(* The first results that differ from [want]: a [Some_nan] wanted matches
   any NaN. *)
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

(* Stateful reductions *)

(* A result: some bits, or a value within a tolerance. *)
type expect = Is of want | Near of float * float

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
  | Logsumexp -> "Logsumexp"
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

(* [c] as a reduction case of Sum, whose result's shape and outputs it
   shares. *)
let as_case c = { scan = false; monoid = Sum; axes = c.axes; x = c.x; views = [] }

let stat_shape c = result_shape (as_case c)

(* A float's value. *)
let value = function F (x, _) -> x | I _ -> invalid_arg "value: an integer"

(* The extreme by [e] of the terms [ts] at [d], then its first position. *)
let arg_expect d (e : Nx_kernel.extreme) ts =
  let f, _ = op d (match e with Max -> Max | Min -> Min) in
  let best = Array.fold_left f ts.(0) ts in
  let same t = if is_nan best then is_nan t else bits_of t = bits_of best in
  let i = Option.get (Array.find_index same ts) in
  [| Is (want_of ts.(i)); Is (Bits (Int64.of_int i)) |]

(* Arg and Logsumexp scans *)

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
      [| Is (want_of !best); Is (Bits (Int64.of_int !at)) |])
    ts

