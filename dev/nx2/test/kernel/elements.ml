(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Elements as their bits, for the suites whose references move elements:
   gathers, scatters and sorts. Arrays are drawn through views, and their
   elements read back in C order of indices as strings of their bytes. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move

let pp_ints = Nx_array_gen.pp_ints
let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let total s = Array.fold_left ( * ) 1 s
let shape_of (A.Any x) = L.shape (A.layout x)
let dtype_of (A.Any x) = D.Any (A.dtype x)
let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = )

(* Elements as bits *)

(* [x]'s elements in C order of indices, each as its bytes in the host's order,
   a sub-byte element as its code in one byte. *)
let elements (A.Any x) =
  let bytes (type v s) (c : (v, s) A.t) =
    match A.dtype c with
    | D.Bit -> Array.map (fun b -> if b then "\001" else "\000") (A.to_array c)
    | dt when D.bits dt = 4 ->
        Array.map
          (fun v -> String.make 1 (Char.chr v))
          (A.to_array (Option.get (A.bitcast D.Uint4 c)))
    | dt ->
        let w = D.bits dt / 8 in
        let b = A.to_array (Option.get (A.bitcast D.Uint8 c)) in
        Array.init
          (Array.length b / w)
          (fun i -> String.init w (fun j -> Char.chr b.((w * i) + j)))
  in
  bytes (A.copy x)

(* A fresh C-contiguous array of [dt] and shape [s] holding [es]. *)
let of_elements (D.Any dt) s es =
  let bits = D.bits dt in
  let b = Bytes.make (max 1 (D.bytes dt (Array.length es))) '\000' in
  Array.iteri
    (fun i e ->
      if bits >= 8 then
        Bytes.blit_string e 0 b (i * String.length e) (String.length e)
      else
        let at = i * bits in
        let code = Char.code e.[0] land ((1 lsl bits) - 1) in
        let k = at / 8 in
        Bytes.set b k
          (Char.chr (Char.code (Bytes.get b k) lor (code lsl (at mod 8)))))
    es;
  A.Any (A.v dt (L.contiguous s) (Rig.Buffer.of_string (Bytes.to_string b)))

(* The element of zero bits. *)
let zero (D.Any dt) = String.make (max 1 (D.bits dt / 8)) '\000'

(* Values *)

let f32 x =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.bits_of_float x);
  Bytes.to_string b

let f64 x =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.bits_of_float x);
  Bytes.to_string b

let nans32 = [| 0x7FC00000l; 0xFFC12345l; 0x7F800001l; 0xFFA00002l |]
let nans64 = [| 0x7FF8000000000000L; 0x7FF0000000000001L; 0xFFF4000000000002L |]
let specials = [| 0.; -0.; 1.; -1.; 2.; 0.5; infinity; neg_infinity |]

(* A random element of [dt]: small values often, so that targets tie and sums
   cancel, then specials, NaNs of drawn payloads, and any bits. *)
let element (D.Any dt) r =
  let pick a = a.(Random.State.int r (Array.length a)) in
  let any n = String.init n (fun _ -> Char.chr (Random.State.int r 256)) in
  let float32 () =
    match Random.State.int r 10 with
    | 0 | 1 | 2 | 3 -> f32 (Float.of_int (Random.State.int r 7 - 3))
    | 4 | 5 -> f32 (pick specials)
    | 6 ->
        let b = Bytes.create 4 in
        Bytes.set_int32_le b 0 (pick nans32);
        Bytes.to_string b
    | _ -> any 4
  in
  let float64 () =
    match Random.State.int r 10 with
    | 0 | 1 | 2 | 3 -> f64 (Float.of_int (Random.State.int r 7 - 3))
    | 4 | 5 -> f64 (pick specials)
    | 6 ->
        let b = Bytes.create 8 in
        Bytes.set_int64_le b 0 (pick nans64);
        Bytes.to_string b
    | _ -> any 8
  in
  match dt with
  | D.Float32 -> float32 ()
  | D.Float64 -> float64 ()
  | D.Complex64 -> float32 () ^ float32 ()
  | D.Complex128 -> float64 () ^ float64 ()
  | D.Bool | D.Bit -> String.make 1 (Char.chr (Random.State.int r 2))
  | _ when D.bits dt = 4 -> String.make 1 (Char.chr (Random.State.int r 16))
  | _ ->
      let w = D.bits dt / 8 in
      if Random.State.bool r then any w
      else
        let v = Random.State.int r 7 - 3 in
        String.init w (fun j -> Char.chr ((v asr (8 * j)) land 0xFF))

(* Views *)

(* How an array of shape [s] is made and viewed: its axes made in another order,
   stepped by two along its last made axis, reversed along a made axis, and
   broadcast along the axes [broadcast] from one element. *)
type view = {
  perm : int array;
  stepped : bool;
  reversed : int option;
  broadcast : int list;
}

let view_names v =
  (if v.perm <> Array.init (Array.length v.perm) Fun.id then [ "permuted" ]
   else [])
  @ (if v.stepped then [ "stepped" ] else [])
  @ (if v.reversed <> None then [ "reversed" ] else [])
  @ if v.broadcast <> [] then [ "broadcast" ] else []

let view_of ?(broadcast = []) r =
  let open Gen in
  let* p = permutation ~pp:Format.pp_print_int (List.init r Fun.id) in
  let* stepped = bool in
  let+ reversed = option (int_range 0 (max 0 (r - 1))) in
  {
    perm = Array.of_list p;
    stepped = stepped && r > 0;
    reversed = (if r > 0 then reversed else None);
    broadcast;
  }

let plain r =
  {
    perm = Array.init r Fun.id;
    stepped = false;
    reversed = None;
    broadcast = [];
  }

(* An array of shape [s] through the view [v] whose elements in C order are [f
   k], k counted in C order of the array made before broadcasting. *)
let operand dt s v f =
  let r = Array.length s in
  let s0 = Array.mapi (fun i e -> if List.mem i v.broadcast then 1 else e) s in
  let inv = Array.make r 0 in
  Array.iteri (fun i m -> inv.(m) <- i) v.perm;
  let made = Array.init r (fun m -> s0.(inv.(m))) in
  let last = r - 1 in
  let made' =
    Array.mapi (fun i e -> if v.stepped && i = last then 2 * e else e) made
  in
  let whole e = { M.start = 0; count = e; step = 1 } in
  let move m (A.Any x) = A.Any (Option.get (A.move m x)) in
  let x = of_elements dt made' (Array.init (total made') f) in
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
  if v.broadcast = [] then x else move (M.Broadcast s) x

(* Positions: in range mostly, and at -1, the extent, past it, and the extremes
   of int64. *)
let position r d =
  match Random.State.int r 12 with
  | 0 -> -1L
  | 1 -> Int64.of_int d
  | 2 -> Int64.of_int (d + 3)
  | 3 -> Int64.min_int
  | 4 -> Int64.max_int
  | _ -> if d = 0 then 0L else Int64.of_int (Random.State.int r d)

(* Positions of shape [s] through the view [v], each drawn by [f]. *)
let int64s s v f =
  operand (D.Any D.Int64) s v (fun _ ->
      let b = Bytes.create 8 in
      Bytes.set_int64_le b 0 (f ());
      Bytes.to_string b)

let positions (A.Any p) = A.to_array (A.expect D.Int64 (A.Any p))

(* Indices in C order *)

let index_of s k =
  let r = Array.length s in
  let i = Array.make r 0 in
  let k = ref k in
  for a = r - 1 downto 0 do
    i.(a) <- !k mod s.(a);
    k := !k / s.(a)
  done;
  i

let flat s i =
  let k = ref 0 in
  Array.iteri (fun a x -> k := (!k * s.(a)) + x) i;
  !k

(* Arithmetic on bits *)

let get32 e at = Int32.float_of_bits (String.get_int32_le e at)
let get64 e at = Int64.float_of_bits (String.get_int64_le e at)
let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Prog.Add on two floats of 32 or 64 bits, from their bits: the first NaN
   operand's bits, else the rounded sum, +0 for a zero. *)
let add_bits ~w a b =
  let get, put, round =
    if w = 4 then (get32, f32, round32) else (get64, f64, Fun.id)
  in
  let x = get a 0 and y = get b 0 in
  if Float.is_nan x then a
  else if Float.is_nan y then b
  else
    let s = round (x +. y) in
    put (if s = 0. then 0. else s)

(* The order of two floats that are not NaN, -0 below +0. *)
let compare_floats x y =
  match compare x y with
  | 0 -> compare (Float.sign_bit y) (Float.sign_bit x)
  | c -> c

(* An integer element's value, sign- or zero-extended. *)
let int_value (D.Any dt) e =
  let w = String.length e in
  let v = ref 0L in
  for j = w - 1 downto 0 do
    v := Int64.logor (Int64.shift_left !v 8) (Int64.of_int (Char.code e.[j]))
  done;
  let bits = if D.bits dt < 8 then D.bits dt else 8 * w in
  if bits >= 64 then !v
  else
    let s = 64 - bits in
    if D.is D.Signed dt then Int64.shift_right (Int64.shift_left !v s) s
    else Int64.shift_right_logical (Int64.shift_left !v s) s

let int_bits (D.Any dt) w v =
  if D.bits dt < 8 then
    String.make 1 (Char.chr (Int64.to_int v land ((1 lsl D.bits dt) - 1)))
  else
    String.init w (fun j ->
        Char.chr (Int64.to_int (Int64.shift_right_logical v (8 * j)) land 0xFF))

let is_narrow (D.Any dt) = D.is D.Float dt && D.bits dt < 32

(* A narrow float's code as a float32 value. *)
let decode (D.Any dt) e =
  Int32.float_of_bits
    (Int32.of_int
       (Nx_array_support.decode (D.code dt)
          (Char.code e.[0]
          lor if String.length e > 1 then Char.code e.[1] lsl 8 else 0)))

(* The code of the narrow float nearest [x], by nx.cpu's cast. *)
let encode (D.Any dt) x =
  let src = A.of_array D.Float32 [| 1 |] [| x |] in
  let dst = A.create Rig.host dt [| 1 |] in
  (match Nx_cpu.apply1 Nx_kernel.Prog.Cast ~dst src with
  | A.Done -> ()
  | r -> failf "a cast answered %a" Nx_array_support.pp_answer r);
  (elements (A.Any dst)).(0)

(* The order sorts follow over elements of [d]: -0 below +0, every NaN, and
   every complex number with a NaN part, above +inf and equal to each other;
   complex numbers by real part, then imaginary part; integers by value; [false]
   first. *)
let compare_elements (D.Any dt as d) a b =
  let nans na nb k =
    match (na, nb) with
    | true, true -> 0
    | true, false -> 1
    | false, true -> -1
    | false, false -> k ()
  in
  if D.is D.Complex dt then
    let h = String.length a / 2 in
    let get = if h = 4 then get32 else get64 in
    let nan e = Float.is_nan (get e 0) || Float.is_nan (get e h) in
    nans (nan a) (nan b) (fun () ->
        match compare_floats (get a 0) (get b 0) with
        | 0 -> compare_floats (get a h) (get b h)
        | c -> c)
  else if D.is D.Float dt then
    let get e =
      if is_narrow d then decode d e
      else if String.length e = 4 then get32 e 0
      else get64 e 0
    in
    let x = get a and y = get b in
    nans (Float.is_nan x) (Float.is_nan y) (fun () -> compare_floats x y)
  else if D.is D.Signed dt then Int64.compare (int_value d a) (int_value d b)
  else Int64.unsigned_compare (int_value d a) (int_value d b)
