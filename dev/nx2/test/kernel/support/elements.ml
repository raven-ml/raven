(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Elements as their bits, for the suites whose references move elements:
   gathers, scatters and sorts. Arrays are drawn through views, and their
   elements read back in C order of indices as strings of their bytes.
   Shapes near a fit draw the refusals of nx.cpu's suites. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move

let pp_ints = Nx_array_gen.pp_ints
let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let total s = Array.fold_left ( * ) 1 s
let shape_of (A.Any x) = Iarray.to_array (L.shape (A.layout x))
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
  let l = L.contiguous (Iarray.of_array s) in
  A.Any (A.v dt l (Rig.Buffer.of_string (Bytes.to_string b)))

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
  | D.Bit -> String.make 1 (Char.chr (Random.State.int r 2))
  (* A boolean is true where its byte is not zero: any byte. *)
  | D.Bool -> String.make 1 (Char.chr [| 0; 1; 2; 255 |].(Random.State.int r 4))
  | _ when D.bits dt = 4 -> String.make 1 (Char.chr (Random.State.int r 16))
  | _ ->
      let w = D.bits dt / 8 in
      if Random.State.bool r then any w
      else
        let v = Random.State.int r 7 - 3 in
        String.init w (fun j -> Char.chr ((v asr (8 * j)) land 0xFF))

(* Views *)

(* How an array of shape [s] is made and viewed. Axis [i] of [s] is made
   axis [perm.(i)]; the made axes are stored in the order [stored] names,
   outermost first; the last made axis is made twice as long and stepped by
   two where [stepped]; the last two made axes, [p] and [q], are the [p]
   windows of [q] taps of one made axis, the windows [step] apart and the
   taps [dilation], where [windowed = Some (step, dilation)]; made axis
   [reversed] is read backwards; and the axes [broadcast] of [s] are made
   with one element and broadcast. *)
type view = {
  perm : int array;
  stored : int array;
  stepped : bool;
  windowed : (int * int) option;
  reversed : int option;
  broadcast : int list;
}

let identity p = p = Array.init (Array.length p) Fun.id

let inverse p =
  let q = Array.make (Array.length p) 0 in
  Array.iteri (fun i m -> q.(m) <- i) p;
  q

let view_names v =
  (if identity v.perm then [] else [ "permuted" ])
  @ (if identity v.stored then [] else [ "stored apart" ])
  @ (if v.stepped then [ "stepped" ] else [])
  @ (if v.windowed <> None then [ "windowed" ] else [])
  @ (if v.reversed <> None then [ "reversed" ] else [])
  @ if v.broadcast <> [] then [ "broadcast" ] else []

let plain r =
  {
    perm = Array.init r Fun.id;
    stored = Array.init r Fun.id;
    stepped = false;
    windowed = None;
    reversed = None;
    broadcast = [];
  }

(* A view of rank [r], broadcast along [broadcast]: windows one time in
   four where [windows], the made axes stored in another order one time in
   three where [apart]. *)
let view_of ?(broadcast = []) ?(windows = false) ?(apart = false) r =
  let open Gen in
  let axes = List.init r Fun.id in
  let* perm = permutation ~pp:Format.pp_print_int axes in
  let* windowed =
    if windows && r >= 2 then
      frequency
        [
          (3, constant None);
          (1, map Option.some (pair (int_range 1 2) (int_range 1 2)));
        ]
    else constant None
  in
  let* stored =
    if apart && windowed = None then
      frequency [ (2, constant axes); (1, permutation ~pp:Format.pp_print_int axes) ]
    else constant axes
  in
  let* stepped = bool in
  let+ reversed = option (int_range 0 (max 0 (r - 1))) in
  {
    perm = Array.of_list perm;
    stored = Array.of_list stored;
    stepped = stepped && r > 0;
    windowed;
    reversed = (if r > 0 then reversed else None);
    broadcast;
  }

(* The shape of the array made for [s] through [v], before any move: [s] in
   made order, a broadcast axis of one element, the windows' two axes one,
   the last twice as long where stepped. *)
let made v s =
  let r = Array.length s in
  let inv = inverse v.perm in
  let m =
    Array.init r (fun a ->
        if List.mem inv.(a) v.broadcast then 1 else s.(inv.(a)))
  in
  let m =
    match v.windowed with
    | Some (step, dilation) when r >= 2 && m.(r - 2) > 0 && m.(r - 1) > 0 ->
        let p = m.(r - 2) and q = m.(r - 1) in
        Array.append (Array.sub m 0 (r - 2))
          [| ((p - 1) * step) + (dilation * (q - 1)) + 1 |]
    | _ -> m
  in
  let last = Array.length m - 1 in
  Array.mapi (fun i e -> if v.stepped && i = last then 2 * e else e) m

(* An array of shape [s] through the view [v], its elements those of
   [make m], a C-contiguous array of the made shape [m] ({!made}). *)
let operand_with make s v =
  let r = Array.length s in
  let inv = inverse v.perm in
  let shape =
    Array.init r (fun a ->
        if List.mem inv.(a) v.broadcast then 1 else s.(inv.(a)))
  in
  let m = made v s in
  let whole e = { M.start = 0; count = e; step = 1 } in
  let move mv (A.Any x) = A.Any (Option.get (A.move mv x)) in
  let x = make m in
  (* The elements copied into an array stored in [v.stored]'s order, read
     back in the made axes' order. *)
  let x =
    if identity v.stored then x
    else if Array.length v.stored <> Array.length m then
      invalid_arg "operand: stored names other axes than the made ones"
    else
      let copy (A.Any t) = A.Any (A.copy t) in
      move
        (M.Permute (Iarray.of_array (inverse v.stored)))
        (copy (move (M.Permute (Iarray.of_array v.stored)) x))
  in
  let last = Array.length m - 1 in
  let x =
    if not v.stepped then x
    else
      move
        (M.Slice
           (Iarray.mapi
              (fun i e ->
                if i = last then { M.start = 1; count = e / 2; step = 2 }
                else whole e)
              (Iarray.of_array m)))
        x
  in
  let x =
    match v.windowed with
    | Some (step, dilation) when Array.length m < r ->
        let w : M.window = { axis = r - 2; size = shape.(r - 1); step; dilation } in
        move (M.Window (Iarray.of_array [| w |])) x
    | _ -> x
  in
  let x =
    match v.reversed with
    | Some a when shape.(a) > 0 ->
        move
          (M.Slice
             (Iarray.mapi
                (fun i e ->
                  if i = a then { M.start = e - 1; count = e; step = -1 }
                  else whole e)
                (Iarray.of_array shape)))
          x
    | _ -> x
  in
  let x = move (M.Permute (Iarray.of_array v.perm)) x in
  if v.broadcast = [] then x else move (M.Broadcast (Iarray.of_array s)) x

(* An array of [dt] and shape [s] through the view [v] whose elements are
   [f k], k counted in C order of the made array. *)
let operand dt s v f =
  operand_with (fun m -> of_elements dt m (Array.init (total m) f)) s v

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

(* Prog.Add on two floats of 32 or 64 bits, from their bits: the IEEE 754
   sum, rounded once, -0 for two -0's, a NaN where an operand is or opposite
   infinities meet. *)
let add_bits ~w a b =
  let get, put, round =
    if w = 4 then (get32, f32, round32) else (get64, f64, Fun.id)
  in
  put (round (get a 0 +. get b 0))

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
  (match Nx_kernels_support.map1 (module Nx_cpu) Cast ~dst src with
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
  else if D.is D.Boolean dt then compare (a <> "\000") (b <> "\000")
  else if D.is D.Signed dt then Int64.compare (int_value d a) (int_value d b)
  else Int64.unsigned_compare (int_value d a) (int_value d b)

(* Whether the element [e] of [d] is a NaN, a complex number where a part
   is. *)
let nan_of (D.Any dt as d) e =
  if D.is D.Complex dt then
    let h = String.length e / 2 in
    let get = if h = 4 then get32 else get64 in
    Float.is_nan (get e 0) || Float.is_nan (get e h)
  else if is_narrow d then Float.is_nan (decode d e)
  else if D.is D.Float dt then
    Float.is_nan (if String.length e = 4 then get32 e 0 else get64 e 0)
  else false

(* Elements of [d] that have the same bits or are both NaN, a complex
   number's parts each so: a NaN result promises a NaN, never which one,
   and a NaN part leaves the other part's result to be checked. *)
let same (D.Any dt as d) =
  let blind d a b = String.equal a b || (nan_of d a && nan_of d b) in
  let parts a b =
    let h = String.length a / 2 in
    let part = if h = 4 then D.Any D.Float32 else D.Any D.Float64 in
    let at e i = String.sub e i h in
    blind part (at a 0) (at b 0) && blind part (at a h) (at b h)
  in
  let equal = if D.is D.Complex dt then parts else blind d in
  Testable.make ~pp:(Testable.pp Testable.string) ~equal

(* Shapes near a fit *)

(* [s] mostly, else [s] missed by one change: an extent one larger, the last
   axis dropped, or an axis of 1 appended. *)
let near s =
  let r = Array.length s in
  let grow i = Array.mapi (fun j d -> if j = i then d + 1 else d) s in
  let misses =
    (1, Gen.constant (Array.append s [| 1 |]))
    ::
    (if r = 0 then []
     else
       [
         (1, Gen.map grow (Gen.int_range 0 (r - 1)));
         (1, Gen.constant (Array.sub s 0 (r - 1)));
       ])
  in
  Gen.frequency ((6, Gen.constant s) :: misses)

(* An axis of [s] mostly, else -1 or its rank. *)
let axis_near s =
  let r = Array.length s in
  let inside = if r = 0 then Gen.constant 0 else Gen.int_range 0 (r - 1) in
  Gen.frequency [ (8, inside); (1, Gen.constant (-1)); (1, Gen.constant r) ]

(* [s] with [n] along [axis], where it has that axis. *)
let along s axis n = Array.mapi (fun i d -> if i = axis then n else d) s

(* Padded loads *)

(* The array [x] padded with the element [fill] (its bits) as [pad] says,
   C-contiguous, built element by element: along each axis, padded index
   [i] holds element [(i - lo) / (interior + 1)] where that divides and lies
   in the operand, the fill elsewhere; then the windows' view. *)
let pad_array (A.Any x) fill (pad : Nx_kernel.pad) =
  let dt = A.dtype x in
  let w = D.bits dt / 8 in
  let s = Iarray.to_array (L.shape (A.layout x)) in
  let r = Array.length s in
  let padded =
    Array.init r (fun i ->
        let d = s.(i) in
        Iarray.get pad.lo i + Iarray.get pad.hi i + d
        + if d > 0 then Iarray.get pad.interior i * (d - 1) else 0)
  in
  (* The operand's index at padded index [idx], if it lands on one. *)
  let source idx =
    let src = Array.make r 0 and inside = ref true in
    Array.iteri
      (fun i j ->
        let c = j - Iarray.get pad.lo i
        and step = Iarray.get pad.interior i + 1 in
        if c < 0 || c mod step <> 0 || c / step >= s.(i) then inside := false
        else src.(i) <- c / step)
      idx;
    if !inside then Some src else None
  in
  let indices = Nx_array_gen.indices padded in
  let a =
    if w = 0 then begin
      (* A sub-byte element is no byte of its own: element by element. *)
      let one = A.v dt (L.contiguous [||]) (Rig.Buffer.of_string fill) in
      let f = A.get one [||] in
      let a = A.create Rig.host dt (Iarray.of_array padded) in
      List.iter
        (fun idx ->
          A.set a (Iarray.of_array idx)
            (match source idx with
            | Some src -> A.get x (Iarray.of_array src)
            | None -> f))
        indices;
      a
    end
    else begin
      let bytes = A.to_array (Option.get (A.bitcast D.Uint8 (A.copy x))) in
      let b = Bytes.create (max 1 (w * total padded)) in
      List.iteri
        (fun k idx ->
          let at = Option.map (flat s) (source idx) in
          for q = 0 to w - 1 do
            Bytes.set b ((k * w) + q)
              (match at with
              | Some at -> Char.chr bytes.((at * w) + q)
              | None -> fill.[q])
          done)
        indices;
      let l = L.contiguous (Iarray.of_array padded) in
      A.v dt l (Rig.Buffer.of_string (Bytes.to_string b))
    end
  in
  if Iarray.length pad.windows = 0 then A.Any a
  else A.Any (Option.get (A.move (M.Window pad.windows) a))

(* Guard pages *)

let pp_side ppf s =
  Format.pp_print_string ppf
    (match s with Nx_kernels_support.Before -> "before" | After -> "after")

let sides =
  Gen.with_pp pp_side
    (Gen.of_list [ Nx_kernels_support.Before; Nx_kernels_support.After ])

(* [f guard place], where [guard x] is a copy of [x] against an inaccessible
   page on [side], laid out as [x] is, and [place d s] a C-contiguous array
   of the dtype [d] and shape [s] placed likewise; their memory freed once [f]
   returns. *)
let with_guards side f =
  let frees = ref [] in
  let keep (a, free) =
    frees := free :: !frees;
    A.Any a
  in
  let guard (A.Any x) = keep (Nx_kernels_support.guard side x) in
  let place (D.Any dt) s =
    let l = L.contiguous (Iarray.of_array s) in
    keep (Nx_kernels_support.guarded side dt l)
  in
  Fun.protect
    ~finally:(fun () -> List.iter (fun free -> free ()) !frees)
    (fun () -> f guard place)

(* Extents of one element to a few vectors past the widest. *)
let tails =
  Gen.frequency
    [
      (6, Gen.int_range 1 72);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 127; 128; 129; 255; 256; 257 ]);
    ]

(* Shapes of up to two short outer axes and a last extent of {!tails}. *)
let tail_shapes =
  let open Gen in
  let* n = tails in
  let+ outer = array ~size:(int_range 0 2) (int_range 1 4) in
  Array.append outer [| n |]

(* [call xs dsts] into fresh destinations of the dtypes and shapes [dsts], then
   with [xs] and the destinations as {!with_guards} places them: the same
   answer and, where it is [Done], the same elements, bit for bit, or any NaN
   for a NaN where [any_nan]. The answer. *)
let same_guarded ?(any_nan = false) side ~call xs dsts =
  let fresh =
    Array.map
      (fun (D.Any dt, s) -> A.Any (A.create Rig.host dt (Iarray.of_array s)))
      dsts
  in
  let want = call xs fresh in
  with_guards side (fun guard place ->
      let ds = Array.map (fun (d, s) -> place d s) dsts in
      equal answer want (call (Array.map guard xs) ds);
      if want = A.Done then
        Array.iteri
          (fun i d ->
            let elt = if any_nan then same (fst dsts.(i)) else string in
            equal (array elt) (elements fresh.(i)) (elements d))
          ds);
  want
