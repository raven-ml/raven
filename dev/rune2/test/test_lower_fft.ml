(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fourier transforms lowered, against nx's eager float64 transform of the same
   points: within the sum of both bounds, Lower_fft's at the operand's
   precision and nx's CPU kernels' at float64, normwise over the whole
   result. *)

open Windtrap
module D = Nx_array.Dtype
module K = Nx_kernel
module P = Nx.Prim

let numel s = Array.fold_left ( * ) 1 s

type kind = C2c of K.direction | R2c | C2r of int

type case = {
  kind : kind;
  wide : bool;  (** float64 parts. *)
  shape : int array;  (** The operand's. *)
  axes : int list;
  transposed : bool;
  re : float array;
  im : float array;  (** Empty for [R2c]. *)
}

let last c = List.nth c.axes (List.length c.axes - 1)

let pp_case ppf c =
  let ints l = String.concat ";" (List.map string_of_int l) in
  let floats a =
    String.concat ";" (Array.to_list (Array.map (Printf.sprintf "%h") a))
  in
  Format.fprintf ppf "%s %s [%s] axes [%s]%s re [%s] im [%s]"
    (match c.kind with
    | C2c Forward -> "c2c forward"
    | C2c Inverse -> "c2c inverse"
    | R2c -> "r2c"
    | C2r n -> Printf.sprintf "c2r of %d points" n)
    (if c.wide then "float64" else "float32")
    (ints (Array.to_list c.shape))
    (ints c.axes)
    (if c.transposed then " transposed" else "")
    (floats c.re) (floats c.im)

(* Operands

   A case's values are in C order of the operand's shape. A transposed
   operand is a strided view of a copy laid out in the reverse order. *)

let f32 v = Int32.float_of_bits (Int32.bits_of_float v)
let round wide v = if wide then v else f32 v

let viewed c x =
  if c.transposed then Nx.transpose (Nx.copy (Nx.transpose x)) else x

(* The bins whose imaginary parts a [C2r] along one axis never reads: [0],
   and [n/2] for an even [n]. Along several, it reads them through the
   others' transforms. *)
let unread c i =
  match (c.kind, c.axes) with
  | C2r n, [ _ ] ->
      let l = last c in
      let r = Array.length c.shape in
      let stride = numel (Array.sub c.shape (l + 1) (r - l - 1)) in
      let k = i / stride mod c.shape.(l) in
      k = 0 || (n mod 2 = 0 && k = n / 2)
  | _ -> false

(* The imaginary parts, those a [C2r] never reads [0] where [zeroed]. *)
let imag c ~zeroed =
  Array.mapi (fun i v -> if zeroed && unread c i then 0. else v) c.im

let complexes (type s) (dt : (Complex.t, s) D.t) c ~zeroed =
  let im = imag c ~zeroed in
  let z = Array.mapi (fun i re -> { Complex.re; im = im.(i) }) c.re in
  viewed c (Nx.create dt c.shape z)

let reals (type s) (dt : (float, s) D.t) c =
  viewed c (Nx.create dt c.shape c.re)

(* Results *)

(* [results c ~wide ~lowered] is the result of [c]'s transform as float64
   parts, real parts then imaginary ones, at float64 parts where [wide],
   lowered where [lowered]. *)
let results c ~wide ~lowered =
  let get : type v s. (unit -> (v, s, Nx.host) Nx.t) -> v array =
   fun op -> if lowered then Lowered.run op else Nx.to_array (op ())
  in
  let split zs =
    Array.append
      (Array.map (fun z -> z.Complex.re) zs)
      (Array.map (fun z -> z.Complex.im) zs)
  in
  let zeroed = not lowered in
  let axes = Iarray.of_list c.axes in
  let fft o () = P.eval ~by:"t" (P.Fft o) in
  match (c.kind, wide) with
  | C2c direction, true ->
      let x = complexes D.Complex128 c ~zeroed in
      split (get (fft (C2c { direction; axes; x })))
  | C2c direction, false ->
      let x = complexes D.Complex64 c ~zeroed in
      split (get (fft (C2c { direction; axes; x })))
  | R2c, true ->
      let x = reals D.Float64 c in
      split (get (fft (R2c { dtype = D.Complex128; axes; x })))
  | R2c, false ->
      let x = reals D.Float32 c in
      split (get (fft (R2c { dtype = D.Complex64; axes; x })))
  | C2r n, true ->
      let x = complexes D.Complex128 c ~zeroed in
      get (fft (C2r { dtype = D.Float64; n; axes; x }))
  | C2r n, false ->
      let x = complexes D.Complex64 c ~zeroed in
      get (fft (C2r { dtype = D.Float32; n; axes; x }))

(* Bounds

   [Lower_fft]'s [e(n)], and nx's CPU kernels' (Nx_cpu), whose lines here
   reach neither Bluestein's convolution nor the real lines of 512 points. *)

let rec factors n p =
  if n <= 1 then []
  else if p * p > n then [ n ]
  else if n mod p = 0 then p :: factors (n / p) p
  else factors n (p + 1)

let lowered_e n =
  let fs = factors n 2 in
  let c p =
    let p = Float.of_int p in
    if p = 2. then 1. else ((2. *. p) +. 15.) *. sqrt p
  in
  List.fold_left (fun a p -> a +. c p) 0. fs
  +. (15. *. Float.of_int (max 0 (List.length fs - 1)))

let eager_e n =
  let p32 p = Float.pow (Float.of_int p) 1.5 in
  (8. *. List.fold_left (fun a p -> a +. p32 p) 0. (factors n 2)) +. 8.

(* The points along each of [c]'s axes. *)
let points c =
  List.map
    (fun a ->
      match c.kind with C2r n when a = last c -> n | _ -> c.shape.(a))
    c.axes

let norm a = sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0. a)

(* [‖ŷ - y‖₂ <= ε u √N ‖x‖₂] on each side, [‖x‖₂] [√2] times larger for a
   [C2r], whose unread imaginary parts count as [0], read [0] by eager. *)
let check c =
  let l = results c ~wide:c.wide ~lowered:true in
  let e = results c ~wide:true ~lowered:false in
  equal int (Array.length e) (Array.length l);
  let ps = points c in
  let sum f = List.fold_left (fun a n -> a +. f n) 0. ps in
  let u = if c.wide then 0x1p-53 else 0x1p-24 in
  let x = norm (Array.append c.re (imag c ~zeroed:true)) in
  let x = match c.kind with C2r _ -> sqrt 2. *. x | _ -> x in
  let n = Float.of_int (List.fold_left ( * ) 1 ps) in
  let eps = (sum lowered_e *. u) +. (sum eager_e *. 0x1p-53) in
  let bound = eps *. sqrt n *. x in
  let diff = norm (Array.mapi (fun i e -> e -. l.(i)) e) in
  let msg = Format.asprintf "%a" pp_case c in
  less ~msg float_exact ~than:(Float.succ bound) diff

(* Generators *)

let among xs = Gen.one_of (List.map Gen.constant xs)

let extent =
  Gen.frequency
    [
      (1, Gen.constant 0);
      (1, Gen.constant 1);
      (8, among [ 2; 3; 4; 5; 6; 8; 9; 10; 12; 16 ]);
    ]

let part wide =
  let open Gen in
  frequency
    [
      (6, map (round wide) (float_range (-4.) 4.));
      (1, map (round wide) (float_range (-1e3) 1e3));
      (1, constant 0.);
      (1, constant (-0.));
    ]

(* [case kind] draws a transform of [kind]: an operand of rank 1 to 3, a
   nonempty set of its axes, and for a [C2r] along one axis the unread
   imaginary parts NaN, infinite or huge half the time. *)
let case kind =
  let open Gen in
  let* kind = kind in
  let* wide = bool in
  let* r = int_range 1 3 in
  let* s = map Array.of_list (list ~size:(constant r) extent) in
  let* keep = list ~size:(constant r) bool in
  let* first = int_range 0 (r - 1) in
  let axes =
    List.filter (fun a -> a = first || List.nth keep a) (List.init r Fun.id)
  in
  let* kind =
    match kind with
    | C2r _ ->
        let+ n = extent in
        C2r n
    | k -> constant k
  in
  let c =
    { kind; wide; shape = s; axes; transposed = false; re = [||]; im = [||] }
  in
  let shape =
    match kind with
    | C2r n -> Array.mapi (fun i e -> if i = last c then (n / 2) + 1 else e) s
    | _ -> s
  in
  let* transposed = if r > 1 then bool else constant false in
  let k = numel shape in
  let* re = array ~size:(constant k) (part wide) in
  let* im =
    match kind with
    | R2c -> constant [||]
    | _ -> array ~size:(constant k) (part wide)
  in
  let c = { c with shape; transposed; re; im } in
  let* wild =
    match (kind, axes) with C2r _, [ _ ] -> bool | _ -> constant false
  in
  let+ odd = among [ Float.nan; Float.infinity; 1e30 ] in
  let im = Array.mapi (fun i v -> if wild && unread c i then odd else v) im in
  { c with im }

let case kind = Gen.with_pp pp_case (case kind)

(* Covers *)

let covers c =
  let ps = points c in
  cover "float32 parts" (not c.wide);
  cover "float64 parts" c.wide;
  cover "a transposed operand" c.transposed;
  cover "two axes" (List.length c.axes >= 2);
  let some f = List.exists (fun n -> f (factors n 2)) ps in
  cover "a power of two"
    (some (fun fs -> List.length fs >= 2 && List.for_all (( = ) 2) fs));
  cover "an odd prime factor" (some (List.exists (fun p -> p > 2)));
  cover "mixed factors"
    (some (fun fs -> List.length (List.sort_uniq Int.compare fs) > 1))

let example kind shape axes re im =
  { kind; wide = false; shape; axes; transposed = false; re; im }

let ramp n = Array.init n (fun i -> Float.of_int (i mod 7) -. 3.)

(* A transform of float64 parts along both axes of a transposed operand,
   which reaches the covers each run needs. *)
let both kind shape =
  let k = numel shape in
  let im = match kind with R2c -> [||] | _ -> Array.map Float.neg (ramp k) in
  let re = ramp k in
  { kind; wide = true; shape; axes = [ 0; 1 ]; transposed = true; re; im }

let c2cs =
  let examples =
    [
      example (C2c Forward) [| 0; 3 |] [ 0; 1 ] [||] [||];
      example (C2c Inverse) [| 1; 3 |] [ 0 ] [| 1.; 2.; 3. |]
        [| 0.; -0.; 1. |];
      both (C2c Forward) [| 8; 6 |];
      { (both (C2c Inverse) [| 8; 6 |]) with wide = false };
    ]
  in
  prop ~count:20 ~examples "complex transforms hold their bound"
    (case (among [ C2c Forward; C2c Inverse ]))
    (fun c ->
      covers c;
      cover "a transform of no point" (List.mem 0 (points c));
      check c)

let r2cs =
  let examples =
    [
      example R2c [| 2; 0 |] [ 1 ] [||] [||];
      example R2c [| 1 |] [ 0 ] [| -3. |] [||];
      both R2c [| 8; 6 |];
      { (both R2c [| 8; 6 |]) with wide = false };
    ]
  in
  prop ~count:20 ~examples "real transforms hold their bound"
    (case (Gen.constant R2c))
    (fun c ->
      covers c;
      cover "a last axis of no point"
        (List.nth (points c) (List.length c.axes - 1) = 0);
      check c)

let c2rs =
  let nan = Float.nan in
  let examples =
    [
      example (C2r 0) [| 2; 1 |] [ 1 ] [| 1.; 2. |] [| 3.; 4. |];
      example (C2r 4) [| 3 |] [ 0 ] [| 1.; 2.; 3. |]
        [| nan; 1.; Float.infinity |];
      example (C2r 5) [| 2; 3 |] [ 1 ] [| 1.; 2.; 3.; 4.; 5.; 6. |]
        [| nan; 1.; 2.; 1e30; 3.; 4. |];
      both (C2r 6) [| 8; 4 |];
      { (both (C2r 6) [| 8; 4 |]) with wide = false };
    ]
  in
  prop ~count:20 ~examples "real inverse transforms hold their bound"
    (case (Gen.constant (C2r 0)))
    (fun c ->
      covers c;
      cover "an odd number of points"
        (match c.kind with C2r n -> n mod 2 = 1 | _ -> false);
      cover "an unread NaN imaginary part" (Array.exists Float.is_nan c.im);
      check c)

let () =
  exit (run "Lower_fft" [ group "transforms" [ c2cs; r2cs; c2rs ] ])
