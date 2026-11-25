(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Fourier transforms of each kernel library: each result against the sums
   nx_kernel.mli's c2c, r2c and c2r state, computed in double-double (or in
   double, for single-precision transforms), within the bound the library states
   (Support's [transform_factor]), forward and inverse, along one axis or
   several, through views, at every length up to 70 and on long lines of each
   kind, every destination element written; the bins a c2r ignores; the same
   bits under every layout, on one thread as on the job's, and for a line alone
   as in its batch; nx.cpu's bits on every target and host architecture; empty
   transforms; the refusals; no byte read or written past the operand or the
   result; and the device memory nx.cuda keeps for later calls. Each backend
   runs its own kernels on its device, opened in the test. *)

open Windtrap
open Elements
open Exact
module Support = Nx_kernels_support

(* sin φ and cos φ by their series, φ in [0, π/4]. *)
let sincos phi =
  let s = ref (0., 0.) and c = ref (0., 0.) and t = ref (1., 0.) in
  for k = 0 to 40 do
    (match k mod 4 with
    | 0 -> c := dd_add !c !t
    | 1 -> s := dd_add !s !t
    | 2 -> c := dd_add !c (dd_neg !t)
    | _ -> s := dd_add !s (dd_neg !t));
    t := dd_div (dd_mul !t phi) (Float.of_int (k + 1), 0.)
  done;
  (!s, !c)

(* e^(-2πi m/n), 0 <= m < n: the angle within its quarter taken from the nearer
   end, then turned by whole quarters. *)
let unit m n =
  let q = 4 * m / n in
  let r = (4 * m) - (q * n) in
  let far = 2 * r > n in
  let pi_2 = (1.5707963267948966, 6.123233995736766e-17) in
  let phi =
    dd_mul pi_2
      (dd_div
         (Float.of_int (if far then n - r else r), 0.)
         (Float.of_int n, 0.))
  in
  let s, c = sincos phi in
  let c, s = if far then (s, c) else (c, s) in
  let re, im =
    match q with
    | 0 -> (c, s)
    | 1 -> (dd_neg s, c)
    | 2 -> (dd_neg c, dd_neg s)
    | _ -> (s, dd_neg c)
  in
  (re, dd_neg im)

let units = Hashtbl.create 16

let units_of n =
  match Hashtbl.find_opt units n with
  | Some w -> w
  | None ->
      let w = Array.init n (fun m -> unit m n) in
      Hashtbl.add units n w;
      w

(* The exact transforms *)

(* A complex number of double-doubles. *)
type dd = { re : float * float; im : float * float }

let dd_of (z : Complex.t) = { re = (z.re, 0.); im = (z.im, 0.) }

let complex_of z : Complex.t =
  { re = fst z.re +. snd z.re; im = fst z.im +. snd z.im }

(* [acc.(i)] and [acc.(i + 1)], a double-double (h, l), plus the product of the
   double-doubles (ah, al) and (bh, bl): its high part by TwoSum into h, the
   rest, the product's error by fma and its cross terms, into l. *)
let[@inline] add_product acc i ah al bh bl =
  let p = ah *. bh in
  let e = Float.fma ah bh (-.p) +. (ah *. bl) +. (al *. bh) in
  let h = Float.Array.get acc i in
  let s = h +. p in
  let bb = s -. h in
  Float.Array.set acc (i + 1)
    (Float.Array.get acc (i + 1) +. (h -. (s -. bb)) +. (p -. bb) +. e);
  Float.Array.set acc i s

(* The sum Σ_j x_j e^(∓2πi jk/n) of a line, in double-double where [wide] (each
   sum's low parts added without renormalising, which errs by about n u² of
   double), in double else, its twiddles rounded once. The parts are unboxed
   floats, so the O(n²) sums allocate nothing. *)
let dft ~wide direction (x : dd array) =
  let n = Array.length x in
  let w = units_of n in
  let sign = match direction with Nx_kernel.Forward -> 1. | Inverse -> -1. in
  let parts f a = Float.Array.init n (fun j -> f a.(j)) in
  let zrh = parts (fun z -> fst z.re) x and zrl = parts (fun z -> snd z.re) x in
  let zih = parts (fun z -> fst z.im) x and zil = parts (fun z -> snd z.im) x in
  let wrh = parts (fun ((h, _), _) -> h) w in
  let wrl = parts (fun ((_, l), _) -> l) w in
  let wih = parts (fun (_, (h, _)) -> sign *. h) w in
  let wil = parts (fun (_, (_, l)) -> sign *. l) w in
  let get = Float.Array.get in
  Array.init n (fun k ->
      (* The real part's high and low parts, then the imaginary part's. *)
      let acc = Float.Array.make 4 0. in
      let rh = ref 0. and ih = ref 0. in
      for j = 0 to n - 1 do
        let t = j * k mod n in
        let xh = get zrh j and xl = get zrl j in
        let yh = get zih j and yl = get zil j in
        let ch = get wrh t and cl = get wrl t in
        let sh = get wih t and sl = get wil t in
        if wide then begin
          add_product acc 0 xh xl ch cl;
          add_product acc 0 (-.yh) (-.yl) sh sl;
          add_product acc 2 xh xl sh sl;
          add_product acc 2 yh yl ch cl
        end
        else begin
          rh := !rh +. (xh *. ch) -. (yh *. sh);
          ih := !ih +. (xh *. sh) +. (yh *. ch)
        end
      done;
      if wide then { re = (get acc 0, get acc 1); im = (get acc 2, get acc 3) }
      else { re = (!rh, 0.); im = (!ih, 0.) })

(* [f] applied to each line along [axis] of [v], of shape [s], giving lines of
   [m] points: the result and its shape. *)
let along s axis m f v =
  let n = s.(axis) in
  let t = Array.copy s in
  t.(axis) <- m;
  let out = Array.make (total t) { re = (0., 0.); im = (0., 0.) } in
  Array.iteri
    (fun k _ ->
      let i = index_of t k in
      if i.(axis) = 0 then begin
        let line =
          Array.init n (fun j ->
              let ij = Array.copy i in
              ij.(axis) <- j;
              v.(flat s ij))
        in
        let y = f line in
        Array.iteri
          (fun j z ->
            let ij = Array.copy i in
            ij.(axis) <- j;
            out.(flat t ij) <- z)
          y
      end)
    out;
  (out, t)

(* What a case transforms *)
type kind =
  | C2c of Nx_kernel.direction
  | R2c
  | C2r of int  (** Of [n] points. *)

(* The four transforms of a line of [n] points. *)
let kinds n = [ C2c Nx_kernel.Forward; C2c Inverse; R2c; C2r n ]

let kind_name = function
  | C2c Forward -> "forward c2c"
  | C2c Inverse -> "inverse c2c"
  | R2c -> "r2c"
  | C2r n -> Printf.sprintf "c2r of %d points" n

let width wide = if wide then "64-bit" else "32-bit"

(* The exact result of [kind] of [v], of shape [s], along [axes]. *)
let exact ~wide kind axes s v =
  let last = axes.(Array.length axes - 1) in
  let rest = Array.sub axes 0 (Array.length axes - 1) in
  let each d axes (v, s) =
    Array.fold_left
      (fun (v, s) a -> along s a s.(a) (dft ~wide d) v)
      (v, s) axes
  in
  match kind with
  | C2c d -> fst (each d axes (v, s))
  | R2c ->
      let v, s = each Forward axes (v, s) in
      (* Of no point, the empty sum: one bin of +0. *)
      let half l =
        Array.init
          ((s.(last) / 2) + 1)
          (fun k ->
            if k < Array.length l then l.(k)
            else { re = (0., 0.); im = (0., 0.) })
      in
      fst (along s last ((s.(last) / 2) + 1) half v)
  | C2r n ->
      let v, s = each Inverse rest (v, s) in
      let complete line =
        Array.init n (fun k ->
            if k <= n / 2 then line.(k)
            else
              let z = line.(n - k) in
              { re = z.re; im = dd_neg z.im })
      in
      let real line =
        Array.map
          (fun z -> { z with im = (0., 0.) })
          (dft ~wide Inverse (complete line))
      in
      fst (along s last n real v)

(* The bound *)

(* The prime factors of [n], least first, by which the cases are covered. *)
let rec factors n d =
  if n = 1 then []
  else if d * d > n then [ n ]
  else if n mod d = 0 then d :: factors (n / d) d
  else factors n (d + 1)

let by_bluestein n = List.exists (fun p -> p > 250) (factors n 2)
let composite n = List.length (factors n 2) > 1

(* [b]'s ε of a transform along [axes] of [s], [n] points along the last for
   c2r. *)
let epsilon (b : Support.backend) kind axes s =
  let last = axes.(Array.length axes - 1) in
  Array.fold_left
    (fun eps a ->
      let n = match kind with C2r n when a = last -> n | _ -> s.(a) in
      let axis : Support.axis =
        match kind with
        | R2c when a = last -> R2c_last
        | C2r _ when a = last -> C2r_last
        | R2c | C2r _ | C2c _ -> Complex
      in
      eps +. b.transform_factor axis n)
    0. axes

(* Cases *)

type case = {
  kind : kind;
  wide : bool;  (** float64 parts. *)
  shape : int array;  (** The operand's. *)
  axes : int array;
  view : view;
  xs : Complex.t array;  (** Its points, none where {!drawn} draws them. *)
}

let pp_case ppf c =
  Format.fprintf ppf "%s %s of %a along %a (%s)" (kind_name c.kind)
    (width c.wide) pp_ints c.shape pp_ints c.axes
    (String.concat ", " (view_names c.view))

(* Shapes, with the axes they must transform: small ones of any axes; one long
   line among short ones, transformed with any others, of a composite length of
   several primes, of a prime past 250 or twice one (514 and 1018 halved for a
   real line), or of a length about a power of two; and long composite lines
   side by side. A long line's companions hold at most 3 lines, the cost of its
   O(n²) reference. *)
let four = [ 289; 300; 360; 512; 625; 1000; 1024 ]
let blue = [ 251; 257; 502; 509; 514; 1018; 1031 ]
let edge = [ 53; 64; 65; 97; 100; 241; 243; 255; 256 ]

(* Lengths of nx.cpu's four steps (fft.c: past 64 points, composite, no prime
   past 250), with 8 | n, whose rows are of 8 points (fft_body.h's [four_make]),
   and without, side by side as its lanes take them. *)
let side_lengths = [ 72; 120; 100; 255 ]

let shape_gen =
  let open Gen in
  let int_of = of_list ~pp:Format.pp_print_int in
  let small =
    let+ s = array ~size:(int_range 1 3) (int_range 0 9) in
    (s, [||])
  in
  let one_long =
    let* n =
      frequency [ (1, int_of four); (1, int_of blue); (1, int_of edge) ]
    in
    let* k = int_range 0 2 in
    let* e = int_range 1 3 in
    let* past_one = int_range 0 (max 0 (k - 1)) in
    let rest = Array.init k (fun a -> if a = past_one then e else 1) in
    let+ at = int_range 0 k in
    ( Array.concat [ Array.sub rest 0 at; [| n |]; Array.sub rest at (k - at) ],
      [| at |] )
  in
  let side_by_side =
    let* n = int_of side_lengths in
    let+ k = int_range 8 9 in
    ([| n; k |], [| 0 |])
  in
  frequency [ (2, small); (2, one_long); (1, side_by_side) ]

let value wide r =
  let x = Random.State.float r 2. -. 1. in
  if wide then x else Int32.float_of_bits (Int32.bits_of_float x)

(* [n] points of [kind]'s operand drawn from [seed], each part in (-1, 1), of
   [wide]'s width, real for an r2c. *)
let draw kind ~wide n seed =
  let rs = Random.State.make seed in
  Array.init n (fun _ ->
      let re = value wide rs in
      let im = match kind with R2c -> 0. | _ -> value wide rs in
      { Complex.re; im })

let case_gen =
  let open Gen in
  with_pp pp_case
  @@ let* s, must = shape_gen in
     let r = Array.length s in
     let* chosen = array ~size:(constant r) bool in
     let* last =
       if must = [||] then int_range 0 (r - 1) else constant must.(0)
     in
     (* The axis that must be transformed is the last. *)
     let axes =
       Array.of_list
         (List.filter
            (fun a -> a = last || (chosen.(a) && a < last))
            (List.init r Fun.id))
     in
     let* kind = of_list [ C2c Forward; C2c Inverse; R2c; C2r 0 ] in
     let* kind, s =
       match kind with
       | C2r _ ->
           let l = axes.(Array.length axes - 1) in
           let+ odd = bool in
           let n = max 0 ((2 * (s.(l) / 2)) + if odd then 1 else 0) in
           (C2r n, Array.mapi (fun a e -> if a = l then (n / 2) + 1 else e) s)
       | k -> constant (k, s)
     in
     let* wide = bool in
     let* broadcast =
       if total s = 0 then constant []
       else
         frequency
           [ (4, constant []); (1, map (fun a -> [ a ]) (int_range 0 (r - 1))) ]
     in
     let* view = view_of ~broadcast r in
     let+ seed = int in
     let xs = draw kind ~wide (total s) [| seed |] in
     { kind; wide; shape = s; axes; view; xs }

(* Real lines of 514 and 1018 points, whose halves have a prime factor past 250,
   in each width and direction: the generator draws them too seldom for every
   seed. *)
let halved_cases =
  let case (n, wide, kind) =
    let s = match kind with C2r _ -> (n / 2) + 1 | _ -> n in
    let xs = draw kind ~wide s [| n |] in
    { kind; wide; shape = [| s |]; axes = [| 0 |]; view = plain 1; xs }
  in
  List.concat_map
    (fun n ->
      List.concat_map
        (fun wide -> List.map case [ (n, wide, R2c); (n, wide, C2r n) ])
        [ false; true ])
    [ 514; 1018 ]

(* A long line along the second of two transformed axes, and an operand
   broadcast along its transformed axis, which the generator draws in one case
   of ten each. *)
let bound_examples =
  let case kind wide shape axes view =
    let xs = draw kind ~wide (total shape) [| total shape |] in
    { kind; wide; shape; axes; view; xs }
  in
  halved_cases
  @ [
      case (C2c Forward) false [| 3; 257; 2 |] [| 0; 1 |] (plain 3);
      case (C2c Forward) true [| 289; 3 |] [| 0 |]
        { (plain 2) with broadcast = [ 0 ] };
    ]

let points c =
  let last = c.axes.(Array.length c.axes - 1) in
  Array.mapi
    (fun a e -> match c.kind with C2r n when a = last -> n | _ -> e)
    c.shape

(* The operand: complex or real, through the case's view, made from the values
   [xs]. *)
let operand c =
  let s = Iarray.of_array c.shape in
  let re () = Array.map (fun (z : Complex.t) -> z.re) c.xs in
  let element (z : Complex.t) =
    match (c.kind, c.wide) with
    | R2c, false -> f32 z.re
    | R2c, true -> f64 z.re
    | _, false -> f32 z.re ^ f32 z.im
    | _, true -> f64 z.re ^ f64 z.im
  in
  let dt =
    match (c.kind, c.wide) with
    | R2c, false -> D.Any D.Float32
    | R2c, true -> D.Any D.Float64
    | _, false -> D.Any D.Complex64
    | _, true -> D.Any D.Complex128
  in
  (* A plain view's array holds the values in order, made without their
     elements' strings; a stepped view makes twice the elements, whose values
     repeat. *)
  if c.view = plain (Array.length c.shape) then
    match (c.kind, c.wide) with
    | R2c, false -> A.Any (A.of_array D.Float32 s (re ()))
    | R2c, true -> A.Any (A.of_array D.Float64 s (re ()))
    | _, false -> A.Any (A.of_array D.Complex64 s c.xs)
    | _, true -> A.Any (A.of_array D.Complex128 s c.xs)
  else
    operand dt c.shape c.view (fun k -> element c.xs.(k mod Array.length c.xs))

(* The result's shape and dtype. *)
let result c =
  let last = c.axes.(Array.length c.axes - 1) in
  let s = Array.copy c.shape in
  (match c.kind with
  | R2c -> s.(last) <- (s.(last) / 2) + 1
  | C2r n -> s.(last) <- n
  | C2c _ -> ());
  let dt =
    match (c.kind, c.wide) with
    | C2r _, false -> D.Any D.Float32
    | C2r _, true -> D.Any D.Float64
    | _, false -> D.Any D.Complex64
    | _, true -> D.Any D.Complex128
  in
  (dt, s)

(* [Done] or failing: every backend computes every transform. *)
let check got = equal answer A.Done got

(* A fresh array of [dt] and shape [s] on [b]'s device, every element NaN, so
   that one a transform leaves unwritten shows. *)
let poisoned (b : Support.backend) (D.Any dt) s =
  let fill (type v w) (dt : (v, w) D.t) (nan : v) =
    let y = A.create Rig.host dt (Iarray.of_array s) in
    A.write y (fun v -> Bigarray.Genarray.fill v nan);
    A.Any (Support.on b y)
  in
  let nan = { Complex.re = Float.nan; im = Float.nan } in
  match dt with
  | D.Complex64 -> fill D.Complex64 nan
  | D.Complex128 -> fill D.Complex128 nan
  | D.Float32 -> fill D.Float32 Float.nan
  | D.Float64 -> fill D.Float64 Float.nan
  | _ -> invalid_arg "poisoned"

(* [b]'s [kind] of [x] along [axes] into [dst], each of any of the transforms'
   dtypes, whether it fits or not. *)
let transform (b : Support.backend) kind ~axes (A.Any dst) (A.Any x) =
  let module K = (val b.kernels) in
  let axes = Iarray.of_array axes in
  let x = A.Any x and y = A.Any dst in
  match (kind, A.dtype dst) with
  | C2c d, D.Complex64 ->
      K.c2c d ~axes ~dst:(A.expect D.Complex64 y) (A.expect D.Complex64 x)
  | C2c d, D.Complex128 ->
      K.c2c d ~axes ~dst:(A.expect D.Complex128 y) (A.expect D.Complex128 x)
  | R2c, (D.Complex64 | D.Complex128) -> (
      let r2c (type s) (dt : (Complex.t, s) D.t) =
        let dst = A.expect dt y in
        match dtype_of x with
        | D.Any D.Float32 -> K.r2c ~axes ~dst (A.expect D.Float32 x)
        | _ -> K.r2c ~axes ~dst (A.expect D.Float64 x)
      in
      match A.dtype dst with
      | D.Complex64 -> r2c D.Complex64
      | _ -> r2c D.Complex128)
  | C2r _, (D.Float32 | D.Float64) -> (
      let c2r (type s) (dt : (float, s) D.t) =
        let dst = A.expect dt y in
        match dtype_of x with
        | D.Any D.Complex64 -> K.c2r ~axes ~dst (A.expect D.Complex64 x)
        | _ -> K.c2r ~axes ~dst (A.expect D.Complex128 x)
      in
      match A.dtype dst with D.Float32 -> c2r D.Float32 | _ -> c2r D.Float64)
  | _ -> invalid_arg "transform"

(* [b]'s result of [c] on [x], on its device into a fresh array, or into [dst]
   with [x] where they lie; [Done] or failing. [one_thread] runs nx.cpu on one
   thread. *)
let run (b : Support.backend) ?(one_thread = false) ?dst c (A.Any x) =
  let d, s = result c in
  let x, y =
    match dst with
    | Some y -> (A.Any x, y)
    | None -> (A.Any (Support.on b x), poisoned b d s)
  in
  let call () = transform b c.kind ~axes:c.axes y x in
  check (if one_thread then Nx_cpu.one_thread call else call ());
  y

let complexes (A.Any y) : Complex.t array =
  let y = Support.host y in
  match A.dtype y with
  | D.Complex64 -> A.to_array (A.expect D.Complex64 (A.Any y))
  | D.Complex128 -> A.to_array (A.expect D.Complex128 (A.Any y))
  | D.Float32 ->
      Array.map
        (fun re -> { Complex.re; im = 0. })
        (A.to_array (A.expect D.Float32 (A.Any y)))
  | D.Float64 ->
      Array.map
        (fun re -> { Complex.re; im = 0. })
        (A.to_array (A.expect D.Float64 (A.Any y)))
  | _ -> invalid_arg "complexes"

(* An error [e] within [bound]. windtrap orders NaN below every float, so a NaN
   error, from a NaN result element, reads as infinity and fails, as
   Nx_support.gap reads it; this library does not link nx. *)
let error_within bound e =
  at_most float_exact ~than:bound (if Float.is_nan e then infinity else e)

let norm (zs : Complex.t array) =
  sqrt
    (Array.fold_left
       (fun s (z : Complex.t) -> s +. (z.re *. z.re) +. (z.im *. z.im))
       0. zs)

(* ‖x‖₂ as the bounds read it: a c2r's √2 times its operand's, at least its
   completed bins'. *)
let operand_norm c =
  match c.kind with C2r _ -> sqrt 2. *. norm c.xs | _ -> norm c.xs

(* [b]'s result of [c] within its bound. *)
let within_bound b c =
  let x = operand c in
  let xs = complexes x in
  let got = complexes (run b c x) in
  let want =
    Array.map complex_of
      (exact ~wide:c.wide c.kind c.axes c.shape (Array.map dd_of xs))
  in
  let err = norm (Array.map2 Complex.sub want got) in
  let u = if c.wide then epsilon_float /. 2. else 0x1p-24 in
  let n = Array.fold_left (fun k a -> k * (points c).(a)) 1 c.axes in
  let bound =
    epsilon b c.kind c.axes c.shape
    *. u
    *. sqrt (Float.of_int n)
    *. operand_norm c
  in
  error_within bound err

(* Within the bound, by the prime factors of the points along each axis. *)
let law_bound b c =
  let lengths = Array.map (fun a -> (points c).(a)) c.axes in
  cover "a prime factor past 250" (Array.exists by_bluestein lengths);
  cover "a prime length"
    (Array.exists (fun n -> n > 1 && not (composite n)) lengths);
  cover "a composite length of several primes"
    (Array.exists
       (fun n -> List.length (List.sort_uniq compare (factors n 2)) > 1)
       lengths);
  cover "several axes" (Array.length c.axes > 1);
  cover "an operand broadcast along a transformed axis"
    (List.exists (fun a -> Array.mem a c.axes) c.view.broadcast);
  cover "a real line through complex ones of n points"
    (match c.kind with
    | R2c | C2r _ ->
        let n = (points c).(c.axes.(Array.length c.axes - 1)) in
        n mod 2 = 1 || n < 512
    | C2c _ -> false);
  cover "a real line halved, its half with a prime factor past 250"
    (match c.kind with
    | R2c | C2r _ ->
        let n = (points c).(c.axes.(Array.length c.axes - 1)) in
        n mod 2 = 0 && n >= 512 && by_bluestein (n / 2)
    | C2c _ -> false);
  within_bound b c

(* c2r cases along several axes whose bins 0 and n/2 hold, along the other axes,
   X[-j] = -conj X[j] and no other bin anything: the other axes' inverse
   transforms of those columns are imaginary, and the exact result is +0. *)
let opposite_gen =
  let open Gen in
  let length =
    of_list ~pp:Format.pp_print_int [ 2; 3; 4; 5; 6; 7; 9; 12; 15 ]
  in
  with_pp pp_case
  @@ let* others = array ~size:(int_range 1 2) length in
     let* n = int_range 1 4 in
     let* wide = bool in
     let+ seed = int in
     let rs = Random.State.make [| seed |] in
     let r = Array.length others in
     let s = Array.append others [| (n / 2) + 1 |] in
     let xs = Array.make (total s) Complex.zero in
     let mirror i =
       Array.mapi (fun a e -> if a = r then e else (s.(a) - e) mod s.(a)) i
     in
     Array.iteri
       (fun k _ ->
         let i = index_of s k in
         let o = flat s (mirror i) in
         if (i.(r) = 0 || 2 * i.(r) = n) && k <= o then begin
           let im = value wide rs in
           let re = if k = o then 0. else value wide rs in
           xs.(k) <- { Complex.re; im };
           xs.(o) <- { Complex.re = -.re; im }
         end)
       xs;
     let axes = Array.init (r + 1) Fun.id in
     { kind = C2r n; wide; shape = s; axes; view = plain (r + 1); xs }

(* Within the bound by √2 times the operand's norm: the completed bins' norm,
   that of the exact result over √N, is +0, and rounding in the other axes'
   passes leaves real parts. *)
let law_opposite b c =
  cover "several other axes" (Array.length c.shape > 2);
  cover "bin n/2" (match c.kind with C2r n -> n mod 2 = 0 | _ -> false);
  let x = operand c in
  let err = norm (complexes (run b c x)) in
  let u = if c.wide then epsilon_float /. 2. else 0x1p-24 in
  let n = Array.fold_left (fun k a -> k * (points c).(a)) 1 c.axes in
  let bound =
    epsilon b c.kind c.axes c.shape
    *. u
    *. sqrt (Float.of_int n)
    *. operand_norm c
  in
  error_within bound err

let specials = [| Float.nan; Float.infinity; Float.neg_infinity |]

(* Cases whose operand holds, half the time, NaNs and infinities at up to three
   drawn points. *)
let special_gen =
  let open Gen in
  with_pp pp_case
  @@ let* c = case_gen in
     let* count = frequency [ (1, constant 0); (1, int_range 1 3) ] in
     let+ seed = int in
     let n = Array.length c.xs in
     if n = 0 then c
     else begin
       let rs = Random.State.make [| seed |] and xs = Array.copy c.xs in
       for _ = 1 to count do
         let x = specials.(Random.State.int rs 3) in
         let j = Random.State.int rs n in
         xs.(j) <-
           (if c.kind <> R2c && Random.State.bool rs then
              { (xs.(j)) with im = x }
            else { (xs.(j)) with re = x })
       done;
       { c with xs }
     end

(* A result's parts as bits, every NaN as one: nx_cpu.mli leaves a NaN's sign
   and payload unspecified, and fixes the rest. *)
let bits_of y =
  let bits x =
    if Float.is_nan x then 0x7ff8000000000000L else Int64.bits_of_float x
  in
  Array.map (fun (z : Complex.t) -> (bits z.re, bits z.im)) (complexes y)

(* The same bits through every view, on one thread as on the job's, and for each
   line transformed alone, NaNs and infinities included. *)
let law_bits b c =
  let last = c.axes.(Array.length c.axes - 1) in
  let n = (points c).(last) and r = Array.length c.shape in
  cover "long lines side by side"
    (n > 64 && last < r - 1 && c.shape.(r - 1) >= 8);
  cover "a long line alone, its points apart"
    (n > 64 && last < r - 1 && c.shape.(r - 1) < 8);
  cover "a NaN or an infinity among the points"
    (Array.exists
       (fun (z : Complex.t) ->
         not (Float.is_finite z.re && Float.is_finite z.im))
       c.xs);
  let x = operand c in
  let plain = { c with view = plain r; xs = complexes x } in
  let px = operand plain in
  let (A.Any placed) = x in
  let placed = Support.on b placed in
  cover "an operand with gaps" (Support.gapped (A.layout placed));
  let bits = array (pair int64 int64) in
  let want = bits_of (run b plain px) in
  equal ~msg:"through the view" bits want (bits_of (run b c (A.Any placed)));
  equal ~msg:"on one thread" bits want
    (bits_of (run b ~one_thread:true plain px));
  (* Each line along a single axis, alone: its bits in the batch's. *)
  let s = snd (result c) in
  if Array.length c.axes = 1 && total s > 0 then begin
    let lines = total s / s.(last) in
    List.iter
      (fun l ->
        let i =
          index_of (Array.mapi (fun a e -> if a = last then 1 else e) s) l
        in
        let at j shape =
          let ij = Array.copy i in
          ij.(last) <- j;
          flat shape ij
        in
        let alone =
          {
            plain with
            shape = [| c.shape.(last) |];
            axes = [| 0 |];
            view = Elements.plain 1;
            xs = Array.init c.shape.(last) (fun j -> plain.xs.(at j c.shape));
          }
        in
        let got = bits_of (run b alone (operand alone)) in
        equal
          ~msg:(Printf.sprintf "line %d alone" l)
          bits
          (Array.init s.(last) (fun j -> want.(at j s)))
          got)
      (List.sort_uniq compare [ 0; lines - 1 ])
  end

(* A long line alone, its points 3 apart, which the generator draws in about one
   case of ten, and one of fifteen with its result's other axes past one
   element. *)
let bits_examples =
  [
    {
      kind = C2c Forward;
      wide = false;
      shape = [| 300; 3 |];
      axes = [| 0 |];
      view = plain 2;
      xs = draw (C2c Forward) ~wide:false 900 [| 300 |];
    };
  ]

(* Long lines *)

(* Lines past the generated cases' lengths, each a transform of three impulses,
   whose exact transform is a sum of twiddles: powers of two and other 7-smooth
   lengths, primes whose predecessor is 7-smooth or not (1009, 2017, 2161, 4051;
   4099, 10007, 65537), and real lines of even lengths whose halves are prime or
   7-smooth, each kind and width within the bound. 2161 and 4051 take Rader's
   convolution in one of nx.amd's complex64 groups and Bluestein's at
   complex128, whose groups hold half the points. *)
let long_lengths =
  [ 1009; 2017; 2161; 3125; 4051; 4096; 4099; 6561; 8192; 10007; 16384; 16807;
    19683; 20014; 30030; 65536; 65537; 131072 ]

(* A line of [n] points: its transform of [kind] and width. *)
let line_id (n, kind, wide) =
  match kind with
  | C2r _ -> Printf.sprintf "%s, %s" (kind_name kind) (width wide)
  | _ -> Printf.sprintf "%s of %d points, %s" (kind_name kind) n (width wide)

(* The roots e^(-2πi m/n), m < n, of a long line, made once for its transforms:
   past 4096 points in double, whose error is far below either width's bound at
   these lengths. *)
let line_units n =
  if n <= 4096 then units_of n
  else
    Array.init n (fun m ->
        let phi = 2. *. Float.pi *. (Float.of_int m /. Float.of_int n) in
        ((cos phi, 0.), (-.sin phi, 0.)))

(* [kind] of [n] points from three impulses at [ats] of coefficients [cs]:
   within [b]'s bound of their exact transform. *)
let check_impulses b ~wide kind n w ats cs =
  let shape, xs =
    match kind with
    | C2r m ->
        let h = (m / 2) + 1 in
        let xs = Array.make h Complex.zero in
        List.iter2
          (fun at z -> xs.(at mod h) <- Complex.add xs.(at mod h) z)
          ats cs;
        ([| h |], xs)
    | R2c | C2c _ ->
        let xs = Array.make n Complex.zero in
        List.iter2
          (fun at (z : Complex.t) ->
            let z = if kind = R2c then { z with im = 0. } else z in
            xs.(at) <- Complex.add xs.(at) z)
          ats cs;
        ([| n |], xs)
  in
  let c = { kind; wide; shape; axes = [| 0 |]; view = plain 1; xs } in
  let got = complexes (run b c (operand c)) in
  let impulses =
    List.filter
      (fun (_, (z : Complex.t)) -> z.re <> 0. || z.im <> 0.)
      (Array.to_list (Array.mapi (fun j z -> (j, z)) xs))
  in
  (* Each point against its sum of the impulses' terms, z_j e^(∓2πi jk/n). *)
  let err = ref 0. in
  Array.iteri
    (fun k (y : Complex.t) ->
      let re = ref 0. and im = ref 0. in
      List.iter
        (fun (j, (z : Complex.t)) ->
          let (ch, cl), (sh, sl) = w.(j * k mod n) in
          let c = ch +. cl and s = sh +. sl in
          match kind with
          | C2c Forward | R2c ->
              re := !re +. ((z.re *. c) -. (z.im *. s));
              im := !im +. ((z.re *. s) +. (z.im *. c))
          | C2c Inverse ->
              re := !re +. ((z.re *. c) +. (z.im *. s));
              im := !im +. ((z.im *. c) -. (z.re *. s))
          | C2r m ->
              (* Bin j and its conjugate n - j, or one bin at 0 or n / 2, whose
                 imaginary part takes no part. *)
              re :=
                !re
                +.
                if j = 0 || 2 * j = m then z.re *. c
                else 2. *. ((z.re *. c) +. (z.im *. s)))
        impulses;
      let dre = y.re -. !re and dim = y.im -. !im in
      err := !err +. (dre *. dre) +. (dim *. dim))
    got;
  let u = if wide then epsilon_float /. 2. else 0x1p-24 in
  let bound =
    epsilon b kind [| 0 |] shape *. u *. sqrt (Float.of_int n) *. operand_norm c
  in
  error_within bound (sqrt !err)

let impulses =
  Complex.
    [
      { re = 0.75; im = -0.5 };
      { re = -1.25; im = 0.25 };
      { re = 0.5; im = 1. };
    ]

(* Each kind and width of each long line, the line's roots made once for its
   transforms. *)
let long_lines =
  List.concat_map
    (fun n ->
      let w = lazy (line_units n) in
      List.concat_map
        (fun wide -> List.map (fun kind -> (n, kind, wide, w)) (kinds n))
        [ false; true ])
    long_lengths

let check_long b (n, kind, wide, w) =
  check_impulses b ~wide kind n (Lazy.force w) [ 3; n / 3; n - 1 ] impulses

(* An impulse at [at] of a line of [n] points of [dt], forward and inverse, each
   within the bound of its exact transform, the twiddles. The twiddles are
   computed in double, whose error is far below either bound at these
   lengths. *)
let impulse_line (type s) (b : Support.backend) (dt : (Complex.t, s) D.t) n at =
  let module K = (val b.kernels) in
  let x = A.of_array dt [| n |] (Array.make n Complex.zero) in
  A.set x [| at |] Complex.one;
  let x = Support.on b x in
  let check_line direction =
    let y = A.create b.device dt [| n |] in
    check (K.c2c direction ~axes:[| 0 |] ~dst:y x);
    let sign =
      match direction with Nx_kernel.Forward -> -1. | Inverse -> 1.
    in
    let err = ref 0. in
    A.read (Support.host y) @@ fun y ->
    let y = Bigarray.reshape_1 y n in
    for k = n - 1 downto 0 do
      let (z : Complex.t) = y.{k} in
      let phi =
        sign *. 2. *. Float.pi *. float_of_int (at * k mod n) /. float_of_int n
      in
      let dre = z.re -. cos phi and dim = z.im -. sin phi in
      err := !err +. (dre *. dre) +. (dim *. dim)
    done;
    let u = match dt with D.Complex64 -> 0x1p-24 | _ -> epsilon_float /. 2. in
    let bound =
      epsilon b (C2c direction) [| 0 |] [| n |] *. u *. sqrt (float_of_int n)
    in
    error_within bound (sqrt !err)
  in
  check_line Forward;
  check_line Inverse

(* A complex64 line of 2^22 points, 32 MiB, where on Linux nx.cpu's four-step
   grid lies in its result where the largest cache is at most 32 MiB. *)
let test_past_caches b () = impulse_line b D.Complex64 (1 lsl 22) 12345

(* A complex128 line of 4,199,040 = 2^7 3^8 5 points, 64 MiB, the least 7-smooth
   length past 2048^2. *)
let test_wide_line b () = impulse_line b D.Complex128 4199040 54321

(* A complex64 line of 2^23 points, 64 MiB, of two factors of 4096 and 2048
   points. *)
let test_long_line b () = impulse_line b D.Complex64 (1 lsl 23) 7654321

(* The first element where [y] and [z], of one dtype and shape, differ in their
   bits, if any. *)
let differs (A.Any y) (A.Any z) =
  let w = D.bytes (A.dtype y) 1 in
  let y = A.copy (Support.host y) and z = A.copy (Support.host z) in
  Rig.Buffer.read (A.buffer y) Bigarray.char @@ fun y ->
  Rig.Buffer.read (A.buffer z) Bigarray.char @@ fun z ->
  let n = Bigarray.Array1.dim y in
  let rec from k =
    if k = n then None else if y.{k} = z.{k} then from (k + 1) else Some (k / w)
  in
  from 0

(* A random complex64 line of 3^14 points, 38 MiB, with the same bits as its
   values shaped [1; n] and transformed along both axes, forward and inverse, on
   the job's threads and on one. On Linux, where the largest cache is under 38
   MiB, the line's grid lies in its result, and the second axis's, transformed
   in place, in scratch. Its n1 and n2 are odd, so both steps fill their last
   lanes in part. *)
let test_grid_homes (b : Support.backend) () =
  let module K = (val b.kernels) in
  let n = 4782969 in
  let v = Bigarray.Array1.create Bigarray.complex32 Bigarray.c_layout n in
  let rs = Random.State.make [| 14 |] in
  let part () = Random.State.float rs 2. -. 1. in
  for k = 0 to n - 1 do
    v.{k} <- { Complex.re = part (); im = part () }
  done;
  let v = Bigarray.genarray_of_array1 v in
  let line = Support.on b (A.of_bigarray D.Complex64 v)
  and row =
    Support.on b (A.of_bigarray D.Complex64 (Bigarray.reshape v [| 1; n |]))
  in
  let c2c direction axes x s () =
    let y = A.create b.device D.Complex64 s in
    check (K.c2c direction ~axes ~dst:y x);
    A.Any (Support.host y)
  in
  List.iter
    (fun direction ->
      let by_row = c2c direction [| 0; 1 |] row [| 1; n |]
      and by_line = c2c direction [| 0 |] line [| n |] in
      let want = by_row () in
      let check msg y = equal ~msg (option int) None (differs want y) in
      check "the row on one thread" (Nx_cpu.one_thread by_row);
      check "the line" (by_line ());
      check "the line on one thread" (Nx_cpu.one_thread by_line))
    [ Nx_kernel.Forward; Inverse ]

(* [kind]'s operand of shape [s], plain, its points drawn from [seed] as [draw]
   draws them, written through a bigarray: a long line's, made without its
   elements' strings. *)
let drawn kind ~wide s seed =
  let rs = Random.State.make seed in
  let fill (type v w) (dt : (v, w) D.t) (point : unit -> v) =
    let x = A.create Rig.host dt (Iarray.of_array s) in
    A.write x (fun v ->
        let v = Bigarray.reshape_1 v (total s) in
        for k = 0 to total s - 1 do
          v.{k} <- point ()
        done);
    A.Any x
  in
  let part () = value wide rs in
  let point () =
    let re = part () in
    { Complex.re; im = part () }
  in
  match (kind, wide) with
  | R2c, false -> fill D.Float32 part
  | R2c, true -> fill D.Float64 part
  | _, false -> fill D.Complex64 point
  | _, true -> fill D.Complex128 point

(* [kind] along the first axis of an operand of shape [s] that {!drawn}
   makes. *)
let along_first kind ~wide s =
  {
    kind;
    wide;
    shape = s;
    axes = [| 0 |];
    view = plain (Array.length s);
    xs = [||];
  }

(* A forward c2c of [lines] lines of [n] points, side by side, drawn from
   [seed]. *)
let line ~wide n lines seed =
  let shape = if lines = 1 then [| n |] else [| n; lines |] in
  let xs = draw (C2c Forward) ~wide (n * lines) [| seed |] in
  { (along_first (C2c Forward) ~wide shape) with xs }

(* Transforms of a MiB and past, which nx.cpu runs with the lock released: a
   line of each complex dtype and lines side by side. *)
let streamed () =
  [
    line ~wide:true (1 lsl 16) 1 19;
    line ~wide:false (1 lsl 17) 1 22;
    line ~wide:false 1024 136 10;
  ]

let pp_line ppf (kind, wide, s) =
  Format.fprintf ppf "%s, %s, of %a" (kind_name kind) (width wide) pp_ints s

(* The streamed transforms, and one of each regime whose steps are jobs of their
   own: Bluestein's convolution (65537 points), a real line through complex ones
   of half its points, each way (131072), and three steps, an inverse (4,199,040
   complex128 points). *)
let threaded =
  [
    (C2c Forward, true, [| 1 lsl 16 |]);
    (C2c Forward, false, [| 1 lsl 17 |]);
    (C2c Forward, false, [| 1024; 136 |]);
    (C2c Forward, false, [| 65537 |]);
    (R2c, false, [| 131072 |]);
    (C2r 131072, true, [| 65537 |]);
    (C2c Inverse, true, [| 4199040 |]);
  ]

(* The same bits on the job's threads as on one. *)
let check_threads b (kind, wide, s) =
  let c = along_first kind ~wide s in
  let x = drawn kind ~wide s [| total s |] in
  equal ~msg:"the first element whose bits differ on one thread" (option int)
    None
    (differs (run b c x) (run b ~one_thread:true c x))

(* For [c]'s lines side by side along its first axis: the bits of each
   line in the batch, the same as the lines' one after another, and as the
   first, middle and last line's alone. *)
let same_alone b c =
  let bits = array (pair int64 int64) in
  let n = c.shape.(0) and lines = c.shape.(1) in
  let ys = bits_of (run b c (operand c)) in
  let rows =
    {
      c with
      shape = [| lines; n |];
      axes = [| 1 |];
      xs = Array.init (n * lines) (fun i -> c.xs.((i mod n * lines) + (i / n)));
    }
  in
  equal ~msg:"one after another" bits
    (Array.init (n * lines) (fun i -> ys.((i mod n * lines) + (i / n))))
    (bits_of (run b rows (operand rows)));
  List.iter
    (fun l ->
      let alone =
        {
          c with
          shape = [| n |];
          view = plain 1;
          xs = Array.init n (fun j -> c.xs.((j * lines) + l));
        }
      in
      equal
        ~msg:(Printf.sprintf "line %d alone" l)
        bits
        (Array.init n (fun j -> ys.((j * lines) + l)))
        (bits_of (run b alone (operand alone))))
    [ 0; lines / 2; lines - 1 ]

(* Lines side by side past a GPU group's points, 4096 at complex64 and 2048 at
   complex128 on nx.amd, which take several steps there, and a MiB of lines,
   which nx.cpu transforms with the lock released. *)
let batches = [ (6561, 8, false); (3125, 8, true); (1024, 136, false) ]
let check_batch b (n, lines, wide) = same_alone b (line ~wide n lines n)

(* Every short length *)

(* Three lines of every length from 0 to 70, of each kind and width, along the
   last axis or, at odd lengths, side by side along the first: within the bound,
   every destination element written. *)
let short_lines =
  List.concat_map
    (fun n ->
      List.concat_map
        (fun wide -> List.map (fun kind -> (n, kind, wide)) (kinds n))
        [ false; true ])
    (List.init 71 Fun.id)

let check_short b (n, kind, wide) =
  let e = match kind with C2r _ -> (n / 2) + 1 | _ -> n in
  let shape, axes =
    if n mod 2 = 0 then ([| 3; e |], [| 1 |]) else ([| e; 3 |], [| 0 |])
  in
  let xs = draw kind ~wide (total shape) [| n |] in
  within_bound b { kind; wide; shape; axes; view = plain 2; xs }

(* Complex arrays at their parts' alignment *)

(* [x]'s elements in a fresh array on [b]'s device whose first byte lies one
   part past twice a part's alignment: 4 mod 8 for complex64, 8 mod 16 for
   complex128; a real array as it is. nx_array.mli promises an array only its
   parts' alignment. *)
let off_pairs (b : Support.backend) (A.Any x) =
  let dt = A.dtype x in
  let part = match dt with D.Complex64 -> 4 | D.Complex128 -> 8 | _ -> 0 in
  let s = Iarray.to_array (L.shape (A.layout x)) in
  let bytes = D.bytes dt (total s) in
  let buf = Rig.Buffer.create b.device (bytes + (2 * part) + 16) in
  let view = Rig.Buffer.view buf ~first:part ~length:bytes in
  let a = A.v dt (L.contiguous (Iarray.of_array s)) view in
  A.transfer ~src:x ~dst:a;
  A.Any a

(* Each kind and width from an operand and into a destination off their pairs'
   alignment, of lines that fit a group and of long ones, which take a real
   line's split or pre-pass launch: the bits of the aligned transform. *)
let off_pair_lines =
  List.concat_map
    (fun wide ->
      List.map
        (fun (kind, shape, axes) -> (kind, wide, shape, axes))
        [
          (C2c Forward, [| 3; 64 |], [| 1 |]);
          (C2c Inverse, [| 70000 |], [| 0 |]);
          (C2c Forward, [| 64; 5 |], [| 0; 1 |]);
          (R2c, [| 3; 64 |], [| 1 |]);
          (R2c, [| 131072 |], [| 0 |]);
          (C2r 64, [| 3; 33 |], [| 1 |]);
          (C2r 131072, [| 65537 |], [| 0 |]);
          (C2r 9, [| 4; 5 |], [| 0; 1 |]);
        ])
    [ false; true ]

let pp_lines ppf (kind, wide, shape, axes) =
  Format.fprintf ppf "%s %s of %a along %a" (kind_name kind) (width wide)
    pp_ints shape pp_ints axes

let check_off_pairs b (kind, wide, shape, axes) =
  let xs = draw kind ~wide (total shape) [| total shape |] in
  let c = { kind; wide; shape; axes; view = plain (Array.length shape); xs } in
  let x = operand c in
  let want = bits_of (run b c x) in
  let D.Any dt, s = result c in
  let dst = off_pairs b (A.Any (A.create b.device dt (Iarray.of_array s))) in
  equal (array (pair int64 int64)) want (bits_of (run b ~dst c (off_pairs b x)))

(* The bins a c2r ignores *)

(* The imaginary parts of bin 0 and, at an even n, of bin n / 2 take no part in
   a c2r along one axis: NaN there gives the bits +0 there gives, which are
   finite. *)
let ignored_lengths = [ 1; 2; 3; 8; 9; 64; 1000; 1024; 20014; 32768 ]

let check_ignored b (n, wide) =
  let h = (n / 2) + 1 in
  let xs = draw (C2r n) ~wide (3 * h) [| n |] in
  let with_im v =
    Array.mapi
      (fun i (z : Complex.t) ->
        let k = i mod h in
        if k = 0 || (n mod 2 = 0 && k = n / 2) then { z with im = v } else z)
      xs
  in
  let got xs =
    let c =
      {
        kind = C2r n;
        wide;
        shape = [| 3; h |];
        axes = [| 1 |];
        view = plain 2;
        xs;
      }
    in
    bits_of (run b c (operand c))
  in
  equal (array (pair int64 int64)) (got (with_im 0.)) (got (with_im Float.nan))

(* A c2r along axes [0; 1] of [3; 2] bins whose column 0 is i c (1, w, w²), w =
   e^(-2πi/3), and column 1 zero: the first axis's inverse transform leaves
   column 0 all imaginary, which the last axis drops, so the exact result is 0
   and every rounding of the first axis errs against it. Within the bound, at c
   = 1 and c = 2^40. *)
let check_dropped b (wide, c) =
  let h = sqrt 3. /. 2. *. c in
  let xs =
    Complex.
      [|
        { re = 0.; im = c };
        zero;
        { re = h; im = -0.5 *. c };
        zero;
        { re = -.h; im = -0.5 *. c };
        zero;
      |]
  in
  within_bound b
    {
      kind = C2r 2;
      wide;
      shape = [| 3; 2 |];
      axes = [| 0; 1 |];
      view = plain 2;
      xs;
    }

(* Long lines on later axes *)

(* Long lines along the axis transformed second, so read from the destination,
   and a c2r's other axis into its copy of the operand. The long axis holds two
   impulses, each a line of drawn values along the short one, so the long axis's
   exact transform is a sum of twiddles. *)
let check_later b (kind, n0, n1, wide) =
  let e1 = match kind with C2r m -> (m / 2) + 1 | _ -> n1 in
  let shape = [| n0; e1 |] and rows = [ 1; (n0 / 3) + 1 ] in
  let rs = Random.State.make [| n0; n1 |] in
  let xs = Array.make (n0 * e1) Complex.zero in
  List.iter
    (fun p ->
      for l = 0 to e1 - 1 do
        let re = value wide rs in
        let im = match kind with R2c -> 0. | _ -> value wide rs in
        xs.((p * e1) + l) <- { Complex.re; im }
      done)
    rows;
  let c = { kind; wide; shape; axes = [| 0; 1 |]; view = plain 2; xs } in
  let w = line_units n0 in
  let root m =
    complex_of
      (let re, im = w.(m mod n0) in
       { re; im })
  in
  (* Along the long axis: Σ over the impulses' rows of their values times the
     twiddles, conjugated for an inverse. *)
  let long ~inverse m (v : Complex.t array) =
    Array.init (n0 * m) (fun i ->
        let k = i / m and l = i mod m in
        List.fold_left
          (fun acc p ->
            let r = root (p * k) in
            let r = if inverse then Complex.conj r else r in
            Complex.add acc (Complex.mul r v.((p * m) + l)))
          Complex.zero rows)
  in
  let short kind s v =
    Array.map complex_of (exact ~wide kind [| 1 |] s (Array.map dd_of v))
  in
  let want =
    match kind with
    | C2c d -> long ~inverse:(d = Inverse) n1 (short kind shape xs)
    | R2c -> long ~inverse:false ((n1 / 2) + 1) (short R2c shape xs)
    | C2r _ -> short kind shape (long ~inverse:true e1 xs)
  in
  let got = complexes (run b c (operand c)) in
  let err = norm (Array.map2 Complex.sub want got) in
  let u = if wide then epsilon_float /. 2. else 0x1p-24 in
  let n = n0 * match kind with C2r m -> m | _ -> n1 in
  let bound =
    epsilon b kind [| 0; 1 |] shape
    *. u
    *. sqrt (Float.of_int n)
    *. operand_norm c
  in
  error_within bound err

let later_lines =
  List.concat_map
    (fun wide ->
      List.map
        (fun (kind, n0, n1) -> (kind, n0, n1, wide))
        [
          (C2c Forward, 16384, 3);
          (C2c Inverse, 10007, 3);
          (R2c, 10007, 4);
          (C2r 5, 16384, 5);
          (C2r 3, 10007, 3);
          (C2r 10, 10007, 10);
        ])
    [ false; true ]

(* Guard pages *)

(* The same bits with the operand, through its view, and the result each against
   an inaccessible page. *)
let law_guards b (c, side) =
  let after = side = Support.After in
  cover "a stepped operand's first element at the page before"
    (c.view.stepped && not after);
  cover "a reversed operand at the page after" (c.view.reversed <> None && after);
  let a = c.axes.(0) in
  let n = (points c).(a) in
  let s = snd (result c) and r = Array.length c.shape in
  cover "a long line" (n > 64);
  cover "long lines side by side" (n > 64 && a < r - 1 && s.(r - 1) >= 8);
  cover "a long line alone, its points apart"
    (n > 64
    && a < r - 1
    && s.(r - 1) < 8
    && total (Array.sub s (a + 1) (r - a - 1)) > 1);
  let call xs ds =
    ignore (run b ~dst:ds.(0) c xs.(0));
    A.Done
  in
  ignore (same_guarded side ~call [| operand c |] [| result c |])

(* The streamed transforms, on one thread and on the job's, each against the
   page. *)
let test_streamed_guards b () =
  List.iter
    (fun c ->
      List.iter
        (fun side ->
          List.iter
            (fun one_thread ->
              let call xs ds =
                ignore (run b ~one_thread ~dst:ds.(0) c xs.(0));
                A.Done
              in
              ignore (same_guarded side ~call [| operand c |] [| result c |]))
            [ false; true ])
        Support.[ Before; After ])
    (streamed ())

(* Every target's bits *)

(* Digests of lines whose lengths take each stated regime of nx.cpu: small
   primes and their products, powers of two, primes past 250 and their doubles,
   real lines halved and not, odd radices and long lines of one prime; each the
   same on every target and every host architecture, as nx_cpu.mli promises, so
   one baseline holds on every host. *)
let lengths =
  [
    7;
    60;
    64;
    241;
    100;
    1000;
    1024;
    4096;
    257;
    1031;
    514;
    512;
    1001;
    59049;
    65536;
    255;
    504;
  ]

(* A line of each kind, forward, inverse, r2c and c2r, of [n] points drawn from
   a seed of [n]. *)
let lines wide n =
  List.map
    (fun kind ->
      let s = match kind with C2r n -> [| (n / 2) + 1 |] | _ -> [| n |] in
      let xs = draw kind ~wide s.(0) [| n |] in
      { kind; wide; shape = s; axes = [| 0 |]; view = plain 1; xs })
    (kinds n)

(* [table digest] is a row of [digest wide n] per length and width. *)
let table digest =
  List.concat_map
    (fun n ->
      List.map
        (fun wide ->
          let d = Digest.to_hex (Digest.string (digest wide n)) in
          Printf.sprintf "%d %s %s" n
            (if wide then "f64" else "f32")
            (String.sub d 0 16))
        [ false; true ])
    lengths
  |> String.concat "\n"

let test_digests b () =
  let one c =
    String.concat "" (Array.to_list (elements (run b c (operand c))))
  in
  let t =
    table (fun wide n -> String.concat "" (List.map one (lines wide n)))
  in
  expect t
  @@ __POS_OF__
       {|
    7 f32 25fc4490bddde05e
    7 f64 103173224a3c62a3
    60 f32 d3a7aaecc9170471
    60 f64 a09269231c08e518
    64 f32 03980dbe5fc31474
    64 f64 4e11c0ee2fcd11fa
    241 f32 b35843e8d1006799
    241 f64 7c89018ba17c56a4
    100 f32 3e380cd2344fc918
    100 f64 843090a5d39338e3
    1000 f32 e602dba7a6c0bceb
    1000 f64 429f0b3827908fdc
    1024 f32 8e7ebe6021ecff18
    1024 f64 32a0959d1c06a257
    4096 f32 565572e3e5171eca
    4096 f64 d61210f2d5bb445a
    257 f32 1299a99be866a7a0
    257 f64 0deee0529736803a
    1031 f32 82ef31e36a702475
    1031 f64 ee8d9a2b3e80e82a
    514 f32 2fb192795424f835
    514 f64 ad9633a6b7805e61
    512 f32 632bd1f0c75e1b18
    512 f64 deadd438cd4674f9
    1001 f32 505c62414d34ca77
    1001 f64 90073e7e2eedfe4e
    59049 f32 b14bbed8caf9add0
    59049 f64 bad69298c8cb58ad
    65536 f32 daf6c7d86798f76a
    65536 f64 a781a7e486c3fc67
    255 f32 d21f36d15c90f6d4
    255 f64 d180e6b9aaee8921
    504 f32 e20bf75bdb2c7c07
    504 f64 95a710c363c304dc
    |}

(* The same lines with an infinity at their second point, every NaN as one as in
   [bits_of]. Which results are infinite or NaN, and the infinities' signs,
   follow from the order of each line's operations, which the finite digests do
   not see. *)
let test_infinity_digests b () =
  let with_inf c =
    let xs = Array.copy c.xs in
    xs.(1) <- { Complex.re = Float.infinity; im = 0.5 };
    { c with xs }
  in
  let bits buf c =
    Array.iter
      (fun (re, im) ->
        Buffer.add_int64_le buf re;
        Buffer.add_int64_le buf im)
      (bits_of (run b c (operand c)))
  in
  let t =
    table (fun wide n ->
        let buf = Buffer.create 4096 in
        List.iter (fun c -> bits buf (with_inf c)) (lines wide n);
        Buffer.contents buf)
  in
  expect t
  @@ __POS_OF__
       {|
    7 f32 469a74b37861fb26
    7 f64 da88bd8c1e14d4bc
    60 f32 767a346ef50146f7
    60 f64 924303ca70291f73
    64 f32 f83260284e23a5e9
    64 f64 64bfeeccc0be915c
    241 f32 2505f0b74c41baee
    241 f64 dcd8f319d8479071
    100 f32 27748f9f8de48ff3
    100 f64 27748f9f8de48ff3
    1000 f32 4b2096ae677b758d
    1000 f64 352d09137fc230bc
    1024 f32 fa9888ed238f1aef
    1024 f64 289c35530d58f9c2
    4096 f32 868a3d6c27445ed4
    4096 f64 868a3d6c27445ed4
    257 f32 6a6ec49f03c4f2fb
    257 f64 6a6ec49f03c4f2fb
    1031 f32 8567975d9fe983ff
    1031 f64 8567975d9fe983ff
    514 f32 42d19b8019ba0f47
    514 f64 42d19b8019ba0f47
    512 f32 27ffd1dd0269d789
    512 f64 27ffd1dd0269d789
    1001 f32 409f5cbd2bcbfbce
    1001 f64 409f5cbd2bcbfbce
    59049 f32 0d6240a261907a87
    59049 f64 0d6240a261907a87
    65536 f32 9a9785afd0d8a399
    65536 f64 9a9785afd0d8a399
    255 f32 9087cf9d303bf0e6
    255 f64 9087cf9d303bf0e6
    504 f32 3617586aea1783a1
    504 f64 3617586aea1783a1
    |}

(* Empty transforms and refusals *)

(* Of no point along a transformed axis: an r2c's one bin of +0 into a
   destination of NaNs; a c2c and a c2r with no result element, whose claim is
   Done. *)
let empty_cases =
  [
    {
      kind = R2c;
      wide = false;
      shape = [| 3; 0 |];
      axes = [| 0; 1 |];
      view = plain 2;
      xs = [||];
    };
    {
      kind = C2c Inverse;
      wide = true;
      shape = [| 0; 4 |];
      axes = [| 1 |];
      view = plain 2;
      xs = [||];
    };
    {
      kind = C2r 0;
      wide = false;
      shape = [| 2; 1 |];
      axes = [| 1 |];
      view = plain 2;
      xs = Array.make 2 Complex.one;
    };
  ]

let check_empty b c =
  let d, s = result c in
  equal (array string)
    (Array.make (total s) (Elements.zero d))
    (elements (run b c (operand c)))

(* Transforms that do not fit: [kind] of an operand of [x] along [axes] into a
   destination of [dst], a misfit of the class [what]. *)
type misfit = {
  msg : string;
  what : string;
  of_kind : kind;
  x : D.any * int array;
  along : int array;
  dst : D.any * int array;
}

let misfits =
  let row msg what of_kind x along dst =
    { msg; what; of_kind; x; along; dst }
  in
  let c64 = D.Any D.Complex64 and c128 = D.Any D.Complex128 in
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 in
  [
    row "r2c of float32 into complex128" "a dtype" R2c (f32, [| 4 |]) [| 0 |]
      (c128, [| 3 |]);
    row "c2r of complex64 into float64" "a dtype" (C2r 4) (c64, [| 3 |]) [| 0 |]
      (f64, [| 4 |]);
    row "c2c along no axis" "an attribute" (C2c Forward) (c64, [| 4 |]) [||]
      (c64, [| 4 |]);
    row "c2c along decreasing axes" "an attribute" (C2c Forward)
      (c64, [| 2; 2 |])
      [| 1; 0 |]
      (c64, [| 2; 2 |]);
    row "c2c along an axis past the rank" "an attribute" (C2c Forward)
      (c64, [| 4 |]) [| 1 |] (c64, [| 4 |]);
    row "c2c into another shape" "a shape" (C2c Inverse) (c128, [| 4 |]) [| 0 |]
      (c128, [| 5 |]);
    row "r2c into all the bins" "a shape" R2c (f32, [| 4 |]) [| 0 |]
      (c64, [| 4 |]);
    row "c2r of other bins than its points'" "a shape" (C2r 4) (c128, [| 4 |])
      [| 0 |] (f64, [| 4 |]);
    row "c2r along an axis past the rank" "an attribute" (C2r 4) (c128, [| 3 |])
      [| 2 |] (f64, [| 4 |]);
  ]

let check_misfit (b : Support.backend) m =
  let D.Any dt, s = m.x in
  let x = A.create b.device dt (Iarray.of_array s) in
  Misfit.check ~msg:m.msg ~what:m.what b [| m.dst |] (fun g ds ->
      transform b m.of_kind ~axes:m.along ds.(0) (A.Any (g.Misfit.g x)))

(* A c2c into the bytes of its operand shifted by one point: the door refuses
   such sharing as Overlapping. *)
let test_overlapping (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.create b.device D.Complex64 [| 5 |] in
  let from start =
    Option.get (A.move (M.Slice [| { M.start; count = 4; step = 1 } |]) x)
  in
  equal answer A.Overlapping
    (K.c2c Forward ~axes:[| 0 |] ~dst:(from 1) (from 0))

(* Past 2^31 elements *)

(* A c2r into float32 [2^21 + 1; 1024], 8 GiB, its last element at 2^31 + 1023,
   from bins broadcast along the first axis: sampled lines have the first line's
   bits, and the first is within the bound. A device holds it where the host
   would need 8 GiB. *)
let test_past_2_31 (b : Support.backend) () =
  let module K = (val b.kernels) in
  let lines = (1 lsl 21) + 1 and n = 1024 in
  let h = (n / 2) + 1 in
  let rs = Random.State.make [| 31 |] in
  let row =
    Array.init h (fun _ ->
        let re = value false rs in
        { Complex.re; im = value false rs })
  in
  let bins = Support.on b (A.of_array D.Complex64 [| h |] row) in
  let x =
    A.v D.Complex64 (L.v ~strides:[| 0; 1 |] [| lines; h |]) (A.buffer bins)
  in
  let y = A.create b.device D.Float32 [| lines; n |] in
  check (K.c2r ~axes:[| 1 |] ~dst:y x);
  let line l =
    A.to_array
      (Support.host
         (A.v D.Float32
            (L.v ~offset:(l * n) ~strides:[| 1 |] [| n |])
            (A.buffer y)))
  in
  let first = line 0 in
  let bits = Array.map Int32.bits_of_float in
  List.iter
    (fun l -> equal (array int32) (bits first) (bits (line l)))
    [ 1; lines / 2; lines - 1 ];
  let c =
    {
      kind = C2r n;
      wide = false;
      shape = [| h |];
      axes = [| 0 |];
      view = plain 1;
      xs = row;
    }
  in
  let want =
    Array.map complex_of
      (exact ~wide:false c.kind c.axes c.shape (Array.map dd_of row))
  in
  let err =
    norm
      (Array.map2
         (fun (w : Complex.t) re -> Complex.sub w { re; im = 0. })
         want first)
  in
  let bound =
    epsilon b c.kind c.axes c.shape
    *. 0x1p-24
    *. sqrt (Float.of_int n)
    *. operand_norm c
  in
  error_within bound err

(* The bytes of device memory nx_cuda.mli states nx.cuda keeps for later calls
   on a device. *)
let cuda_kept = 192 * 1024 * 1024

(* Lines of primes past 2^19, each by Bluestein's convolution, whose plans at
   complex128, a chirp of 8 MiB and a kernel of 16 MiB or more each, take more
   than the 64 MiB nx_cuda.mli states the plans keep. *)
let large_plans = [ 524309; 524341; 524347; 524351; 524353 ]

let[@inline never] transform_line (b : Support.backend) n =
  let module K = (val b.kernels) in
  let x = A.create b.device D.Complex128 [| n |] in
  let y = A.create b.device D.Complex128 [| n |] in
  check (K.c2c Forward ~axes:[| 0 |] ~dst:y x);
  Rig.Buffer.wait (A.buffer y) Read_write

(* nx.cuda keeps at most [cuda_kept] bytes of device memory for later calls,
   however many large lines it has planned. *)
let test_kept_plans (b : Support.backend) () =
  (* The first call loads the library's code on the device. *)
  transform_line b 64;
  let before = Support.settled b.device in
  List.iter (transform_line b) large_plans;
  at_most ~msg:"bytes kept" int ~than:cuda_kept
    (Support.settled b.device - before)

(* 32 lines of the prime 131101, whose predecessor is not 7-smooth, each by
   Bluestein's convolution: about the workspace of four lines of a prime past
   2^20, whose plan the host makes in a tenth of the time. nx_cuda.mli states at
   most [3m] complex points a line, [m <= 4n]. The bound counts that workspace
   in the driver's units ({!Rig.allocation}); the line's plan is one the device
   keeps. *)
let test_loop_workspace (b : Support.backend) () =
  let module K = (val b.kernels) in
  let lines = 32 and n = 131101 in
  let x = A.create b.device D.Complex64 [| lines; n |] in
  let y = A.create b.device D.Complex64 [| lines; n |] in
  let m = 4 * n in
  let bound = Rig.allocation b.device (lines * 3 * m * 8) in
  Kernels_cases.loop_held b ~bound (fun () ->
      check (K.c2c Forward ~axes:[| 1 |] ~dst:y x);
      Rig.Buffer.wait (A.buffer y) Read_write)

let laws (d : Support.declared) =
  let around f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  let test name f = test name (fun () -> around f ()) in
  let cpu = d.label = Support.cpu.name in
  (* Guard pages lie in host memory: a backend computing elsewhere has none. *)
  let guards =
    if not cpu then []
    else
      [
        group
          "reads and writes no byte past the operand and the result, whatever \
           pages lie beside them"
          [
            prop ~count:300
              ~examples:(List.map (fun c -> (c, Support.Before)) bits_examples)
              "through views" (Gen.pair case_gen sides) (around law_guards);
            test "past the size the lock is released for" test_streamed_guards;
          ];
      ]
  in
  (* nx.cpu's bits, the same on every target. *)
  let targets =
    if not cpu then []
    else
      [
        test "on every target and host architecture" test_digests;
        test
          "on every target and host architecture, infinities among the points"
          test_infinity_digests;
      ]
  in
  group d.label
    ([
       group "transforms are within the library's bound"
         [
           prop ~examples:bound_examples "by the prime factors of the points"
             case_gen (around law_bound);
           prop "bins whose other axes' transforms are imaginary" opposite_gen
             (around law_opposite);
           cases ~id:line_id "every length to 70, every element written"
             short_lines (around check_short);
           cases
             ~id:(fun (n, kind, wide, _) -> line_id (n, kind, wide))
             "long lines" long_lines (around check_long);
           cases
             ~id:(fun (kind, n0, n1, wide) ->
               line_id (n0, kind, wide)
               ^ Printf.sprintf ", lines of %d points" n1)
             "long lines on later axes" later_lines (around check_later);
           cases
             ~id:(fun (wide, c) -> Printf.sprintf "%s, c = %h" (width wide) c)
             "a c2r of bins no real line has"
             (List.concat_map
                (fun wide -> [ (wide, 1.); (wide, 0x1p40) ])
                [ false; true ])
             (around check_dropped);
           test "a line past the caches" test_past_caches;
           test "a complex128 line of three steps" test_wide_line;
           test "a complex64 line of 2^23 points" test_long_line;
           cases
             ~id:(Format.asprintf "%a" pp_case)
             "a transform of no point is the empty sum" empty_cases
             (around check_empty);
         ];
       cases
         ~id:(fun (n, wide) -> line_id (n, C2r n, wide))
         "a c2r ignores the imaginary parts of bins 0 and n / 2"
         (List.concat_map (fun n -> [ (n, false); (n, true) ]) ignored_lengths)
         (around check_ignored);
       cases
         ~id:(Format.asprintf "%a" pp_lines)
         "operands and destinations at their parts' alignment" off_pair_lines
         (around check_off_pairs);
       group "bits are a function of the values alone"
         ([
            prop ~examples:bits_examples "whatever the views, threads or batch"
              special_gen (around law_bits);
            cases
              ~id:(Format.asprintf "%a" pp_line)
              "whatever the threads, past the size the lock is released for"
              threaded (around check_threads);
            cases
              ~id:(fun (n, lines, wide) ->
                Printf.sprintf "%d lines of %d points, %s" lines n (width wide))
              "whatever the batch, lines of several steps or of a MiB" batches
              (around check_batch);
            test "whatever the home of a long line's grid" test_grid_homes;
          ]
         @ targets);
       cases
         ~id:(fun m -> m.msg)
         "a transform of dtypes, axes or shapes that do not fit is not Done \
          and writes nothing outside dst"
         misfits (around check_misfit);
       test "a destination over its operand's bytes is refused as Overlapping"
         test_overlapping;
     ]
    @ guards
    @ (if cpu then [] else [ test "past 2^31 elements" test_past_2_31 ])
    @
    if d.label <> "cuda" then []
    else
      [
        test "keeps at most 192 MiB of device memory for later calls"
          test_kept_plans;
        test "calls in a loop hold one call's workspace" test_loop_workspace;
      ])

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so a transform through it has CUDA's bits:
   what it can get wrong, its launches, memory and order, shows in them. Each
   case runs through CUDA, then through rig.nv. A transform past 2^31 elements
   adds the kernels' index arithmetic, and runs through CUDA alone. A NaN agrees
   with any NaN: nx_cuda.mli leaves a NaN result's sign and payload
   unspecified. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let agree c x =
    Elements.agree (Support.opened first) (Support.opened nv) (fun b ->
        [| run b c x |])
  in
  (* A line of [n] points of [kind], drawn. *)
  let line (n, kind, wide) =
    let s = [| (match kind with C2r m -> (m / 2) + 1 | _ -> n) |] in
    agree (along_first kind ~wide s) (drawn kind ~wide s [| n |])
  in
  group nv.label
    [
      group "bits are cuda's, any NaN for a NaN"
        [
          prop ~examples:bound_examples
            "of transforms by the prime factors of the points" case_gen
            (fun c ->
              let lengths = Array.map (fun a -> (points c).(a)) c.axes in
              cover "a prime factor past 250"
                (Array.exists by_bluestein lengths);
              cover "several axes" (Array.length c.axes > 1);
              cover "a real transform"
                (match c.kind with C2c _ -> false | _ -> true);
              agree c (operand c));
          cases ~id:line_id "of long lines of each kind"
            (List.map (fun (n, kind, wide, _) -> (n, kind, wide)) long_lines)
            line;
          cases ~id:line_id
            "of lines of 2^22 and 2^23 points and of three steps"
            [
              (1 lsl 22, C2c Forward, false);
              (1 lsl 23, C2c Inverse, false);
              (4199040, C2c Forward, true);
            ]
            line;
        ];
    ]

(* nx.metal *)

(* nx.metal computes the complex64 c2c and the float32 r2c and c2r along axes
   whose lines run 7-smooth, Rader's, or Bluestein's of m up to 2^20, and declines
   every other transform (nx_metal.mli), so its laws draw those alone, from a
   cut of the lengths that reaches every feature of a line's first step, a long
   line and each convolution ([metal_features]): one axis among others
   of up to 3 lines for the bound, whose exact sums cost [n] products a point;
   and for the bits, batches of up to 20 lines, side by side or apart, through
   every view. Longer lines are impulses ([test_metal_long]). Each step is its
   own pipeline, so the suite makes them together before its first call. *)
let metal_lengths =
  [
    1;
    2;
    3;
    5;
    7;
    8;
    12;
    16;
    49;
    60;
    90;
    343;
    1000;
    1024;
    2048;
    2187;
    3675;
    4096;
    4374;
    11;
    22;
    23;
    1031;
    2039;
  ]

module Fft = Nx_metal.Fft

(* The features of a line's first step that a cut reaches: its radices, its
   first and last pass's, its count of passes, its points a thread, a line's
   threads within a SIMD group or past it, one line a group or several, and its
   points' parity; whether the line takes several steps, and its
   convolution. *)
let metal_features n =
  let line = Fft.line n ~real:false in
  let steps = line.steps in
  let s = steps.(0) in
  let r = s.radices and k = Array.length s.radices in
  List.map
    (Printf.sprintf "radix %d")
    (List.sort_uniq compare (Array.to_list r))
  @ (if k = 0 then [ "no pass" ]
     else
       [
         Printf.sprintf "first radix %d" r.(0);
         Printf.sprintf "last radix %d" r.(k - 1);
       ])
  @ [
      (if k > 1 then "several passes" else "one pass or none");
      Printf.sprintf "%d points a thread" s.p;
      (if s.tl <= 32 then "a line in a SIMD group" else "a line of SIMD groups");
      (if Fft.rows s > 1 then "lines a group" else "a line a group");
      (if n mod 2 = 0 then "even points" else "odd points");
      (if Array.length steps > 1 then "a long line" else "a line of one step");
      (match line.how with
      | Smooth -> "a 7-smooth line"
      | Rader -> "Rader's line"
      | Bluestein _ -> "Bluestein's line");
    ]

let metal_all_features =
  List.sort_uniq compare (List.concat_map metal_features metal_lengths)

(* Makes every key the cut's lines take, once, before the suite's first call. *)
let metal_prepared (b : Support.backend) =
  lazy
    (Nx_metal.prepare b.device
       (List.filter_map
          (fun (plan, n) ->
            let line = Nx_array.Layout.contiguous [| n |]
            and bins = Nx_array.Layout.contiguous [| (n / 2) + 1 |] in
            match plan with
            | `C2c -> Fft.c2c ~bases:(0, 0) ~axes:[| 0 |] line ~dst:line
            | `R2c -> Fft.r2c ~bases:(0, 0) ~axes:[| 0 |] line ~dst:bins
            | `C2r -> Fft.c2r ~bases:(0, 0) ~axes:[| 0 |] bins ~dst:line)
          (List.concat_map
             (fun n -> [ (`C2c, n); (`R2c, n); (`C2r, n) ])
             metal_lengths)))

let metal_case ~lines =
  let open Gen in
  with_pp pp_case
  @@ let* n = of_list ~pp:Format.pp_print_int metal_lengths in
     let* rest = array ~size:(int_range 0 1) lines in
     let* at = int_range 0 (Array.length rest) in
     let s =
       Array.concat
         [
           Array.sub rest 0 at;
           [| n |];
           Array.sub rest at (Array.length rest - at);
         ]
     in
     let r = Array.length s in
     let* kind = of_list [ C2c Nx_kernel.Forward; C2c Inverse; R2c; C2r n ] in
     let s =
       match kind with
       | C2r _ -> Array.mapi (fun a e -> if a = at then (n / 2) + 1 else e) s
       | C2c _ | R2c -> s
     in
     let* broadcast =
       frequency
         [ (2, constant []); (1, map (fun a -> [ a ]) (int_range 0 (r - 1))) ]
     in
     let+ view = view_of ~broadcast r and+ seed = int in
     let rs = Random.State.make [| seed |] in
     let xs =
       Array.init (total s) (fun _ ->
           let re = value false rs in
           { Complex.re; im = value false rs })
     in
     { kind; wide = false; shape = s; axes = [| at |]; view; xs }

let metal_bound_gen = metal_case ~lines:(Gen.int_range 1 3)

(* A line of each length of the cut, so that every feature is covered whatever
   the seed: at odd positions an inverse of two lines through a reversed view,
   the operand broadcast along the batch, at even ones one line. *)
let metal_bound_examples =
  let real kind n =
    let s = [| (match kind with C2r _ -> (n / 2) + 1 | _ -> n) |] in
    let rs = Random.State.make [| n; 1 |] in
    {
      kind;
      wide = false;
      shape = s;
      axes = [| 0 |];
      view = Elements.plain 1;
      xs =
        Array.init (total s) (fun _ ->
            let re = value false rs in
            { Complex.re; im = value false rs });
    }
  in
  List.concat_map
    (fun n -> [ real R2c n; real (C2r n) n ])
    [ 1; 2; 3; 11; 12; 22; 23; 343; 1000; 2187; 4096 ]
  @ List.mapi
      (fun i n ->
        let rs = Random.State.make [| n |] in
        let two = i mod 2 = 1 in
        let s = if two then [| n; 2 |] else [| n |] in
        {
          kind = C2c (if two then Inverse else Forward);
          wide = false;
          shape = s;
          axes = [| 0 |];
          view =
            (if two then
               { (Elements.plain 2) with reversed = Some 0; broadcast = [ 1 ] }
             else Elements.plain 1);
          xs =
            Array.init (total s) (fun _ ->
                let re = value false rs in
                { Complex.re; im = value false rs });
        })
      metal_lengths

let law_metal_bound b c =
  let n = (points c).(c.axes.(0)) in
  let fs = match c.kind with C2c _ -> metal_features n | R2c | C2r _ -> [] in
  List.iter (fun f -> cover f (List.mem f fs)) metal_all_features;
  let real = match c.kind with C2c _ -> false | R2c | C2r _ -> true in
  cover "an r2c" (c.kind = R2c);
  cover "a c2r" (match c.kind with C2r _ -> true | _ -> false);
  cover "a halved real line" (real && n mod 2 = 0);
  cover "an odd real line" (real && n mod 2 = 1);
  let halved = real && n mod 2 = 0 in
  let how = (Fft.line (if halved then n / 2 else n) ~real:halved).how in
  cover "a real Rader line" (real && how = Rader);
  cover "a real Bluestein line" (real && how <> Smooth && how <> Rader);
  cover "the step of 4096 points" (n = 4096);
  cover "an inverse" (c.kind = C2c Inverse);
  cover "a batch of lines" (Array.length c.shape > 1);
  cover "an operand broadcast" (c.view.broadcast <> []);
  cover "a view off its lines' order" (c.view.stepped || c.view.reversed <> None);
  within_bound b c

let metal_bits_gen =
  let open Gen in
  let lines =
    frequency [ (1, int_range 1 3); (1, int_range 8 9); (1, int_range 16 20) ]
  in
  with_pp pp_case
  @@ let* c = metal_case ~lines in
     let* count = frequency [ (1, constant 0); (1, int_range 1 3) ] in
     let+ seed = int in
     let rs = Random.State.make [| seed |] and xs = Array.copy c.xs in
     for _ = 1 to count do
       let x = specials.(Random.State.int rs 3) in
       let j = Random.State.int rs (Array.length xs) in
       xs.(j) <-
         (if Random.State.bool rs then { (xs.(j)) with im = x }
          else { (xs.(j)) with re = x })
     done;
     { c with xs }

(* Lines whose bits law_bits covers whatever the seed: 16 lines side by side
   with a NaN among their points, and 2 lines apart, of 4096 points, of 1000 and
   of 4374, a long line. *)
let metal_bits_examples =
  let case ?(kind = C2c Forward) n lines nan =
    let s = [| (match kind with C2r _ -> (n / 2) + 1 | _ -> n); lines |] in
    let rs = Random.State.make [| lines |] in
    let xs =
      Array.init (total s) (fun _ ->
          let re = value false rs in
          { Complex.re; im = value false rs })
    in
    if nan then xs.(lines + 1) <- { (xs.(lines + 1)) with re = Float.nan };
    {
      kind;
      wide = false;
      shape = s;
      axes = [| 0 |];
      view = Elements.plain 2;
      xs;
    }
  in
  [
    case 4096 16 true;
    case 4096 2 false;
    case 1000 16 true;
    case 1000 2 false;
    case 4374 2 true;
    case ~kind:R2c 1000 16 true;
    case ~kind:R2c 343 2 false;
    case ~kind:(C2r 1000) 1000 2 true;
    case ~kind:(C2r 343) 343 16 false;
  ]

(* Long lines of two steps, from 6561 = 3^8 to 2^17, complex, odd real and
   halved, and of three, 3,828,125 = 5^7 7^2, through their transposes, of
   impulses whose exact transform costs a few products a point; and long
   Bluestein lines, of 4099 and of the prime 65537, whose 65536 takes no Rader
   line, and halved of 8198, h 4099. The line of 2^23 points
   ([test_long_line]) takes the step of 4096 points. *)
let metal_long_lengths =
  [ 6561; 8192; 16807; 19683; 65536; 131072; 4099; 8198; 65537 ]

let test_metal_long b () =
  List.iter
    (fun n ->
      let w = line_units n in
      List.iter
        (fun kind ->
          check_impulses b ~wide:false kind n w [ 3; n / 3; n - 1 ] impulses)
        [ C2c Forward; C2c Inverse; R2c; C2r n ])
    metal_long_lengths;
  impulse_line b D.Complex64 3828125 1234567

let metal_suite (d : Support.declared) =
  let prepared = lazy (Lazy.force (metal_prepared (Support.opened d))) in
  let run f x =
    let b = Support.opened d in
    Lazy.force prepared;
    b.around (fun () -> f b x)
  in
  group d.label
    [
      group "transforms are within the library's bound"
        [
          prop ~count:6 ~examples:metal_bound_examples "along a 7-smooth axis"
            metal_bound_gen (run law_metal_bound);
          test "long lines" (fun () -> run test_metal_long ());
          test "a line of 2^23 points" (fun () -> run test_long_line ());
        ];
      group "bits are a function of the values alone"
        [
          prop ~examples:metal_bits_examples ~count:40
            "whatever the views or batch" metal_bits_gen (run law_bits);
        ];
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.fft"
       (List.map laws
          (Support.declare Support.cpu
          :: List.filter_map Fun.id
               [ Support.declared_cuda; Support.declared_amd ])
       @ nv
       @ Option.to_list (Option.map metal_suite Support.declared_metal)))
