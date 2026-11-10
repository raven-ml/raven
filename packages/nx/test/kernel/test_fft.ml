(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Fourier transforms of each kernel library: each result against the
   sums nx_kernel.mli's c2c, r2c and c2r state, computed in double-double
   (or in double, for single-precision transforms), within the bound the
   library states (Support's [transform_factor]), forward and inverse,
   along one axis or several, through views, at every length up to 70 and
   on long lines of each kind, every destination element written; the
   bins a c2r ignores; the same bits under every layout, on one thread as
   on the job's, and for a line alone as in its batch; nx.cpu's bits on
   every target; empty transforms; the refusals; no byte read or written
   past the operand or the result; and the device memory nx.cuda keeps for
   later calls. Each backend runs its own kernels on its device. *)

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

(* The sum Σ_j x_j e^(∓2πi jk/n) of a line, in double-double where [wide] (each
   sum's low parts added without renormalising, which errs by about n u² of
   double), in double else, its twiddles rounded once. *)
let dft ~wide direction (x : dd array) =
  let n = Array.length x in
  let w = units_of n in
  let sign = match direction with Nx_kernel.Forward -> 1. | Inverse -> -1. in
  Array.init n (fun k ->
      let rh = ref 0. and rl = ref 0. and ih = ref 0. and il = ref 0. in
      (* (h, l) += a + b, b small. *)
      let add h l a b =
        let s = !h +. a in
        let bb = s -. !h in
        l := !l +. (!h -. (s -. bb)) +. (a -. bb) +. b;
        h := s
      in
      for j = 0 to n - 1 do
        let (zrh, zrl), (zih, zil) = (x.(j).re, x.(j).im) in
        let (wrh, wrl), (wih, wil) = w.(j * k mod n) in
        let wih = sign *. wih and wil = sign *. wil in
        if wide then begin
          let prod ah al bh bl =
            let p = ah *. bh in
            (p, Float.fma ah bh (-.p) +. (ah *. bl) +. (al *. bh))
          in
          let p1, e1 = prod zrh zrl wrh wrl and p2, e2 = prod zih zil wih wil in
          let p3, e3 = prod zrh zrl wih wil and p4, e4 = prod zih zil wrh wrl in
          add rh rl p1 e1;
          add rh rl (-.p2) (-.e2);
          add ih il p3 e3;
          add ih il p4 e4
        end
        else begin
          rh := !rh +. (zrh *. wrh) -. (zih *. wih);
          ih := !ih +. (zrh *. wih) +. (zih *. wrh)
        end
      done;
      { re = (!rh, !rl); im = (!ih, !il) })

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

(* [b]'s ε of a transform along [axes] of [s], [n] points along the last
   for c2r. *)
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
  xs : Complex.t array;
}

let pp_case ppf c =
  Format.fprintf ppf "%s %s of %a along %a (%s)"
    (match c.kind with
    | C2c Forward -> "forward c2c"
    | C2c Inverse -> "inverse c2c"
    | R2c -> "r2c"
    | C2r n -> Printf.sprintf "c2r of %d points" n)
    (if c.wide then "64-bit" else "32-bit")
    pp_ints c.shape pp_ints c.axes
    (String.concat ", " (view_names c.view))

(* Shapes, with the axes they must transform: small ones of any axes; one long
   line among short ones, transformed with any others, of a composite length
   of several primes, of a prime past 250 or twice one (514 and 1018 halved
   for a real line), or of a length about a power of two; and long composite
   lines side by side. *)
let four = [ 289; 300; 360; 512; 625; 1000; 1024 ]
let blue = [ 251; 257; 502; 509; 514; 1018; 1031 ]
let edge = [ 53; 64; 65; 97; 100; 241; 243; 255; 256 ]

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
    let* rest = array ~size:(int_range 0 2) (int_range 1 3) in
    let+ at = int_range 0 (Array.length rest) in
    ( Array.concat
        [
          Array.sub rest 0 at;
          [| n |];
          Array.sub rest at (Array.length rest - at);
        ],
      [| at |] )
  in
  let side_by_side =
    let* n = int_of [ 289; 300; 360; 512 ] in
    let+ k = int_range 8 9 in
    ([| n; k |], [| 0 |])
  in
  frequency [ (2, small); (2, one_long); (1, side_by_side) ]

let value wide r =
  let x = Random.State.float r 2. -. 1. in
  if wide then x else Int32.float_of_bits (Int32.bits_of_float x)

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
     let rs = Random.State.make [| seed |] in
     let xs =
       Array.init (total s) (fun _ ->
           let re = value wide rs in
           let im = match kind with R2c -> 0. | _ -> value wide rs in
           { Complex.re; im })
     in
     { kind; wide; shape = s; axes; view; xs }

(* Real lines of 514 and 1018 points, whose halves have a prime factor past
   250, in each width and direction: the generator draws them too seldom for
   every seed. *)
let halved_cases =
  let case (n, wide, kind) =
    let s = match kind with C2r _ -> (n / 2) + 1 | _ -> n in
    let rs = Random.State.make [| n |] in
    let xs =
      Array.init s (fun _ ->
          let re = value wide rs in
          let im = match kind with R2c -> 0. | _ -> value wide rs in
          { Complex.re; im })
    in
    { kind; wide; shape = [| s |]; axes = [| 0 |]; view = plain 1; xs }
  in
  List.concat_map
    (fun n ->
      List.concat_map
        (fun wide -> List.map case [ (n, wide, R2c); (n, wide, C2r n) ])
        [ false; true ])
    [ 514; 1018 ]

let points c =
  let last = c.axes.(Array.length c.axes - 1) in
  Array.mapi
    (fun a e -> match c.kind with C2r n when a = last -> n | _ -> e)
    c.shape

(* The operand: complex or real, through the case's view, made from the values
   [xs]. *)
let operand c =
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
  (* A stepped view makes twice the elements: the values repeat. *)
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

(* A fresh array of [dt] and shape [s] on [b]'s device, every element NaN,
   so that one a transform leaves unwritten shows. *)
let poisoned (b : Support.backend) (D.Any dt as d) s =
  let nan =
    match dt with
    | D.Complex64 -> f32 Float.nan ^ f32 Float.nan
    | D.Complex128 -> f64 Float.nan ^ f64 Float.nan
    | D.Float32 -> f32 Float.nan
    | D.Float64 -> f64 Float.nan
    | _ -> invalid_arg "poisoned"
  in
  let (A.Any y) = of_elements d s (Array.make (total s) nan) in
  A.Any (Support.on b y)

(* [b]'s result of [c] on [x], on its device into a fresh array, or into
   [dst] with [x] where they lie; [Done] or failing. [one_thread] runs nx.cpu
   on one thread. *)
let run (b : Support.backend) ?(one_thread = false) ?dst c (A.Any x) =
  let module K = (val b.kernels) in
  let (D.Any dt as d), s = result c in
  let x, y =
    match dst with
    | Some d -> (x, A.expect dt d)
    | None -> (Support.on b x, A.expect dt (poisoned b d s))
  in
  let axes = Iarray.of_array c.axes in
  let call () =
    match (c.kind, A.dtype x, dt) with
    | C2c d, (D.Complex64 as cx), D.Complex64 ->
        K.c2c d ~axes ~dst:y (A.expect cx (A.Any x))
    | C2c d, (D.Complex128 as cx), D.Complex128 ->
        K.c2c d ~axes ~dst:y (A.expect cx (A.Any x))
    | R2c, (D.Float32 as fx), D.Complex64 ->
        K.r2c ~axes ~dst:y (A.expect fx (A.Any x))
    | R2c, (D.Float64 as fx), D.Complex128 ->
        K.r2c ~axes ~dst:y (A.expect fx (A.Any x))
    | C2r _, (D.Complex64 as cx), D.Float32 ->
        K.c2r ~axes ~dst:(A.expect D.Float32 (A.Any y)) (A.expect cx (A.Any x))
    | C2r _, (D.Complex128 as cx), D.Float64 ->
        K.c2r ~axes ~dst:(A.expect D.Float64 (A.Any y)) (A.expect cx (A.Any x))
    | _ -> invalid_arg "run"
  in
  let got = if one_thread then Nx_cpu.one_thread call else call () in
  check got;
  A.Any y

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

let norm (zs : Complex.t array) =
  sqrt
    (Array.fold_left
       (fun s (z : Complex.t) -> s +. (z.re *. z.re) +. (z.im *. z.im))
       0. zs)

(* ‖x‖₂ as the bounds read it: a c2r's √2 times its operand's, at least
   its completed bins'. *)
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
  at_most float_exact ~than:bound err

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

(* c2r cases along several axes whose bins 0 and n/2 hold, along the other
   axes, X[-j] = -conj X[j] and no other bin anything: the other axes' inverse
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
  at_most float_exact ~than:bound err

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

(* The same bits through every view, on one thread as on the job's, and for
   each line transformed alone, NaNs and infinities included. *)
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
  let bits = array (pair int64 int64) in
  let want = bits_of (run b plain px) in
  equal bits want (bits_of (run b c x));
  equal bits want (bits_of (run b ~one_thread:true plain px));
  (* Each line along a single axis, alone: its bits in the batch's. *)
  let s = snd (result c) in
  if Array.length c.axes = 1 && total s > 0 then begin
    let ys = bits_of (run b plain px) in
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
        equal bits (Array.init s.(last) (fun j -> ys.(at j s))) got)
      (List.sort_uniq compare [ 0; lines - 1 ])
  end

(* Long lines *)

(* Lines past the generated cases' lengths, each a transform of three
   impulses, whose exact transform is a sum of twiddles: powers of two and
   other 7-smooth lengths, primes whose predecessor is 7-smooth or not
   (1009, 2017; 4099, 10007, 65537), and real lines of even
   lengths whose halves are prime or 7-smooth, each kind and width within
   the bound, with the same bits on one thread. *)
let long_lengths =
  [ 1009; 2017; 3125; 4096; 4099; 6561; 8192; 10007; 16384; 16807; 19683;
    20014; 30030; 65536; 65537; 131072 ]

(* The roots e^(-2πi m/n), m < n, of a long line, made once for its
   transforms: past 4096 points in double, whose error is far below either
   width's bound at these lengths. *)
let line_units n =
  if n <= 4096 then units_of n
  else
    Array.init n (fun m ->
        let phi = 2. *. Float.pi *. (Float.of_int m /. Float.of_int n) in
        ((cos phi, 0.), (-.sin phi, 0.)))

(* [kind] of [n] points from three impulses at [ats] of coefficients
   [cs]: within [b]'s bound of their exact transform, and the same bits on
   one thread. *)
let check_impulses b ~wide kind n w ats cs =
  let shape, xs =
    match kind with
    | C2r m ->
        let h = (m / 2) + 1 in
        let xs = Array.make h Complex.zero in
        List.iter2 (fun at z -> xs.(at mod h) <- Complex.add xs.(at mod h) z) ats cs;
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
  let x = operand c in
  let y = run b c x in
  let got = complexes y in
  let alone = complexes (run b ~one_thread:true c x) in
  let same (y : Complex.t) (z : Complex.t) =
    Int64.bits_of_float y.re = Int64.bits_of_float z.re
    && Int64.bits_of_float y.im = Int64.bits_of_float z.im
  in
  let rec differs k =
    if k = Array.length got then None
    else if same got.(k) alone.(k) then differs (k + 1)
    else Some k
  in
  equal (option int) None (differs 0);
  let root m = complex_of (let re, im = w.(m mod n) in { re; im }) in
  let impulses =
    List.filter
      (fun (_, (z : Complex.t)) -> z.re <> 0. || z.im <> 0.)
      (Array.to_list (Array.mapi (fun j z -> (j, z)) xs))
  in
  let terms k =
    List.fold_left
         (fun acc (j, (z : Complex.t)) ->
           match kind with
           | C2c Forward | R2c -> Complex.add acc (Complex.mul z (root (j * k)))
           | C2c Inverse ->
               Complex.add acc (Complex.mul z (Complex.conj (root (j * k))))
           | C2r m ->
               (* Bin j and its conjugate n - j, or one bin at 0 or n / 2,
                  whose imaginary part takes no part. *)
               let e = Complex.conj (root (j * k)) in
               let re =
                 if j = 0 || 2 * j = m then z.re *. e.re
                 else 2. *. ((z.re *. e.re) -. (z.im *. e.im))
               in
               { acc with re = acc.re +. re })
      Complex.zero impulses
  in
  let want = Array.init (total (snd (result c))) terms in
  let err = norm (Array.map2 Complex.sub want got) in
  let u = if wide then epsilon_float /. 2. else 0x1p-24 in
  let bound =
    epsilon b kind [| 0 |] shape *. u *. sqrt (Float.of_int n) *. operand_norm c
  in
  at_most float_exact ~than:bound err

let test_long b () =
  List.iter
    (fun n ->
      let w = line_units n in
      let ats = [ 3; n / 3; n - 1 ] in
      let cs =
        Complex.[ { re = 0.75; im = -0.5 }; { re = -1.25; im = 0.25 }; { re = 0.5; im = 1. } ]
      in
      List.iter
        (fun wide ->
          List.iter
            (fun kind -> check_impulses b ~wide kind n w ats cs)
            [ C2c Forward; C2c Inverse; R2c; C2r n ])
        [ false; true ])
    long_lengths

(* An impulse at [at] of a line of [n] points of [dt], forward and
   inverse, each within the bound of its exact transform, the twiddles,
   and with the same bits on one thread. The twiddles are computed in
   double, whose error is far below either bound at these lengths. *)
let impulse_line (type s) (b : Support.backend) (dt : (Complex.t, s) D.t) n at =
  let module K = (val b.kernels) in
  let x = A.of_array dt [| n |] (Array.make n Complex.zero) in
  A.set x [| at |] Complex.one;
  let x = Support.on b x in
  let check_line direction =
    let c2c () =
      let y = A.create b.device dt [| n |] in
      check (K.c2c direction ~axes:[| 0 |] ~dst:y x);
      Support.host y
    in
    let y = c2c () and alone = Nx_cpu.one_thread c2c in
    let sign = match direction with Nx_kernel.Forward -> -1. | Inverse -> 1. in
    let err = ref 0. and differs = ref None in
    for k = n - 1 downto 0 do
      let (z : Complex.t) = A.get y [| k |] in
      let phi =
        sign *. 2. *. Float.pi *. float_of_int (at * k mod n) /. float_of_int n
      in
      let dre = z.re -. cos phi and dim = z.im -. sin phi in
      err := !err +. (dre *. dre) +. (dim *. dim);
      let (w : Complex.t) = A.get alone [| k |] in
      if
        Int64.bits_of_float z.re <> Int64.bits_of_float w.re
        || Int64.bits_of_float z.im <> Int64.bits_of_float w.im
      then differs := Some k
    done;
    equal (option int) None !differs;
    let u = match dt with D.Complex64 -> 0x1p-24 | _ -> epsilon_float /. 2. in
    let bound =
      epsilon b (C2c direction) [| 0 |] [| n |] *. u *. sqrt (float_of_int n)
    in
    at_most float_exact ~than:bound (sqrt !err)
  in
  check_line Forward;
  check_line Inverse

(* A complex64 line of 2^22 points, 32 MiB, where on Linux nx.cpu's
   four-step grid lies in its result where the largest cache is at most
   32 MiB. *)
let test_past_caches b () = impulse_line b D.Complex64 (1 lsl 22) 12345

(* A complex128 line of 4,199,040 = 2^7 3^8 5 points, 64 MiB, the least
   7-smooth length past 2048^2. *)
let test_wide_line b () = impulse_line b D.Complex128 4199040 54321

(* A complex64 line of 2^23 points, 64 MiB, of two factors of 4096 and
   2048 points. *)
let test_long_line b () = impulse_line b D.Complex64 (1 lsl 23) 7654321

(* The first of [n] complex64 points where [y] and [z] differ in their
   bits, if any. *)
let differs n y z =
  A.read y @@ fun y ->
  A.read z @@ fun z ->
  let y = Bigarray.reshape_1 y n and z = Bigarray.reshape_1 z n in
  let same (a : Complex.t) (b : Complex.t) =
    Int64.bits_of_float a.re = Int64.bits_of_float b.re
    && Int64.bits_of_float a.im = Int64.bits_of_float b.im
  in
  let rec from k =
    if k = n then None else if same y.{k} z.{k} then from (k + 1) else Some k
  in
  from 0

(* A random complex64 line of 3^14 points, 38 MiB, with the same bits as
   its values shaped [1; n] and transformed along both axes, forward and
   inverse, on the job's threads and on one. On Linux, where the largest
   cache is under 38 MiB, the line's grid lies in its result, and the
   second axis's, transformed in place, in scratch. Its n1 and n2 are odd,
   so both steps fill their last lanes in part. *)
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
    Support.host y
  in
  List.iter
    (fun direction ->
      let by_row = c2c direction [| 0; 1 |] row [| 1; n |]
      and by_line = c2c direction [| 0 |] line [| n |] in
      let want = by_row () in
      let check y = equal (option int) None (differs n want y) in
      check (Nx_cpu.one_thread by_row);
      check (by_line ());
      check (Nx_cpu.one_thread by_line))
    [ Nx_kernel.Forward; Inverse ]

(* A forward c2c of [lines] lines of [n] points, drawn from [seed]. *)
let line ~wide n lines seed =
  let rs = Random.State.make [| seed |] in
  let xs =
    Array.init (n * lines) (fun _ ->
        let re = value wide rs in
        { Complex.re; im = value wide rs })
  in
  let shape = if lines = 1 then [| n |] else [| n; lines |] in
  let r = Array.length shape in
  { kind = C2c Forward; wide; shape; axes = [| 0 |]; view = plain r; xs }

(* Transforms of a MiB and past, which nx.cpu runs with the lock released: a
   line of each dtype, and lines side by side. *)
let streamed =
  [
    line ~wide:true (1 lsl 16) 1 19;
    line ~wide:false (1 lsl 17) 1 22;
    line ~wide:false 1024 136 10;
  ]

(* The same bits on the job's threads as on one, and for lines side by side,
   each line's bits alone. *)
let test_streamed b () =
  let bits = array (pair int64 int64) in
  List.iter
    (fun c ->
      let x = operand c in
      equal bits (bits_of (run b c x)) (bits_of (run b ~one_thread:true c x)))
    streamed;
  let c = List.nth streamed 2 in
  let n = c.shape.(0) and lines = c.shape.(1) in
  let ys = bits_of (run b c (operand c)) in
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
      equal bits
        (Array.init n (fun j -> ys.((j * lines) + l)))
        (bits_of (run b alone (operand alone))))
    [ 0; lines / 2; lines - 1 ]

(* Every short length *)

(* Three lines of every length from 0 to 70, of each kind and width, along
   the last axis or, at odd lengths, side by side along the first: within
   the bound, every destination element written. *)
let test_short_lengths b () =
  for n = 0 to 70 do
    List.iter
      (fun wide ->
        List.iter
          (fun kind ->
            let e = match kind with C2r _ -> (n / 2) + 1 | _ -> n in
            let shape, axes =
              if n mod 2 = 0 then ([| 3; e |], [| 1 |]) else ([| e; 3 |], [| 0 |])
            in
            let rs = Random.State.make [| n |] in
            let xs =
              Array.init (total shape) (fun _ ->
                  let re = value wide rs in
                  let im = match kind with R2c -> 0. | _ -> value wide rs in
                  { Complex.re; im })
            in
            within_bound b { kind; wide; shape; axes; view = plain 2; xs })
          [ C2c Forward; C2c Inverse; R2c; C2r n ])
      [ false; true ]
  done

(* Complex arrays at their parts' alignment *)

(* [x]'s elements in a fresh array on [b]'s device whose first byte lies
   one part past twice a part's alignment: 4 mod 8 for complex64, 8 mod
   16 for complex128; a real array as it is. nx_array.mli promises an
   array only its parts' alignment. *)
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

(* Each kind and width from an operand and into a destination off their
   pairs' alignment, of lines that fit a group and of long ones, which
   take a real line's split or pre-pass launch: the bits of the aligned
   transform. *)
let test_off_pairs b () =
  List.iter
    (fun wide ->
      List.iter
        (fun (kind, shape, axes) ->
          let rs = Random.State.make [| total shape |] in
          let xs =
            Array.init (total shape) (fun _ ->
                let re = value wide rs in
                { Complex.re; im = (match kind with R2c -> 0. | _ -> value wide rs) })
          in
          let c = { kind; wide; shape; axes; view = plain (Array.length shape); xs } in
          let x = operand c in
          let want = bits_of (run b c x) in
          let D.Any dt, s = result c in
          let dst = off_pairs b (A.Any (A.create b.device dt (Iarray.of_array s))) in
          let got = bits_of (run b ~dst c (off_pairs b x)) in
          equal (array (pair int64 int64)) want got)
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

(* The bins a c2r ignores *)

(* The imaginary parts of bin 0 and, at an even n, of bin n / 2 take no
   part in a c2r along one axis: NaN there gives the bits +0 there gives,
   which are finite. *)
let test_ignored_bins b () =
  List.iter
    (fun n ->
      List.iter
        (fun wide ->
          let h = (n / 2) + 1 in
          let rs = Random.State.make [| n |] in
          let xs =
            Array.init (3 * h) (fun _ ->
                let re = value wide rs in
                { Complex.re; im = value wide rs })
          in
          let with_im v =
            Array.mapi
              (fun i (z : Complex.t) ->
                let k = i mod h in
                if k = 0 || (n mod 2 = 0 && k = n / 2) then { z with im = v }
                else z)
              xs
          in
          let got xs =
            let c =
              { kind = C2r n; wide; shape = [| 3; h |]; axes = [| 1 |];
                view = plain 2; xs }
            in
            bits_of (run b c (operand c))
          in
          let want = got (with_im 0.) in
          equal
            (array (pair int64 int64))
            want
            (got (with_im Float.nan)))
        [ false; true ])
    [ 1; 2; 3; 8; 9; 64; 1000; 1024; 20014; 32768 ]

(* A c2r along axes [0; 1] of [3; 2] bins whose column 0 is
   i c (1, w, w²), w = e^(-2πi/3), and column 1 zero: the first axis's
   inverse transform leaves column 0 all imaginary, which the last axis
   drops, so the exact result is 0 and every rounding of the first axis
   errs against it. Within the bound, at c = 1 and c = 2^40. *)
let test_dropped_bins b () =
  List.iter
    (fun wide ->
      List.iter
        (fun c ->
          let h = sqrt 3. /. 2. *. c in
          let xs =
            Complex.
              [| { re = 0.; im = c }; zero; { re = h; im = -0.5 *. c }; zero;
                 { re = -.h; im = -0.5 *. c }; zero |]
          in
          within_bound b
            { kind = C2r 2; wide; shape = [| 3; 2 |]; axes = [| 0; 1 |];
              view = plain 2; xs })
        [ 1.; 0x1p40 ])
    [ false; true ]

(* Long lines on later axes *)

(* Long lines along the axis transformed second, so read from the
   destination, and a c2r's other axis into its copy of the operand. The
   long axis holds two impulses, each a line of drawn values along the
   short one, so the long axis's exact transform is a sum of twiddles. *)
let test_later_axes b () =
  let check ~wide kind n0 n1 =
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
    let root m = complex_of (let re, im = w.(m mod n0) in { re; im }) in
    (* Along the long axis: Σ over the impulses' rows of their values times
       the twiddles, conjugated for an inverse. *)
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
    let n = n0 * (match kind with C2r m -> m | _ -> n1) in
    let bound =
      epsilon b kind [| 0; 1 |] shape *. u *. sqrt (Float.of_int n) *. operand_norm c
    in
    at_most float_exact ~than:bound err
  in
  List.iter
    (fun wide ->
      List.iter
        (fun (kind, n0, n1) -> check ~wide kind n0 n1)
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

(* The same bits with the operand, through its view, and the result each
   against an inaccessible page. *)
let law_guards b (c, side) =
  let after = side = Support.After in
  cover "a stepped operand's first element at the page before"
    (c.view.stepped && not after);
  cover "a reversed operand at the page after"
    (c.view.reversed <> None && after);
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
    streamed

(* Every target's bits *)

(* Digests of lines whose lengths take each stated regime of nx.cpu: small
   primes and their products, powers of two, primes past 250 and their
   doubles, real lines halved and not, odd radices and long lines of one
   prime; each the same on every target, as nx_cpu.mli promises. *)
let lengths =
  [ 7; 60; 64; 241; 100; 1000; 1024; 4096; 257; 1031; 514; 512; 1001; 59049;
    65536; 255; 504 ]

(* A line of each kind, forward, inverse, r2c and c2r, of [n] points drawn from
   a seed of [n]. *)
let lines wide n =
  List.map
    (fun kind ->
      let s = match kind with C2r n -> [| (n / 2) + 1 |] | _ -> [| n |] in
      let rs = Random.State.make [| n |] in
      let xs =
        Array.init s.(0) (fun _ ->
            let re = value wide rs in
            let im = match kind with R2c -> 0. | _ -> value wide rs in
            { Complex.re; im })
      in
      { kind; wide; shape = s; axes = [| 0 |]; view = plain 1; xs })
    [ C2c Forward; C2c Inverse; R2c; C2r n ]

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
  expect t @@ __POS_OF__ {|
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

(* The same lines with an infinity at their second point, every NaN as one as
   in [bits_of]. Which results are infinite or NaN, and the infinities' signs,
   follow from the order of each line's operations, which the finite digests
   do not see. *)
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
  expect t @@ __POS_OF__ {|
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

let test_empty (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = Support.on b (A.of_array D.Float32 [| 3; 0 |] [||]) in
  let y =
    Support.on b
      (A.of_array D.Complex64 [| 3; 1 |]
         (Array.make 3 { Complex.re = Float.nan; im = 1. }))
  in
  check (K.r2c ~axes:[| 0; 1 |] ~dst:y x);
  equal (array string) (Array.make 3 (f32 0. ^ f32 0.)) (elements (A.Any y));
  let x = Support.on b (A.of_array D.Complex128 [| 0; 4 |] [||]) in
  let y = A.create b.device D.Complex128 [| 0; 4 |] in
  check (K.c2c Inverse ~axes:[| 1 |] ~dst:y x);
  let x =
    Support.on b (A.of_array D.Complex64 [| 2; 1 |] (Array.make 2 Complex.one))
  in
  let y = A.create b.device D.Float32 [| 2; 0 |] in
  check (K.c2r ~axes:[| 1 |] ~dst:y x)

(* Transforms that do not fit, by their dtypes, axes or shapes: not Done,
   and nothing written outside the destination. *)
let test_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let c64 s = A.create b.device D.Complex64 (Iarray.of_array s)
  and c128 s = A.create b.device D.Complex128 (Iarray.of_array s) in
  let f32 s = A.create b.device D.Float32 (Iarray.of_array s) in
  let row (type v s) msg what (dt : (v, s) D.t) dst call =
    Misfit.check ~msg ~what b [| (D.Any dt, dst) |] (fun g ds ->
        call g (A.expect dt ds.(0)))
  in
  let c64' = D.Complex64 and c128' = D.Complex128 and f64' = D.Float64 in
  row "r2c of float32 into complex128" "a dtype" c128' [| 3 |] (fun g dst ->
      K.r2c ~axes:[| 0 |] ~dst (g.Misfit.g (f32 [| 4 |])));
  row "c2r of complex64 into float64" "a dtype" f64' [| 4 |] (fun g dst ->
      K.c2r ~axes:[| 0 |] ~dst (g.Misfit.g (c64 [| 3 |])));
  row "c2c along no axis" "an attribute" c64' [| 4 |] (fun g dst ->
      K.c2c Forward ~axes:[||] ~dst (g.Misfit.g (c64 [| 4 |])));
  row "c2c along decreasing axes" "an attribute" c64' [| 2; 2 |] (fun g dst ->
      K.c2c Forward ~axes:[| 1; 0 |] ~dst (g.Misfit.g (c64 [| 2; 2 |])));
  row "c2c along an axis past the rank" "an attribute" c64' [| 4 |]
    (fun g dst -> K.c2c Forward ~axes:[| 1 |] ~dst (g.Misfit.g (c64 [| 4 |])));
  row "c2c into another shape" "a shape" c128' [| 5 |] (fun g dst ->
      K.c2c Inverse ~axes:[| 0 |] ~dst (g.Misfit.g (c128 [| 4 |])));
  row "r2c into all the bins" "a shape" c64' [| 4 |] (fun g dst ->
      K.r2c ~axes:[| 0 |] ~dst (g.Misfit.g (f32 [| 4 |])));
  row "c2r of other bins than its points'" "a shape" f64' [| 4 |]
    (fun g dst -> K.c2r ~axes:[| 0 |] ~dst (g.Misfit.g (c128 [| 4 |])));
  row "c2r along an axis past the rank" "an attribute" f64' [| 4 |]
    (fun g dst -> K.c2r ~axes:[| 2 |] ~dst (g.Misfit.g (c128 [| 3 |])))

(* Past 2^31 elements *)

(* A c2r into float32 [2^21 + 1; 1024], 8 GiB, its last element at
   2^31 + 1023, from bins broadcast along the first axis: sampled lines
   have the first line's bits, and the first is within the bound. A device
   holds it where the host would need 8 GiB. *)
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
    A.v D.Complex64
      (L.v ~strides:[| 0; 1 |] [| lines; h |])
      (A.buffer bins)
  in
  let y = A.create b.device D.Float32 [| lines; n |] in
  check (K.c2r ~axes:[| 1 |] ~dst:y x);
  let line l =
    A.to_array
      (Support.host
         (A.v D.Float32 (L.v ~offset:(l * n) ~strides:[| 1 |] [| n |]) (A.buffer y)))
  in
  let first = line 0 in
  let bits = Array.map Int32.bits_of_float in
  List.iter
    (fun l -> equal (array int32) (bits first) (bits (line l)))
    [ 1; lines / 2; lines - 1 ];
  let c =
    { kind = C2r n; wide = false; shape = [| h |]; axes = [| 0 |];
      view = plain 1; xs = row }
  in
  let want =
    Array.map complex_of (exact ~wide:false c.kind c.axes c.shape (Array.map dd_of row))
  in
  let err =
    norm (Array.map2 (fun (w : Complex.t) re -> Complex.sub w { re; im = 0. }) want first)
  in
  let bound =
    epsilon b c.kind c.axes c.shape *. 0x1p-24 *. sqrt (Float.of_int n) *. operand_norm c
  in
  at_most float_exact ~than:bound err

(* The bytes of device memory nx_cuda.mli states nx.cuda keeps for later
   calls on a device. *)
let cuda_kept = 192 * 1024 * 1024

(* Lines of primes past 2^19, each by Bluestein's convolution, whose plans
   at complex128, a chirp of 8 MiB and a kernel of 16 MiB or more each,
   take more than the 64 MiB nx_cuda.mli states the plans keep. *)
let large_plans = [ 524309; 524341; 524347; 524351; 524353 ]

let[@inline never] transform_line (b : Support.backend) n =
  let module K = (val b.kernels) in
  let x = A.create b.device D.Complex128 [| n |] in
  let y = A.create b.device D.Complex128 [| n |] in
  check (K.c2c Forward ~axes:[| 0 |] ~dst:y x);
  Rig.Buffer.wait (A.buffer y) Read_write

(* nx.cuda keeps at most [cuda_kept] bytes of device memory for later
   calls, however many large lines it has planned. *)
let test_kept_plans (b : Support.backend) () =
  (* The first call loads the library's code on the device. *)
  transform_line b 64;
  let before = Support.settled b.device in
  List.iter (transform_line b) large_plans;
  at_most ~msg:"bytes kept" int ~than:cuda_kept (Support.settled b.device - before)

(* Lines of a prime past 2^20, each by Bluestein's convolution of
   [m = 2^21] points, take a workspace past the 64 MiB the device keeps:
   nx_cuda.mli states at most [3m] complex points a line. *)
let test_loop_workspace (b : Support.backend) () =
  let module K = (val b.kernels) in
  let lines = 4 and n = 1048583 in
  let x = A.create b.device D.Complex64 [| lines; n |] in
  let y = A.create b.device D.Complex64 [| lines; n |] in
  let m = 4 * n in
  Kernels_cases.loop_held b ~bound:(lines * 3 * m * 8) (fun () ->
      check (K.c2c Forward ~axes:[| 1 |] ~dst:y x);
      Rig.Buffer.wait (A.buffer y) Read_write)

let suite (b : Support.backend) =
  let run f x = b.around (fun () -> f b x) in
  let test name f = test name (fun () -> b.around (f b)) in
  (* Guard pages lie in host memory: a backend computing elsewhere has
     none. *)
  let guards =
    if not (Rig.equal b.device Rig.host) then []
    else
      [
        group
          "reads and writes no byte past the operand and the result, \
           whatever pages lie beside them"
          [
            prop ~count:300 "through views" (Gen.pair case_gen sides)
              (run law_guards);
            test "past the size the lock is released for" test_streamed_guards;
          ];
      ]
  in
  let gpu = List.memq b (Support.gpus ()) in
  (* nx.cpu's bits, the same on every target. *)
  let targets =
    if gpu then []
    else
      [
        test "on every target" test_digests;
        test "on every target, infinities among the points"
          test_infinity_digests;
      ]
  in
  group b.name
    ([
       group "transforms are within the library's bound"
         [
           prop ~examples:halved_cases "by the prime factors of the points"
             case_gen (run law_bound);
           prop "bins whose other axes' transforms are imaginary" opposite_gen
             (run law_opposite);
           test "every length to 70, every element written" test_short_lengths;
           test "long lines" test_long;
           test "long lines on later axes" test_later_axes;
           test "a c2r of bins no real line has" test_dropped_bins;
           test "a line past the caches" test_past_caches;
           test "a complex128 line of three steps" test_wide_line;
           test "a complex64 line of 2^23 points" test_long_line;
           test "a transform of no point is the empty sum" test_empty;
         ];
       test "a c2r ignores the imaginary parts of bins 0 and n / 2"
         test_ignored_bins;
       test "operands and destinations at their parts' alignment"
         test_off_pairs;
       group "bits are a function of the values alone"
         ([
            prop "whatever the views, threads or batch" special_gen
              (run law_bits);
            test "whatever the threads, past the size the lock is released for"
              test_streamed;
            test "whatever the home of a long line's grid" test_grid_homes;
          ]
         @ targets);
       test
         "a transform of dtypes, axes or shapes that do not fit is not Done \
          and writes nothing outside dst"
         test_misfits;
     ]
    @ guards
    @
    if not gpu then []
    else
      [
        test "keeps at most 192 MiB of device memory for later calls"
          test_kept_plans;
        test "calls in a loop hold one call's workspace" test_loop_workspace;
        test "past 2^31 elements" test_past_2_31;
      ])

(* rig.nv *)

(* A line of [n] points of [kind], of random values. *)
let line ~wide kind n =
  let shape = match kind with C2r m -> [| (m / 2) + 1 |] | R2c | C2c _ -> [| n |] in
  let rs = Random.State.make [| n |] in
  let xs =
    Array.init shape.(0) (fun _ ->
        let re = value wide rs in
        { Complex.re; im = value wide rs })
  in
  { kind; wide; shape; axes = [| 0 |]; view = plain 1; xs }

(* rig.nv loads the cubin CUDA loads, so a transform through it has
   CUDA's bits: what it can get wrong, its launches, memory and order,
   shows in them. Each case runs through CUDA, then through rig.nv. A
   transform past 2^31 elements adds the kernels' index arithmetic, and
   runs through CUDA alone. A NaN agrees with any NaN: nx_cuda.mli
   leaves a NaN result's sign and payload unspecified. *)
let nv_laws (first : Support.backend) (nv : Support.backend) =
  let agree call = Elements.agree first nv call in
  let transform c =
    let x = operand c in
    agree (fun b -> [| run b c x |])
  in
  group nv.name
    [
      group "bits are cuda's, any NaN for a NaN"
        [
          prop "of transforms by the prime factors of the points" case_gen
            (fun c ->
              let lengths = Array.map (fun a -> (points c).(a)) c.axes in
              cover "a prime factor past 250" (Array.exists by_bluestein lengths);
              cover "several axes" (Array.length c.axes > 1);
              cover "a real transform" (match c.kind with C2c _ -> false | _ -> true);
              transform c);
          test "of long lines of each kind" (fun () ->
              List.iter
                (fun n ->
                  List.iter
                    (fun wide ->
                      List.iter
                        (fun kind -> transform (line ~wide kind n))
                        [ C2c Forward; C2c Inverse; R2c; C2r n ])
                    [ false; true ])
                long_lengths);
          test "of lines of 2^22 and 2^23 points and of three steps" (fun () ->
              transform (line ~wide:false (C2c Forward) (1 lsl 22));
              transform (line ~wide:false (C2c Inverse) (1 lsl 23));
              transform (line ~wide:true (C2c Forward) 4199040));
        ];
    ]

let () =
  let nv =
    match (Support.gpus (), Support.nv ()) with
    | first :: _, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.fft" (List.map suite (Support.backends ()) @ nv))
