(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's Fourier transforms: each result against the sums nx_kernel.mli's
   c2c, r2c and c2r state, computed in double-double (or in double, for
   single-precision transforms), within the bound nx_cpu.mli states by the
   points' prime factors, forward and inverse, along one axis or several,
   through views; exact transforms of long lines; the same bits under every
   layout, on one thread as on the job's, for a line alone as in its batch,
   and on every target; empty transforms; the refusals; and no byte read or
   written past the operand or the result. *)

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

(* nx_cpu.mli's: e(n) = 8 Σ_p p^(3/2) + 8 over n's prime factors, or
   6 e(m) + 26 for Bluestein's convolution of m points where a prime factor
   exceeds 250. *)
let rec factors n d =
  if n = 1 then []
  else if d * d > n then [ n ]
  else if n mod d = 0 then d :: factors (n / d) d
  else factors n (d + 1)

let by_bluestein n = List.exists (fun p -> p > 250) (factors n 2)
let composite n = List.length (factors n 2) > 1

let pow2 n =
  let rec go p = if p >= n then p else go (2 * p) in
  go 1

let rec e n =
  if by_bluestein n then (6. *. e (pow2 ((2 * n) - 1))) +. 26.
  else
    List.fold_left
      (fun s p -> s +. (8. *. (Float.of_int p ** 1.5)))
      8. (factors n 2)

(* ε of a transform along [axes] of [s], [n] points along the last for c2r. *)
let epsilon kind axes s =
  let last = axes.(Array.length axes - 1) in
  Array.fold_left
    (fun eps a ->
      let n = match kind with C2r n when a = last -> n | _ -> s.(a) in
      let real = a = last && kind <> C2c Forward && kind <> C2c Inverse in
      eps
      +. if real && n mod 2 = 0 && n >= 512 then e (n / 2) +. 8. else e n)
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
     let* view = view_of r in
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

(* The kernels' result of [c] on [x], into [dst] or a fresh array, answering
   [Done] or failing. *)
let run ?(one_thread = false) ?dst c (A.Any x) =
  let D.Any dt, s = result c in
  let y =
    match dst with
    | Some d -> A.expect dt d
    | None -> A.create Rig.host dt (Iarray.of_array s)
  in
  let call () =
    match (c.kind, A.dtype x, dt) with
    | C2c d, (D.Complex64 as cx), D.Complex64 ->
        Nx_cpu.c2c d ~axes:(Iarray.of_array c.axes) ~dst:y
          (A.expect cx (A.Any x))
    | C2c d, (D.Complex128 as cx), D.Complex128 ->
        Nx_cpu.c2c d ~axes:(Iarray.of_array c.axes) ~dst:y
          (A.expect cx (A.Any x))
    | R2c, (D.Float32 as fx), D.Complex64 ->
        Nx_cpu.r2c ~axes:(Iarray.of_array c.axes) ~dst:y
          (A.expect fx (A.Any x))
    | R2c, (D.Float64 as fx), D.Complex128 ->
        Nx_cpu.r2c ~axes:(Iarray.of_array c.axes) ~dst:y
          (A.expect fx (A.Any x))
    | C2r _, (D.Complex64 as cx), D.Float32 ->
        Nx_cpu.c2r ~axes:(Iarray.of_array c.axes)
          ~dst:(A.expect D.Float32 (A.Any y))
          (A.expect cx (A.Any x))
    | C2r _, (D.Complex128 as cx), D.Float64 ->
        Nx_cpu.c2r ~axes:(Iarray.of_array c.axes)
          ~dst:(A.expect D.Float64 (A.Any y))
          (A.expect cx (A.Any x))
    | _ -> invalid_arg "run"
  in
  let got = if one_thread then Nx_cpu.one_thread call else call () in
  equal answer A.Done got;
  A.Any y

let complexes (A.Any y) : Complex.t array =
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

(* ‖x‖₂ as the bound reads it: a c2r's completed bins', at most √2 times its
   operand's. *)
let operand_norm c =
  match c.kind with C2r _ -> sqrt 2. *. norm c.xs | _ -> norm c.xs

(* Within the bound, by the prime factors of the points along each axis. *)
let law_bound c =
  let lengths = Array.map (fun a -> (points c).(a)) c.axes in
  cover "a prime factor past 250" (Array.exists by_bluestein lengths);
  cover "a prime length"
    (Array.exists (fun n -> n > 1 && not (composite n)) lengths);
  cover "a composite length of several primes"
    (Array.exists
       (fun n -> List.length (List.sort_uniq compare (factors n 2)) > 1)
       lengths);
  cover "several axes" (Array.length c.axes > 1);
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
  let x = operand c in
  let xs = complexes x in
  let got = complexes (run c x) in
  let want =
    Array.map complex_of
      (exact ~wide:c.wide c.kind c.axes c.shape (Array.map dd_of xs))
  in
  let err = norm (Array.map2 Complex.sub want got) in
  let u = if c.wide then epsilon_float /. 2. else 0x1p-24 in
  let n = Array.fold_left (fun k a -> k * (points c).(a)) 1 c.axes in
  let bound =
    epsilon c.kind c.axes c.shape
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
let law_bits c =
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
  let want = bits_of (run plain px) in
  equal bits want (bits_of (run c x));
  equal bits want (bits_of (run ~one_thread:true plain px));
  (* Each line along a single axis, alone: its bits in the batch's. *)
  let s = snd (result c) in
  if Array.length c.axes = 1 && total s > 0 then begin
    let ys = bits_of (run plain px) in
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
        let got = bits_of (run alone (operand alone)) in
        equal bits (Array.init s.(last) (fun j -> ys.(at j s))) got)
      (List.sort_uniq compare [ 0; lines - 1 ])
  end

(* Long lines *)

(* Lines past the generated cases' lengths, each a transform of an impulse,
   whose exact transform is the twiddles: powers of two, a prime past 250,
   real ones of an even length and a composite of every small prime, within
   the bound, with the same bits on one thread. *)
let test_long () =
  let check ~wide kind n at =
    let shape, xs =
      match kind with
      | C2r m ->
          ( [| (m / 2) + 1 |],
            Array.init
              ((m / 2) + 1)
              (fun k -> if k = at then Complex.one else Complex.zero) )
      | _ ->
          ( [| n |],
            Array.init n (fun j -> if j = at then Complex.one else Complex.zero)
          )
    in
    let c = { kind; wide; shape; axes = [| 0 |]; view = plain 1; xs } in
    let x = operand c in
    let y = run c x in
    equal (array string) (elements y) (elements (run ~one_thread:true c x));
    let w = units_of n in
    let want =
      Array.init
        (total (snd (result c)))
        (fun k ->
          match kind with
          | C2c Forward | R2c ->
              complex_of
                (let re, im = w.(at * k mod n) in
                 { re; im })
          | C2c Inverse ->
              complex_of
                (let re, im = w.(at * k mod n) in
                 { re; im = dd_neg im })
          | C2r _ ->
              (* Bins at and n - at, conjugate: 2 cos (2π at k / n), or one bin
                 at 0 or n / 2. *)
              let re, _ = w.(at * k mod n) in
              let c = if at = 0 || 2 * at = n then 1. else 2. in
              { Complex.re = c *. fst re; im = 0. })
    in
    let err = norm (Array.map2 Complex.sub want (complexes y)) in
    let u = if wide then epsilon_float /. 2. else 0x1p-24 in
    let bound =
      epsilon kind [| 0 |] shape *. u *. sqrt (Float.of_int n) *. operand_norm c
    in
    at_most float_exact ~than:bound err
  in
  List.iter
    (fun wide ->
      check ~wide (C2c Forward) 65536 3;
      check ~wide (C2c Inverse) 65537 65000;
      check ~wide R2c 131072 7;
      check ~wide (C2r 131072) 131072 11;
      check ~wide (C2r 30030) 30030 5)
    [ false; true ]

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
let test_streamed () =
  let bits = array (pair int64 int64) in
  List.iter
    (fun c ->
      let x = operand c in
      equal bits (bits_of (run c x)) (bits_of (run ~one_thread:true c x)))
    streamed;
  let c = List.nth streamed 2 in
  let n = c.shape.(0) and lines = c.shape.(1) in
  let ys = bits_of (run c (operand c)) in
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
        (bits_of (run alone (operand alone))))
    [ 0; lines / 2; lines - 1 ]

(* Guard pages *)

(* The same bits with the operand, through its view, and the result each
   against an inaccessible page. *)
let law_guards (c, side) =
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
    ignore (run ~dst:ds.(0) c xs.(0));
    A.Done
  in
  ignore (same_guarded side ~call [| operand c |] [| result c |])

(* The streamed transforms, on one thread and on the job's, each against the
   page. *)
let test_streamed_guards () =
  List.iter
    (fun c ->
      List.iter
        (fun side ->
          List.iter
            (fun one_thread ->
              let call xs ds =
                ignore (run ~one_thread ~dst:ds.(0) c xs.(0));
                A.Done
              in
              ignore (same_guarded side ~call [| operand c |] [| result c |]))
            [ false; true ])
        Support.[ Before; After ])
    streamed

(* Every target's bits *)

(* Digests of lines whose lengths take each stated regime: small primes and
   their products, powers of two, primes past 250 and their doubles, real
   lines halved and not, odd radices and long lines of one prime; each the
   same on every target, as nx_cpu.mli promises. *)
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

let test_digests () =
  let one c = String.concat "" (Array.to_list (elements (run c (operand c)))) in
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
let test_infinity_digests () =
  let with_inf c =
    let xs = Array.copy c.xs in
    xs.(1) <- { Complex.re = Float.infinity; im = 0.5 };
    { c with xs }
  in
  let bits b c =
    Array.iter
      (fun (re, im) ->
        Buffer.add_int64_le b re;
        Buffer.add_int64_le b im)
      (bits_of (run c (operand c)))
  in
  let t =
    table (fun wide n ->
        let b = Buffer.create 4096 in
        List.iter (fun c -> bits b (with_inf c)) (lines wide n);
        Buffer.contents b)
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

let test_empty () =
  let x = A.of_array D.Float32 [| 3; 0 |] [||] in
  let y =
    A.of_array D.Complex64 [| 3; 1 |]
      (Array.make 3 { Complex.re = Float.nan; im = 1. })
  in
  equal answer A.Done (Nx_cpu.r2c ~axes:[| 0; 1 |] ~dst:y x);
  equal (array string) (Array.make 3 (f32 0. ^ f32 0.)) (elements (A.Any y));
  let x = A.of_array D.Complex128 [| 0; 4 |] [||] in
  let y = A.create Rig.host D.Complex128 [| 0; 4 |] in
  equal answer A.Done (Nx_cpu.c2c Inverse ~axes:[| 1 |] ~dst:y x);
  let x = A.of_array D.Complex64 [| 2; 1 |] (Array.make 2 Complex.one) in
  let y = A.create Rig.host D.Float32 [| 2; 0 |] in
  equal answer A.Done (Nx_cpu.c2r ~axes:[| 1 |] ~dst:y x)

let test_refusals () =
  let c64 s = A.create Rig.host D.Complex64 s
  and c128 s = A.create Rig.host D.Complex128 s in
  let f32 s = A.create Rig.host D.Float32 s
  and f64 s = A.create Rig.host D.Float64 s in
  equal answer A.Wrong_dtype
    (Nx_cpu.r2c ~axes:[| 0 |] ~dst:(c128 [| 3 |]) (f32 [| 4 |]));
  equal answer A.Wrong_dtype
    (Nx_cpu.c2r ~axes:[| 0 |] ~dst:(f64 [| 4 |]) (c64 [| 3 |]));
  equal answer A.Shape_mismatch
    (Nx_cpu.c2c Forward ~axes:[||] ~dst:(c64 [| 4 |]) (c64 [| 4 |]));
  equal answer A.Shape_mismatch
    (Nx_cpu.c2c Forward ~axes:[| 1; 0 |] ~dst:(c64 [| 2; 2 |]) (c64 [| 2; 2 |]));
  equal answer A.Shape_mismatch
    (Nx_cpu.c2c Forward ~axes:[| 1 |] ~dst:(c64 [| 4 |]) (c64 [| 4 |]));
  equal answer A.Shape_mismatch
    (Nx_cpu.c2c Inverse ~axes:[| 0 |] ~dst:(c128 [| 5 |]) (c128 [| 4 |]));
  equal answer A.Shape_mismatch
    (Nx_cpu.r2c ~axes:[| 0 |] ~dst:(c64 [| 4 |]) (f32 [| 4 |]));
  equal answer A.Shape_mismatch
    (Nx_cpu.c2r ~axes:[| 0 |] ~dst:(f64 [| 4 |]) (c128 [| 4 |]));
  equal answer A.Shape_mismatch
    (Nx_cpu.c2r ~axes:[| 2 |] ~dst:(f64 [| 4 |]) (c128 [| 3 |]))

let suite (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  let test name f = test name (fun () -> b.around f) in
  group b.name
    [
      group "transforms are within nx_cpu.mli's bound"
        [
          prop ~examples:halved_cases "by the prime factors of the points"
            case_gen (run law_bound);
          test "long lines" test_long;
          test "a transform of no point is the empty sum" test_empty;
        ];
      group "bits are a function of the values alone"
        [
          prop "whatever the views, threads or batch" special_gen
            (run law_bits);
          test "whatever the threads, past the size the lock is released for"
            test_streamed;
          test "on every target" test_digests;
          test "on every target, infinities among the points"
            test_infinity_digests;
        ];
      test "refuses dtypes and shapes" test_refusals;
      group
        "reads and writes no byte past the operand and the result, whatever \
         pages lie beside them"
        [
          prop ~count:300 "through views" (Gen.pair case_gen sides)
            (run law_guards);
          test "past the size the lock is released for" test_streamed_guards;
        ];
    ]

let () = exit (Windtrap.run "nx_cpu.fft" (List.map suite Support.backends))
