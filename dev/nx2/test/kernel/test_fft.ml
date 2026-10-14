(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's Fourier transforms: each result against the sums nx_kernel.mli's
   c2c, r2c and c2r state, computed in double-double (or in double, for
   single-precision transforms), within the bound nx_cpu.mli states, along lines
   computed by passes, by the four steps and by Bluestein's convolution, forward
   and inverse, along one axis or several, through views; exact transforms of
   long lines; the same bits under every layout, on one thread as on the job's,
   and for a line alone as in its batch; empty transforms, and the refusals. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

(* Double-double numbers *)

let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  (s, a -. (s -. bb) +. (b -. bb))

let two_prod a b =
  let p = a *. b in
  (p, Float.fma a b (-.p))

let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  two_sum s (e +. al +. bl)

let dd_mul (ah, al) (bh, bl) =
  let p, e = two_prod ah bh in
  two_sum p (e +. (ah *. bl) +. (al *. bh))

let dd_neg (h, l) = (-.h, -.l)

let dd_div (h, l) d =
  let q = h /. d in
  let p, e = two_prod q d in
  two_sum q ((h -. p -. e +. l) /. d)

(* sin φ and cos φ by their series, φ in [0, π/4]. *)
let sincos phi =
  let s = ref (0., 0.) and c = ref (0., 0.) and t = ref (1., 0.) in
  for k = 0 to 40 do
    (match k mod 4 with
    | 0 -> c := dd_add !c !t
    | 1 -> s := dd_add !s !t
    | 2 -> c := dd_add !c (dd_neg !t)
    | _ -> s := dd_add !s (dd_neg !t));
    t := dd_div (dd_mul !t phi) (Float.of_int (k + 1))
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
      (dd_div (Float.of_int (if far then n - r else r), 0.) (Float.of_int n))
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

let four_steps n = n > 64 && List.length (factors n 2) > 1

let by_bluestein n = List.exists (fun p -> p > 250) (factors n 2)

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
   line among short ones, transformed with any others, of a length at the edge
   of the passes, of Bluestein's or of the four steps' (with factors not
   multiples of a vector); and lines of the four steps' lengths side by side. *)
let four = [ 289; 300; 360; 512; 625; 1000; 1024 ]
let blue = [ 251; 257; 502; 509; 1031 ]
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

(* The kernels' result of [c] on [x], answering [Done] or failing. *)
let run ?(threads = 0) c (A.Any x) =
  let D.Any dt, s = result c in
  let y = A.create Rig.host dt s in
  let got =
    match (c.kind, A.dtype x, dt) with
    | C2c d, (D.Complex64 as cx), D.Complex64 ->
        Nx_cpu.c2c_on ~threads d ~axes:c.axes ~dst:y (A.expect cx (A.Any x))
    | C2c d, (D.Complex128 as cx), D.Complex128 ->
        Nx_cpu.c2c_on ~threads d ~axes:c.axes ~dst:y (A.expect cx (A.Any x))
    | R2c, (D.Float32 as fx), D.Complex64 ->
        Nx_cpu.r2c_on ~threads ~axes:c.axes ~dst:y (A.expect fx (A.Any x))
    | R2c, (D.Float64 as fx), D.Complex128 ->
        Nx_cpu.r2c_on ~threads ~axes:c.axes ~dst:y (A.expect fx (A.Any x))
    | C2r _, (D.Complex64 as cx), D.Float32 ->
        Nx_cpu.c2r_on ~threads ~axes:c.axes
          ~dst:(A.expect D.Float32 (A.Any y))
          (A.expect cx (A.Any x))
    | C2r _, (D.Complex128 as cx), D.Float64 ->
        Nx_cpu.c2r_on ~threads ~axes:c.axes
          ~dst:(A.expect D.Float64 (A.Any y))
          (A.expect cx (A.Any x))
    | _ -> invalid_arg "run"
  in
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

let law_bound c =
  cover "by Bluestein"
    (Array.exists (fun a -> by_bluestein (points c).(a)) c.axes);
  cover "by the four steps"
    (Array.exists
       (fun a ->
         let n = (points c).(a) in
         four_steps n && not (by_bluestein n))
       c.axes);
  cover "several axes" (Array.length c.axes > 1);
  cover "a real line through complex ones of n points"
    (match c.kind with
    | R2c | C2r _ ->
        let n = (points c).(c.axes.(Array.length c.axes - 1)) in
        n mod 2 = 1 || n < 512
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

(* The same bits through every view, on one thread as on the job's, in place,
   and for each line transformed alone. *)
let law_bits c =
  let lanes = if c.wide then 4 else 8 in
  let last = c.axes.(Array.length c.axes - 1) in
  let n = (points c).(last) and r = Array.length c.shape in
  cover "the four steps, lines side by side"
    (four_steps n
    && (not (by_bluestein n))
    && last < r - 1
    && c.shape.(r - 1) >= lanes);
  let x = operand c in
  let plain = { c with view = plain r; xs = complexes x } in
  let px = operand plain in
  let want = elements (run plain px) in
  equal (array string) want (elements (run c x));
  equal (array string) want (elements (run ~threads:1 plain px));
  (match c.kind with
  | C2c d ->
      let (A.Any y) = px in
      let y = A.copy y in
      let got =
        match A.dtype y with
        | D.Complex64 as dt ->
            let y = A.expect dt (A.Any y) in
            ignore (Nx_cpu.c2c d ~axes:c.axes ~dst:y y);
            A.Any y
        | D.Complex128 as dt ->
            let y = A.expect dt (A.Any y) in
            ignore (Nx_cpu.c2c d ~axes:c.axes ~dst:y y);
            A.Any y
        | _ -> invalid_arg "law_bits"
      in
      equal (array string) want (elements got)
  | R2c | C2r _ -> ());
  (* Each line along a single axis, alone: its bits in the batch's. *)
  let s = snd (result c) in
  if Array.length c.axes = 1 && total s > 0 then begin
    let ys = complexes (run plain px) in
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
        let got = complexes (run alone (operand alone)) in
        let want = Array.init s.(last) (fun j -> ys.(at j s)) in
        let bits (z : Complex.t) =
          (Int64.bits_of_float z.re, Int64.bits_of_float z.im)
        in
        equal
          (array (pair int64 int64))
          (Array.map bits want) (Array.map bits got))
      (List.sort_uniq compare [ 0; lines - 1 ])
  end

(* Long lines *)

(* Lines past the generated cases' lengths, each a transform of an impulse,
   whose exact transform is the twiddles: by the four steps, by Bluestein's
   convolution, real ones of an even length, each line's steps jobs of their
   own. The same bits on one thread. *)
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
    equal (array string) (elements y) (elements (run ~threads:1 c x));
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

(* Every target's bits *)

(* A digest of each length's transforms, forward, inverse, r2c and c2r, at
   both widths: the same on each target this CPU runs, and on every CPU, since
   a line's arithmetic is a function of its length alone. *)
let test_digests () =
  let lengths = [ 7; 60; 64; 241; 100; 1000; 1024; 4096; 257; 1031; 514 ] in
  let digest wide n =
    let one kind =
      let s = match kind with C2r n -> [| (n / 2) + 1 |] | _ -> [| n |] in
      let rs = Random.State.make [| n |] in
      let xs =
        Array.init s.(0) (fun _ ->
            let re = value wide rs in
            let im = match kind with R2c -> 0. | _ -> value wide rs in
            { Complex.re; im })
      in
      let c = { kind; wide; shape = s; axes = [| 0 |]; view = plain 1; xs } in
      String.concat "" (Array.to_list (elements (run c (operand c))))
    in
    let all = List.map one [ C2c Forward; C2c Inverse; R2c; C2r n ] in
    String.sub (Digest.to_hex (Digest.string (String.concat "" all))) 0 16
  in
  let table () =
    List.concat_map
      (fun n ->
        List.map
          (fun wide ->
            Printf.sprintf "%d %s %s" n
              (if wide then "f64" else "f32")
              (digest wide n))
          [ false; true ])
      lengths
    |> String.concat "\n"
  in
  match Support.cpus with
  | [] -> fail "no target"
  | first :: rest ->
      let t = first.around table in
      List.iter
        (fun (b : Support.backend) -> equal text t (b.around table) ~msg:b.name)
        rest;
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
        1000 f32 9f7496c9f569e4b4
        1000 f64 429f0b3827908fdc
        1024 f32 348239c8ad312acd
        1024 f64 06a68c714648e330
        4096 f32 565572e3e5171eca
        4096 f64 d61210f2d5bb445a
        257 f32 cc59d550e2e47f14
        257 f64 0deee0529736803a
        1031 f32 82ef31e36a702475
        1031 f64 ee8d9a2b3e80e82a
        514 f32 05d1f0b872a90809
        514 f64 ad9633a6b7805e61
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

let () =
  exit
    (Windtrap.run "nx_kernel.fft"
       [
         group "nx.cpu"
           [
             prop "transforms are within nx_cpu.mli's bound" case_gen law_bound;
             prop "bits do not depend on views, threads, place or batch"
               case_gen law_bits;
             test "long lines are within the bound, one thread's bits" test_long;
             test "a transform of no point is the empty sum" test_empty;
             test "refuses dtypes and shapes" test_refusals;
             test "digests of transforms, the same on every target"
               test_digests;
           ];
       ])
