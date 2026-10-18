(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fft through Nx.Prim, on a library whose transforms are the sums
   Nx_kernel.S.c2c, r2c and c2r state, computed term by term: the values they
   give for small operands, the shapes and dtypes of each transform, its rule
   before any kernel, a split placement, and a library that declines. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

(* What a transform reads and writes. *)
type kind = C2c of Nx_kernel.direction | R2c | C2r of int  (** Of [n] points. *)

(* nx.cpu, with each transform the direct sum of its definition. *)
module Dft = struct
  include Nx_cpu

  let name = "nx.dft"
  let calls = Atomic.make 0

  let complexes (type v s) (a : (v, s) A.t) : Complex.t array =
    match D.kind (A.dtype a) with
    | D.Complex -> A.to_array a
    | D.Float -> Array.map (fun re -> { Complex.re; im = 0. }) (A.to_array a)
    | _ -> invalid_arg "Dft: a dtype of no transform"

  (* [f] of each line of [z], of shape [s], along [axis], each [n] long. *)
  let lines z s axis n f =
    let r = Array.length s in
    let inner =
      Array.fold_left ( * ) 1 (Array.sub s (axis + 1) (r - axis - 1))
    in
    let outer = Array.fold_left ( * ) 1 (Array.sub s 0 axis) in
    let m = s.(axis) in
    let out = Array.make (outer * n * inner) Complex.zero in
    for o = 0 to outer - 1 do
      for i = 0 to inner - 1 do
        let line = Array.init m (fun j -> z.((((o * m) + j) * inner) + i)) in
        Array.iteri (fun t y -> out.((((o * n) + t) * inner) + i) <- y) (f line)
      done
    done;
    let s = Array.copy s in
    s.(axis) <- n;
    (out, s)

  (* [Σ_j x[j] e^(sign 2πi jk/n)] at each [k] below [n]. *)
  let sum sign x n k =
    let acc = ref Complex.zero in
    Array.iteri
      (fun j xj ->
        let t =
          sign *. 2. *. Float.pi *. Float.of_int (j * k) /. Float.of_int n
        in
        acc := Complex.add !acc (Complex.mul xj (Complex.polar 1. t)))
      x;
    !acc

  let dft sign x =
    let n = Array.length x in
    Array.init n (sum sign x n)

  let transform kind ~axes ~dst x =
    Atomic.incr calls;
    let axes = Iarray.to_array axes in
    let last = axes.(Array.length axes - 1) in
    let z = complexes x and shape = L.shape (A.layout x) in
    let along sign (z, shape) a = lines z shape a shape.(a) (dft sign) in
    let z, _ =
      match kind with
      | C2c d ->
          let sign = match d with Forward -> -1. | Inverse -> 1. in
          Array.fold_left (along sign) (z, shape) axes
      | R2c ->
          let z, shape = Array.fold_left (along (-1.)) (z, shape) axes in
          lines z shape last
            ((shape.(last) / 2) + 1)
            (fun l ->
              (* No point is one bin of +0: the empty sum. *)
              Array.init
                ((Array.length l / 2) + 1)
                (fun k -> if k < Array.length l then l.(k) else Complex.zero))
      | C2r n ->
          let rest = Array.sub axes 0 (Array.length axes - 1) in
          let z, shape = Array.fold_left (along 1.) (z, shape) rest in
          lines z shape last n (fun bins ->
              let full =
                Array.init n (fun k ->
                    if k <= n / 2 then bins.(k) else Complex.conj bins.(n - k))
              in
              Array.map (fun c -> { c with Complex.im = 0. }) (dft 1. full))
    in
    let store (type v s) (d : (v, s) A.t) =
      let s = L.shape (A.layout d) in
      let at = Array.make (Array.length s) 0 in
      Array.iteri
        (fun i (c : Complex.t) ->
          let k = ref i in
          for a = Array.length s - 1 downto 0 do
            at.(a) <- !k mod s.(a);
            k := !k / s.(a)
          done;
          match D.kind (A.dtype d) with
          | D.Complex -> A.set d at c
          | D.Float -> A.set d at c.re
          | _ -> invalid_arg "Dft: a dtype of no transform")
        z
    in
    store dst;
    A.Done

  let c2c d ~axes ~dst x = transform (C2c d) ~axes ~dst x
  let r2c ~axes ~dst x = transform R2c ~axes ~dst x

  let c2r ~axes ~dst x =
    let n =
      (L.shape (A.layout dst)).(Iarray.get axes (Iarray.length axes - 1))
    in
    transform (C2r n) ~axes ~dst x
end

let m = Nx_support.memory

module One = (val Nx.devices ~kernels:(module Dft) [ m 0 ])
module Two = (val Nx.devices ~kernels:(module Dft) [ m 0; m 1 ])

let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))
let on a = Nx.place One.on (Nx.Repr.of_array Nx.Host.v a)
let z re im = { Complex.re; im }
let fft f = Nx.Prim.eval ~by:"t" (Fft f)

let complex =
  Testable.make
    ~pp:(fun ppf (c : Complex.t) -> Format.fprintf ppf "%g%+gi" c.re c.im)
    ~equal:(fun (a : Complex.t) b ->
      Float.abs (a.re -. b.re) <= 1e-9 && Float.abs (a.im -. b.im) <= 1e-9)

let near = float 1e-5

let test_c2c () =
  let x =
    on
      (A.of_array D.Complex128 [| 4 |] [| z 1. 0.; z 2. 0.; z 3. 0.; z 4. 0. |])
  in
  let y = fft (C2c { direction = Forward; axes = [| 0 |]; x }) in
  equal ~msg:"forward" (array complex)
    [| z 10. 0.; z (-2.) 2.; z (-2.) 0.; z (-2.) (-2.) |]
    (read y);
  equal ~msg:"inverse, undivided" (array complex)
    [| z 4. 0.; z 8. 0.; z 12. 0.; z 16. 0. |]
    (read (fft (C2c { direction = Inverse; axes = [| 0 |]; x = y })));
  let impulse =
    on
      (A.of_array D.Complex128 [| 2; 3 |]
         (Array.init 6 (fun i -> z (if i = 0 then 1. else 0.) 0.)))
  in
  equal ~msg:"an impulse over both axes" (array complex)
    (Array.make 6 (z 1. 0.))
    (read (fft (C2c { direction = Forward; axes = [| 0; 1 |]; x = impulse })))

let test_real () =
  let x = on (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |]) in
  let y = fft (R2c { dtype = Nx.complex64; axes = [| 0 |]; x }) in
  equal ~msg:"r2c: bins 0 to n/2" (array int) [| 3 |] (Nx.shape y);
  equal ~msg:"r2c" (array complex)
    [| z 10. 0.; z (-2.) 2.; z (-2.) 0. |]
    (read y);
  let back = fft (C2r { dtype = Nx.float32; n = 4; axes = [| 0 |]; x = y }) in
  equal ~msg:"c2r of 4 points, undivided" (array near) [| 4.; 8.; 12.; 16. |]
    (read back);
  let odd = fft (C2r { dtype = Nx.float32; n = 5; axes = [| 0 |]; x = y }) in
  equal ~msg:"c2r of 5 points" (array int) [| 5 |] (Nx.shape odd)

let test_split () =
  let xs =
    Array.init 8 (fun i -> z (Float.of_int i) (Float.of_int (i mod 3)))
  in
  let a = A.of_array D.Complex128 [| 2; 4 |] xs in
  let one = fft (C2c { direction = Forward; axes = [| 1 |]; x = on a }) in
  let x = Nx.place (Two.split ~axis:0) (Nx.Repr.of_array Nx.Host.v a) in
  let two = fft (C2c { direction = Forward; axes = [| 1 |]; x }) in
  equal ~msg:"as on one device" (array complex) (read one) (read two);
  equal ~msg:"split as its operand" bool true
    (Option.equal Nx.Placement.equal (Nx.placement two)
       (Some (Two.split ~axis:0)))

let test_rules () =
  let x = on (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |]) in
  let c = on (A.of_array D.Complex64 [| 3 |] (Array.make 3 Complex.zero)) in
  Atomic.set Dft.calls 0;
  let raises f = raises_match (Exn.invalid_arg ~substring:"t: ") f in
  raises (fun () -> fft (R2c { dtype = Nx.complex128; axes = [| 0 |]; x }));
  raises (fun () ->
      fft
        (R2c { dtype = Nx.complex64; axes = [| 0 |]; x = Nx.cast Nx.float16 x }));
  raises (fun () ->
      fft (C2r { dtype = Nx.float32; n = 6; axes = [| 0 |]; x = c }));
  raises (fun () ->
      fft (C2r { dtype = Nx.float64; n = 4; axes = [| 0 |]; x = c }));
  raises (fun () -> fft (C2c { direction = Forward; axes = [||]; x = c }));
  raises (fun () -> fft (C2c { direction = Forward; axes = [| 1 |]; x = c }));
  equal ~msg:"kernel calls" int 0 (Atomic.get Dft.calls)

(* Declines: on a library that declines every transform, a transform is nx's
   expansion. Against the direct sums, within the bound Expand states: [‖ŷ - y‖₂
   <= 20 u P √N ‖x‖₂], [N] the points transformed, [P] the passes, [log₂ n] for
   an axis of a power of two [n] and [3 log₂ M + 2] for another, [M] the power
   of two its convolution runs at, and [u] the parts' unit roundoff; a C2r's
   [‖x‖₂] is its completed bins', at most [√2] times its operand's. *)

(* nx.cpu, declining every transform. *)
module Untransformed = struct
  include Nx_cpu

  let name = "nx.untransformed"
  let c2c _ ~axes:_ ~dst:_ _ = A.Declined
  let r2c ~axes:_ ~dst:_ _ = A.Declined
  let c2r ~axes:_ ~dst:_ _ = A.Declined
end

module Dec = (val Nx.devices ~kernels:(module Untransformed) [ m 2 ])

(* A declined C2r reads only the real parts of bins 0 and, for an even n, n/2,
   as {!Nx_kernel.S.c2r} states: those of NaN and infinite imaginary parts are
   those of +0 ones. *)
let test_declined_c2r () =
  List.iter
    (fun n ->
      let h = (n / 2) + 1 in
      let bins im =
        Array.init h (fun k ->
            z (Float.of_int (k + 1)) (if k = 0 || 2 * k = n then im k else 0.5))
      in
      let c2r xs =
        let x =
          Nx.place Dec.on
            (Nx.Repr.of_array Nx.Host.v (A.of_array D.Complex128 [| h |] xs))
        in
        read (fft (C2r { dtype = Nx.float64; n; axes = [| 0 |]; x }))
      in
      equal
        ~msg:(Printf.sprintf "%d points" n)
        (array float_exact)
        (c2r (bins (fun _ -> 0.)))
        (c2r (bins (fun k -> if k = 0 then Float.nan else Float.infinity))))
    [ 6; 7; 8 ]

type transform =
  | Complex of Nx_kernel.direction
  | Real
  | Back of int  (** C2r of [n] points. *)

type case = {
  shape : int array;
  axes : int array;
  kind : transform;
  wide : bool;  (** complex128 or float64. *)
  xs : Complex.t array;
}

let pp_case ppf c =
  let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "[%s] axes [%s] %s %s" (ints c.shape) (ints c.axes)
    (match c.kind with
    | Complex Forward -> "forward"
    | Complex Inverse -> "inverse"
    | Real -> "r2c"
    | Back n -> Printf.sprintf "c2r %d" n)
    (if c.wide then "64-bit parts" else "32-bit parts")

let case =
  let open Gen in
  with_pp pp_case
  @@ let* shape = array ~size:(int_range 1 3) (int_range 0 12) in
     let r = Array.length shape in
     let* chosen = array ~size:(constant r) bool in
     let* last = int_range 0 (r - 1) in
     let axes =
       Array.of_list
         (List.filter (fun a -> a = last || chosen.(a)) (List.init r Fun.id))
     in
     let* kind = of_list [ Complex Forward; Complex Inverse; Real; Back 0 ] in
     let* kind =
       match kind with
       | Back _ ->
           let+ n = int_range 0 12 in
           Back n
       | k -> constant k
     in
     (* A C2r reads n/2 + 1 bins along its last axis. *)
     let shape =
       match kind with
       | Back n ->
           let l = axes.(Array.length axes - 1) in
           Array.mapi (fun a e -> if a = l then (n / 2) + 1 else e) shape
       | Complex _ | Real -> shape
     in
     let* wide = bool in
     let part = map (fun i -> Float.of_int i /. 8.) (int_range (-16) 16) in
     let+ xs =
       array
         ~size:(constant (Array.fold_left ( * ) 1 shape))
         (let+ re = part and+ im = part in
          { Complex.re; im })
     in
     { shape; axes; kind; wide; xs }

let log2_up n =
  let rec go p k = if p >= n then k else go (2 * p) (k + 1) in
  go 1 0

(* The passes of an axis of [n] points. *)
let passes n =
  if n land (n - 1) = 0 then log2_up n else (3 * log2_up ((2 * n) - 1)) + 2

let law_declined c =
  let l = c.axes.(Array.length c.axes - 1) in
  let points a =
    match c.kind with Back n when a = l -> n | _ -> c.shape.(a)
  in
  cover "an axis not a power of two"
    (Array.exists (fun a -> points a land (points a - 1) <> 0) c.axes);
  cover "several axes" (Array.length c.axes > 1);
  let p = Array.fold_left (fun k a -> k + passes (points a)) 0 c.axes in
  let n = Array.fold_left (fun k a -> k * points a) 1 c.axes in
  let norm2 =
    Array.fold_left
      (fun m (z : Complex.t) ->
        match c.kind with
        | Real -> m +. (z.re *. z.re)
        | Complex _ -> m +. (z.re *. z.re) +. (z.im *. z.im)
        | Back _ -> m +. (2. *. ((z.re *. z.re) +. (z.im *. z.im))))
      0. c.xs
  in
  let u = if c.wide then epsilon_float /. 2. else 0x1p-24 in
  let bound = 20. *. u *. Float.of_int p *. sqrt (Float.of_int n *. norm2) in
  let reals xs = Array.map (fun (z : Complex.t) -> z.re) xs in
  let complexes xs = Array.map (fun re -> { Complex.re; im = 0. }) xs in
  let compute p =
    let on a = Nx.place p (Nx.Repr.of_array Nx.Host.v a) in
    let axes = Iarray.of_array c.axes and s = c.shape in
    match (c.kind, c.wide) with
    | Complex direction, true ->
        let x = on (A.of_array D.Complex128 s c.xs) in
        read (fft (C2c { direction; axes; x }))
    | Complex direction, false ->
        let x = on (A.of_array D.Complex64 s c.xs) in
        read (fft (C2c { direction; axes; x }))
    | Real, true ->
        let x = on (A.of_array D.Float64 s (reals c.xs)) in
        read (fft (R2c { dtype = Nx.complex128; axes; x }))
    | Real, false ->
        let x = on (A.of_array D.Float32 s (reals c.xs)) in
        read (fft (R2c { dtype = Nx.complex64; axes; x }))
    | Back n, true ->
        let x = on (A.of_array D.Complex128 s c.xs) in
        complexes (read (fft (C2r { dtype = Nx.float64; n; axes; x })))
    | Back n, false ->
        let x = on (A.of_array D.Complex64 s c.xs) in
        complexes (read (fft (C2r { dtype = Nx.float32; n; axes; x })))
  in
  let want = compute One.on and got = compute Dec.on in
  equal ~msg:"shape" int (Array.length want) (Array.length got);
  let err =
    sqrt
      (Array.fold_left ( +. ) 0.
         (Array.map2
            (fun (a : Complex.t) (b : Complex.t) ->
              let d = Complex.sub a b in
              (d.re *. d.re) +. (d.im *. d.im))
            want got))
  in
  at_most ~msg:"‖ŷ - y‖₂ within the bound" float_exact ~than:bound err

(* Accuracy: nx.cpu's transforms are no less accurate than the expansion, on
   each of its routes, against sums in double-double. Errors are summed in
   squares over fixed seeds, so the law is deterministic, and over at least
   4096 points, which few roundings would sway on a short line. *)

let two_sum a b =
  let s = a +. b in
  let bb = s -. a in
  (s, a -. (s -. bb) +. (b -. bb))

let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  two_sum s (e +. al +. bl)

let dd_mul (ah, al) (bh, bl) =
  let p = ah *. bh in
  two_sum p (Float.fma ah bh (-.p) +. (ah *. bl) +. (al *. bh))

let dd_neg (h, l) = (-.h, -.l)

let dd_div (h, l) d =
  let q = h /. d in
  let p = q *. d in
  two_sum q ((h -. p -. Float.fma q d (-.p) +. l) /. d)

(* e^(-2πi m/n), 0 <= m < n, in double-double: the angle within its quarter
   taken from the nearer end, sin and cos by their series. *)
let unit m n =
  let q = 4 * m / n in
  let r = (4 * m) - (q * n) in
  let far = 2 * r > n in
  let pi_2 = (1.5707963267948966, 6.123233995736766e-17) in
  let phi =
    dd_mul pi_2
      (dd_div (Float.of_int (if far then n - r else r), 0.) (Float.of_int n))
  in
  let s = ref (0., 0.) and c = ref (0., 0.) and t = ref (1., 0.) in
  for k = 0 to 27 do
    (match k mod 4 with
    | 0 -> c := dd_add !c !t
    | 1 -> s := dd_add !s !t
    | 2 -> c := dd_add !c (dd_neg !t)
    | _ -> s := dd_add !s (dd_neg !t));
    t := dd_div (dd_mul !t phi) (Float.of_int (k + 1))
  done;
  let c, s = if far then (!s, !c) else (!c, !s) in
  let re, im =
    match q with
    | 0 -> (c, s)
    | 1 -> (dd_neg s, c)
    | 2 -> (dd_neg c, dd_neg s)
    | _ -> (s, dd_neg c)
  in
  (re, dd_neg im)

(* A sum in double-double, its low parts added without renormalising, which
   errs by about n u² of double over n terms. *)
type sum = { mutable hi : float; mutable lo : float }

(* [s] plus [a·(bh + bl)], [a] a double. *)
let[@inline] add s a bh bl =
  let p = a *. bh in
  let e = Float.fma a bh (-.p) +. (a *. bl) in
  let t = s.hi +. p in
  let bb = t -. s.hi in
  s.lo <- s.lo +. (s.hi -. (t -. bb)) +. (p -. bb) +. e;
  s.hi <- t

(* The transform of [direction] of [x], each part a double-double. *)
let exact direction (x : Complex.t array) =
  let n = Array.length x in
  let w = Array.init n (fun m -> unit m n) in
  let sign = match direction with Nx_kernel.Forward -> 1. | Inverse -> -1. in
  let wrh = Array.map (fun ((h, _), _) -> h) w
  and wrl = Array.map (fun ((_, l), _) -> l) w
  and wih = Array.map (fun (_, (h, _)) -> sign *. h) w
  and wil = Array.map (fun (_, (_, l)) -> sign *. l) w in
  Array.init n (fun k ->
      let re = { hi = 0.; lo = 0. } and im = { hi = 0.; lo = 0. } in
      for j = 0 to n - 1 do
        let t = j * k mod n and xr = x.(j).re and xi = x.(j).im in
        add re xr wrh.(t) wrl.(t);
        add re xi (-.wih.(t)) (-.wil.(t));
        add im xr wih.(t) wil.(t);
        add im xi wrh.(t) wrl.(t)
      done;
      ((re.hi, re.lo), (im.hi, im.lo)))

module Cpu = (val Nx.devices [ m 3 ])

(* What a route's case transforms: [n] points, of 64-bit parts where [wide]. *)
type accuracy = { route : string; kind : transform; n : int; wide : bool }

(* ‖ŷ - y‖₂² of nx.cpu's and of the expansion's [c] on the values of [seed]. *)
let errors c seed =
  let rs = Random.State.make [| seed; c.n |] in
  let part () =
    let v = Random.State.float rs 2. -. 1. in
    if c.wide then v else Int32.float_of_bits (Int32.bits_of_float v)
  in
  let n = c.n in
  let count = match c.kind with Back _ -> (n / 2) + 1 | _ -> n in
  let xs =
    Array.init count (fun _ ->
        let re = part () in
        { Complex.re; im = (if c.kind = Real then 0. else part ()) })
  in
  let want =
    match c.kind with
    | Complex d -> exact d xs
    | Real -> Array.sub (exact Forward xs) 0 ((n / 2) + 1)
    | Back _ ->
        let full =
          Array.init n (fun k ->
              if k <= n / 2 then xs.(k) else Complex.conj xs.(n - k))
        in
        Array.map (fun (re, _) -> (re, (0., 0.))) (exact Inverse full)
  in
  let compute p =
    let on a = Nx.place p (Nx.Repr.of_array Nx.Host.v a) in
    let axes = [| 0 |] and s = [| count |] in
    let reals = Array.map (fun (z : Complex.t) -> z.re) in
    let complexes = Array.map (fun re -> { Complex.re; im = 0. }) in
    let axes = Iarray.of_array axes in
    match (c.kind, c.wide) with
    | Complex direction, true ->
        let x = on (A.of_array D.Complex128 s xs) in
        read (fft (C2c { direction; axes; x }))
    | Complex direction, false ->
        let x = on (A.of_array D.Complex64 s xs) in
        read (fft (C2c { direction; axes; x }))
    | Real, true ->
        let x = on (A.of_array D.Float64 s (reals xs)) in
        read (fft (R2c { dtype = Nx.complex128; axes; x }))
    | Real, false ->
        let x = on (A.of_array D.Float32 s (reals xs)) in
        read (fft (R2c { dtype = Nx.complex64; axes; x }))
    | Back n, true ->
        let x = on (A.of_array D.Complex128 s xs) in
        complexes (read (fft (C2r { dtype = Nx.float64; n; axes; x })))
    | Back n, false ->
        let x = on (A.of_array D.Complex64 s xs) in
        complexes (read (fft (C2r { dtype = Nx.float32; n; axes; x })))
  in
  let error (got : Complex.t array) =
    let sum = ref 0. in
    Array.iteri
      (fun k ((rh, rl), (ih, il)) ->
        let dr = got.(k).re -. rh -. rl and di = got.(k).im -. ih -. il in
        sum := !sum +. (dr *. dr) +. (di *. di))
      want;
    !sum
  in
  (error (compute Cpu.on), error (compute Dec.on))

let accuracy_cases =
  let each route ns kinds =
    List.concat_map
      (fun n ->
        List.concat_map
          (fun kind ->
            List.map
              (fun wide -> { route; kind = kind n; n; wide })
              [ false; true ])
          kinds)
      ns
  in
  let complex = [ (fun _ -> Complex Forward); (fun _ -> Complex Inverse) ] in
  let real = [ (fun _ -> Real); (fun n -> Back n) ] in
  each "passes" [ 7; 60; 64; 241 ] complex
  @ each "the four steps" [ 100; 1000; 1024 ] complex
  @ each "Bluestein's" [ 257; 1031 ] (complex @ real)
  @ each "Bluestein's, halved" [ 514 ] real
  @ each "real lines" [ 6; 8; 16; 32; 64; 128; 256; 512; 1024 ] real

let test_accuracy () =
  List.iter
    (fun c ->
      let k = ref 0. and e = ref 0. in
      for seed = 1 to Int.max 8 (4096 / c.n) do
        let ek, ee = errors c seed in
        k := !k +. ek;
        e := !e +. ee
      done;
      let msg =
        Format.asprintf "%s, %d points, %s %s" c.route c.n
          (match c.kind with
          | Complex Forward -> "forward"
          | Complex Inverse -> "inverse"
          | Real -> "r2c"
          | Back _ -> "c2r")
          (if c.wide then "64-bit parts" else "32-bit parts")
      in
      at_most ~msg float_exact ~than:(sqrt !e) (sqrt !k))
    accuracy_cases

(* The functions, through Nx: each against its sum computed in OCaml, at
   float64, and the identities between them, over lengths with no point, one,
   and odd and even ones. *)
module Laws = struct
  let pi = Float.pi

  (* The sum [Σ_j x[j] e^(sign 2πi jk/n)] for each [k]. *)
  let dft ~sign (x : Complex.t array) =
    let n = Array.length x in
    Array.init n (fun k ->
        let acc = ref Complex.zero in
        Array.iteri
          (fun j v ->
            let t = sign *. 2. *. pi *. Float.of_int (j * k mod n) /. Float.of_int n in
            acc := Complex.add !acc (Complex.mul v { re = cos t; im = sin t }))
          x;
        !acc)

  let pp_complex ppf (z : Complex.t) = Format.fprintf ppf "%g%+gi" z.re z.im

  let close tol =
    Testable.make ~pp:pp_complex ~equal:(fun (a : Complex.t) (b : Complex.t) ->
        Complex.norm (Complex.sub a b) <= tol *. Float.max 1. (Complex.norm b))

  let close_floats tol =
    Testable.make ~pp:Format.pp_print_float ~equal:(fun a b ->
        Float.abs (a -. b) <= tol *. Float.max 1. (Float.abs b))

  let signal =
    let open Gen in
    let* n = int_range 0 9 in
    let+ re = array ~size:(constant n) (float_range (-1.) 1.)
    and+ im = array ~size:(constant n) (float_range (-1.) 1.) in
    Array.map2 (fun re im -> { Complex.re; im }) re im

  let complex_of a = Nx.create Nx.complex128 [| Array.length a |] a

  (* Demands that a law meet each length the generator draws. *)
  let cover_lengths n =
    cover "no point" (n = 0);
    cover "one point" (n = 1);
    cover "an odd length" (n > 1 && n mod 2 = 1);
    cover "an even length" (n > 0 && n mod 2 = 0)

  (* The factor of a transform of [n] points, forward or not, under [norm]. *)
  let factor norm ~forward n =
    let f = Float.of_int (Stdlib.max 1 n) in
    match (norm, forward) with
    | Nx.Backward, true | Nx.Forward, false -> 1.
    | Nx.Backward, false | Nx.Forward, true -> 1. /. f
    | Nx.Ortho, _ -> 1. /. Float.sqrt f

  let norms = [ Nx.Backward; Nx.Forward; Nx.Ortho ]

  let law_fft =
    prop "fft and ifft are the sums, ifft (fft x) is x under each norm" signal
      (fun x ->
        let n = Array.length x in
        cover_lengths n;
        let t = complex_of x in
        equal ~msg:"fft" (array (close 1e-12)) (dft ~sign:(-1.) x) (Nx.to_array (Nx.fft t));
        equal ~msg:"ifft" (array (close 1e-12))
          (Array.map (fun z -> Complex.div z { re = Float.of_int (Stdlib.max 1 n); im = 0. }) (dft ~sign:1. x))
          (Nx.to_array (Nx.ifft t));
        List.iter
          (fun norm ->
            equal ~msg:"round trip" (array (close 1e-12)) x
              (Nx.to_array (Nx.ifft ~norm (Nx.fft ~norm t))))
          [ Nx.Backward; Nx.Forward; Nx.Ortho ];
        let energy a = Array.fold_left (fun s z -> s +. Complex.norm2 z) 0. a in
        equal ~msg:"Parseval" (close_floats 1e-12) (energy x)
          (energy (Nx.to_array (Nx.fft ~norm:Ortho t))))

  let scale c a = Array.map (fun z -> Complex.mul { Complex.re = c; im = 0. } z) a

  let law_real =
    prop "rfft is fft's first bins, irfft (rfft x) is x, under each norm" signal
      (fun x ->
        let n = Array.length x in
        cover_lengths n;
        let re = Array.map (fun (z : Complex.t) -> z.re) x in
        let t = Nx.create Nx.float64 [| n |] re in
        let full = dft ~sign:(-1.) (Array.map (fun re -> { Complex.re; im = 0. }) re) in
        List.iter
          (fun norm ->
            let bins = Nx.to_array (Nx.rfft Nx.complex128 ~norm t) in
            equal ~msg:"bins" int ((n / 2) + 1) (Array.length bins);
            if n > 0 then begin
              let want = scale (factor norm ~forward:true n) (Array.sub full 0 ((n / 2) + 1)) in
              equal ~msg:"rfft" (array (close 1e-12)) want bins;
              equal ~msg:"irfft" (array (close_floats 1e-12)) re
                (Nx.to_array (Nx.irfft Nx.float64 ~norm ~s:[ n ] (Nx.rfft Nx.complex128 ~norm t)))
            end)
          norms;
        let narrow = Nx.rfft Nx.complex64 (Nx.cast Nx.float16 t) in
        equal ~msg:"float16 in" int ((n / 2) + 1) (Nx.numel narrow))

  (* irfft's linear map, as nx.mli states it: the bins cut or extended to
     [n/2 + 1], bin [k] above [n/2] the conjugate of bin [n - k], the
     imaginary parts of bin 0 and, for an even [n], bin [n/2] unread. *)
  let irfft_map (z : Complex.t array) n =
    let h = (n / 2) + 1 in
    let bin k =
      let b = if k < Array.length z then z.(k) else Complex.zero in
      if k = 0 || 2 * k = n then { b with im = 0. } else b
    in
    let full = Array.init n (fun k -> if k < h then bin k else Complex.conj (bin (n - k))) in
    Array.map (fun (c : Complex.t) -> c.re) (dft ~sign:1. full)

  let law_irfft =
    prop "irfft is its linear map, Hermitian operand or not"
      Gen.(pair (int_range 1 7) (int_range 1 10))
      (fun (m, n) ->
        cover "bins cut" (m > (n / 2) + 1);
        cover "bins extended" (m < (n / 2) + 1);
        cover "an even n" (n mod 2 = 0);
        cover "an odd n" (n mod 2 = 1);
        let z = Array.init m (fun k -> { Complex.re = cos (Float.of_int (3 * k + 1)); im = sin (Float.of_int (5 * k + 2)) }) in
        let t = complex_of z in
        List.iter
          (fun norm ->
            let want = Array.map (fun v -> v *. factor norm ~forward:false n) (irfft_map z n) in
            equal ~msg:"irfft" (array (close_floats 1e-12)) want
              (Nx.to_array (Nx.irfft Nx.float64 ~norm ~s:[ n ] t)))
          norms;
        let n' = 2 * (m - 1) in
        if n' > 0 then
          equal ~msg:"2 (m - 1) points" (array (close_floats 1e-12)) (Array.map (fun v -> v /. Float.of_int n') (irfft_map z n'))
            (Nx.to_array (Nx.irfft Nx.float64 t)))

  (* Lengths past the generator's, powers of two and others, Bluestein's
     among them: complex128 within 1e-13 and complex64 within 1e-5 of the
     sums, in the 2-norm relative to the sums'. *)
  let test_long () =
    let norm2 a = Float.sqrt (Array.fold_left (fun s z -> s +. Complex.norm2 z) 0. a) in
    let within ~msg tol want got =
      let err = norm2 (Array.map2 Complex.sub got want) in
      less ~msg float_exact ~than:(tol *. norm2 want) err
    in
    List.iter
      (fun n ->
        let x = Array.init n (fun j -> { Complex.re = sin (Float.of_int (7 * j)); im = cos (Float.of_int (3 * j)) }) in
        let msg = Printf.sprintf "%d points" n in
        let t = complex_of x in
        let back = Array.map (fun z -> Complex.div z { re = Float.of_int n; im = 0. }) (dft ~sign:1. x) in
        within ~msg:(msg ^ ", fft") 1e-13 (dft ~sign:(-1.) x) (Nx.to_array (Nx.fft t));
        within ~msg:(msg ^ ", ifft") 1e-13 back (Nx.to_array (Nx.ifft t));
        let t64 = Nx.cast Nx.complex64 t in
        let x64 = Nx.to_array t64 in
        within ~msg:(msg ^ ", complex64 fft") 1e-5 (dft ~sign:(-1.) x64) (Nx.to_array (Nx.fft t64));
        within ~msg:(msg ^ ", complex64 ifft") 1e-5
          (Array.map (fun z -> Complex.div z { re = Float.of_int n; im = 0. }) (dft ~sign:1. x64))
          (Nx.to_array (Nx.ifft t64)))
      [ 10; 16; 17; 100; 128; 257; 514; 1031 ]

  (* [x] of shape [[m; n]], transformed along both axes in either order. *)
  let law_axes =
    prop "a transform along two axes is one along each in turn"
      Gen.(pair (int_range 1 4) (int_range 1 5))
      (fun (m, n) ->
        let x = Nx.init Nx.float64 [| m; n |] (fun i -> sin (Float.of_int ((i.(0) * 7) + i.(1)))) in
        let z = Nx.cast Nx.complex128 x in
        let both = Nx.fft ~axes:[ 0; 1 ] z in
        let turns = Nx.fft ~axes:[ 0 ] (Nx.fft ~axes:[ 1 ] z) in
        equal ~msg:"fft" (array (close 1e-12)) (Nx.to_array turns) (Nx.to_array both);
        equal ~msg:"axes in any order" (array (close 1e-12)) (Nx.to_array both)
          (Nx.to_array (Nx.fft ~axes:[ 1; 0 ] z));
        let first = Nx.rfft Nx.complex128 ~axes:[ 1; 0 ] x in
        let want = Nx.slice [ Nx.R (0, (m / 2) + 1); Nx.A ] both in
        equal ~msg:"rfft, real axis first" (array (close 1e-12)) (Nx.to_array want) (Nx.to_array first);
        equal ~msg:"irfft, real axis first" (array (close_floats 1e-12)) (Nx.to_array x)
          (Nx.to_array (Nx.irfft Nx.float64 ~axes:[ 1; 0 ] ~s:[ n; m ] first)))

  let test_sizes () =
    let x = Nx.create Nx.complex128 [| 3 |] [| { re = 1.; im = 0. }; { re = 2.; im = 0. }; { re = 3.; im = 0. } |] in
    let padded = Nx.create Nx.complex128 [| 5 |] [| { re = 1.; im = 0. }; { re = 2.; im = 0. }; { re = 3.; im = 0. }; Complex.zero; Complex.zero |] in
    equal ~msg:"extended" (array (close 1e-12)) (Nx.to_array (Nx.fft padded)) (Nx.to_array (Nx.fft ~s:[ 5 ] x));
    equal ~msg:"cut" (array (close 1e-12)) (Nx.to_array (Nx.fft (Nx.slice [ Nx.R (0, 2) ] x)))
      (Nx.to_array (Nx.fft ~s:[ 2 ] x));
    equal ~msg:"no point" int 0 (Nx.numel (Nx.fft ~s:[ 0 ] x));
    (* A part scaled alone: (inf + 0i) / 2 keeps its zero. *)
    let inf = Nx.create Nx.complex128 [| 2 |] [| { re = Float.infinity; im = 0. }; Complex.zero |] in
    let y = Nx.to_array (Nx.fft ~norm:Forward inf) in
    let parts = Array.concat (Array.to_list (Array.map (fun (z : Complex.t) -> [| z.re; z.im |]) y)) in
    equal ~msg:"infinite part" (array float_exact)
      [| Float.infinity; 0.; Float.infinity; 0. |] parts

  (* The cosine and sine transforms' sums, at float64, as nx.mli states them. *)
  let reference family kind (x : float array) =
    let n = Array.length x in
    let f = Float.of_int n in
    let sum lo hi g = let s = ref 0. in for j = lo to hi do s := !s +. g j done; !s in
    let sign k = if k mod 2 = 0 then 1. else -1. in
    Array.init n (fun k ->
        let k' = Float.of_int k in
        match (family, kind) with
        | `C, 1 -> x.(0) +. (sign k *. x.(n - 1)) +. (2. *. sum 1 (n - 2) (fun j -> x.(j) *. cos (pi *. Float.of_int j *. k' /. (f -. 1.))))
        | `C, 2 -> 2. *. sum 0 (n - 1) (fun j -> x.(j) *. cos (pi *. k' *. ((2. *. Float.of_int j) +. 1.) /. (2. *. f)))
        | `C, 3 -> x.(0) +. (2. *. sum 1 (n - 1) (fun j -> x.(j) *. cos (pi *. Float.of_int j *. ((2. *. k') +. 1.) /. (2. *. f))))
        | `C, _ -> 2. *. sum 0 (n - 1) (fun j -> x.(j) *. cos (pi *. ((2. *. Float.of_int j) +. 1.) *. ((2. *. k') +. 1.) /. (4. *. f)))
        | `S, 1 -> 2. *. sum 0 (n - 1) (fun j -> x.(j) *. sin (pi *. (Float.of_int j +. 1.) *. (k' +. 1.) /. (f +. 1.)))
        | `S, 2 -> 2. *. sum 0 (n - 1) (fun j -> x.(j) *. sin (pi *. (k' +. 1.) *. ((2. *. Float.of_int j) +. 1.) /. (2. *. f)))
        | `S, 3 -> (sign k *. x.(n - 1)) +. (2. *. sum 0 (n - 2) (fun j -> x.(j) *. sin (pi *. (Float.of_int j +. 1.) *. ((2. *. k') +. 1.) /. (2. *. f))))
        | `S, _ -> 2. *. sum 0 (n - 1) (fun j -> x.(j) *. sin (pi *. ((2. *. Float.of_int j) +. 1.) *. ((2. *. k') +. 1.) /. (4. *. f))))

  let real_signal =
    let open Gen in
    let* n = int_range 0 9 in
    array ~size:(constant n) (float_range (-1.) 1.)

  let types = [ (Nx.Type_1, 1); (Nx.Type_2, 2); (Nx.Type_3, 3); (Nx.Type_4, 4) ]

  let law_trigonometric =
    prop "dct and dst are their sums, and invert under each norm" real_signal (fun x ->
        let n = Array.length x in
        cover_lengths n;
        let t = Nx.create Nx.float64 [| n |] x in
        let t32 = Nx.cast Nx.float32 t in
        List.iter
          (fun family ->
            List.iter
              (fun (type_, kind) ->
                if not (family = `C && kind = 1 && n = 1) then begin
                  let forward, inverse = match family with `C -> (Nx.dct, Nx.idct) | `S -> (Nx.dst, Nx.idst) in
                  let want = reference family kind x in
                  equal ~msg:"sum" (array (close_floats 1e-12)) want
                    (Nx.to_array (forward ~type_ t));
                  List.iter
                    (fun norm ->
                      equal ~msg:"round trip" (array (close_floats 1e-12)) x
                        (Nx.to_array (inverse ~type_ ~norm (forward ~type_ ~norm t))))
                    norms;
                  let energy a = Array.fold_left (fun s v -> s +. (v *. v)) 0. a in
                  equal ~msg:"orthonormal" (close_floats 1e-12) (energy x)
                    (energy (Nx.to_array (forward ~type_ ~norm:Ortho t)));
                  equal ~msg:"float32" (array (close_floats 1e-5)) want
                    (Nx.to_array (Nx.cast Nx.float64 (forward ~type_ t32)));
                  (* A narrow format computes at float32 and rounds once. *)
                  let narrow (type s) (dt : (float, s) Nx.dtype) =
                    let h = Nx.cast dt t in
                    equal ~msg:("at float32, " ^ Nx.Dtype.name dt) (array float_exact)
                      (Nx.to_array (Nx.cast dt (forward ~type_ (Nx.cast Nx.float32 h))))
                      (Nx.to_array (forward ~type_ h))
                  in
                  narrow Nx.float16;
                  narrow Nx.bfloat16
                end)
              types)
          [ `C; `S ])

  let test_trigonometric_empty () =
    List.iter
      (fun (msg, f) ->
        List.iter
          (fun (type_, _) ->
            List.iter
              (fun (s, axes) ->
                let y = f ~type_ ~axes (Nx.zeros Nx.float32 s) in
                equal ~msg (array int) s (Nx.shape y);
                equal ~msg:(msg ^ ", its dtype") string "float32" (Nx.Dtype.name (Nx.dtype y)))
              [ ([| 0 |], [ 0 ]); ([| 0; 3 |], [ 0; 1 ]); ([| 3; 0 |], [ 1 ]); ([| 2; 0 |], [ 0 ]) ])
          types)
      [ ("dct", fun ~type_ ~axes x -> Nx.dct ~type_ ~axes x);
        ("idct", fun ~type_ ~axes x -> Nx.idct ~type_ ~axes x);
        ("dst", fun ~type_ ~axes x -> Nx.dst ~type_ ~axes x);
        ("idst", fun ~type_ ~axes x -> Nx.idst ~type_ ~axes x) ]

  let test_hann () =
    equal ~msg:"hann 4" (array (close_floats 1e-15)) [| 0.; 0.5; 1.; 0.5 |]
      (Nx.to_array (Nx.hann Nx.float64 4));
    equal ~msg:"hann 1" (array (close_floats 0.)) [| 0. |] (Nx.to_array (Nx.hann Nx.float32 1));
    equal ~msg:"hann 0" (array int) [| 0 |] (Nx.shape (Nx.hann Nx.float32 0))

  let law_stft =
    prop "istft (stft x) is x where a taper holds it"
      Gen.(triple (int_range 1 9) (int_range 1 9) (int_range 9 40))
      (fun (window, step, n) ->
        let step = Stdlib.min step window in
        assume (window <= n);
        let x = Nx.init Nx.float64 [| 2; n |] (fun i -> cos (Float.of_int ((i.(0) * 13) + (i.(1) * 3)))) in
        let z = Nx.stft Nx.complex128 ~window ~step x in
        let frames = ((n - window) / step) + 1 in
        equal ~msg:"shape" (array int) [| 2; frames; (window / 2) + 1 |] (Nx.shape z);
        let covered = ((frames - 1) * step) + window in
        let ones = Nx.ones Nx.float64 [| window |] in
        let flat = Nx.istft Nx.float64 ~window ~step ~win:ones ~length:n
            (Nx.stft Nx.complex128 ~window ~step ~win:ones x) in
        let want = Nx.pad [| (0, 0); (0, n - covered) |] 0. (Nx.slice [ Nx.A; Nx.R (0, covered) ] x) in
        equal ~msg:"untapered" (array (close_floats 1e-12)) (Nx.to_array want) (Nx.to_array flat);
        let y = Nx.istft Nx.float64 ~window ~step z in
        let k = (covered / 2) + 1 in
        cover "length cuts" (k < covered);
        equal ~msg:"length cuts" (array (close_floats 0.))
          (Nx.to_array (Nx.slice [ Nx.A; Nx.R (0, k) ] y))
          (Nx.to_array (Nx.istft Nx.float64 ~window ~step ~length:k z));
        let w = Nx.to_array (Nx.hann Nx.float64 window) in
        let xs = Nx.to_array x and ys = Nx.to_array y in
        for row = 0 to 1 do
          for i = 0 to covered - 1 do
            (* A sample is held where a frame's taper there is not zero. *)
            let held = ref false in
            for f = 0 to frames - 1 do
              let j = i - (f * step) in
              if j >= 0 && j < window && w.(j) <> 0. then held := true
            done;
            let want = if !held then xs.((row * n) + i) else 0. in
            equal ~msg:"tapered" (close_floats 1e-12) want ys.((row * covered) + i)
          done
        done)

  let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

  let test_refusals () =
    let z = Nx.zeros Nx.complex64 [| 4 |] and x = Nx.zeros Nx.float32 [| 4 |] in
    invalid ~by:"Nx.fft" (fun () -> Nx.fft (Nx.zeros Nx.complex64 [||]));
    invalid ~by:"Nx.fft" (fun () -> Nx.fft ~axes:[] z);
    invalid ~by:"Nx.fft" (fun () -> Nx.fft ~axes:[ 1 ] z);
    invalid ~by:"Nx.ifft" (fun () -> Nx.ifft ~axes:[ 0; 0 ] (Nx.zeros Nx.complex64 [| 2; 2 |]));
    invalid ~by:"Nx.fft" (fun () -> Nx.fft ~s:[ 1; 2 ] z);
    invalid ~by:"Nx.rfft" (fun () -> Nx.rfft Nx.complex64 ~s:[ -1 ] x);
    invalid ~by:"Nx.irfft" (fun () -> Nx.irfft Nx.float32 (Nx.zeros Nx.complex64 [| 0 |]));
    invalid ~by:"Nx.dct" (fun () -> Nx.dct ~type_:Type_1 (Nx.zeros Nx.float32 [| 1 |]));
    invalid ~by:"Nx.idct" (fun () -> Nx.idct ~type_:Type_1 (Nx.zeros Nx.float32 [| 0; 1 |]));
    invalid ~by:"Nx.dst" (fun () -> Nx.dst ~axes:[ 1 ] x);
    invalid ~by:"Nx.idst" (fun () -> Nx.idst ~axes:[ 0; 0 ] x);
    invalid ~by:"Nx.hann" (fun () -> Nx.hann Nx.float32 (-1));
    invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:5 x);
    invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:2 ~win:x x);
    invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 ~step:5 (Nx.zeros Nx.complex64 [| 2; 3 |]));
    invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 (Nx.zeros Nx.complex64 [| 2; 4 |]))

  (* A composite's refusal from a function it calls names the composite: a
     taper on another device than the signal. *)
  let test_composites_named () =
    let on k x = Nx.place (Nx.Placement.device Two.v (m k)) x in
    let x = on 0 (Nx.zeros Nx.float32 [| 8 |]) in
    let win = on 1 (Nx.ones Nx.float32 [| 4 |]) in
    invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:4 ~win x);
    let z = on 0 (Nx.zeros Nx.complex64 [| 2; 3 |]) in
    invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 ~win z)
end

let () =
  exit
    (run "nx fft"
       [
         group "transforms"
           [
             test "complex to complex" test_c2c;
             test "real to complex and back" test_real;
             test "a split value transforms as on one device" test_split;
           ];
         group "rules and declines"
           [
             test "the rule raises before any kernel" test_rules;
             test "a declined c2r ignores bins 0's and n/2's imaginary parts"
               test_declined_c2r;
             prop "a declined transform is within its bound" case law_declined;
             test "nx.cpu is as accurate as the expansion on each route"
               test_accuracy;
           ];
         group "functions"
           [
             Laws.law_fft;
             Laws.law_real;
             Laws.law_irfft;
             Laws.law_axes;
             test "long lengths at complex128 and complex64" Laws.test_long;
             test "s cuts and extends, a part scales alone" Laws.test_sizes;
             Laws.law_trigonometric;
             test "an axis of no element" Laws.test_trigonometric_empty;
             test "hann" Laws.test_hann;
             Laws.law_stft;
             test "each refusal names its function" Laws.test_refusals;
             test "a composite names its function" Laws.test_composites_named;
           ];
       ])
