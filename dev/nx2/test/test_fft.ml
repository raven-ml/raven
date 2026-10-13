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
    let n = (L.shape (A.layout dst)).(axes.(Array.length axes - 1)) in
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
    let axes = c.axes and s = c.shape in
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
             prop "a declined transform is within its bound" case law_declined;
             test "nx.cpu is as accurate as the expansion on each route"
               test_accuracy;
           ];
       ])
