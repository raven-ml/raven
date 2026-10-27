(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fft through Nx.Prim, on a library whose transforms are the sums
   Nx_kernel.S.c2c, r2c and c2r state, computed term by term: the values they
   give for small operands, the shapes and dtypes of each transform, its rule
   before any kernel, a split placement, and nx.cpu's reading of a c2r's
   bins. *)

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
    let z = complexes x and shape = Iarray.to_array (L.shape (A.layout x)) in
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
      let s = Iarray.to_array (L.shape (A.layout d)) in
      let at = Array.make (Array.length s) 0 in
      Array.iteri
        (fun i (c : Complex.t) ->
          let k = ref i in
          for a = Array.length s - 1 downto 0 do
            at.(a) <- !k mod s.(a);
            k := !k / s.(a)
          done;
          let at = Iarray.of_array at in
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
      let shape = L.shape (A.layout dst) in
      Iarray.get shape (Iarray.get axes (Iarray.length axes - 1))
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

(* nx.cpu's C2r reads only the real parts of bins 0 and, for an even n, n/2,
   as {!Nx_kernel.S.c2r} states: those of NaN and infinite imaginary parts are
   those of +0 ones. *)
let test_c2r_bins () =
  List.iter
    (fun n ->
      let h = (n / 2) + 1 in
      let bins im =
        Array.init h (fun k ->
            z (Float.of_int (k + 1)) (if k = 0 || 2 * k = n then im k else 0.5))
      in
      let c2r xs =
        let x =
          Nx.Repr.of_array Nx.Host.v (A.of_array D.Complex128 [| h |] xs)
        in
        read (fft (C2r { dtype = Nx.float64; n; axes = [| 0 |]; x }))
      in
      equal
        ~msg:(Printf.sprintf "%d points" n)
        (array float_exact)
        (c2r (bins (fun _ -> 0.)))
        (c2r (bins (fun k -> if k = 0 then Float.nan else Float.infinity))))
    [ 6; 7; 8 ]

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

  (* A float narrower than float32 transforms at float32: rfft of it has
     the bits of rfft of its float32 cast, and irfft into it is the float32
     result rounded once. The signal is one where a transform at float64
     rounded once gives other bits: for rfft at every format, for irfft at
     float16 and bfloat16, while a float8 result's three bits of fraction
     store the same from either precision, so there it holds the rounding
     alone. *)
  let test_narrow () =
    let x =
      Nx.create Nx.float32 [| 256 |]
        (Array.init 256 (fun i -> 3. *. sin (Float.of_int (i * i))))
    in
    let narrow (type s) (dt : (float, s) Nx.dtype) =
      let h = Nx.cast dt x in
      let wide = Nx.cast Nx.float32 h in
      let bins z = Nx.to_array (Nx.bitcast Nx.int32 (Nx.bitcast Nx.float32 z)) in
      equal ~msg:("rfft of " ^ Nx.Dtype.name dt) (array int32)
        (bins (Nx.rfft Nx.complex64 wide))
        (bins (Nx.rfft Nx.complex64 h));
      let z = Nx.rfft Nx.complex64 wide in
      let points y = Nx.to_array (Nx.bitcast Nx.int64 (Nx.cast Nx.float64 y)) in
      equal ~msg:("irfft into " ^ Nx.Dtype.name dt) (array int64)
        (points (Nx.cast dt (Nx.irfft Nx.float32 z)))
        (points (Nx.irfft dt z))
    in
    narrow Nx.float16;
    narrow Nx.bfloat16;
    narrow Nx.float8_e4m3fn;
    narrow Nx.float8_e5m2

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

  let ints x = Array.map Int32.to_int (Nx.to_array x)

  let test_frequencies () =
    let freqs = Nx.to_array in
    equal ~msg:"fftfreq 4" (array float_exact) [| 0.; 0.25; -0.5; -0.25 |]
      (freqs (Nx.fftfreq Nx.float64 4));
    equal ~msg:"fftfreq 5, d 0.5" (array float_exact)
      [| 0.; 0.4; 0.8; -0.8; -0.4 |]
      (freqs (Nx.fftfreq Nx.float64 ~d:0.5 5));
    equal ~msg:"fftfreq 1" (array float_exact) [| 0. |] (freqs (Nx.fftfreq Nx.float32 1));
    equal ~msg:"rfftfreq 4" (array float_exact) [| 0.; 0.25; 0.5 |]
      (freqs (Nx.rfftfreq Nx.float64 4));
    equal ~msg:"rfftfreq 5" (array float_exact) [| 0.; 0.2; 0.4 |]
      (freqs (Nx.rfftfreq Nx.float64 5));
    equal ~msg:"fftshift (fftfreq 5)" (array float_exact)
      [| -0.4; -0.2; 0.; 0.2; 0.4 |]
      (freqs (Nx.fftshift (Nx.fftfreq Nx.float64 5)));
    let x = Nx.arange Nx.int32 0 6 1 |> Nx.reshape [| 2; 3 |] in
    equal ~msg:"fftshift, last axis" (array int) [| 2; 0; 1; 5; 3; 4 |] (ints (Nx.fftshift x));
    equal ~msg:"fftshift, both axes" (array int) [| 5; 3; 4; 2; 0; 1 |]
      (ints (Nx.fftshift ~axes:[ 0; 1 ] x));
    equal ~msg:"ifftshift, last axis" (array int) [| 1; 2; 0; 4; 5; 3 |] (ints (Nx.ifftshift x))

  (* Each frequency is [i / (d n)] at float64, [i] wrapped past the middle,
     stored once; and fftshift orders them, zero at [n / 2]. *)
  let law_frequencies =
    prop "fftfreq is i / (d n) at float64, ascending after fftshift"
      Gen.(pair (int_range 1 64) (int_range 1 16))
      (fun (n, k) ->
        let d = Float.of_int k /. 8. and f = Float.of_int n in
        let i_of j = if j < (n + 1) / 2 then j else j - n in
        let want = Array.init n (fun j -> Float.of_int (i_of j) /. (d *. f)) in
        equal ~msg:"fftfreq" (array float_exact) want (Nx.to_array (Nx.fftfreq Nx.float64 ~d n));
        let real = Array.init ((n / 2) + 1) (fun j -> Float.of_int j /. (d *. f)) in
        equal ~msg:"rfftfreq" (array float_exact) real (Nx.to_array (Nx.rfftfreq Nx.float64 ~d n));
        let shifted = Nx.to_array (Nx.fftshift (Nx.fftfreq Nx.float64 ~d n)) in
        equal ~msg:"zero at n / 2" float_exact 0. shifted.(n / 2);
        Array.iteri
          (fun j v -> if j > 0 then less ~msg:"ascending" float_exact ~than:v shifted.(j - 1))
          shifted)

  (* ifftshift undoes fftshift along any distinct axes, at any extents. *)
  let law_shift_round_trip =
    prop "ifftshift undoes fftshift"
      Gen.(pair (array ~size:(int_range 1 3) (int_range 0 5)) (int_range 0 7))
      (fun (s, pick) ->
        let r = Array.length s in
        let axes = List.filter (fun a -> (pick lsr a) land 1 = 1) (List.init r Fun.id) in
        let axes = if axes = [] then [ r - 1 ] else axes in
        cover "an odd extent shifted" (List.exists (fun a -> s.(a) mod 2 = 1) axes);
        let x = Nx.arange Nx.int32 0 (Array.fold_left ( * ) 1 s) 1 |> Nx.reshape s in
        let y = Nx.fftshift ~axes x in
        equal ~msg:"round trip" (array int32) (Nx.to_array x) (Nx.to_array (Nx.ifftshift ~axes y)))

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
    invalid ~by:"Nx.hfft" (fun () -> Nx.hfft Nx.float32 (Nx.zeros Nx.complex64 [| 0 |]));
    invalid ~by:"Nx.hfft" (fun () -> Nx.hfft Nx.float32 ~axes:[ 1 ] z);
    invalid ~by:"Nx.ihfft" (fun () -> Nx.ihfft Nx.complex64 ~s:[ -1 ] x);
    invalid ~by:"Nx.dct" (fun () -> Nx.dct ~type_:Type_1 (Nx.zeros Nx.float32 [| 1 |]));
    invalid ~by:"Nx.idct" (fun () -> Nx.idct ~type_:Type_1 (Nx.zeros Nx.float32 [| 0; 1 |]));
    invalid ~by:"Nx.dst" (fun () -> Nx.dst ~axes:[ 1 ] x);
    invalid ~by:"Nx.idst" (fun () -> Nx.idst ~axes:[ 0; 0 ] x);
    invalid ~by:"Nx.hann" (fun () -> Nx.hann Nx.float32 (-1));
    invalid ~by:"Nx.fftfreq" (fun () -> Nx.fftfreq Nx.float32 0);
    invalid ~by:"Nx.fftfreq" (fun () -> Nx.fftfreq Nx.float32 ~d:0. 4);
    invalid ~by:"Nx.rfftfreq" (fun () -> Nx.rfftfreq Nx.float32 ~d:Float.nan 4);
    invalid ~by:"Nx.fftshift" (fun () -> Nx.fftshift (Nx.zeros Nx.float32 [||]));
    invalid ~by:"Nx.fftshift" (fun () -> Nx.fftshift ~axes:[] x);
    invalid ~by:"Nx.ifftshift" (fun () -> Nx.ifftshift ~axes:[ 0; 0 ] x);
    invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:5 x);
    invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:2 ~win:x x);
    invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 ~step:5 (Nx.zeros Nx.complex64 [| 2; 3 |]));
    invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 (Nx.zeros Nx.complex64 [| 2; 4 |]))

  (* The Hermitian signal of [n] points whose first bins are [x], as irfft
     reads them: bin [k] above [n/2] the conjugate of bin [n - k], bins past
     [x]'s zero, and the imaginary parts of bin [0] and, for an even [n], of
     bin [n/2] dropped. *)
  let hermitian (x : Complex.t array) n =
    Array.init n (fun j ->
        let k = if j > n / 2 then n - j else j in
        let z = if k < Array.length x then x.(k) else Complex.zero in
        let z = if k = 0 || 2 * k = n then { z with im = 0. } else z in
        if j > n / 2 then Complex.conj z else z)

  let bins_and_length =
    let open Gen in
    let* m = frequency [ (1, constant 1); (2, int_range 2 6) ] and+ odd = bool in
    let+ re = array ~size:(constant m) (float_range (-1.) 1.)
    and+ im = array ~size:(constant m) (float_range (-1.) 1.) in
    (Array.map2 (fun re im -> { Complex.re; im }) re im,
     if odd then (2 * m) - 1 else 2 * (m - 1))

  let law_hermitian =
    prop "hfft is the forward sum of the Hermitian signal, ihfft its inverse"
      bins_and_length (fun (x, n) ->
        cover_lengths n;
        let m = Array.length x in
        cover "one bin" (m = 1);
        let t = complex_of x in
        let y = Array.init n (fun j -> x.(j mod m).re -. x.(j mod m).im) in
        let ty = Nx.create Nx.float64 [| n |] y in
        let reals = Array.map (fun re -> { Complex.re; im = 0. }) in
        List.iter
          (fun norm ->
            let want =
              Array.map
                (fun (z : Complex.t) -> z.re *. factor norm ~forward:true n)
                (dft ~sign:(-1.) (hermitian x n))
            in
            equal ~msg:"hfft" (array (close_floats 1e-12)) want
              (Nx.to_array (Nx.hfft Nx.float64 ~norm ~s:[ n ] t));
            if n mod 2 = 0 then
              equal ~msg:"hfft's default points" (array (close_floats 1e-12))
                want (Nx.to_array (Nx.hfft Nx.float64 ~norm t));
            if n > 0 then begin
              let inverse =
                scale (factor norm ~forward:false n)
                  (Array.sub (dft ~sign:1. (reals y)) 0 ((n / 2) + 1))
              in
              let h = Nx.ihfft Nx.complex128 ~norm ty in
              equal ~msg:"ihfft" (array (close 1e-12)) inverse (Nx.to_array h);
              equal ~msg:"round trip" (array (close_floats 1e-12)) y
                (Nx.to_array (Nx.hfft Nx.float64 ~norm ~s:[ n ] h))
            end)
          norms)

  (* Over two axes, in either order: the half of a Hermitian signal, the
     inverse sums of a real spectrum, transforms back to that spectrum, and
     ihfft of the spectrum is that half. *)
  let test_hermitian_axes () =
    let rows = 3 and cols = 4 in
    let spectrum =
      Array.init (rows * cols) (fun j -> Float.of_int ((j * 7) mod 5) -. 2.)
    in
    (* The inverse transform of [y], laid out [rows × cols], along both. *)
    let along_rows y =
      Array.concat
        (List.init rows (fun r -> dft ~sign:1. (Array.sub y (r * cols) cols)))
    in
    let along_cols y =
      let col c = Array.init rows (fun r -> y.((r * cols) + c)) in
      let t = Array.init cols (fun c -> dft ~sign:1. (col c)) in
      Array.init (rows * cols) (fun j -> t.(j mod cols).(j / cols))
    in
    let n = Float.of_int (rows * cols) in
    let h =
      Array.map
        (fun z -> Complex.div z { re = n; im = 0. })
        (along_cols (along_rows (Array.map (fun re -> { Complex.re; im = 0. }) spectrum)))
    in
    let ty = Nx.create Nx.float64 [| rows; cols |] spectrum in
    let th = Nx.create Nx.complex128 [| rows; cols |] h in
    let half axis =
      match axis with
      | 1 -> Nx.slice [ A; R (0, (cols / 2) + 1) ] th
      | _ -> Nx.slice [ R (0, (rows / 2) + 1); A ] th
    in
    List.iter
      (fun (axes, s) ->
        let msg = Printf.sprintf "axes [%s]" (String.concat "; " (List.map string_of_int axes)) in
        let last = List.nth axes 1 in
        equal ~msg:(msg ^ ": hfft") (array (close_floats 1e-12)) spectrum
          (Nx.to_array (Nx.hfft Nx.float64 ~axes ~s (half last)));
        equal ~msg:(msg ^ ": ihfft") (array (close 1e-12))
          (Nx.to_array (half last))
          (Nx.to_array (Nx.ihfft Nx.complex128 ~axes ty)))
      [ ([ 0; 1 ], [ rows; cols ]); ([ 1; 0 ], [ cols; rows ]) ]

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
         group "rules"
           [
             test "the rule raises before any kernel" test_rules;
             test "a c2r ignores bins 0's and n/2's imaginary parts"
               test_c2r_bins;
           ];
         group "functions"
           [
             Laws.law_fft;
             Laws.law_real;
             Laws.law_hermitian;
             test "hfft and ihfft over two axes, in either order"
               Laws.test_hermitian_axes;
             Laws.law_irfft;
             Laws.law_axes;
             test "long lengths at complex128 and complex64" Laws.test_long;
             test "s cuts and extends, a part scales alone" Laws.test_sizes;
             Laws.law_trigonometric;
             test "an axis of no element" Laws.test_trigonometric_empty;
             test "a narrow float transforms at float32, rounded once"
               Laws.test_narrow;
             test "hann" Laws.test_hann;
             test "frequencies and shifts" Laws.test_frequencies;
             Laws.law_frequencies;
             Laws.law_shift_round_trip;
             Laws.law_stft;
             test "each refusal names its function" Laws.test_refusals;
             test "a composite names its function" Laws.test_composites_named;
           ];
       ])
