(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fft through Nx.Prim, on a library whose transforms are the sums
   Nx_kernel.Spec.fft states, computed term by term: the values the spec gives
   for small operands, the shapes and dtypes of each transform, its rule before
   any kernel, a split placement, and a library that declines. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module S = Nx_kernel.Spec

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

  let fft s ~dst (A.Any x) =
    Atomic.incr calls;
    let axes = S.axes s in
    let last = axes.(Array.length axes - 1) in
    let z = complexes x and shape = L.shape (A.layout x) in
    let along sign (z, shape) a = lines z shape a shape.(a) (dft sign) in
    let z, _ =
      match S.transform s with
      | C2c d ->
          let sign = match d with Forward -> -1. | Inverse -> 1. in
          Array.fold_left (along sign) (z, shape) axes
      | R2c ->
          let z, shape = Array.fold_left (along (-1.)) (z, shape) axes in
          lines z shape last
            ((shape.(last) / 2) + 1)
            (fun l -> Array.sub l 0 ((Array.length l / 2) + 1))
      | C2r { n } ->
          let rest = Array.sub axes 0 (Array.length axes - 1) in
          let z, shape = Array.fold_left (along 1.) (z, shape) rest in
          lines z shape last n (fun bins ->
              let full =
                Array.init n (fun k ->
                    if k <= n / 2 then bins.(k) else Complex.conj bins.(n - k))
              in
              Array.map (fun c -> { c with Complex.im = 0. }) (dft 1. full))
    in
    let (A.Any d) = dst in
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
    store d;
    A.Done
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

let test_declined () =
  let x =
    Nx.Repr.of_array Nx.Host.v
      (A.of_array D.Complex64 [| 2 |] (Array.make 2 Complex.one))
  in
  match fft (C2c { direction = Forward; axes = [| 0 |]; x }) with
  | _ -> fail "a declined transform computed"
  | exception Invalid_argument e ->
      List.iter
        (fun sub -> contains ~msg:sub ~sub e)
        [ "t: "; "nx.cpu"; "Fft"; "complex64"; "not available yet"; "Nx.place" ]

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
             test "a declined transform raises naming the move" test_declined;
           ];
       ])
