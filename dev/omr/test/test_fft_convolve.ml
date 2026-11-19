(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module F = Omr.Fft_convolve

let bits a =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (Nx.to_array (Nx.cast Nx.float64 a))

let cbits a =
  Array.concat [ bits (Nx.real Nx.float64 a); bits (Nx.imag Nx.float64 a) ]

(* The transform size of a kernel of [m] points, as the .mli states it. *)
let size m =
  let cost n =
    Float.of_int n *. Float.log2 (Float.of_int n) /. Float.of_int (n - m + 1)
  in
  let rec first n = if n >= m then n else first (2 * n) in
  let best = ref (first 1) and n = ref (2 * first 1) in
  for _ = 1 to 24 do
    if cost !n < cost !best then best := !n;
    n := 2 * !n
  done;
  !best

(* nx.cpu's transform constant for a power of two [n], its last axis of a real
   transform when [real]: [e(n) = 8 Σ p^(3/2) + 8] over [n]'s prime factors,
   [e(n/2) + 8] for a real last axis of [n >= 512]. *)
let rec e ~real n =
  if real && n >= 512 then e ~real:false (n / 2) +. 8.
  else
    let k = Float.log2 (Float.of_int n) in
    (8. *. k *. (2. ** 1.5)) +. 8.

(* The .mli's bound for a slice, from the kernel extents [ms]. *)
let bound ~complex ~u ms nx nk =
  let ns = List.map size ms in
  let last = List.length ns - 1 in
  let eps =
    List.fold_left ( +. ) 0.
      (List.mapi (fun i n -> e ~real:((not complex) && i = last) n) ns)
  in
  let c =
    List.fold_left2 (fun a n m -> a * ((n + (n - m)) / (n - m + 1))) 1 ns ms
  in
  let n' = List.fold_left ( * ) 1 ns in
  let k =
    if complex then (2. *. eps) +. eps +. (2. *. Float.sqrt 2.)
    else (2. *. Float.sqrt 2. *. eps) +. (Float.sqrt 2. *. eps) +. 4.
  in
  k *. u *. Float.sqrt (Float.of_int (c * n')) *. nx *. nk

let norm a = Nx.item [] (Nx.sqrt (Nx.sum (Nx.square a)))

(* Direct convolution at float64 of [x] extended by [ends] with [k], by
   [Nx.correlate] of the flipped kernel over the last axes, conjugated since
   correlate conjugates its kernel. *)
let direct zero x k ends =
  let r = Nx.ndim x and a = List.length ends in
  let pads =
    Array.init r (fun i ->
        if i < r - a then (0, 0) else List.nth ends (i - (r - a)))
  in
  Nx.correlate ~padding:`Valid (Nx.pad pads zero x) (Nx.conjugate (Nx.flip k))

let random seed shape =
  let r = Random.State.make [| seed |] in
  Nx.init Nx.float64 shape (fun _ -> Random.State.float r 2. -. 1.)

type case = {
  name : string;
  x : int array;
  k : int array;
  ends : (int * int) list;
}

let rows =
  [
    {
      name = "1-d, m 31, full";
      x = [| 1000 |];
      k = [| 31 |];
      ends = [ (30, 30) ];
    };
    { name = "1-d, m 1"; x = [| 300 |]; k = [| 1 |]; ends = [ (0, 0) ] };
    { name = "1-d, m 3, a tie"; x = [| 300 |]; k = [| 3 |]; ends = [ (2, 0) ] };
    {
      name = "1-d, m 17, a tie";
      x = [| 500 |];
      k = [| 17 |];
      ends = [ (8, 8) ];
    };
    {
      name = "1-d, m 100, short";
      x = [| 60 |];
      k = [| 100 |];
      ends = [ (99, 99) ];
    };
    {
      name = "2-d, 9 by 7, full";
      x = [| 60; 50 |];
      k = [| 9; 7 |];
      ends = [ (8, 8); (6, 6) ];
    };
    {
      name = "2-d, 5 by 40, valid";
      x = [| 30; 300 |];
      k = [| 5; 40 |];
      ends = [ (0, 0); (0, 0) ];
    };
  ]

let within_bound (type c) (cdt : (Complex.t, c) Nx.dtype) complex =
  cases
    (Printf.sprintf "within the normwise bound of direct convolution (%s)"
       (if complex then "complex" else "real"))
    ~id:(fun c -> c.name)
    rows
    (fun c ->
      let axes = List.init (Array.length c.x) Fun.id in
      let x = random 1 c.x and k = random 2 c.k in
      let ms = Array.to_list c.k in
      if complex then begin
        let xi = random 3 c.x and ki = random 4 c.k in
        let zx = Nx.complex Nx.complex128 ~re:x ~im:xi
        and zk = Nx.complex Nx.complex128 ~re:k ~im:ki in
        let got =
          Omr.fft_convolve ~axes ~ends:c.ends (Nx.cast cdt zx) (Nx.cast cdt zk)
        in
        let want = direct Complex.zero zx zk c.ends in
        let err = Nx.sub (Nx.cast Nx.complex128 got) want in
        let n z =
          Float.sqrt
            ((norm (Nx.real Nx.float64 z) ** 2.)
            +. (norm (Nx.imag Nx.float64 z) ** 2.))
        in
        less float_exact
          ~than:(bound ~complex ~u:(Float.ldexp 1. (-24)) ms (n zx) (n zk))
          (n err)
      end
      else begin
        let got =
          Omr.fft_convolve ~axes ~ends:c.ends (Nx.cast Nx.float32 x)
            (Nx.cast Nx.float32 k)
        in
        let want = direct 0. x k c.ends in
        less float_exact
          ~than:(bound ~complex ~u:(Float.ldexp 1. (-24)) ms (norm x) (norm k))
          (norm (Nx.sub (Nx.cast Nx.float64 got) want))
      end)

(* At float64 the result is direct convolution's to a few units of the norm's
   rounding. *)
let per_slice =
  test "a kernel per slice, broadcast or not" (fun () ->
      let x = random 5 [| 3; 200 |] in
      List.iter
        (fun k ->
          let got = Omr.fft_convolve ~axes:[ 1 ] ~ends:[ (14, 0) ] x k in
          for s = 0 to 2 do
            let ks = if Nx.dim 0 k = 1 then 0 else s in
            let want =
              direct 0.
                (Nx.slice [ Nx.I s; Nx.A ] x)
                (Nx.slice [ Nx.I ks; Nx.A ] k)
                [ (14, 0) ]
            in
            equal
              (array (float 1e-12))
              (Nx.to_array want)
              (Nx.to_array (Nx.slice [ Nx.I s; Nx.A ] got))
          done)
        [ random 6 [| 3; 15 |]; random 7 [| 1; 15 |] ])

(* Streams *)

type scase = {
  dims : int array;
  m : int array;
  axis : int;
  ends : (int * int) list;
  seed : int;
}

let pp_scase ppf c =
  Format.fprintf ppf "{x [%s]; k [%s]; axis %d; seed %d}"
    (String.concat "; " (Array.to_list (Array.map string_of_int c.dims)))
    (String.concat "; " (Array.to_list (Array.map string_of_int c.m)))
    c.axis c.seed

let scase =
  let open Gen in
  with_pp pp_scase
  @@
  let* two = bool in
  let* m0 = int_range 1 40 and+ m1 = int_range 1 6 in
  let* n0 = int_range 0 300 and+ n1 = int_range 1 12 in
  let* l0 = int_range 0 40 and+ h0 = int_range 0 40 in
  let* axis = if two then int_range 0 1 else constant 0 in
  let+ seed = int_range 0 1_000_000 in
  if two then
    {
      dims = [| n0; n1 |];
      m = [| m0; m1 |];
      axis;
      ends = [ (l0, h0); (m1 - 1, 0) ];
      seed;
    }
  else { dims = [| n0 |]; m = [| m0 |]; axis; ends = [ (l0, h0) ]; seed }

let stream_case (type c) (cdt : (Complex.t, c) Nx.dtype) complex c =
  let axes = List.init (Array.length c.dims) Fun.id in
  let n = c.dims.(c.axis) in
  let r = Random.State.make [| c.seed |] in
  let rec split left =
    if left = 0 then [ 0 ]
    else
      let k = Int.min left (Random.State.int r 90) in
      k :: split (left - k)
  in
  let run (type v s) (x : (v, s, Nx.host) Nx.t) (k : (v, s, Nx.host) Nx.t) eq =
    let piece a b =
      Nx.slice
        (List.init (Nx.ndim x) (fun i ->
             if i = c.axis then Nx.R (a, b) else Nx.A))
        x
    in
    let s, outs, _ =
      List.fold_left
        (fun (s, outs, at) len ->
          let s, y =
            F.feed ~axis:c.axis ~axes ~ends:c.ends k s (piece at (at + len))
          in
          (s, y :: outs, at + len))
        (F.start, [], 0) (split n)
    in
    cover ~share:0.2 "blocks from feeds"
      (List.exists (fun y -> Nx.dim c.axis y > 0) outs);
    let last = F.finish ~axis:c.axis ~axes ~ends:c.ends k s in
    eq
      (Omr.fft_convolve ~axes ~ends:c.ends x k)
      (Nx.concatenate ~axis:c.axis (List.rev (last :: outs)))
  in
  let x = random c.seed c.dims and k = random (c.seed + 1) c.m in
  if complex then
    let z a b = Nx.cast cdt (Nx.complex Nx.complex128 ~re:a ~im:b) in
    run
      (z x (random (c.seed + 2) c.dims))
      (z k (random (c.seed + 3) c.m))
      (fun a b ->
        equal (array int) (Nx.shape a) (Nx.shape b);
        equal (array int64) (cbits a) (cbits b))
  else
    run (Nx.cast Nx.float32 x) (Nx.cast Nx.float32 k) (fun a b ->
        equal (array int) (Nx.shape a) (Nx.shape b);
        equal (array int64) (bits a) (bits b))

let stream_real =
  prop "every split gives the whole call's bits (float32)" ~count:150 scase
    (stream_case Nx.complex64 false)

let stream_complex =
  prop "every split gives the whole call's bits (complex64)" ~count:100 scase
    (stream_case Nx.complex64 true)

let errors =
  let x = Nx.zeros Nx.float64 [| 8 |] and k = Nx.ones Nx.float64 [| 3 |] in
  let raises f = raises_match (Exn.invalid_arg ~sub:"Omr.fft_convolve") f in
  group "misuse"
    [
      test "a repeated axis" (fun () ->
          raises (fun () ->
              Omr.fft_convolve ~axes:[ 0; 0 ] ~ends:[ (0, 0); (0, 0) ] x k));
      test "ends of another length" (fun () ->
          raises (fun () -> Omr.fft_convolve ~axes:[ 0 ] ~ends:[] x k));
      test "a kernel of another rank" (fun () ->
          raises (fun () ->
              Omr.fft_convolve ~axes:[ 0 ]
                ~ends:[ (0, 0) ]
                x
                (Nx.ones Nx.float64 [| 1; 3 |])));
      test "integers" (fun () ->
          raises (fun () ->
              Omr.fft_convolve ~axes:[ 0 ]
                ~ends:[ (0, 0) ]
                (Nx.zeros Nx.int32 [| 8 |])
                (Nx.ones Nx.int32 [| 3 |])));
      test "a stream along an axis not among the axes" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Fft_convolve.feed") (fun () ->
              F.feed ~axis:1 ~axes:[ 0 ]
                ~ends:[ (0, 0) ]
                (Nx.ones Nx.float64 [| 3; 1 |])
                F.start
                (Nx.zeros Nx.float64 [| 8; 2 |])));
    ]

let () =
  exit
    (run "omr.fft_convolve"
       [
         within_bound Nx.complex64 false;
         within_bound Nx.complex64 true;
         per_slice;
         stream_real;
         stream_complex;
         errors;
       ])
