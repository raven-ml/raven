(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

let archive file = lazy (Nx_io.load_npz file)

(* torch's antialiased interpolate in float64 on a [9; 11; 2] image, from
   golden/sample.npz (contrib/mynd/test/gen/sample.py); torch's in float32 and
   float64 along one axis and Pillow's 8-bit resize, from golden/resize.npz
   (contrib/mynd/test/gen/resize.py) *)
let sample_npz = archive "golden/sample.npz"
let resize_npz = archive "golden/resize.npz"

let entry npz dt key =
  match Nx_io.Archive.find key (Lazy.force npz) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden has no %s" key

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let same_shape want got = equal (array int) (Nx.shape want) (Nx.shape got)

let close tol want got =
  same_shape want got;
  equal (array (float tol)) (floats want) (floats got)

(* NaN equals NaN here: bit-for-bit comparison of float arrays. *)
let bits x =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (floats x)

let same want got =
  same_shape want got;
  equal (array int64) (bits want) (bits got)

let kernels : (string * V.kernel) list =
  [
    ("box", `Box);
    ("triangle", `Triangle);
    ("keys -0.5", `Cubic (-0.5));
    ("keys -0.75", `Cubic (-0.75));
    ("lanczos2", `Lanczos 2);
    ("lanczos3", `Lanczos 3);
  ]

let edges : (string * V.edge) list =
  [
    ("zero", `Zero);
    ("replicate", `Replicate);
    ("symmetric whole", `Symmetric_whole);
    ("symmetric half", `Symmetric_half);
    ("periodic", `Periodic);
    ("renormalise", `Renormalise);
  ]

let image ~h ~w ~c seed =
  let st = Random.State.make [| seed |] in
  Nx.init Nx.float64 [| h; w; c |] (fun _ -> Random.State.float st 2. -. 1.)

(* [rows ~edge k n n'] is {!V.resize_weights}' rows, each its taps' pixels and
   weights, the padding left out. *)
let rows ~edge k n n' =
  let idx, w = V.resize_weights ~edge k n n' in
  let t = Nx.dim 1 idx in
  let idx = Nx.to_array idx and w = Nx.to_array w in
  Array.init n' (fun j ->
      List.filter_map
        (fun s ->
          let i = idx.((j * t) + s) in
          if i < 0L then None else Some (Int64.to_int i, w.((j * t) + s)))
        (List.init t Fun.id))

(* Against torch *)

let torch64 =
  cases "torch's antialiased interpolate in float64, renormalised"
    ~id:(fun (m, (h, w)) -> Printf.sprintf "%s to %dx%d" m h w)
    (List.concat_map
       (fun m ->
         List.map (fun s -> (m, s)) [ (4, 5); (13, 17); (6, 23); (9, 4) ])
       [ "bilinear"; "bicubic" ])
    (fun (m, (h, w)) ->
      let k = if m = "bilinear" then `Triangle else `Cubic (-0.5) in
      let x = entry sample_npz Nx.float64 "rz.x" in
      close 1e-13
        (entry sample_npz Nx.float64 (Printf.sprintf "rz.%s.%dx%d" m h w))
        (V.resize ~edge:`Renormalise k (h, w) x))

(* Against torch in float32. The values lie in [[0, 1)], and [u = 2⁻²⁴].

   mynd's float32 pass rounds each weight, each product and the sum of a new
   pixel's [t] taps of absolute weights [s]: within [(t + 1) u s] of the exact
   resize, which torch's float64 result is to far below that.

   torch computes in float32 what mynd takes exactly: the scale [n / n'],
   rounded, each centre [scale (j + ½)], rounded, and each tap's argument [(i -
   c + ½) / s'] for [s' = max 1 scale], whose rounding of [i - c + ½] and of [1
   / s'] and the product move it by at most [3u (h + 1)]. A centre off by [δc]
   moves the arguments by [δc / s'], so each by at most [δ = δc / s' + 3u (h +
   1)], and each kernel value by [L δ], [L] the kernel's largest slope: [1] for
   Triangle and [25/18] for Keys' [-½], whose slope is [4.5τ² - 5τ] on [[0, 1]]
   and [-(3τ² - 10τ + 8) / 2] on [[1, 2]]. For [t + 1] taps (torch's support may
   hold one more, of weight below [L δ]) and the values' sum [K], the weights
   move by at most [(t + 1) L δ (1 + s) / K] in all, summing to [0], so a pixel
   moves by half that. torch's own roundings add [(t + 5) u s]: the kernel's
   value, the division and the sum. Where [n / n'] is dyadic, as in 480 to 640,
   [δc] is [0]. *)
let kernel_value (k : V.kernel) t =
  let a = Float.abs t in
  match k with
  | `Triangle -> 1. -. a
  | `Cubic c ->
      if a <= 1. then ((c +. 2.) *. a *. a *. a) -. ((c +. 3.) *. a *. a) +. 1.
      else
        (c *. a *. a *. a) -. (5. *. c *. a *. a) +. (8. *. c *. a) -. (4. *. c)
  | `Box | `Lanczos _ -> invalid_arg "kernel_value"

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

let torch32 =
  cases "torch's antialiased interpolate in float32, per size pair"
    ~id:(fun (n, n', m) -> Printf.sprintf "%s %d to %d" m n n')
    (List.concat_map
       (fun (n, n') -> [ (n, n', "bilinear"); (n, n', "bicubic") ])
       [
         (427, 640);
         (480, 640);
         (480, 1024);
         (640, 1024);
         (1500, 1024);
         (2000, 640);
         (333, 640);
       ])
    (fun (n, n', m) ->
      let k, h, slope =
        if m = "bilinear" then (`Triangle, 1., 1.)
        else (`Cubic (-0.5), 2., 25. /. 18.)
      in
      let key s = Printf.sprintf "torch.%d.%d.%s" n n' s in
      let x =
        Nx.reshape [| 1; n; 1 |] (entry resize_npz Nx.float32 (key "x"))
      in
      let got =
        floats (Nx.reshape [| n' |] (V.resize ~edge:`Renormalise k (1, n') x))
      in
      let t32 = floats (entry resize_npz Nx.float32 (key (m ^ ".f32"))) in
      let t64 = floats (entry resize_npz Nx.float64 (key (m ^ ".f64"))) in
      let u = 0x1p-24 in
      let scale = Float.of_int n /. Float.of_int n' in
      let s' = Float.max 1. scale and scale32 = round32 scale in
      let e = if n > n' then 2 * n else 2 * n' in
      Array.iteri
        (fun j r ->
          let t = List.length r in
          let s = List.fold_left (fun a (_, w) -> a +. Float.abs w) 0. r in
          let centre = (Float.of_int j +. 0.5) *. scale in
          let dc =
            Float.abs (round32 (scale32 *. (Float.of_int j +. 0.5)) -. centre)
          in
          let d = (dc /. s') +. (3. *. u *. (h +. 1.)) in
          let big_k =
            List.fold_left
              (fun a (i, _) ->
                let num = (((2 * j) + 1) * n) - (((2 * i) + 1) * n') in
                a +. kernel_value k (Float.of_int num /. Float.of_int e))
              0. r
          in
          let ours = Float.of_int (t + 1) *. u *. s in
          let theirs =
            (Float.of_int (t + 1) *. slope *. d *. (1. +. s) /. (2. *. big_k))
            +. (Float.of_int (t + 5) *. u *. s)
          in
          let msg = Printf.sprintf "new pixel %d" j in
          at_most ~msg ~than:(ours *. 1.01) float_exact
            (Float.abs (got.(j) -. t64.(j)));
          at_most ~msg
            ~than:((ours +. theirs) *. 1.01)
            float_exact
            (Float.abs (got.(j) -. t32.(j))))
        (rows ~edge:`Renormalise k n n'))

(* Pillow's 8-bit resize as a composition over the weights

   Pillow rounds each weight to 22 fraction bits, half away from zero, and adds
   each pass's products in integers from [2²¹], shifting the sum back and
   clamping it to [[0, 255]]. It resizes the columns first unless the height
   shrinks by more than twice what the width does, and an RGBA image through its
   premultiplied form, each colour [c α / 255] rounded, then divided back by the
   resized alpha, truncated, where alpha is neither 0 nor 255 (Pillow 13's
   Resample.c and Convert.c). *)

let precision = 22

let fixed w =
  let v = w *. Float.of_int (1 lsl precision) in
  Float.to_int (if w < 0. then v -. 0.5 else v +. 0.5)

let clip8 v = Int.max 0 (Int.min 255 v)

(* [pass ~axis k n' img] resizes axis [axis], [0] the rows and [1] the columns,
   of an image [[h][w][c]] of bytes to [n'] pixels. *)
let pass ~axis k n' img =
  let h = Array.length img and w = Array.length img.(0) in
  let c = Array.length img.(0).(0) in
  let n = if axis = 0 then h else w in
  let table =
    Array.map
      (List.map (fun (i, wt) -> (i, fixed wt)))
      (rows ~edge:`Renormalise k n n')
  in
  let px i j ch t = if axis = 0 then img.(t).(j).(ch) else img.(i).(t).(ch) in
  let h', w' = if axis = 0 then (n', w) else (h, n') in
  Array.init h' (fun i ->
      Array.init w' (fun j ->
          Array.init c (fun ch ->
              let taps = table.(if axis = 0 then i else j) in
              let sum =
                List.fold_left
                  (fun s (t, wt) -> s + (wt * px i j ch t))
                  (1 lsl (precision - 1))
                  taps
              in
              clip8 (sum asr precision))))

let pillow k (h', w') img =
  let h = Array.length img and w = Array.length img.(0) in
  let rows img = if h' = h then img else pass ~axis:0 k h' img in
  let cols img = if w' = w then img else pass ~axis:1 k w' img in
  let columns_first = not (h - h' > 0 && h - h' > (w - w') * 2) in
  if columns_first then rows (cols img) else cols (rows img)

let muldiv255 a b =
  let t = (a * b) + 128 in
  ((t lsr 8) + t) lsr 8

let premultiply =
  Array.map
    (Array.map (fun p ->
         Array.mapi (fun c v -> if c = 3 then v else muldiv255 v p.(3)) p))

let unpremultiply =
  Array.map
    (Array.map (fun p ->
         let a = p.(3) in
         Array.mapi
           (fun c v ->
             if c = 3 || a = 0 || a = 255 then v else clip8 (255 * v / a))
           p))

let bytes x =
  let s = Nx.shape x in
  let a = Array.map Int64.to_int (Nx.to_array (Nx.cast Nx.int64 x)) in
  Array.init s.(0) (fun i ->
      Array.init s.(1) (fun j ->
          Array.init s.(2) (fun c -> a.((((i * s.(1)) + j) * s.(2)) + c))))

let pillow_cases =
  List.concat_map
    (fun i ->
      List.concat_map
        (fun mode ->
          List.map (fun f -> (i, mode, f)) [ "bilinear"; "bicubic"; "lanczos" ])
        [ "L"; "RGB"; "RGBA" ])
    [ 0; 1; 2; 3; 4 ]

let pillow8 =
  cases "Pillow's 8-bit resize is a composition over resize_weights"
    ~id:(fun (i, mode, f) -> Printf.sprintf "%d %s %s" i mode f)
    pillow_cases
    (fun (i, mode, f) ->
      let k =
        match f with
        | "bilinear" -> `Triangle
        | "bicubic" -> `Cubic (-0.5)
        | _ -> `Lanczos 3
      in
      let size =
        Nx.to_array (entry resize_npz Nx.int64 (Printf.sprintf "pil.%d.size" i))
      in
      let h' = Int64.to_int size.(2) and w' = Int64.to_int size.(3) in
      let key s = Printf.sprintf "pil.%d.%s.%s" i mode s in
      let x = bytes (entry resize_npz Nx.uint8 (key "x")) in
      let got =
        if mode = "RGBA" then unpremultiply (pillow k (h', w') (premultiply x))
        else pillow k (h', w') x
      in
      equal
        (array (array (array int)))
        (bytes (entry resize_npz Nx.uint8 (key f)))
        got)

(* The table's values, from the formulas *)

(* Triangle from 4 pixels to 2: new pixel 0's centre is 1, its kernel stretched
   by 2, so pixels -1 to 2 lie at τ = ¾, ¼, -¼, -¾ and weigh ¼, ¾, ¾, ¼ before
   the sum. *)
let table_values =
  test "a table from the formulas" (fun () ->
      let check ?(n = 4) ?(n' = 2) edge want_i want_w =
        let i, w = V.resize_weights ~edge `Triangle n n' in
        equal (array int64) want_i (Nx.to_array i);
        equal (array float_exact) want_w (Nx.to_array w)
      in
      let q = 0.75 /. 1.75 and e = 0.25 /. 1.75 in
      check `Renormalise [| 0L; 1L; 2L; 1L; 2L; 3L |] [| q; q; e; e; q; q |];
      check `Zero
        [| 0L; 1L; 2L; 1L; 2L; 3L |]
        [| 0.375; 0.375; 0.125; 0.125; 0.375; 0.375 |];
      check `Replicate
        [| 0L; 0L; 1L; 2L; 1L; 2L; 3L; 3L |]
        [| 0.125; 0.375; 0.375; 0.125; 0.125; 0.375; 0.375; 0.125 |];
      (* Enlarging 2 to 4, the kernel unstretched, centres at ¼, ¾, 5/4 and 7/4:
         the outer two have one tap inside, and their rows end in padding. *)
      check ~n:2 ~n':4 `Renormalise
        [| 0L; -1L; 0L; 1L; 0L; 1L; 1L; -1L |]
        [| 1.; 0.; 0.75; 0.25; 0.25; 0.75; 1.; 0. |];
      (* Shrinking 3 to 1: pixels -1 to 4 lie at τ = (2 - 2i) / 6, each rounded
         once, and pixel 4, at τ = -1, weighs 0 and is left out; pixels -1 and 3
         fold onto 0 and 2. *)
      let k i = 1. -. Float.abs (Float.of_int (2 - (2 * i)) /. 6.) in
      let sum = List.fold_left (fun s i -> s +. k i) 0. [ -1; 0; 1; 2; 3; 4 ] in
      check ~n:3 ~n':1 `Replicate [| 0L; 0L; 1L; 2L; 2L |]
        (Array.map (fun i -> k i /. sum) [| -1; 0; 1; 2; 3 |]))

(* Laws *)

let gen_sizes = Gen.(pair (int_range 1 24) (int_range 1 24))

(* A flat image resizes to itself under every kernel and every edge but [`Zero],
   shrinking and enlarging. *)
let flat =
  prop "a flat image resizes to itself" ~count:200
    Gen.(
      quad (int_range 1 5) (int_range 0 5) (pair gen_sizes gen_sizes)
        (float_range (-4.) 4.))
    (fun (e, k, ((h, w), (h', w')), c) ->
      let _, edge = List.nth edges e and _, k = List.nth kernels k in
      cover "shrinking" (h' < h || w' < w);
      cover "enlarging" (h' > h || w' > w);
      let x = Nx.full Nx.float64 [| h; w; 1 |] c in
      let y = floats (V.resize ~edge k (h', w') x) in
      Array.iter (fun v -> equal (float_rel ~rel:1e-14 ~abs:0x1p-1000) c v) y)

(* In float32 a flat row resizes to itself within rounding: each new pixel is [Σ
   fl(wᵢ) c] over its [t] taps, the float64 weights summing to [1] within
   float64's rounding, so the weights', the products' and the sum's roundings
   bound it by [(t + 1) u s |c|], [s] the row's absolute weights, [u = 2⁻²⁴]. *)
let flat32 =
  prop "a flat float32 row resizes to itself within rounding" ~count:200
    Gen.(quad (int_range 1 5) (int_range 0 5) gen_sizes (float_range 0.5 1.))
    (fun (e, k, (n, n'), c) ->
      let _, edge = List.nth edges e and _, k = List.nth kernels k in
      let c = Int32.float_of_bits (Int32.bits_of_float c) in
      let x = Nx.full Nx.float32 [| 1; n; 1 |] c in
      let y = floats (V.resize ~edge k (1, n') x) in
      let table = if n = n' then [||] else rows ~edge k n n' in
      Array.iteri
        (fun j v ->
          let t, s =
            if n = n' then (0, 1.)
            else
              ( List.length table.(j),
                List.fold_left (fun a (_, w) -> a +. Float.abs w) 0. table.(j)
              )
          in
          let bound = Float.of_int (t + 1) *. 0x1p-24 *. s *. c *. 1.01 in
          at_most ~than:bound float_exact (Float.abs (v -. c)))
        y)

(* An infinite pixel reaches exactly the new pixels whose rows and columns name
   it in the table with a weight other than [0]. *)
let local =
  prop "a pixel reaches only the new pixels that weigh it" ~count:200
    Gen.(
      quad (int_range 0 5) (int_range 0 5) (pair gen_sizes gen_sizes)
        (pair (int_range 0 1000) (int_range 0 1000)))
    (fun (e, k, ((h, w), (h', w')), (pi, pj)) ->
      let _, edge = List.nth edges e and _, k = List.nth kernels k in
      let pi = pi mod h and pj = pj mod w in
      let x = Nx.to_array (image ~h ~w ~c:1 9) in
      x.((pi * w) + pj) <- infinity;
      let x = Nx.create Nx.float64 [| h; w; 1 |] x in
      let names n n' p =
        if n = n' then Array.init n (fun j -> j = p)
        else
          Array.map
            (fun r -> List.exists (fun (i, w) -> i = p && w <> 0.) r)
            (rows ~edge k n n')
      in
      let ri = names h h' pi and cj = names w w' pj in
      cover "some new pixel finite" (Array.exists not ri || Array.exists not cj);
      let want =
        Array.init (h' * w') (fun q -> not (ri.(q / w') && cj.(q mod w')))
      in
      let y = floats (V.resize ~edge k (h', w') x) in
      equal (array bool) want (Array.map Float.is_finite y))

(* Enlarging reads as sampling at the new centres, [(j + ½) n / n'], under every
   edge. *)
let enlarge_is_sample =
  prop "an enlarging resize is a sample at the new centres" ~count:200
    Gen.(
      triple (int_range 0 5) (int_range 0 5)
        (quad (int_range 1 9) (int_range 1 9) (int_range 0 9) (int_range 0 9)))
    (fun (e, k, (h, w, dh, dw)) ->
      let _, edge = List.nth edges e and _, k = List.nth kernels k in
      let h' = h + dh and w' = w + dw in
      let x = image ~h ~w ~c:2 10 in
      let centre n n' j =
        Float.of_int (((2 * j) + 1) * n) /. Float.of_int (2 * n')
      in
      let p =
        Nx.init Nx.float64 [| h'; w'; 2 |] (fun ix ->
            if ix.(2) = 0 then centre w w' ix.(1) else centre h h' ix.(0))
      in
      close 1e-13 (V.sample ~edge k x p) (V.resize ~edge k (h', w') x))

(* Under [`Renormalise] each row's weights sum to [1] and the pixel holding the
   new centre weighs more than [0], so the sum the weights were divided by was
   positive: the kernels' values in [[-½, ½]] are positive for these. *)
let renormalised_rows =
  prop "renormalised rows divide by a positive sum" ~count:300
    Gen.(
      triple
        (one_of
           [
             constant `Box;
             constant `Triangle;
             map (fun a -> `Cubic a) (float_range (-1.) 0.);
             map (fun n -> `Lanczos n) (int_range 1 4);
           ])
        (int_range 1 80) (int_range 1 80))
    (fun (k, n, n') ->
      cover "shrinking" (n' < n);
      cover "enlarging" (n' > n);
      Array.iteri
        (fun j r ->
          let c = Float.of_int (((2 * j) + 1) * n) /. Float.of_int (2 * n') in
          let i = Float.to_int c in
          equal (float 1e-12) 1. (List.fold_left (fun s (_, w) -> s +. w) 0. r);
          let w = List.assoc i r in
          less ~than:w float_exact 0.)
        (rows ~edge:`Renormalise k n n'))

(* [spread ~edge k n n'] is an axis's most taps and largest sum of absolute
   weights, [(0, 1.)] where its extent is unchanged and resizing leaves it. *)
let spread ~edge k n n' =
  if n = n' then (0, 1.)
  else
    Array.fold_left
      (fun (t, s) r ->
        ( Int.max t (List.length r),
          Float.max s (List.fold_left (fun a (_, w) -> a +. Float.abs w) 0. r)
        ))
      (0, 0.) (rows ~edge k n n')

(* The formats narrower than 32 bits compute at float32 and round once: within
   half an ulp of the float64 resize of the same values, plus float32's error on
   the two passes. A pass of [t] taps of absolute sum [s] over values of
   magnitude [m] rounds each weight, each product and its sum, within [(t + 1) u
   s m], [u = 2⁻²⁴]; the second pass reads the first's results, of magnitude [s₁
   m], and carries its error times [s₂]. For values in [[0, 1)] that is [(t₁ +
   t₂ + 2) u s₁ s₂] to first order, up to 144 taps a pass here ([`Lanczos 3]
   from 24 pixels to 1). *)
let narrow =
  prop "a narrow format rounds a float32 resize once" ~count:100
    Gen.(
      quad bool (int_range 0 5)
        (frequency
           [
             (1, pair gen_sizes gen_sizes);
             (* Shrinking by over 5, where the kernels stretch the most. *)
             (1, pair (pair (int_range 16 24) (int_range 1 3)) gen_sizes);
           ])
        (int_range 0 1000))
    (fun (half, k, ((h, w), (h', w')), seed) ->
      let _, k = List.nth kernels k in
      let t1, s1 = spread ~edge:`Renormalise k w w'
      and t2, s2 = spread ~edge:`Renormalise k h h' in
      cover "over 16 taps a pass" (Int.max t1 t2 > 16);
      let float32 = Float.of_int (t1 + t2 + 2) *. 0x1p-24 *. s1 *. s2 *. 1.01 in
      let check (type s) (dt : (float, s) Nx.dtype) =
        let x = Nx.cast dt (Nx.abs (image ~h ~w ~c:1 seed)) in
        let want =
          floats (V.resize ~edge:`Renormalise k (h', w') (Nx.cast Nx.float64 x))
        in
        let got = floats (V.resize ~edge:`Renormalise k (h', w') x) in
        let p = (Nx.Dtype.float_format dt).fraction_bits + 1 in
        Array.iteri
          (fun q r ->
            let _, e = Float.frexp (Float.abs r +. float32) in
            let ulp = Float.ldexp 1. (e - p) in
            at_most
              ~than:((ulp /. 2.) +. float32)
              float_exact
              (Float.abs (got.(q) -. r)))
          want
      in
      if half then check Nx.float16 else check Nx.bfloat16)

(* Linear in the image: the derivative at each pixel is the resize of the image
   that is 1 there and 0 elsewhere. *)
let derivative =
  cases "the derivative in the image" ~id:fst kernels (fun (_, k) ->
      let x = image ~h:5 ~w:6 ~c:1 11 in
      let r = image ~h:3 ~w:8 ~c:1 12 in
      let f x = Nx.sum (Nx.mul (V.resize ~edge:`Symmetric_half k (3, 8) x) r) in
      let g = floats (Rune.grad' f x) in
      let unit q =
        Nx.init Nx.float64 [| 5; 6; 1 |] (fun ix ->
            if (ix.(0) * 6) + ix.(1) = q then 1. else 0.)
      in
      equal
        (array (float 1e-15))
        (Array.init 30 (fun q -> Nx.item [] (f (unit q))))
        g)

let compiled =
  cases "compiled is eager" ~id:fst kernels (fun (_, k) ->
      let x = Nx.cast Nx.float32 (image ~h:7 ~w:9 ~c:2 13) in
      let f x = V.resize ~edge:`Periodic k (4, 13) x in
      same (f x) (Rune.jit' f x))

let unchanged =
  test "an unchanged extent is left as it is" (fun () ->
      let x = image ~h:5 ~w:6 ~c:2 14 in
      same x (V.resize ~edge:`Zero (`Lanczos 3) (5, 6) x);
      let y = V.resize ~edge:`Zero (`Lanczos 3) (5, 9) x in
      let row i a = Nx.slice [ Nx.R (i, i + 1) ] a in
      same (row 2 y) (V.resize ~edge:`Zero (`Lanczos 3) (1, 9) (row 2 x)))

let refusals =
  group "misuse"
    [
      test "resize to no pixel" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.resize") (fun () ->
              V.resize ~edge:`Zero `Box (0, 3)
                (Nx.zeros Nx.float32 [| 4; 4; 1 |])));
      test "resize an image of no row" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.resize") (fun () ->
              V.resize ~edge:`Zero `Box (2, 3)
                (Nx.zeros Nx.float32 [| 0; 4; 1 |])));
      test "an invalid kernel" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.resize") (fun () ->
              V.resize ~edge:`Zero (`Lanczos 0) (2, 3)
                (Nx.zeros Nx.float32 [| 4; 4; 1 |])));
      test "a table from no pixel" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.resize_weights") (fun () ->
              V.resize_weights ~edge:`Zero `Triangle 0 3));
    ]

let () =
  exit
    (run "mynd.resize"
       [
         torch64;
         torch32;
         pillow8;
         table_values;
         flat;
         flat32;
         local;
         enlarge_is_sample;
         renormalised_rows;
         narrow;
         derivative;
         compiled;
         unchanged;
         refusals;
       ])
