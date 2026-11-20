(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

(* Equal as numbers, a zero of either sign and NaN equal to themselves. *)
let numbers =
  Array.map (fun v ->
      if Float.is_nan v then 0x7ff8L else Int64.bits_of_float (v +. 0.))

let same ?msg want got = equal ?msg (array int64) (numbers want) (numbers got)

let rules : (string * Nx.extension) list =
  [
    ("zero", `Zero);
    ("replicate", `Replicate);
    ("symmetric_whole", `Symmetric_whole);
    ("symmetric_half", `Symmetric_half);
    ("periodic", `Periodic);
    ("antisymmetric_whole", `Antisymmetric_whole);
  ]

let stencils : (string * V.Filter.stencil) list =
  [ ("central", `Central); ("sobel", `Sobel); ("scharr", `Scharr) ]

(* Values in [-1, 1) from the test's own [Random] state. *)
let random s =
  Nx.create Nx.float64 s
    (Array.init (Array.fold_left ( * ) 1 s) (fun _ -> Random.float 2. -. 1.))

type flt = F : string * (float, _) Nx.dtype -> flt

let floats_dt =
  [
    F ("float16", Nx.float16);
    F ("bfloat16", Nx.bfloat16);
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
  ]

(* Gradients *)

(* The weights across each stencil, and the 3 × 3 weights of [gx] and [gy] as a
   column of those times the row [-½ 0 ½], and its transpose. *)
let weights = function
  | `Central -> [| 1. |]
  | `Sobel -> [| 0.25; 0.5; 0.25 |]
  | `Scharr -> [| 0.1875; 0.625; 0.1875 |]

let products s =
  let a = weights s in
  let n = Array.length a in
  let d = [| -0.5; 0.; 0.5 |] in
  let kx =
    Nx.create Nx.float64 [| n; 3 |]
      (Array.init (n * 3) (fun k -> a.(k / 3) *. d.(k mod 3)))
  in
  let ky =
    Nx.create Nx.float64 [| 3; n |]
      (Array.init (3 * n) (fun k -> d.(k / n) *. a.(k mod n)))
  in
  (kx, ky)

(* Against correlate by the product weights. Values within [[-1, 1]] read as at
   most [9] through two antisymmetric folds; each result sums at most nine
   products, so either is within [10 ε] of the exact sum of terms below [9]. *)
let against_correlate =
  cases "gradient is correlate by the product weights, to rounding"
    ~id:(fun ((r, _), (s, _)) -> Printf.sprintf "%s %s" r s)
    (List.concat_map (fun r -> List.map (fun s -> (r, s)) stencils) rules)
    (fun ((_, e), (_, s)) ->
      let x = random [| 2; 5; 7; 2 |] in
      let kx, ky = products s in
      let gx, gy = V.Filter.gradient ~edge:e s x in
      let tol = float (2. *. 10. *. epsilon_float *. 9.) in
      equal ~msg:"gx" (array tol)
        (floats (V.Filter.correlate ~edge:e kx x))
        (floats gx);
      equal ~msg:"gy" (array tol)
        (floats (V.Filter.correlate ~edge:e ky x))
        (floats gy))

(* A ramp [a j + b i + c] with slopes and offset in eighths and values below
   [16] is exact in every format here, as are its reads and the sums the
   stencils take of them, at float32 for the narrow formats. *)
let ramp =
  prop "a ramp's gradient is its slope" ~count:300
    Gen.(
      quad (of_list floats_dt)
        (pair (of_list stencils) (of_list rules))
        (pair (int_range 2 7) (int_range 2 7))
        (triple (int_range (-8) 8) (int_range (-8) 8) (int_range (-8) 8)))
    (fun (F (_, dt), ((_, s), (rn, e)), (h, w), (ka, kb, kc)) ->
      let a = Float.of_int ka /. 8. and b = Float.of_int kb /. 8. in
      let c = Float.of_int kc /. 8. in
      let x =
        Nx.init Nx.float64 [| h; w; 1 |] (fun i ->
            (a *. Float.of_int i.(1)) +. (b *. Float.of_int i.(0)) +. c)
      in
      let gx, gy = V.Filter.gradient ~edge:e s (Nx.cast dt x) in
      let gx = floats gx and gy = floats gy in
      let all = e = `Antisymmetric_whole in
      cover "antisymmetric_whole" all;
      let smoothed = s <> `Central in
      for i = 0 to h - 1 do
        for j = 0 to w - 1 do
          let inner_i = i > 0 && i < h - 1 and inner_j = j > 0 && j < w - 1 in
          let k = (i * w) + j in
          if all || (inner_j && (inner_i || not smoothed)) then
            equal
              ~msg:(Printf.sprintf "%s gx at (%d, %d)" rn i j)
              float_exact a gx.(k);
          if all || (inner_i && (inner_j || not smoothed)) then
            equal
              ~msg:(Printf.sprintf "%s gy at (%d, %d)" rn i j)
              float_exact b gy.(k)
        done
      done)

(* Flips and transposes of the images give those of the gradients, equal as
   numbers. *)
let symmetries =
  prop "flips and transposes commute with the gradient" ~count:200
    Gen.(
      quad
        (of_list
           [
             F ("float16", Nx.float16);
             F ("float32", Nx.float32);
             F ("float64", Nx.float64);
           ])
        (of_list stencils) (of_list rules)
        (pair (int_range 1 6) (int_range 1 6)))
    (fun (F (_, dt), (_, s), (_, e), (h, w)) ->
      let x = Nx.cast dt (random [| 2; h; w; 2 |]) in
      let g = V.Filter.gradient ~edge:e s in
      let gx, gy = g x in
      let flip a v = Nx.flip ~axes:[ a ] v in
      let t v = Nx.transpose ~axes:[ 0; 2; 1; 3 ] v in
      let fx, fy = g (flip 2 x) in
      same (floats (Nx.neg (flip 2 gx))) (floats fx);
      same (floats (flip 2 gy)) (floats fy);
      let fx, fy = g (flip 1 x) in
      same (floats (flip 1 gx)) (floats fx);
      same (floats (Nx.neg (flip 1 gy))) (floats fy);
      let tx, ty = g (t x) in
      same (floats (t gy)) (floats tx);
      same (floats (t gx)) (floats ty))

(* Narrow formats compute at float32 and round once. *)
let narrow =
  cases "narrow formats round once"
    ~id:(fun ((r, _), F (n, _)) -> Printf.sprintf "%s %s" r n)
    (List.concat_map
       (fun r ->
         List.map
           (fun d -> (r, d))
           [ F ("float16", Nx.float16); F ("bfloat16", Nx.bfloat16) ])
       rules)
    (fun ((_, e), F (_, dt)) ->
      let x = Nx.cast dt (random [| 5; 6; 2 |]) in
      let wx, wy = V.Filter.gradient ~edge:e `Scharr (Nx.cast Nx.float32 x) in
      let gx, gy = V.Filter.gradient ~edge:e `Scharr x in
      same (floats (Nx.cast dt wx)) (floats gx);
      same (floats (Nx.cast dt wy)) (floats gy))

(* The derivative is the product weights' correlation's, to rounding. A pixel's
   derivative sums at most [36] terms, nine outputs' weights of [gx] and of
   [gy], each up to twice through an antisymmetric fold, whose magnitudes sum to
   at most [6]: either sum is within [37 ε 6] of the exact one. *)
let derivative =
  cases "the derivative is the correlation's"
    ~id:(fun ((r, _), (s, _)) -> Printf.sprintf "%s %s" r s)
    (List.concat_map (fun r -> List.map (fun s -> (r, s)) stencils) rules)
    (fun ((_, e), (_, s)) ->
      let x = random [| 4; 5; 1 |] in
      let r1 = random [| 4; 5; 1 |] and r2 = random [| 4; 5; 1 |] in
      let kx, ky = products s in
      let loss (gx, gy) =
        Nx.add (Nx.sum (Nx.mul gx r1)) (Nx.sum (Nx.mul gy r2))
      in
      let want =
        Rune.grad'
          (fun x ->
            loss
              (V.Filter.correlate ~edge:e kx x, V.Filter.correlate ~edge:e ky x))
          x
      in
      equal
        (array (float (2. *. 37. *. 6. *. epsilon_float)))
        (floats want)
        (floats (Rune.grad' (fun x -> loss (V.Filter.gradient ~edge:e s x)) x)))

let compiled =
  test "compiled is eager" (fun () ->
      let x = Nx.cast Nx.float32 (random [| 6; 7; 3 |]) in
      List.iter
        (fun (name, e) ->
          List.iter
            (fun (_, s) ->
              let f x = V.Filter.gradient ~edge:e s x in
              let sig_ = Nx.Ptree.(tensor @-> returns (pair tensor tensor)) in
              let gx, gy = f x and cx, cy = Rune.jit sig_ f x in
              same ~msg:name (floats gx) (floats cx);
              same ~msg:name (floats gy) (floats cy))
            stencils)
        rules)

(* Against scipy, OpenCV and numpy, from golden/edges.npz
   (contrib/mynd/test/gen/edges.py) *)
let golden = lazy (Nx_io.load_npz "golden/edges.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/edges.npz has no %s" key

(* The libraries' stencils are [n] times mynd's, [n] a power of two, and add
   their products in other orders: each sum of at most nine terms of weights
   summing to [n] in magnitude is within [10 ε n] of [x]'s largest magnitude of
   the exact one. *)
let libraries =
  let rows lib stencil n edges =
    List.map (fun e -> (lib, stencil, n, e)) edges
  in
  let opencv =
    List.filter
      (fun (n, _) -> n <> "periodic" && n <> "antisymmetric_whole")
      rules
  in
  let scipy = List.filter (fun (n, _) -> n <> "antisymmetric_whole") rules in
  cases "gradient agrees with scipy, OpenCV and numpy"
    ~id:(fun (lib, (s, _), _, (e, _)) -> Printf.sprintf "%s %s %s" lib s e)
    (rows "scipy" ("sobel", `Sobel) 8. scipy
    @ rows "opencv" ("sobel", `Sobel) 8. opencv
    @ rows "opencv" ("scharr", `Scharr) 32. opencv
    @ rows "numpy" ("central", `Central) 1.
        [ ("antisymmetric_whole", `Antisymmetric_whole) ])
    (fun (lib, (sn, s), n, (en, e)) ->
      let x = entry Nx.float64 "x" in
      let key axis =
        if lib = "numpy" then Printf.sprintf "central.numpy.%s" axis
        else Printf.sprintf "%s.%s.%s.%s" sn lib en axis
      in
      let big = Nx.item [] (Nx.max (Nx.abs x)) in
      (* numpy's one-sided difference at the border reads [x(1) - x(0)], which
         mynd forms from a read of up to [3] times [x]'s largest. *)
      let reach = if lib = "numpy" then 3. else 1. in
      let tol = float (2. *. 10. *. epsilon_float *. n *. reach *. big) in
      let gx, gy = V.Filter.gradient ~edge:e s x in
      equal ~msg:"x" (array tol)
        (floats (entry Nx.float64 (key "x")))
        (floats (Nx.mul_s gx n));
      equal ~msg:"y" (array tol)
        (floats (entry Nx.float64 (key "y")))
        (floats (Nx.mul_s gy n)))

(* Maxima *)

(* [reference h w c gx gy m] is the rule read from the .mli at each pixel of
   [[h; w; c]] gradients and their strengths [m], [None] where rounding may
   decide it. A strength read at [w = 0] or [1], or between two ring pixels that
   tie, is exact, and between [∞] and a finite one is [∞]; a NaN gives NaN.
   Between two others the .mli fixes no rounding: [a + w (b - a)] at float64 is
   within [4 ε max |a| |b|] of the exact interpolation, so a comparison within
   that distance is left undecided. *)
let reference h w c gx gy m =
  let at i j l = (((i * w) + j) * c) + l in
  let sign v = if v > 0. then 1 else if v < 0. then -1 else 0 in
  Array.init
    (h * w * c)
    (fun o ->
      let i = o / (w * c) and j = o / c mod w and l = o mod c in
      let mp = m.(o) in
      (* Whether [mp] passes the comparison with the strength at [p + sg u],
         [None] if rounding may decide it. *)
      let passes sg cmp =
        let ux = sg *. gx.(o) and uy = sg *. gy.(o) in
        let ax = Float.abs ux and ay = Float.abs uy in
        let si = sign uy and sj = sign ux in
        let major = ax >= ay in
        let wt = if major then ay /. ax else ax /. ay in
        let ai, aj = if major then (i, j + sj) else (i + si, j) in
        let bi, bj = (i + si, j + sj) in
        let inside r s = r >= 0 && r < h && s >= 0 && s < w in
        if (not (inside ai aj)) || (wt > 0. && not (inside bi bj)) then
          Some true
        else
          let a = m.(at ai aj l) and b = m.(at bi bj l) in
          if wt = 0. then Some (cmp mp a)
          else if wt = 1. || a = b then Some (cmp mp b)
          else if Float.is_nan a || Float.is_nan b then Some (cmp mp Float.nan)
          else if a = Float.infinity || b = Float.infinity then
            Some (cmp mp Float.infinity)
          else
            let v = a +. (wt *. (b -. a)) in
            let e =
              4. *. epsilon_float *. Float.max (Float.abs a) (Float.abs b)
            in
            if Float.is_nan v || Float.abs (mp -. v) > e then Some (cmp mp v)
            else None
      in
      if not (mp > 0.) then Some false
      else
        match (passes (-1.) ( > ), passes 1. ( >= )) with
        | Some false, _ | _, Some false -> Some false
        | Some true, Some true -> Some true
        | _ -> None)

(* Gradients of small integers tie often; a few are NaN, and some [(1.6e308,
   1e308)], whose strength overflows to [∞] and whose step interpolates at [w =
   0.625]. *)
let huge = (1.6e308, 1e308)

let gen_field =
  Gen.(
    let* h, (w, c) =
      pair (int_range 1 7) (pair (int_range 1 7) (int_range 1 2))
    in
    let small = map Float.of_int (int_range (-3) 3) in
    let v =
      frequency
        [
          (30, pair small small);
          (1, pair (constant Float.nan) small);
          (1, pair small (constant Float.nan));
          (6, constant huge);
        ]
    in
    let+ g = array ~size:(constant (h * w * c)) v in
    (h, w, c, Array.map fst g, Array.map snd g))

let field (h, w, c, gx, gy) =
  (Nx.create Nx.float64 [| h; w; c |] gx, Nx.create Nx.float64 [| h; w; c |] gy)

let maxima_rule =
  prop "gradient_maxima keeps what the rule keeps" ~count:400 gen_field
    (fun ((h, w, c, gx, gy) as f) ->
      let tx, ty = field f in
      let m = floats (Nx.hypot tx ty) in
      let want = reference h w c gx gy m in
      let got = Nx.to_array (V.Filter.gradient_maxima (tx, ty)) in
      cover ~share:0.9 "decided" (Array.for_all Option.is_some want);
      (* A neighbour [a = p - u] along an axis or a diagonal, inside, with [p]'s
         strength: [p], pointed toward, is never kept. *)
      let tie = ref false in
      Array.iteri
        (fun o mp ->
          let i = o / (w * c) and j = o / c mod w in
          let ax = Float.abs gx.(o) and ay = Float.abs gy.(o) in
          let line = mp > 0. && (ax = 0. || ay = 0. || ax = ay) in
          let si = compare gy.(o) 0. and sj = compare gx.(o) 0. in
          let ai = i - si and aj = j - sj in
          if line && ai >= 0 && ai < h && aj >= 0 && aj < w then
            if m.(o - (si * w * c) - (sj * c)) = mp then begin
              tie := true;
              equal
                ~msg:(Printf.sprintf "pixel %d, pointed toward" o)
                bool false got.(o)
            end)
        m;
      cover "a tie along an axis or a diagonal" !tie;
      (* An infinite strength between two infinite ring pixels ahead, kept where
         nothing behind it rules it out. *)
      cover "an infinite strength kept between infinite ones"
        (Array.exists
           (fun o ->
             let i = o / (w * c) and j = o / c mod w in
             (gx.(o), gy.(o)) = huge
             && i + 1 < h
             && j + 1 < w
             && m.(o + (w * c) + c) = Float.infinity
             && m.(o + c) = Float.infinity
             && want.(o) = Some true)
           (Array.init (h * w * c) Fun.id));
      cover "a NaN" (Array.exists Float.is_nan gx);
      Array.iteri
        (fun o w ->
          match w with
          | Some w -> equal ~msg:(Printf.sprintf "pixel %d" o) bool w got.(o)
          | None -> ())
        want)

let maxima_symmetries =
  prop "flips and transposes of the gradients commute with the maxima"
    ~count:200 gen_field (fun f ->
      let gx, gy = field f in
      let g = V.Filter.gradient_maxima in
      let k = g (gx, gy) in
      let flip a v = Nx.flip ~axes:[ a ] v in
      let t v = Nx.transpose ~axes:[ 1; 0; 2 ] v in
      let arr = Nx.to_array in
      equal (array bool)
        (arr (flip 1 k))
        (arr (g (Nx.neg (flip 1 gx), flip 1 gy)));
      equal (array bool)
        (arr (flip 0 k))
        (arr (g (flip 0 gx, Nx.neg (flip 0 gy))));
      equal (array bool) (arr (t k)) (arr (g (t gy, t gx))))

(* [(1, 0)] beside [(1, 2⁻⁶)] along [x]: the second's strength, [√(1 + 2⁻¹²)],
   is above [1] at float32 and rounds to [1] in float16 and bfloat16, where the
   first would tie it and be kept. *)
let maxima_narrow =
  test "narrow formats form the strength at float32" (fun () ->
      List.iter
        (fun (F (n, dt)) ->
          let gx = Nx.create dt [| 1; 2; 1 |] [| 1.; 1. |] in
          let gy = Nx.create dt [| 1; 2; 1 |] [| 0.; 0x1p-6 |] in
          equal ~msg:n (array bool) [| false; true |]
            (Nx.to_array (V.Filter.gradient_maxima (gx, gy))))
        [ F ("float16", Nx.float16); F ("bfloat16", Nx.bfloat16) ];
      let x = random [| 2; 9; 11; 1 |] in
      let g = V.Filter.gradient ~edge:`Symmetric_whole `Central x in
      let cast dt (a, b) = (Nx.cast dt a, Nx.cast dt b) in
      List.iter
        (fun (F (n, dt)) ->
          let gn = cast dt g in
          equal ~msg:n (array bool)
            (Nx.to_array (V.Filter.gradient_maxima (cast Nx.float32 gn)))
            (Nx.to_array (V.Filter.gradient_maxima gn)))
        [ F ("float16", Nx.float16); F ("bfloat16", Nx.bfloat16) ])

let maxima_compiled =
  test "compiled maxima are eager's" (fun () ->
      let small _ = Float.of_int (Random.int 7 - 3) in
      let gx = Nx.init Nx.float32 [| 2; 7; 8; 2 |] small in
      let gy = Nx.init Nx.float32 [| 2; 7; 8; 2 |] small in
      let f gx gy = V.Filter.gradient_maxima (gx, gy) in
      let sig_ = Nx.Ptree.(tensor @-> tensor @-> returns tensor) in
      equal (array bool)
        (Nx.to_array (f gx gy))
        (Nx.to_array (Rune.jit sig_ f gx gy)))

(* scikit-image keeps both pixels of a tie and reads no ring past the border;
   smooth random gradients have neither. *)
let maxima_skimage =
  test "gradient_maxima agrees with scikit-image inside the border ring"
    (fun () ->
      let gx = entry Nx.float64 "maxima.gx"
      and gy = entry Nx.float64 "maxima.gy" in
      let h = Nx.dim 0 gx and w = Nx.dim 1 gx in
      let k =
        V.Filter.gradient_maxima
          (Nx.reshape [| h; w; 1 |] gx, Nx.reshape [| h; w; 1 |] gy)
      in
      let inner x =
        Nx.to_array (Nx.slice [ Nx.R (1, h - 1); Nx.R (1, w - 1) ] x)
      in
      equal (array bool)
        (inner (entry Nx.bool "maxima.skimage"))
        (inner (Nx.reshape [| h; w |] k)))

(* Edges *)

let blurred_gradient sigma x =
  let edge = `Symmetric_whole in
  let radius = Float.to_int (Float.ceil (4. *. sigma)) in
  let b = V.Filter.gaussian ~edge ~radius (Nx.scalar Nx.float64 sigma) x in
  V.Filter.gradient ~edge `Central b

(* A step between columns 5 and 6 keeps one pixel a row at every σ. Unblurred,
   the two pixels of the step tie, and the darker is kept. *)
let step =
  test "a step keeps one pixel a row, the darker at a tie" (fun () ->
      let x =
        Nx.init Nx.float64 [| 6; 12; 1 |] (fun i ->
            if i.(1) >= 6 then 1. else 0.)
      in
      let columns k =
        let k = Nx.to_array (Nx.reshape [| 6; 12 |] k) in
        List.init 6 (fun i ->
            List.filter (fun j -> k.((i * 12) + j)) (List.init 12 Fun.id))
      in
      List.iter
        (fun sigma ->
          let ((gx, gy) as g) = blurred_gradient sigma x in
          let strong = Nx.greater_s (Nx.hypot gx gy) 1e-3 in
          let k = Nx.logical_and (V.Filter.gradient_maxima g) strong in
          List.iter
            (fun cols ->
              equal ~msg:(Printf.sprintf "σ %g" sigma) int 1 (List.length cols))
            (columns k))
        [ 0.6; 1.; 1.4; 2.; 3. ];
      let keep x =
        columns
          (V.Filter.gradient_maxima
             (V.Filter.gradient ~edge:`Replicate `Central x))
      in
      equal (list (list int)) (List.init 6 (fun _ -> [ 5 ])) (keep x);
      equal
        (list (list int))
        (List.init 6 (fun _ -> [ 6 ]))
        (keep (Nx.sub_s (Nx.neg x) (-1.))))

(* A square turned by 30° keeps its outline in one 8-connected piece. *)
let square =
  test "a rotated square's outline is one piece" (fun () ->
      let n = 40 in
      let c = Float.cos (Float.pi /. 6.) and s = Float.sin (Float.pi /. 6.) in
      let x =
        Nx.init Nx.float64 [| n; n; 1 |] (fun i ->
            let u = Float.of_int i.(1) +. 0.5 -. 20.
            and v = Float.of_int i.(0) +. 0.5 -. 20. in
            let a = (c *. u) +. (s *. v) and b = (c *. v) -. (s *. u) in
            if Float.abs a < 10. && Float.abs b < 10. then 1. else 0.)
      in
      List.iter
        (fun sigma ->
          let ((gx, gy) as g) = blurred_gradient sigma x in
          let k =
            Nx.logical_and
              (V.Filter.gradient_maxima g)
              (Nx.greater_s (Nx.hypot gx gy) 0.05)
          in
          let labels = Nx.to_array (V.Mask.components `Eight k) in
          let pieces =
            List.sort_uniq compare
              (List.filter (fun l -> l >= 0L) (Array.to_list labels))
          in
          equal ~msg:(Printf.sprintf "σ %g" sigma) int 1 (List.length pieces))
        [ 1.; 1.4; 2. ])

(* scikit-image's canny at its defaults, composed: a blur under [`Zero] divided
   by the blur of ones plus float64's epsilon, of radius [⌊4σ + ½⌋]; [`Sobel]
   under [`Symmetric_half], scipy's reflect; the border ring dropped; its
   thresholds [0.1] and [0.2] of raw Sobel strengths, eight times mynd's, met
   with [>=]; hysteresis over 8-connected candidates. *)
let canny_skimage =
  cases "Canny's edges compose to scikit-image's at its defaults"
    ~id:(fun s -> Printf.sprintf "σ %d" s)
    [ 1; 2 ]
    (fun s ->
      let x = entry Nx.float64 "canny.x" in
      let h = Nx.dim 0 x and w = Nx.dim 1 x in
      let x = Nx.reshape [| h; w; 1 |] x in
      let sigma = Float.of_int s in
      let radius = Float.to_int ((4. *. sigma) +. 0.5) in
      let blur =
        V.Filter.gaussian ~edge:`Zero ~radius (Nx.scalar Nx.float64 sigma)
      in
      let b =
        Nx.div (blur x) (Nx.add_s (blur (Nx.ones_like x)) epsilon_float)
      in
      let ((gx, gy) as g) = V.Filter.gradient ~edge:`Symmetric_half `Sobel b in
      let m = Nx.hypot gx gy in
      let ring =
        Nx.init Nx.bool [| h; w; 1 |] (fun i ->
            i.(0) = 0 || i.(0) = h - 1 || i.(1) = 0 || i.(1) = w - 1)
      in
      let thin =
        Nx.logical_and (V.Filter.gradient_maxima g) (Nx.logical_not ring)
      in
      let candidates = Nx.logical_and thin (Nx.greater_equal_s m (0.1 /. 8.)) in
      let edges =
        V.Mask.reconstruct `Eight
          ~marker:(Nx.greater_equal_s m (0.2 /. 8.))
          candidates
      in
      equal (array bool)
        (Nx.to_array (entry Nx.bool (Printf.sprintf "canny.%d" s)))
        (Nx.to_array edges))

let refusals =
  cases "misuse raises"
    ~id:(fun (n, _) -> n)
    [
      ( "two axes",
        fun () ->
          ignore
            (V.Filter.gradient ~edge:`Zero `Sobel
               (Nx.zeros Nx.float32 [| 4; 4 |])) );
      ( "no row past the edge",
        fun () ->
          ignore
            (V.Filter.gradient ~edge:`Replicate `Central
               (Nx.zeros Nx.float32 [| 0; 4; 1 |])) );
      ( "maxima of two axes",
        fun () ->
          let g = Nx.zeros Nx.float32 [| 4; 4 |] in
          ignore (V.Filter.gradient_maxima (g, g)) );
      ( "maxima of two shapes",
        fun () ->
          ignore
            (V.Filter.gradient_maxima
               ( Nx.zeros Nx.float32 [| 4; 4; 1 |],
                 Nx.zeros Nx.float32 [| 4; 5; 1 |] )) );
    ]
    (fun (_, f) -> raises_match (Exn.invalid_arg ~sub:"Mynd.") f)

let () =
  exit
    (run "mynd.gradient"
       [
         against_correlate;
         ramp;
         symmetries;
         narrow;
         derivative;
         compiled;
         libraries;
         maxima_rule;
         maxima_symmetries;
         maxima_narrow;
         maxima_compiled;
         maxima_skimage;
         step;
         square;
         canny_skimage;
         refusals;
       ])
