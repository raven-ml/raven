(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let shape_of x = Array.to_list (Nx.shape x)
let eps (type s) (dt : (float, s) Nx.dtype) = (Nx.Dtype.float_format dt).epsilon

(* NaN equals NaN here: bit-for-bit comparison of float arrays. *)
let bits a =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    a

let same want got = equal (array int64) (bits want) (bits got)

let extensions : (string * V.extension) list =
  [
    ("zero", `Zero);
    ("replicate", `Replicate);
    ("symmetric_whole", `Symmetric_whole);
    ("symmetric_half", `Symmetric_half);
    ("periodic", `Periodic);
  ]

let rules : (string * Nx.extension) list =
  List.map (fun (n, e) -> (n, (e :> Nx.extension))) extensions
  @ [ ("antisymmetric_whole", `Antisymmetric_whole) ]

let image h w c f =
  Nx.create Nx.float64 [| h; w; c |]
    (Array.init (h * w * c) (fun k -> f (k / (w * c)) (k / c mod w) (k mod c)))

let random h w c = image h w c (fun _ _ _ -> Random.float 2. -. 1.)

(* Values in [0, 1) from the test's own [Random] state. *)
let uniform s =
  Nx.create Nx.float64 s
    (Array.init (Array.fold_left ( * ) 1 s) (fun _ -> Random.float 1.))

(* The reads of a window *)

(* [reads e n lo hi] is the index the extension reads at each position from
   [-lo] to [n - 1 + hi] of a signal of [n], [-1] for zero, by
   {!Nx.extension_index}. *)
let reads (e : V.extension) n lo hi =
  let p =
    Nx.create Nx.int64
      [| n + lo + hi |]
      (Array.init (n + lo + hi) (fun i -> Int64.of_int (i - lo)))
  in
  Array.map Int64.to_int (Nx.to_array (Nx.extension_index e n p))

(* [reader e (kh, kw) h w] is, for pixel [(i, j)]'s tap [(p, q)], the pixel it
   reads, or [None] for zero. *)
let reader e (kh, kw) h w =
  let rows = reads e h (kh / 2) ((kh - 1) / 2)
  and cols = reads e w (kw / 2) ((kw - 1) / 2) in
  fun i j p q ->
    let a = rows.(i + p) and b = cols.(j + q) in
    if a < 0 || b < 0 then None else Some (a, b)

(* The image extended as [Nx.extend] reads it, [[h + kh - 1; w + kw - 1; c]]:
   the reads of [`Antisymmetric_whole], which combine two pixels. *)
let extended e (kh, kw) x =
  Nx.extend ~axis:0 e (kh / 2, (kh - 1) / 2) x
  |> Nx.extend ~axis:1 e (kw / 2, (kw - 1) / 2)
  |> floats

(* [tap e (kh, kw) x] is [fun i j l p q -> value] read by the spec. *)
let tap e (kh, kw) x =
  let h = Nx.dim 0 x and w = Nx.dim 1 x and c = Nx.dim 2 x in
  let v = floats x in
  match e with
  | #V.extension as e -> (
      let r = reader e (kh, kw) h w in
      fun i j l p q ->
        match r i j p q with
        | None -> 0.
        | Some (a, b) -> v.((((a * w) + b) * c) + l))
  | `Antisymmetric_whole ->
      let ext = extended e (kh, kw) x in
      let w' = w + kw - 1 in
      fun i j l p q -> ext.(((((i + p) * w') + j + q) * c) + l)

(* [naive e k x] is the correlation summed in float64 in row-major tap order,
   and the bound [n ε Σ |tap k|] on its distance to any order of the sum. *)
let naive e k x =
  let kh = Nx.dim 0 k and kw = Nx.dim 1 k in
  let kv = floats k in
  let h = Nx.dim 0 x and w = Nx.dim 1 x and c = Nx.dim 2 x in
  let t = tap e (kh, kw) x in
  let sum = Array.make (h * w * c) 0. and bound = Array.make (h * w * c) 0. in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      for l = 0 to c - 1 do
        let s = ref 0. and a = ref 0. in
        for p = 0 to kh - 1 do
          for q = 0 to kw - 1 do
            let v = kv.((p * kw) + q) *. t i j l p q in
            s := !s +. v;
            a := !a +. Float.abs v
          done
        done;
        let o = (((i * w) + j) * c) + l in
        sum.(o) <- !s;
        bound.(o) <- Float.of_int ((kh * kw) + 1) *. epsilon_float *. !a
      done
    done
  done;
  (sum, bound)

let within (want, bound) got =
  Array.iteri
    (fun i w ->
      let g = got.(i) in
      if
        not
          ((Float.is_nan w && Float.is_nan g) || Float.abs (g -. w) <= bound.(i))
      then failf "element %d is %h, not %h within %h" i g w bound.(i))
    want

(* Windows *)

let windows_reads =
  let sizes = [ (1, 1); (3, 3); (2, 4); (5, 2); (6, 7) ] in
  cases "windows read the extension's pixels, every dtype"
    ~id:(fun ((n, _), (kh, kw)) -> Printf.sprintf "%s %dx%d" n kh kw)
    (List.concat_map (fun e -> List.map (fun s -> (e, s)) sizes) extensions)
    (fun ((_, e), (kh, kw)) ->
      let h = 4 and w = 5 and c = 2 in
      let x =
        image h w c (fun i j l -> Float.of_int ((((i * w) + j) * c) + l + 1))
      in
      let want = Array.make (h * w * c * kh * kw) 0. in
      let r = reader e (kh, kw) h w in
      for i = 0 to h - 1 do
        for j = 0 to w - 1 do
          for l = 0 to c - 1 do
            for p = 0 to kh - 1 do
              for q = 0 to kw - 1 do
                let o = (((((((i * w) + j) * c) + l) * kh) + p) * kw) + q in
                want.(o) <-
                  (match r i j p q with
                  | None -> 0.
                  | Some (a, b) -> Float.of_int ((((a * w) + b) * c) + l + 1))
              done
            done
          done
        done
      done;
      let check (type a b) (dt : (a, b) Nx.dtype) =
        let y = V.Filter.windows ~edge:e (kh, kw) (Nx.cast dt x) in
        equal (list int) [ h; w; c; kh; kw ] (shape_of y);
        equal (array float_exact) want (floats y)
      in
      check Nx.float32;
      check Nx.int32;
      check Nx.uint8;
      let m = Nx.greater_s x 20. in
      equal (array bool)
        (Array.map (fun v -> v > 20.) want)
        (Nx.to_array (V.Filter.windows ~edge:e (kh, kw) m)))

let windows_derivative =
  test "windows' derivative adds back to the pixel each reads" (fun () ->
      List.iter
        (fun (name, e) ->
          let kh = 3 and kw = 4 and h = 3 and w = 5 and c = 1 in
          let x = random h w c in
          let rw = uniform [| h; w; c; kh; kw |] in
          let g =
            Rune.grad'
              (fun x ->
                Nx.sum (Nx.mul (V.Filter.windows ~edge:e (kh, kw) x) rw))
              x
          in
          let want = Array.make (h * w * c) 0. in
          let rv = Nx.to_array rw and r = reader e (kh, kw) h w in
          for i = 0 to h - 1 do
            for j = 0 to w - 1 do
              for p = 0 to kh - 1 do
                for q = 0 to kw - 1 do
                  match r i j p q with
                  | None -> ()
                  | Some (a, b) ->
                      let o = (a * w) + b in
                      want.(o) <-
                        want.(o) +. rv.((((((i * w) + j) * kh) + p) * kw) + q)
                done
              done
            done
          done;
          equal ~msg:name (array (float 1e-12)) want (Nx.to_array g))
        extensions)

(* Correlation *)

let shapes = [ (1, 1); (3, 3); (4, 4); (2, 5); (1, 6); (7, 7); (11, 13) ]

let correlate_naive =
  cases "correlate is the sum of taps times weights"
    ~id:(fun ((n, _), (kh, kw)) -> Printf.sprintf "%s %dx%d" n kh kw)
    (List.concat_map (fun e -> List.map (fun s -> (e, s)) shapes) rules)
    (fun ((_, e), (kh, kw)) ->
      let x = random 5 7 2 and k = uniform [| kh; kw |] in
      within (naive e k x) (floats (V.Filter.correlate ~edge:e k x)))

let correlate_empty =
  test "an image of no row or no channel" (fun () ->
      let k = Nx.ones Nx.float64 [| 3; 2 |] in
      equal (list int) [ 0; 4; 2 ]
        (shape_of
           (V.Filter.correlate ~edge:`Zero k
              (Nx.zeros Nx.float64 [| 0; 4; 2 |])));
      equal (list int) [ 3; 0; 2 ]
        (shape_of
           (V.Filter.correlate ~edge:`Zero k
              (Nx.zeros Nx.float64 [| 3; 0; 2 |])));
      equal (list int) [ 3; 4; 0 ]
        (shape_of
           (V.Filter.correlate ~edge:`Periodic k
              (Nx.zeros Nx.float64 [| 3; 4; 0 |])));
      equal (list int) [ 0; 3; 4; 2 ]
        (shape_of
           (V.Filter.correlate ~edge:`Replicate
              (Nx.ones Nx.float64 [| 0; 3; 2 |])
              (Nx.zeros Nx.float64 [| 3; 4; 2 |]))))

(* An image's result is the same in any batch: shared, paired, a bank over one
   image and broadcast batches give each image's bits alone, at extents of 49
   taps and more. *)
let batch_law =
  cases "each image's result is the same in any batch"
    ~id:(fun ((n, _), (kh, kw)) -> Printf.sprintf "%s %dx%d" n kh kw)
    (List.concat_map
       (fun e -> List.map (fun s -> (e, s)) [ (3, 5); (7, 7); (9, 9) ])
       [
         ("zero", `Zero);
         ("symmetric_half", `Symmetric_half);
         ("antisymmetric_whole", `Antisymmetric_whole);
       ])
    (fun ((_, (e : Nx.extension)), (kh, kw)) ->
      let n = 3 and h = 6 and w = 8 and c = 2 in
      let xs = uniform [| n; h; w; c |] in
      let ks = uniform [| n; kh; kw |] in
      let alone i k =
        floats (V.Filter.correlate ~edge:e k (Nx.slice [ Nx.I i ] xs))
      in
      let ki i = Nx.slice [ Nx.I i ] ks in
      let x0 = Nx.slice [ Nx.I 0 ] xs in
      let loop f = Array.concat (List.init n f) in
      same
        (loop (fun i -> alone i (ki i)))
        (floats (V.Filter.correlate ~edge:e ks xs));
      same
        (loop (fun i -> alone i (ki 0)))
        (floats (V.Filter.correlate ~edge:e (ki 0) xs));
      same
        (loop (fun i -> floats (V.Filter.correlate ~edge:e (ki i) x0)))
        (floats (V.Filter.correlate ~edge:e ks x0));
      (* [[n; 1]] weights against [[1; n]] images: every pair. *)
      let pairs =
        V.Filter.correlate ~edge:e
          (Nx.reshape [| n; 1; kh; kw |] ks)
          (Nx.reshape [| 1; n; h; w; c |] xs)
      in
      equal (list int) [ n; n; h; w; c ] (shape_of pairs);
      same
        (Array.concat (List.init n (fun a -> loop (fun b -> alone b (ki a)))))
        (floats pairs))

let separable =
  test "a column then a row is the product's correlation to rounding" (fun () ->
      let x = random 6 7 2 in
      let u = uniform [| 5; 1 |] and v = uniform [| 1; 4 |] in
      List.iter
        (fun (name, e) ->
          let one = V.Filter.correlate ~edge:e (Nx.matmul u v) x in
          let two =
            V.Filter.correlate ~edge:e v (V.Filter.correlate ~edge:e u x)
          in
          equal ~msg:name (array (float 1e-13)) (floats one) (floats two))
        rules)

(* The derivative in [k] at [(p, q)] is the sum of [r] times tap [(p, q)]; in
   [x] at a pixel, the sum of [r] times the weight of each tap that reads it. *)
let derivatives =
  cases "derivatives in the weights and the image"
    ~id:(fun (n, _) -> n)
    extensions
    (fun (_, e) ->
      let kh = 4 and kw = 3 and h = 3 and w = 5 and c = 2 in
      let x = random h w c and k = uniform [| kh; kw |] in
      let r = uniform [| h; w; c |] in
      let rv = Nx.to_array r and kv = Nx.to_array k in
      let t = tap (e :> Nx.extension) (kh, kw) x
      and rd = reader e (kh, kw) h w in
      let gk = Array.make (kh * kw) 0. and gx = Array.make (h * w * c) 0. in
      for i = 0 to h - 1 do
        for j = 0 to w - 1 do
          for l = 0 to c - 1 do
            let ro = rv.((((i * w) + j) * c) + l) in
            for p = 0 to kh - 1 do
              for q = 0 to kw - 1 do
                gk.((p * kw) + q) <- gk.((p * kw) + q) +. (ro *. t i j l p q);
                match rd i j p q with
                | None -> ()
                | Some (a, b) ->
                    let o = (((a * w) + b) * c) + l in
                    gx.(o) <- gx.(o) +. (ro *. kv.((p * kw) + q))
              done
            done
          done
        done
      done;
      let loss k x = Nx.sum (Nx.mul (V.Filter.correlate ~edge:e k x) r) in
      equal ~msg:"weights"
        (array (float 1e-12))
        gk
        (Nx.to_array (Rune.grad' (fun k -> loss k x) k));
      equal ~msg:"image"
        (array (float 1e-12))
        gx
        (Nx.to_array (Rune.grad' (fun x -> loss k x) x)))

let ramp =
  test "under Antisymmetric_whole a difference reads a ramp's slope" (fun () ->
      let x =
        image 3 6 1 (fun i j _ -> (0.5 *. Float.of_int j) +. Float.of_int i)
      in
      let d = Nx.create Nx.float64 [| 1; 3 |] [| -0.5; 0.; 0.5 |] in
      equal (array float_exact) (Array.make 18 0.5)
        (floats (V.Filter.correlate ~edge:`Antisymmetric_whole d x)))

(* One NaN pixel makes NaN exactly the results whose window reads it. *)
let nan_reach =
  cases "a NaN reaches the results that read it, compiled too"
    ~id:(fun ((n, _), (kh, kw)) -> Printf.sprintf "%s %dx%d" n kh kw)
    (List.concat_map
       (fun e -> List.map (fun s -> (e, s)) [ (3, 3); (4, 2); (9, 9) ])
       rules)
    (fun ((_, e), (kh, kw)) ->
      let h = 5 and w = 6 in
      List.iter
        (fun (i0, j0) ->
          let x =
            image h w 1 (fun i j _ ->
                if i = i0 && j = j0 then Float.nan else 1.)
          in
          let t = tap e (kh, kw) x in
          let want =
            Array.init (h * w) (fun o ->
                let i = o / w and j = o mod w in
                let hit = ref false in
                for p = 0 to kh - 1 do
                  for q = 0 to kw - 1 do
                    if Float.is_nan (t i j 0 p q) then hit := true
                  done
                done;
                !hit)
          in
          (* Weights of [0] read their taps all the same. *)
          let k = Nx.zeros Nx.float64 [| kh; kw |] in
          let isnan y = Array.map Float.is_nan (floats y) in
          equal ~msg:"eager" (array bool) want
            (isnan (V.Filter.correlate ~edge:e k x));
          equal ~msg:"compiled" (array bool) want
            (isnan (Rune.jit' (V.Filter.correlate ~edge:e k) x)))
        [ (0, 0); (2, 3); (4, 5) ])

let compiled =
  test "compiled is eager to rounding" (fun () ->
      let x = random 6 7 3 and k = uniform [| 4; 5 |] in
      List.iter
        (fun (name, e) ->
          let s = Nx.sum (Nx.abs k) |> Nx.item [] in
          equal ~msg:name
            (array (float (21. *. epsilon_float *. s)))
            (floats (V.Filter.correlate ~edge:e k x))
            (floats (Rune.jit' (V.Filter.correlate ~edge:e k) x)))
        rules)

(* Against scipy and OpenCV, from golden/filter.npz
   (contrib/mynd/test/gen/filter.py) *)
let golden = lazy (Nx_io.load_npz "golden/filter.npz")

let entry key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack Nx.float64 p
  | None -> failf "golden/filter.npz has no %s" key

(* OpenCV's filter2D reads no [`Periodic] and sums the larger extents by a
   transform; the goldens hold its direct sums alone. *)
let correlate_libraries =
  let rows lib sizes edges =
    List.concat_map (fun e -> List.map (fun k -> (lib, e, k)) sizes) edges
  in
  let opencv = List.filter (fun (n, _) -> n <> "periodic") extensions in
  cases "correlate agrees with scipy and OpenCV"
    ~id:(fun (lib, (n, _), k) -> Printf.sprintf "%s %s %s" lib k n)
    (rows "scipy" [ "3x3"; "4x4"; "2x5"; "1x6"; "11x13" ] extensions
    @ rows "opencv" [ "3x3"; "4x4"; "2x5" ] opencv)
    (fun (lib, (n, e), k) ->
      let key = Printf.sprintf "correlate.%s.%s.%s" lib k n in
      let x = entry "x" and k = entry ("k." ^ k) in
      let _, bound = naive (e :> Nx.extension) k x in
      within
        (Nx.to_array (entry key), Array.map (fun b -> 2. *. b) bound)
        (floats (V.Filter.correlate ~edge:e k x)))

(* Gaussian blur *)

let scalar s = Nx.scalar Nx.float64 s

(* A constant stays constant: each pass sums [2r + 1] weights whose own sum is
   [1] within [(2r + 1) u], and rounds [2r + 1] additions, so each moves it by
   at most [2 (2r + 1) u |v|], [4 (2r + 1) ε |v|] over both. A subnormal [v]
   rounds each product by up to the least subnormal [η] instead: [2 (2r + 1) η]
   more. *)
let constant =
  prop "a constant image stays constant to rounding" ~count:200
    Gen.(
      quad
        (of_list (List.tl rules))
        (float_range 0.2 3.) (int_range 0 6)
        (pair (float_range (-100.) 100.) bool))
    (fun ((name, e), sigma, radius, (v, single)) ->
      cover ~share:0.1 "radius 0" (radius = 0);
      let check (type s) (dt : (float, s) Nx.dtype) =
        let x = Nx.full dt [| 4; 5; 2 |] v in
        let y =
          floats (V.Filter.gaussian ~edge:e ~radius (Nx.scalar dt sigma) x)
        in
        let v = (floats x).(0) in
        let f = Nx.Dtype.float_format dt in
        let eta = f.min_normal *. f.epsilon in
        let taps = Float.of_int ((2 * radius) + 1) in
        let bound = 4. *. taps *. ((eps dt *. Float.abs v) +. eta) in
        Array.iter
          (fun y ->
            if Float.abs (y -. v) > bound then
              failf "%s: %h becomes %h" name v y)
          y
      in
      if single then check Nx.float32 else check Nx.float64)

(* Each pass is within [δ = 2 erfc (r / σ√2)] of the samples over every integer,
   here of radius [r + 40], past [13 σ]. The second pass's weights are positive
   and sum to [1], so it carries the first's difference unchanged: [2 δ] in all
   on an image within [[-1, 1]]. [`Antisymmetric_whole] can read past [1], so it
   is left out. *)
let truncation =
  prop "truncation is within Getreuer's bound" ~count:150
    Gen.(
      triple
        (of_list (List.filter (fun (n, _) -> n <> "antisymmetric_whole") rules))
        (float_range 0.3 3.) (int_range 0 8))
    (fun ((name, e), sigma, radius) ->
      let x = random 6 7 1 in
      let blur radius =
        floats (V.Filter.gaussian ~edge:e ~radius (scalar sigma) x)
      in
      let delta =
        2. *. Float.erfc (Float.of_int radius /. (sigma *. Float.sqrt 2.))
      in
      let bound = (2. *. delta) +. 1e-12 in
      let near = blur radius and far = blur (radius + 40) in
      Array.iteri
        (fun i f ->
          if Float.abs (near.(i) -. f) > bound then
            failf "%s σ %g radius %d: %h against %h, bound %h" name sigma radius
              near.(i) f bound)
        far)

type flt = F : string * (float, _) Nx.dtype -> flt

let wide =
  [
    F ("float16", Nx.float16);
    F ("bfloat16", Nx.bfloat16);
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
  ]

(* Where [e^(-1 / 2σ²)] underflows at the dtype the weights are computed in,
   float32 for the narrow formats, the weights are the impulse. *)
let impulse (type s) (dt : (float, s) Nx.dtype) sigma =
  let x =
    if Nx.Dtype.bits dt < 32 then
      Int32.float_of_bits (Int32.bits_of_float sigma)
    else sigma
  in
  let s2 = 2. *. x *. x in
  if Nx.Dtype.bits dt < 64 then
    Int32.float_of_bits
      (Int32.bits_of_float
         (Float.exp (-1. /. Int32.float_of_bits (Int32.bits_of_float s2))))
    = 0.
  else Float.exp (-1. /. s2) = 0.

let tiny_sigma =
  let sigmas =
    [
      0.; -0.; 5e-324; 1e-300; 1e-30; 1e-20; 1e-5; 0.01; 0.026; 0.03; 0.07; 0.1;
    ]
  in
  cases "a tiny σ gives finite values and derivatives"
    ~id:(fun (F (n, _), s) -> Printf.sprintf "%s %g" n s)
    (List.concat_map (fun d -> List.map (fun s -> (d, s)) sigmas) wide)
    (fun (F (_, dt), sigma) ->
      let x = Nx.cast dt (random 4 5 1) in
      let r = Nx.cast dt (uniform [| 4; 5; 1 |]) in
      let s = Nx.scalar dt sigma in
      let sigma = Nx.item [] (Nx.cast Nx.float64 s) in
      let blur s = V.Filter.gaussian ~edge:`Symmetric_half ~radius:3 s x in
      let y = floats (blur s) in
      let g =
        Nx.item []
          (Nx.cast Nx.float64
             (Rune.grad' (fun s -> Nx.sum (Nx.mul (blur s) r)) s))
      in
      if not (Array.for_all Float.is_finite y) then
        failf "a value is not finite";
      if not (Float.is_finite g) then failf "the derivative is %h" g;
      if impulse dt sigma then begin
        equal (array float_exact) (floats x) y;
        equal (float 0x1p-1074) 0. g
      end)

let nan_sigma =
  test "a NaN σ gives NaN" (fun () ->
      let y =
        V.Filter.gaussian ~edge:`Zero ~radius:2 (scalar Float.nan)
          (random 3 4 1)
      in
      equal (array float_exact) (Array.make 12 Float.nan) (floats y))

(* Narrow formats compute at float32 and round once: the float32 blur of the
   same values, cast. Under [`Antisymmetric_whole] that is within one ulp of
   float16 of the float64 blur, whose taps outside round at float64. *)
let narrow =
  cases "narrow formats round once"
    ~id:(fun (n, _) -> n)
    rules
    (fun (_, e) ->
      let blur (type s) (dt : (float, s) Nx.dtype) sigma x =
        V.Filter.gaussian ~edge:e ~radius:3 (Nx.scalar dt sigma) (Nx.cast dt x)
      in
      List.iter
        (fun (F (_, dt)) ->
          let x = Nx.cast dt (random 5 6 2) in
          let sigma = Nx.item [] (Nx.cast Nx.float64 (Nx.scalar dt 1.1)) in
          same
            (floats (Nx.cast dt (blur Nx.float32 sigma x)))
            (floats (blur dt sigma x));
          let oracle = floats (blur Nx.float64 sigma x) in
          let got = floats (blur dt sigma x) in
          Array.iteri
            (fun i o ->
              let ulp = Float.max (Float.abs o) 0x1p-14 *. eps dt in
              if Float.abs (got.(i) -. o) > ulp then
                failf "element %d is %h, not %h within an ulp" i got.(i) o)
            oracle)
        [ F ("float16", Nx.float16); F ("bfloat16", Nx.bfloat16) ])

let sigma_batch =
  test "a σ an image and a bank of σ over one image" (fun () ->
      let xs = uniform [| 3; 5; 6; 2 |] and sigmas = [| 0.4; 1.; 2.5 |] in
      let s = Nx.create Nx.float64 [| 3 |] sigmas in
      let alone x sigma =
        floats (V.Filter.gaussian ~edge:`Periodic ~radius:4 (scalar sigma) x)
      in
      let x i = Nx.slice [ Nx.I i ] xs in
      same
        (Array.concat (List.init 3 (fun i -> alone (x i) sigmas.(i))))
        (floats (V.Filter.gaussian ~edge:`Periodic ~radius:4 s xs));
      same
        (Array.concat (List.init 3 (fun i -> alone (x 0) sigmas.(i))))
        (floats (V.Filter.gaussian ~edge:`Periodic ~radius:4 s (x 0))))

let sigma_derivative =
  prop "the derivative in σ is its central difference" ~count:50
    Gen.(pair (of_list rules) (float_range 0.5 3.))
    (fun ((name, e), sigma) ->
      let x = random 4 5 1 and r = uniform [| 4; 5; 1 |] in
      let f s = Nx.sum (Nx.mul (V.Filter.gaussian ~edge:e ~radius:5 s x) r) in
      let at s = Nx.item [] (f (scalar s)) in
      let d = 1e-6 in
      let fd = (at (sigma +. d) -. at (sigma -. d)) /. (2. *. d) in
      equal ~msg:name
        (float_rel ~rel:1e-6 ~abs:1e-8)
        fd
        (Nx.item [] (Rune.grad' f (scalar sigma))))

let gaussian_compiled =
  test "a compiled blur is eager to rounding" (fun () ->
      let x = Nx.cast Nx.float32 (random 6 7 2) in
      let s = Nx.scalar Nx.float32 1.4 in
      List.iter
        (fun (name, e) ->
          let blur s = V.Filter.gaussian ~edge:e ~radius:4 s x in
          equal ~msg:name
            (array (float (64. *. eps Nx.float32)))
            (floats (blur s))
            (floats (Rune.jit' blur s)))
        rules)

(* Each side is within [(4K + 8) u Σ |tap k|] of the exact correlation by the
   outer product [k] of the normalised samples, for [K = 2 radius + 1] taps: two
   passes of [K] products and weights each rounded [K + 3] times. [naive]'s [(K²
   + 1) ε Σ |tap k|] is at least that, so the two agree within twice it. *)
let gaussian_libraries =
  let rows lib edges =
    List.concat_map
      (fun e ->
        List.map
          (fun t -> (lib, e, t))
          [ ("0.5", 1); ("1.3", 4); ("2.0", 6); ("0.8", 7) ])
      edges
  in
  let opencv = List.filter (fun (n, _) -> n <> "periodic") extensions in
  cases "gaussian agrees with scipy and OpenCV"
    ~id:(fun (lib, (n, _), (s, r)) ->
      Printf.sprintf "%s σ %s radius %d %s" lib s r n)
    (rows "scipy" extensions @ rows "opencv" opencv)
    (fun (lib, (n, e), (sigma, radius)) ->
      let key = Printf.sprintf "gaussian.%s.%s.%d.%s" lib sigma radius n in
      let s = Float.of_string sigma in
      let taps = (2 * radius) + 1 in
      let g =
        Array.init taps (fun i ->
            let d = Float.of_int (i - radius) in
            Float.exp (-.(d *. d) /. (2. *. s *. s)))
      in
      let sum = Array.fold_left ( +. ) 0. g in
      let k =
        Nx.init Nx.float64 [| taps; taps |] (fun i ->
            g.(i.(0)) *. g.(i.(1)) /. (sum *. sum))
      in
      let x = entry "x" in
      let _, bound = naive (e :> Nx.extension) k x in
      within
        (Nx.to_array (entry key), Array.map (fun b -> 2. *. b) bound)
        (floats (V.Filter.gaussian ~edge:e ~radius (scalar s) x)))

(* Median *)

(* nx's sort order on values read as float64: NaN above every number, [-0.]
   below [+0.]. *)
let order a b =
  match (Float.is_nan a, Float.is_nan b) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false ->
      if a < b then -1
      else if a > b then 1
      else compare (Float.sign_bit b) (Float.sign_bit a)

(* The lower median of each window by a stable sort of its taps in row-major
   order: its value, and the pixel its tap reads. *)
let median_ref e (kh, kw) h w c v =
  let r = reader e (kh, kw) h w in
  let n = kh * kw in
  Array.init
    (h * w * c)
    (fun o ->
      let i = o / (w * c) and j = o / c mod w and l = o mod c in
      let taps =
        List.init n (fun t ->
            match r i j (t / kw) (t mod kw) with
            | None -> (0., None)
            | Some (a, b) -> (v.((((a * w) + b) * c) + l), Some (a, b)))
      in
      List.nth
        (List.stable_sort (fun (a, _) (b, _) -> order a b) taps)
        ((n - 1) / 2))

type dt = D : string * ('a, 'b) Nx.dtype * float list -> dt

let special = [ Float.nan; -0.; 0.; Float.infinity; Float.neg_infinity ]
let ties = [ -2.; -1.; 0.; 1.; 1.; 2.; 3. ]

let median_dtypes =
  [
    D ("float64", Nx.float64, ties @ special);
    D ("float32", Nx.float32, ties @ special);
    D ("float16", Nx.float16, ties @ special);
    D ("int32", Nx.int32, ties);
    D ("uint8", Nx.uint8, [ 0.; 1.; 1.; 2.; 7.; 255. ]);
    D ("bool", Nx.bool, [ 0.; 1. ]);
  ]

let pick values = List.nth values (Random.int (List.length values))
let windows_sizes = [ (3, 3); (2, 2); (2, 3); (4, 4); (1, 5) ]

let median_sort =
  cases "the lower median of a stable sort of the window, every dtype"
    ~id:(fun (D (n, _, _), (en, _), (kh, kw)) ->
      Printf.sprintf "%s %s %dx%d" n en kh kw)
    (List.concat_map
       (fun d ->
         List.concat_map
           (fun e -> List.map (fun s -> (d, e, s)) windows_sizes)
           extensions)
       median_dtypes)
    (fun (D (_, dt, values), (_, e), (kh, kw)) ->
      let h = 7 and w = 6 and c = 2 in
      let v = Array.init (h * w * c) (fun _ -> pick values) in
      let x = Nx.cast dt (Nx.create Nx.float64 [| h; w; c |] v) in
      let v = floats x in
      let want = Array.map fst (median_ref e (kh, kw) h w c v) in
      let got = V.Filter.median ~edge:e (kh, kw) x in
      equal (list int) [ h; w; c ] (shape_of got);
      same want (floats got))

let median_tall =
  test "bands of rows cover a tall image" (fun () ->
      let h = 23 and w = 3 and c = 1 in
      let x = random h w c in
      List.iter
        (fun (kh, kw) ->
          let want =
            Array.map fst (median_ref `Replicate (kh, kw) h w c (floats x))
          in
          same want (floats (V.Filter.median ~edge:`Replicate (kh, kw) x)))
        [ (2, 2); (3, 3); (5, 1); (1, 1) ])

let median_nan =
  prop "the median is NaN where more than half the window's taps read NaN"
    Gen.(pair (of_list extensions) (of_list windows_sizes))
    (fun ((_, e), (kh, kw)) ->
      let h = 5 and w = 6 in
      let x =
        image h w 1 (fun _ _ _ ->
            if Random.bool () then Float.nan else Random.float 1.)
      in
      let r = reader e (kh, kw) h w and v = floats x in
      let want =
        Array.init (h * w) (fun o ->
            let i = o / w and j = o mod w in
            let nans = ref 0 in
            for p = 0 to kh - 1 do
              for q = 0 to kw - 1 do
                match r i j p q with
                | Some (a, b) when Float.is_nan v.((a * w) + b) -> incr nans
                | _ -> ()
              done
            done;
            2 * !nans > kh * kw)
      in
      equal (array bool) want
        (Array.map Float.is_nan (floats (V.Filter.median ~edge:e (kh, kw) x))))

let median_derivative =
  cases "the derivative goes to the pixel the chosen tap reads"
    ~id:(fun ((n, _), (kh, kw)) -> Printf.sprintf "%s %dx%d" n kh kw)
    (List.concat_map
       (fun e -> List.map (fun s -> (e, s)) windows_sizes)
       extensions)
    (fun ((_, e), (kh, kw)) ->
      let h = 5 and w = 6 and c = 1 in
      let x = image h w c (fun _ _ _ -> pick ties) in
      let r = uniform [| h; w; c |] in
      let rv = Nx.to_array r in
      let want = Array.make (h * w) 0. in
      Array.iteri
        (fun o (_, src) ->
          match src with
          | None -> ()
          | Some (a, b) -> want.((a * w) + b) <- want.((a * w) + b) +. rv.(o))
        (median_ref e (kh, kw) h w c (floats x));
      equal
        (array (float 1e-12))
        want
        (Nx.to_array
           (Rune.grad'
              (fun x -> Nx.sum (Nx.mul (V.Filter.median ~edge:e (kh, kw) x) r))
              x)))

let median_compiled =
  test "a compiled median is eager" (fun () ->
      let x =
        Nx.cast Nx.float32 (image 6 7 2 (fun _ _ _ -> pick (ties @ special)))
      in
      List.iter
        (fun (name, e) ->
          let f = V.Filter.median ~edge:e (3, 2) in
          equal ~msg:name (array int64)
            (bits (floats (f x)))
            (bits (floats (Rune.jit' f x))))
        extensions)

let median_libraries =
  let rows =
    List.concat_map
      (fun e ->
        List.map (fun s -> (e, s)) [ (3, 3); (2, 2); (2, 3); (4, 4); (5, 1) ])
      extensions
  in
  cases "median agrees with scipy's lower rank"
    ~id:(fun ((n, _), (kh, kw)) -> Printf.sprintf "%s %dx%d" n kh kw)
    rows
    (fun ((n, e), (kh, kw)) ->
      (* scipy orders by value alone, so it may pick either zero. *)
      let zero = Array.map (fun v -> v +. 0.) in
      same
        (zero
           (Nx.to_array
              (entry (Printf.sprintf "median.scipy.%dx%d.%s" kh kw n))))
        (zero (floats (V.Filter.median ~edge:e (kh, kw) (entry "m")))))

(* Batch axes index independent images: a batch's windows and medians are each
   image's. *)
let batches =
  cases "windows and medians of a batch are each image's"
    ~id:(fun (n, _) -> n)
    extensions
    (fun (_, e) ->
      let xs = uniform [| 2; 3; 5; 4; 2 |] in
      let each f =
        Array.concat
          (List.concat_map
             (fun a ->
               List.init 3 (fun b ->
                   floats (f (Nx.slice [ Nx.I a; Nx.I b ] xs))))
             [ 0; 1 ])
      in
      let wins x = V.Filter.windows ~edge:e (3, 2) x in
      let med x = V.Filter.median ~edge:e (3, 3) x in
      same (each wins) (floats (wins xs));
      same (each med) (floats (med xs)))

(* [within budget f] runs [f] with the host's budget [budget] bytes above what
   it holds after collecting, and says whether every allocation fitted: one that
   does not first collects and frees the cache, so only the memory [f] holds
   live at once counts. *)
let within budget f =
  let settle () =
    for _ = 1 to 4 do
      Gc.full_major ();
      Rig.free_cache Rig.host
    done
  in
  ignore (Sys.opaque_identity (f ()));
  settle ();
  Rig.set_budget Rig.host (Rig.allocated Rig.host + budget);
  Fun.protect
    ~finally:(fun () -> Rig.set_budget Rig.host max_int)
    (fun () ->
      match f () with _ -> true | exception Rig.Out_of_memory _ -> false)

(* The median sorts the windows of about [h w / (kh kw)] pixels an image at a
   time, even for an image of fewer rows than a window has taps: beside one
   extended copy and the result, about twice [x]'s elements and as many [int64]
   positions, here within twice that. *)
type sized = Z : string * ('a, 'b) Nx.dtype * int -> sized

let median_memory =
  cases "a short image's median holds about twice its elements"
    ~id:(fun (Z (n, _, _)) -> n)
    [ Z ("float64", Nx.float64, 8); Z ("uint8", Nx.uint8, 1) ]
    (fun (Z (_, dt, e)) ->
      let h = 2 and w = 300 and c = 3 and k = 7 in
      let x =
        Nx.cast dt
          (image h w c (fun i j l ->
               Float.of_int (((i * 7) + (j * 3) + l) mod 17)))
      in
      let elements = h * w * c in
      let extended = e * (h + k - 1) * (w + k - 1) * c in
      let bound =
        extended + (e * elements) + (2 * ((2 * e * elements) + (8 * elements)))
      in
      equal bool true
        (within bound (fun () -> V.Filter.median ~edge:`Replicate (k, k) x)))

let median_opencv =
  test "median agrees with OpenCV's on uint8" (fun () ->
      let x =
        match Nx_io.Archive.find "u8" (Lazy.force golden) with
        | Some p -> Nx.unpack Nx.uint8 p
        | None -> failf "golden/filter.npz has no u8"
      in
      List.iter
        (fun k ->
          let want =
            match
              Nx_io.Archive.find
                (Printf.sprintf "median.opencv.%d" k)
                (Lazy.force golden)
            with
            | Some p -> Nx.unpack Nx.uint8 p
            | None -> failf "no median.opencv.%d" k
          in
          equal (array int)
            (Array.map Int.of_float (floats want))
            (Array.map Int.of_float
               (floats (V.Filter.median ~edge:`Replicate (k, k) x))))
        [ 3; 5 ])

let refusals =
  let k = Nx.ones Nx.float32 [| 3; 3 |]
  and x = Nx.zeros Nx.float32 [| 4; 4; 1 |] in
  cases "misuse raises"
    ~id:(fun (n, _) -> n)
    [
      ( "two axes",
        fun () ->
          V.Filter.correlate ~edge:`Zero k (Nx.zeros Nx.float32 [| 4; 4 |]) );
      ( "weights of one axis",
        fun () -> V.Filter.correlate ~edge:`Zero (Nx.ones Nx.float32 [| 3 |]) x
      );
      ( "an empty extent",
        fun () ->
          V.Filter.correlate ~edge:`Zero (Nx.ones Nx.float32 [| 0; 3 |]) x );
      ( "batches that do not broadcast",
        fun () ->
          V.Filter.correlate ~edge:`Zero
            (Nx.ones Nx.float32 [| 2; 3; 3 |])
            (Nx.zeros Nx.float32 [| 3; 4; 4; 1 |]) );
      ( "no row past the edge",
        fun () ->
          V.Filter.correlate ~edge:`Antisymmetric_whole k
            (Nx.zeros Nx.float32 [| 0; 4; 1 |]) );
      ("windows of no row", fun () -> V.Filter.windows ~edge:`Zero (0, 2) x);
      ( "windows of two axes",
        fun () ->
          V.Filter.windows ~edge:`Zero (2, 2) (Nx.zeros Nx.float32 [| 4; 4 |])
      );
      ( "windows past no column",
        fun () ->
          V.Filter.windows ~edge:`Periodic (2, 2)
            (Nx.zeros Nx.float32 [| 4; 0; 1 |]) );
      ( "a negative radius",
        fun () ->
          V.Filter.gaussian ~edge:`Zero ~radius:(-1) (Nx.scalar Nx.float32 1.) x
      );
      ("a median of no column", fun () -> V.Filter.median ~edge:`Zero (3, 0) x);
    ]
    (fun (_, f) ->
      raises_match (Exn.invalid_arg ~sub:"Mynd.") (fun () -> ignore (f ())))

let () =
  exit
    (run "mynd.filter"
       [
         windows_reads;
         windows_derivative;
         correlate_naive;
         correlate_empty;
         batch_law;
         separable;
         derivatives;
         ramp;
         nan_reach;
         compiled;
         correlate_libraries;
         constant;
         truncation;
         tiny_sigma;
         nan_sigma;
         narrow;
         sigma_batch;
         sigma_derivative;
         gaussian_compiled;
         gaussian_libraries;
         median_sort;
         median_tall;
         median_nan;
         median_derivative;
         median_compiled;
         median_libraries;
         median_opencv;
         median_memory;
         batches;
         refusals;
       ])
