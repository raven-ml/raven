(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

let archive file = (file, lazy (Nx_io.load_npz file))

(* pycocotools' mask IoU, from golden/annot.npz
   (contrib/mynd/test/gen/annot.py); scipy's connected-component labels, from
   golden/select.npz (contrib/mynd/test/gen/select.py) *)
let annot = archive "golden/annot.npz"
let select = archive "golden/select.npz"

let entry (file, npz) dt key =
  match Nx_io.Archive.find key (Lazy.force npz) with
  | Some p -> Nx.unpack dt p
  | None -> failf "%s has no %s" file key

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

let close tol want got =
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal
    (array (if tol = 0. then float_exact else float tol))
    (floats want) (floats got)

let longs x = Nx.to_array (Nx.cast Nx.int64 x)

(* Mask IoU *)

let mask_iou =
  test "mask IoU is pycocotools'" (fun () ->
      let m =
        Nx.unsqueeze ~axes:[ -1 ] (Nx.cast Nx.bool (entry annot Nx.uint8 "m"))
      in
      close 1e-15 (entry annot Nx.float64 "iou") (V.Mask.iou Nx.float64 m m))

(* The quotient of the exact counts, rounded once into the dtype, where counts
   past float16's range or bfloat16's precision still give it: a full 300 × 300
   mask with itself, and with its first 100 rows, a third. *)
let mask_iou_narrow =
  test "mask IoU rounds the exact quotient once" (fun () ->
      let full = Nx.ones Nx.bool [| 1; 300; 300; 1 |] in
      let part =
        Nx.pad
          [| (0, 0); (0, 200); (0, 0); (0, 0) |]
          false
          (Nx.ones Nx.bool [| 1; 100; 300; 1 |])
      in
      let both = Nx.concatenate ~axis:0 [ full; part ] in
      let third = 30000. /. 90000. in
      equal (array float_exact)
        [| 1.; Nx.Dtype.of_float Nx.float16 third |]
        (floats (Nx.slice [ Nx.I 0 ] (V.Mask.iou Nx.float16 full both)));
      equal (array float_exact)
        [| 1.; Nx.Dtype.of_float Nx.bfloat16 third |]
        (floats (Nx.slice [ Nx.I 0 ] (V.Mask.iou Nx.bfloat16 full both))))

(* Components *)

let golden_components =
  cases "components are scipy's labels" ~id:fst
    [ ("four", `Four); ("eight", `Eight) ]
    (fun (name, conn) ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry select Nx.bool "cc.m") in
      let want = entry select Nx.int64 ("cc." ^ name) in
      equal (array int64) (longs want) (longs (V.Mask.components conn m)))

(* [reference conn h w px] labels by a sequential union-find, each component by
   its least row-major index. *)
let reference conn h w px =
  let parent = Array.init (h * w) Fun.id in
  let rec find i = if parent.(i) = i then i else find parent.(i) in
  let union a b =
    let a = find a and b = find b in
    if a < b then parent.(b) <- a else if b < a then parent.(a) <- b
  in
  let offsets =
    match conn with
    | `Four -> [ (0, 1); (1, 0) ]
    | `Eight -> [ (0, 1); (1, -1); (1, 0); (1, 1) ]
  in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      if px.((i * w) + j) then
        List.iter
          (fun (di, dj) ->
            let i' = i + di and j' = j + dj in
            if i' < h && j' >= 0 && j' < w && px.((i' * w) + j') then
              union ((i * w) + j) ((i' * w) + j'))
          offsets
    done
  done;
  Array.init (h * w) (fun k -> if px.(k) then Int64.of_int (find k) else -1L)

let gen_mask =
  Gen.(
    let* h, (w, d) =
      pair (int_range 0 14) (pair (int_range 0 14) (float_range 0.2 0.8))
    in
    let+ px =
      array ~size:(constant (h * w)) (map (fun f -> f < d) (float_range 0. 1.))
    in
    (h, w, px))

let union_find =
  prop "components equal a sequential union-find" ~count:200
    Gen.(pair bool gen_mask)
    (fun (eight, (h, w, px)) ->
      let conn = if eight then `Eight else `Four in
      cover "empty" (h * w = 0);
      let m = Nx.create Nx.bool [| h; w; 1 |] px in
      equal (array int64) (reference conn h w px)
        (longs (V.Mask.components conn m)))

(* A serpentine of one-pixel rows joined at alternate ends: one component whose
   least index is far along its path. *)
let serpentine n =
  Nx.init Nx.bool [| n; n; 1 |] (fun i ->
      let r = i.(0) and c = i.(1) in
      r mod 2 = 0 || c = if r / 2 mod 2 = 0 then n - 1 else 0)

let bound n =
  6 + Float.to_int (Float.floor (3. *. Float.log2 (1.5 *. Float.of_int n)))

let rounds =
  cases "rounds stay under Liu and Tarjan's bound" ~id:fst
    [
      ("serpentine 64", serpentine 64);
      ("serpentine 64, flipped", Nx.flip ~axes:[ 0; 1 ] (serpentine 64));
      ( "noise",
        Nx.unsqueeze ~axes:[ -1 ]
          (Nx.slice [ Nx.I 0 ] (entry select Nx.bool "cc.m")) );
    ]
    (fun (_, m) ->
      let h = Nx.dim 0 m and w = Nx.dim 1 m in
      List.iter
        (fun conn ->
          let labels, k = V.Mask.components_rounds conn m in
          let px = Nx.to_array (Nx.reshape [| h * w |] m) in
          equal (array int64) (reference conn h w px) (longs labels);
          less int ~than:(bound (h * w) + 1) (Int32.to_int (Nx.item [] k)))
        [ `Four; `Eight ])

(* A batch padded with false keeps its components; a label, the least index,
   counts in the padded width. *)
let padded =
  prop "padding keeps components, relabelled by the padded width"
    Gen.(triple bool gen_mask (pair (int_range 0 3) (int_range 0 3)))
    (fun (eight, (h, w, px), (ph, pw)) ->
      let conn = if eight then `Eight else `Four in
      let m = Nx.create Nx.bool [| h; w; 1 |] px in
      let mp = Nx.pad [| (0, ph); (0, pw); (0, 0) |] false m in
      let w' = w + pw in
      let moved l =
        if l < 0L then l
        else Int64.(add (mul (div l (of_int w)) (of_int w')) (rem l (of_int w)))
      in
      let want = Array.map moved (longs (V.Mask.components conn m)) in
      let got =
        longs
          (Nx.slice [ Nx.R (0, h); Nx.R (0, w) ] (V.Mask.components conn mp))
      in
      equal (array int64) want got)

let batch_and_vmap =
  test "each mask of a batch is labelled alone, under vmap too" (fun () ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry select Nx.bool "cc.m") in
      let whole = longs (V.Mask.components `Eight m) in
      let one i = longs (V.Mask.components `Eight (Nx.slice [ Nx.I i ] m)) in
      equal (array int64) (Array.append (one 0) (one 1)) whole;
      equal (array int64) whole
        (longs (Rune.vmap' (V.Mask.components `Eight) m)))

let compiled =
  test "compiled is eager" (fun () ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry select Nx.bool "cc.m") in
      equal (array int64)
        (longs (V.Mask.components `Four m))
        (longs (Rune.jit' (V.Mask.components `Four) m)))

(* Areas per mask of a batch: ones scattered at each mask's own flattened
   labels, read back at them. *)
let areas =
  test "areas compose from the labels, per mask" (fun () ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry select Nx.bool "cc.m") in
      let b = Nx.dim 0 m and n = Nx.dim 1 m * Nx.dim 2 m in
      let labels = Nx.reshape [| b; n |] (V.Mask.components `Four m) in
      let ones = Nx.ones Nx.int64 [| b; n |] in
      let counts =
        Nx.scatter ~combine:Add ~axis:(-1) labels ones
          (Nx.zeros Nx.int64 [| b; n |])
      in
      let areas = Nx.take_along_axis ~axis:(-1) labels counts in
      let l = longs labels in
      let count k x =
        let a = ref 0 in
        for i = k * n to ((k + 1) * n) - 1 do
          if l.(i) = x then incr a
        done;
        !a
      in
      equal (array int64)
        (Array.mapi
           (fun i x -> if x < 0L then 0L else Int64.of_int (count (i / n) x))
           l)
        (longs areas))

(* Reconstruction *)

(* [grown conn h w m marker] is [marker] within [m] grown through next pixels of
   [m] until nothing changes, one mask. *)
let grown conn h w m marker =
  let next =
    match conn with
    | `Four -> [ (-1, 0); (0, -1); (0, 1); (1, 0) ]
    | `Eight ->
        [ (-1, -1); (-1, 0); (-1, 1); (0, -1); (0, 1); (1, -1); (1, 0); (1, 1) ]
  in
  let r = Array.map2 ( && ) marker m in
  let changed = ref true in
  while !changed do
    changed := false;
    for i = 0 to h - 1 do
      for j = 0 to w - 1 do
        let k = (i * w) + j in
        if m.(k) && not r.(k) then
          if
            List.exists
              (fun (di, dj) ->
                let i' = i + di and j' = j + dj in
                i' >= 0 && i' < h && j' >= 0 && j' < w && r.((i' * w) + j'))
              next
          then begin
            r.(k) <- true;
            changed := true
          end
      done
    done
  done;
  r

let gen_batch =
  Gen.(
    let* b, (h, w) =
      pair (int_range 1 3) (pair (int_range 0 9) (int_range 0 9))
    in
    let* d, e = pair (float_range 0.3 0.8) (int_range 0 4) in
    let n = b * h * w in
    let+ px = array ~size:(constant n) (float_range 0. 1.)
    and+ mk = array ~size:(constant n) (float_range 0. 1.) in
    (* e = 0: no marker; 1: every pixel; else a sparse one. *)
    let marker =
      Array.map
        (fun v -> match e with 0 -> false | 1 -> true | _ -> v < 0.08)
        mk
    in
    (b, h, w, Array.map (fun v -> v < d) px, marker))

let reconstruct_fixed_point =
  prop "reconstruct is the marker grown within the mask, per mask" ~count:300
    Gen.(pair bool gen_batch)
    (fun (eight, (b, h, w, m, marker)) ->
      let conn = if eight then `Eight else `Four in
      let n = h * w in
      cover "no marker" (not (Array.exists Fun.id marker));
      cover "every pixel a marker" (n > 0 && Array.for_all Fun.id marker);
      cover "a marker outside the mask"
        (Array.exists2 (fun a k -> k && not a) m marker);
      cover "several masks" (b > 1 && n > 0);
      let want =
        Array.concat
          (List.init b (fun k ->
               grown conn h w
                 (Array.sub m (k * n) n)
                 (Array.sub marker (k * n) n)))
      in
      let t a = Nx.create Nx.bool [| b; h; w; 1 |] a in
      let got = V.Mask.reconstruct conn ~marker:(t marker) (t m) in
      equal (array bool) want (Nx.to_array got))

let reconstruct_padded =
  prop "padding with false keeps the reconstruction" ~count:100
    Gen.(triple bool gen_mask (pair (int_range 0 3) (int_range 0 3)))
    (fun (eight, (h, w, px), (ph, pw)) ->
      let conn = if eight then `Eight else `Four in
      let m = Nx.create Nx.bool [| h; w; 1 |] px in
      let marker =
        Nx.init Nx.bool [| h; w; 1 |] (fun i -> (i.(0) + i.(1)) mod 5 = 0)
      in
      let pad x = Nx.pad [| (0, ph); (0, pw); (0, 0) |] false x in
      equal (array bool)
        (Nx.to_array (pad (V.Mask.reconstruct conn ~marker m)))
        (Nx.to_array (V.Mask.reconstruct conn ~marker:(pad marker) (pad m))))

let reconstruct_compiled =
  test "a compiled and a mapped reconstruct are eager's" (fun () ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry select Nx.bool "cc.m") in
      let marker =
        Nx.init Nx.bool (Nx.shape m) (fun i -> ((i.(1) * 7) + i.(2)) mod 11 = 3)
      in
      let f marker m = V.Mask.reconstruct `Eight ~marker m in
      let want = Nx.to_array (f marker m) in
      let s = Nx.Ptree.(tensor @-> tensor @-> returns tensor) in
      equal (array bool) want (Nx.to_array (Rune.jit s f marker m));
      (* Each lane is a mask and its marker, stacked. *)
      let both = Nx.stack ~axis:1 [ marker; m ] in
      let lane x = f (Nx.slice [ Nx.I 0 ] x) (Nx.slice [ Nx.I 1 ] x) in
      equal (array bool) want (Nx.to_array (Rune.vmap' lane both)))

(* Holes are the background components that reach no border pixel: those
   [reconstruct] does not grow from the frame. scipy's binary_fill_holes, from
   golden/morphology.npz (contrib/mynd/test/gen/morphology.py), takes the
   background's connectivity as its structure. The frame comes from a predicate
   on the index, which holds for masks of one row or column too. *)
let morphology = lazy (Nx_io.load_npz "golden/morphology.npz")

let holes =
  cases "filling holes by reconstruction is scipy's" ~id:fst
    [ ("four", `Four); ("eight", `Eight) ]
    (fun (name, conn) ->
      let get key =
        match Nx_io.Archive.find key (Lazy.force morphology) with
        | Some p -> Nx.unsqueeze ~axes:[ -1 ] (Nx.unpack Nx.bool p)
        | None -> failf "golden/morphology.npz has no %s" key
      in
      let m = get "masks" in
      let h = Nx.dim 1 m and w = Nx.dim 2 m in
      let frame =
        Nx.init Nx.bool (Nx.shape m) (fun i ->
            i.(1) = 0 || i.(1) = h - 1 || i.(2) = 0 || i.(2) = w - 1)
      in
      let filled =
        Nx.logical_not
          (V.Mask.reconstruct conn ~marker:frame (Nx.logical_not m))
      in
      equal (array bool)
        (Nx.to_array (get ("holes." ^ name)))
        (Nx.to_array filled))

(* Distances *)

(* [nearest h w px] is, at each pixel of one mask, the least squared distance
   [n] to a true pixel ([-1] where there is none), the least row-major index
   of a true pixel at it ([-1] for none), and whether two true pixels tie at
   it, by a brute force over the true pixels in integers. *)
let nearest h w px =
  let pts = List.filter (fun k -> px.(k)) (List.init (h * w) Fun.id) in
  let n = Array.make (h * w) (-1) and idx = Array.make (h * w) (-1) in
  let tie = ref false in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      let p = (i * w) + j in
      List.iter
        (fun k ->
          let di = i - (k / w) and dj = j - (k mod w) in
          let q = (di * di) + (dj * dj) in
          if n.(p) < 0 || q < n.(p) then (
            n.(p) <- q;
            idx.(p) <- k)
          else if q = n.(p) then tie := true)
        pts
    done
  done;
  (n, idx, !tie)

(* The float64 distance of a squared distance: the root of the exact integer,
   [infinity] for none. *)
let root64 q = if q < 0 then Float.infinity else Float.sqrt (Float.of_int q)

(* [each b h w px f] applies [nearest] to each of the [b] masks of [px]. *)
let each b h w px f =
  let n = h * w in
  Array.concat (List.init b (fun k -> f (nearest h w (Array.sub px (k * n) n))))

type fill = Empty | Full | Corner | Lattice of int | Random of float

let pixel fill h w k u =
  let i = k / w mod h and j = k mod w in
  match fill with
  | Empty -> false
  | Full -> true
  | Corner -> i = 0 && j = 0
  | Lattice s -> i mod s = 0 && j mod s = 0
  | Random d -> u < d

(* Masks [batch @ [h; w]]: sides of 0 to 64, or a column of 1000 to 1100
   rows; empty, full, one corner pixel, lattices whose pixels tie, or drawn
   at a density from 0.005 to 0.5. *)
let gen_masks =
  Gen.(
    let side = of_list [ 0; 1; 2; 3; 7; 13; 31; 64 ] in
    let* batch = of_list [ [||]; [| 2 |]; [| 2; 3 |] ] in
    let* h, w =
      frequency
        [
          (8, pair side side);
          (2, map (fun n -> (n, 1)) (int_range 1000 1100));
        ]
    in
    let* fill =
      frequency
        [
          (1, constant Empty);
          (1, constant Full);
          (1, constant Corner);
          (1, map (fun s -> Lattice s) (int_range 2 3));
          (4, map (fun d -> Random d) (float_range 0.005 0.5));
        ]
    in
    let n = Array.fold_left ( * ) 1 batch * h * w in
    let+ u = array ~size:(constant n) (float_range 0. 1.) in
    (batch, h, w, Array.mapi (fun k u -> pixel fill h w k u) u))

let mask_of batch h w px =
  Nx.create Nx.bool (Array.append batch [| h; w; 1 |]) px

let oracle =
  prop "distances are the brute force's roots of exact integers" gen_masks
    (fun (batch, h, w, px) ->
      let b = Array.fold_left ( * ) 1 batch in
      let any = Array.exists Fun.id px and n = h * w in
      cover "no row or no column" (n = 0);
      cover "an empty mask" (n > 0 && not any);
      cover "a full mask" (n > 0 && Array.for_all Fun.id px);
      cover "a column of 1000 rows" (h >= 1000);
      let tie k =
        let _, _, t = nearest h w (Array.sub px (k * n) n) in
        t
      in
      cover "a tie" (n > 0 && List.exists tie (List.init b Fun.id));
      let m = mask_of batch h w px in
      equal (array float_exact)
        (each b h w px (fun (q, _, _) -> Array.map root64 q))
        (floats (V.Mask.distance Nx.float64 m)))

(* A float format. *)
type format = F : (float, 's) Nx.dtype -> format

let distance_values =
  let line n ks =
    Nx.init Nx.bool [| 1; n; 1 |] (fun i -> List.mem i.(1) ks)
  in
  cases "distances the docs state" ~id:(fun (name, _, _, _) -> name)
    [
      ("a true pixel", F Nx.float32, Nx.ones Nx.bool [| 1; 1; 1 |],
       [| 0. |]);
      ("an empty pixel", F Nx.float32, Nx.zeros Nx.bool [| 1; 1; 1 |],
       [| Float.infinity |]);
      ("a full mask", F Nx.float32, Nx.ones Nx.bool [| 2; 3; 1 |],
       Array.make 6 0.);
      ("an empty mask, float32", F Nx.float32, Nx.zeros Nx.bool [| 4; 5; 1 |],
       Array.make 20 Float.infinity);
      ("an empty mask, float8_e4m3fn", F Nx.float8_e4m3fn,
       Nx.zeros Nx.bool [| 4; 5; 1 |], Array.make 20 448.);
      ("an empty mask, float4_e2m1fn", F Nx.float4_e2m1fn,
       Nx.zeros Nx.bool [| 4; 5; 1 |], Array.make 20 6.);
      ("a row from its first pixel, float32", F Nx.float32, line 4098 [ 0 ],
       Array.init 4098 Float.of_int);
      ("a row from its first pixel, float64", F Nx.float64, line 4098 [ 0 ],
       Array.init 4098 Float.of_int);
      ("a row between two pixels", F Nx.float64, line 7 [ 1; 5 ],
       [| 1.; 0.; 1.; 2.; 1.; 0.; 1. |]);
    ]
    (fun (_, F dt, m, want) ->
      equal (array float_exact) want (floats (V.Mask.distance dt m)))

(* The narrow formats: every distance, the farthest past float16's and
   float8_e4m3fn's greatest values included, is the float64 one cast. *)
let narrow_formats =
  let formats =
    [
      F Nx.float32; F Nx.float16; F Nx.bfloat16; F Nx.float8_e5m2;
      F Nx.float8_e4m3fn; F Nx.float4_e2m1fn;
    ]
  in
  prop "a narrow format's distances are float64's rounded once"
    Gen.(
      frequency
        [
          (2, map (fun n -> (1, n)) (int_range 1 70000));
          (1, map (fun n -> (1, n)) (int_range 65600 70000));
          (2, pair (int_range 1 64) (int_range 1 64));
        ])
    (fun (h, w) ->
      cover "past float16's greatest value" (w > 65520);
      cover "past float8_e4m3fn's greatest value" (w > 449);
      (* One true pixel at the start of a row, or a sparse lattice. *)
      let m =
        Nx.init Nx.bool [| h; w; 1 |] (fun i ->
            if h = 1 then i.(1) = 0 else i.(0) mod 11 = 0 && i.(1) mod 17 = 0)
      in
      let wide = V.Mask.distance Nx.float64 m in
      List.iter
        (fun (F dt) ->
          equal (array float_exact)
            (floats (Nx.cast dt wide))
            (floats (V.Mask.distance dt m)))
        formats)

(* A rearrangement of an image's pixels. *)
type move = { t : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t }

let distance_symmetric =
  prop "distances commute with flips and the transpose" gen_mask
    (fun (h, w, px) ->
      let m = Nx.create Nx.bool [| h; w; 1 |] px in
      let d = V.Mask.distance Nx.float64 in
      List.iter
        (fun { t } ->
          equal (array float_exact) (floats (t (d m))) (floats (d (t m))))
        [
          { t = (fun x -> Nx.flip ~axes:[ 0 ] x) };
          { t = (fun x -> Nx.flip ~axes:[ 1 ] x) };
          { t = (fun x -> Nx.copy (Nx.transpose ~axes:[ 1; 0; 2 ] x)) };
        ])

let distance_monotone =
  prop "more true pixels never lengthen a distance"
    Gen.(pair gen_mask (float_range 0. 1.))
    (fun ((h, w, px), seed) ->
      let more =
        Array.mapi
          (fun k v -> v || Float.of_int ((k * 7919) mod 101) /. 101. < seed)
          px
      in
      let d a =
        V.Mask.distance Nx.float64 (Nx.create Nx.bool [| h; w; 1 |] a)
      in
      let a = d px and b = d more in
      equal (array float_exact) (floats (Nx.minimum a b)) (floats b))

(* Erosion by the disk of radius [r] keeps the pixels farther than [r] from
   every false pixel, dilation those within [r] of a true one. *)
let distance_morphology =
  let disk r =
    let k = (2 * r) + 1 in
    Nx.init Nx.bool [| k; k |] (fun i ->
        let dy = i.(0) - r and dx = i.(1) - r in
        (dy * dy) + (dx * dx) <= r * r)
  in
  prop "thresholds of distances are erosions and dilations by disks"
    Gen.(pair gen_mask (int_range 0 4))
    (fun ((h, w, px), r) ->
      let m = Nx.create Nx.bool [| h; w; 1 |] px in
      let d m = V.Mask.distance Nx.float64 m in
      let r' = Float.of_int r in
      equal (array bool)
        (Nx.to_array (V.Filter.erode (disk r) m))
        (Nx.to_array (Nx.greater_s (d (Nx.logical_not m)) r'));
      equal (array bool)
        (Nx.to_array (V.Filter.dilate (disk r) m))
        (Nx.to_array (Nx.less_equal_s (d m) r')))

let distance_batch =
  prop "each mask of a batch, mapped or padded, has its own distances"
    ~count:50
    Gen.(pair gen_batch (pair (int_range 0 3) (int_range 0 3)))
    (fun ((b, h, w, px, _), (ph, pw)) ->
      let m = Nx.create Nx.bool [| b; h; w; 1 |] px in
      let d = V.Mask.distance Nx.float64 in
      let whole = floats (d m) in
      let one k = floats (d (Nx.slice [ Nx.I k ] m)) in
      equal (array float_exact) (Array.concat (List.init b one)) whole;
      equal (array float_exact) whole (floats (Rune.vmap' d m));
      let mp = Nx.pad [| (0, 0); (0, ph); (0, pw); (0, 0) |] false m in
      equal (array float_exact) whole
        (floats (Nx.slice [ Nx.A; Nx.R (0, h); Nx.R (0, w) ] (d mp))))

(* Masks for the compiled calls: a batch of drawn ones and a column of 1100
   rows, whose columns take 11 rounds. *)
let compiled_masks () =
  let st = Random.State.make [| 7 |] in
  [
    Nx.init Nx.bool [| 3; 31; 13; 1 |] (fun _ ->
        Random.State.float st 1. < 0.1);
    Nx.init Nx.bool [| 2; 1100; 1; 1 |] (fun i -> (i.(0) * 397) + i.(1) = 900);
  ]

let distance_compiled =
  test "a compiled and a compiled mapped distance are eager's" (fun () ->
      List.iter
        (fun m ->
          let d = V.Mask.distance Nx.float64 in
          let want = floats (d m) in
          equal (array float_exact) want (floats (Rune.jit' d m));
          equal (array float_exact) want (floats (Rune.jit' (Rune.vmap' d) m)))
        (compiled_masks ()))

(* The first NVIDIA GPU that opens, through its runtime or its kernel driver.
   No other GPU is opened: Metal has no float64. *)
let nvidia =
  let first (p : Rig_gpus.path) =
    match p.gpus () with
    | Ok (g :: _) -> Result.to_option (g.open_ ())
    | Ok [] | Error _ -> None
  in
  lazy
    (Rig_gpus.paths ()
    |> List.filter (fun (p : Rig_gpus.path) -> p.kind = "CUDA" || p.kind = "NV")
    |> List.find_map first
    |> Option.map (fun d -> Nx.devices [ d ]))

let distance_gpu =
  test "on an NVIDIA GPU, compiled distances are the host's" (fun () ->
      match Lazy.force nvidia with
      | None -> skip ~reason:"no NVIDIA GPU" ()
      | Some (module S) ->
          List.iter
            (fun m ->
              let d m = V.Mask.distance Nx.float64 m in
              let want = floats (d m) in
              let m' = Nx.place S.on m in
              equal (array float_exact) want (floats (Rune.jit' d m'));
              equal (array float_exact) want
                (floats (Rune.jit' (Rune.vmap' d) m')))
            (compiled_masks ()))

(* scipy's distance_transform_edt of the complement, from golden/distance.npz
   (contrib/mynd/test/gen/distance.py). *)
let distance = archive "golden/distance.npz"
let distance_shapes = [ "9x13"; "32x32"; "1x40"; "40x1"; "64x17" ]

let distance_scipy =
  cases "distances are scipy's" ~id:Fun.id distance_shapes (fun key ->
      let m = Nx.unsqueeze ~axes:[ -1 ] (entry distance Nx.bool (key ^ ".m")) in
      equal (array float_exact)
        (floats (entry distance Nx.float64 (key ^ ".d")))
        (floats (V.Mask.distance Nx.float64 m)))

let refusals =
  group "misuse"
    [
      test "a mask without its channel axis" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Mask.components") (fun () ->
              V.Mask.components `Four (Nx.zeros Nx.bool [| 4; 4 |])));
      test "a marker of another shape" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Mask.reconstruct") (fun () ->
              V.Mask.reconstruct `Four
                ~marker:(Nx.zeros Nx.bool [| 4; 5; 1 |])
                (Nx.zeros Nx.bool [| 4; 4; 1 |])));
      test "a distance to a mask without its channel axis" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Mask.distance") (fun () ->
              V.Mask.distance Nx.float32 (Nx.zeros Nx.bool [| 4; 4 |])));
      test "a distance to masks of 2²⁹ rows or columns" (fun () ->
          List.iter
            (fun s ->
              raises_match (Exn.invalid_arg ~sub:"Mynd.Mask.distance")
                (fun () -> V.Mask.distance Nx.float32 (Nx.zeros Nx.bool s)))
            [ [| 0; 1 lsl 29; 1; 1 |]; [| 0; 1; 1 lsl 29; 1 |] ]);
      test "a distance to masks of two channels" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Mask.distance") (fun () ->
              V.Mask.distance Nx.float32 (Nx.zeros Nx.bool [| 4; 4; 2 |])));
      test "masks of two channels" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Mask.reconstruct") (fun () ->
              let m = Nx.zeros Nx.bool [| 4; 4; 2 |] in
              V.Mask.reconstruct `Four ~marker:m m));
    ]

let () =
  exit
    (run "mynd.mask"
       [
         mask_iou;
         mask_iou_narrow;
         golden_components;
         union_find;
         rounds;
         padded;
         batch_and_vmap;
         compiled;
         areas;
         reconstruct_fixed_point;
         reconstruct_padded;
         reconstruct_compiled;
         holes;
         oracle;
         distance_values;
         narrow_formats;
         distance_symmetric;
         distance_monotone;
         distance_morphology;
         distance_batch;
         distance_compiled;
         distance_gpu;
         distance_scipy;
         refusals;
       ])
