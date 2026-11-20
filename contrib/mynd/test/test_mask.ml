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
         refusals;
       ])
