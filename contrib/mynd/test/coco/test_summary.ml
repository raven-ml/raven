(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Mynd_coco
open Support

let fields s =
  [
    s.ap;
    s.ap50;
    s.ap75;
    s.ap_small;
    s.ap_medium;
    s.ap_large;
    s.ar1;
    s.ar10;
    s.ar100;
    s.ar_small;
    s.ar_medium;
    s.ar_large;
  ]

(* math.fsum over the defined entries of pycocotools' tables, over their
   count. *)
let fixture_means =
  cases "the means of val2017's tables are exactly rounded" ~id:fst
    [ ("bbox", `Boxes); ("segm", `Masks) ]
    (fun (kind, which) ->
      let d = Lazy.force fixture in
      let e =
        match which with
        | `Boxes -> evaluate_boxes d (Lazy.force box_dets)
        | `Masks -> evaluate_masks d (Lazy.force mask_dets)
      in
      equal (list float_exact)
        (Array.to_list (floats "eval24.npz" (kind ^ ".means")))
        (fields (summary e)))

(* Skeptic probe s3's dataset: eight categories of five annotations, one
   detection each, whose AR@100 entries' exact mean lies 9.7e-18 above
   0.3125. pycocotools prints 0.313; the exactly rounded mean is 0.3125. *)
let tie =
  test "a mean near a half rounds once" (fun () ->
      let dx = floats "tie.npz" "dx" in
      let images =
        List.init 40 (fun i -> image ~height:200 ~width:300 (i + 1))
      in
      let anns =
        List.init 40 (fun i ->
            ann ~id:(i + 1) ~image:(i + 1)
              ~category:((i / 5) + 1)
              (box 0. 0. 100. 100.))
      in
      let d =
        dataset ~images
          ~categories:(List.init 8 (fun k -> category (k + 1)))
          anns
      in
      let dets =
        Array.init 40 (fun i ->
            det ~image:(i + 1)
              ~category:((i / 5) + 1)
              0.9
              (box dx.(i) 0. 100. 100.))
      in
      let e = evaluate_boxes d dets in
      equal (array float_exact)
        (of_coco (floats "tie.npz" "recall"))
        (Nx.to_array e.recall);
      equal float_exact (floats "tie.npz" "ar100").(0) (summary e).ar100;
      equal float_exact 0.3125 (summary e).ar100)

(* An exact oracle for values on the grid k 2⁻⁵², k in [0, 2⁵²]: their sum
   in two limbs of integers, rounded once. *)
let grid = 0x1p-52

let exact_mean vs =
  let n = List.length vs in
  if n = 0 then Float.nan
  else begin
    let hi = ref 0 and lo = ref 0 in
    List.iter
      (fun v ->
        let k = Float.to_int (v /. grid) in
        hi := !hi + (k lsr 26);
        lo := !lo + (k land ((1 lsl 26) - 1)))
      vs;
    hi := !hi + (!lo lsr 26);
    lo := !lo land ((1 lsl 26) - 1);
    let sum = (Float.of_int !hi *. 0x1p26) +. Float.of_int !lo in
    sum *. grid /. Float.of_int n
  end

let evaluation nk p r =
  {
    categories = Array.init nk category;
    precision = Nx.create Nx.float64 [| 10; 101; nk; 4; 3 |] p;
    recall = Nx.create Nx.float64 [| 10; nk; 4; 3 |] r;
  }

(* The entries each field selects, by its doc. *)
let selected nk p r =
  let pr ts a =
    List.concat_map
      (fun t ->
        List.concat_map
          (fun ri ->
            List.init nk (fun k ->
                p.((((((((t * 101) + ri) * nk) + k) * 4) + a) * 3) + 2)))
          (List.init 101 Fun.id))
      ts
  in
  let rc a m =
    List.concat_map
      (fun t ->
        List.init nk (fun k -> r.((((((t * nk) + k) * 4) + a) * 3) + m)))
      (List.init 10 Fun.id)
  in
  let every = List.init 10 Fun.id in
  [
    pr every 0;
    pr [ 0 ] 0;
    pr [ 5 ] 0;
    pr every 1;
    pr every 2;
    pr every 3;
    rc 0 0;
    rc 0 1;
    rc 0 2;
    rc 1 2;
    rc 2 2;
    rc 3 2;
  ]

let gen_entry =
  Gen.(
    frequency
      [
        (1, constant Float.nan);
        (3, map (fun k -> Float.of_int k *. grid) (int_range 0 (1 lsl 52)));
      ])

let means =
  prop "each field is the exactly rounded mean of its defined entries"
    Gen.(
      let* nk = int_range 1 3 in
      let+ p = array ~size:(constant (10 * 101 * nk * 12)) gen_entry
      and+ r = array ~size:(constant (10 * nk * 12)) gen_entry in
      (nk, p, r))
    (fun (nk, p, r) ->
      let want =
        List.map
          (fun vs ->
            exact_mean (List.filter (fun v -> not (Float.is_nan v)) vs))
          (selected nk p r)
      in
      equal (list float_exact) want (fields (summary (evaluation nk p r))))

(* Lengths around the sum's block sizes, up to every precision entry of 80
   categories. *)
let lengths =
  cases "the mean of n defined entries among NaNs" ~id:string_of_int
    [ 0; 1; 7; 8; 128; 129; 8192; 8193; 80800 ] (fun n ->
      let nk = 80 in
      let slice = 10 * 101 * nk in
      let pick = Array.init slice Fun.id in
      for i = slice - 1 downto 1 do
        let j = Random.int (i + 1) in
        let t = pick.(i) in
        pick.(i) <- pick.(j);
        pick.(j) <- t
      done;
      let p = Array.make (slice * 12) Float.nan in
      let vs =
        List.init n (fun i ->
            let v = Float.of_int (Random.full_int ((1 lsl 52) + 1)) *. grid in
            let q = pick.(i) in
            (* Position [q] of the slice of all IoU thresholds, all areas and
               limit 100. *)
            p.((q * 12) + 2) <- v;
            v)
      in
      let r = Array.make (10 * nk * 12) Float.nan in
      equal float_exact (exact_mean vs) (summary (evaluation nk p r)).ap)

let order =
  prop "the order of the entries does not change a mean"
    Gen.(
      let* vs = list ~size:(int_range 0 300) (float_range 0. 1.) in
      let+ ps = permutation vs in
      (vs, ps))
    (fun (vs, ps) ->
      cover "many entries" (List.length vs > 100);
      let ar vs =
        let r = Array.make (10 * 30 * 12) Float.nan in
        List.iteri (fun i v -> r.((i * 12) + 2) <- v) vs;
        (summary (evaluation 30 (Array.make (10 * 101 * 30 * 12) Float.nan) r))
          .ar100
      in
      equal float_exact (ar vs) (ar ps))

let refusals =
  let raises e =
    raises_match (Exn.invalid_arg ~sub:"Mynd_coco.summary") (fun () ->
        ignore (summary e))
  in
  let p k = Array.make (10 * 101 * k * 12) 0.5
  and r k = Array.make (10 * k * 12) 0.5 in
  group "misuse"
    [
      test "fewer categories than the tables hold" (fun () ->
          raises
            { (evaluation 2 (p 2) (r 2)) with categories = [| category 1 |] });
      test "more categories than the tables hold" (fun () ->
          raises
            {
              (evaluation 1 (p 1) (r 1)) with
              categories = Array.init 3 category;
            });
      test "tables of different categories" (fun () ->
          raises
            {
              (evaluation 2 (p 2) (r 2)) with
              recall = Nx.create Nx.float64 [| 10; 1; 4; 3 |] (r 1);
            });
      test "a precision table of other axes" (fun () ->
          raises
            {
              (evaluation 1 (p 1) (r 1)) with
              precision =
                Nx.create Nx.float64 [| 10; 100; 1; 4; 3 |]
                  (Array.make 12000 0.5);
            });
      test "an entry outside [0, 1]" (fun () ->
          List.iter
            (fun v ->
              let q = p 1 in
              q.(7) <- v;
              raises (evaluation 1 q (r 1));
              let s = r 1 in
              s.(3) <- v;
              raises (evaluation 1 (p 1) s))
            [ Float.infinity; Float.neg_infinity; 1.5; -0.25; 1e308 ]);
    ]

let () =
  exit
    (run "mynd.coco summary"
       [ fixture_means; tie; means; lengths; order; refusals ])
