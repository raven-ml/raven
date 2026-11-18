(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

let f64 = Nx.float64

(* Rows of [d] values around a centre per class, [n] of them, labels cycling
   through [classes]. *)
let labelled ~seed ~n ~d ~classes ~spread =
  let key = Nx.Rng.key seed in
  let y = Nx.mod_s (Nx.arange Nx.int64 0 n 1) (Int64.of_int classes) in
  let centres = Nx.Rng.normal ~key f64 [| classes; d |] in
  let noise = Nx.Rng.normal ~key:(Nx.Rng.fold_in key 1) f64 [| n; d |] in
  (Nx.add (Nx.take ~axis:0 y centres) (Nx.mul_s noise spread), y)

let problems =
  Gen.(
    map
      (fun ((n, d), (classes, seed)) -> (max n classes, d, classes, seed))
      (pair (pair (int_range 2 40) (int_range 1 6)) (pair (int_range 2 5) int)))

(* The probe's objective, whose gradient its minimum zeroes. *)
let objective ~c x y (w, b) =
  let logits = Nx.add (Nx.matmul x w) b in
  let k = Nx.dim 1 logits in
  let onehot = Nx.cast f64 (Nx.one_hot ~num_classes:k y) in
  Nx.add
    (Nx.sub
       (Nx.sum (Nx.logsumexp ~axes:[ 1 ] logits))
       (Nx.sum (Nx.mul onehot logits)))
    (Nx.mul_s (Nx.sum (Nx.square w)) (1. /. (2. *. c)))

(* A probe's error in its search's own measure: the root mean square of
   [(θ − θ* ) / σ], at most 1 where the probe meets the tolerance
   [Tol.v ~rel:1e-6 ~abs:1e-6] that {!Eval.fit} asks for, with [σ = 10⁻⁶ (1 +
   |θ|)] its scale. The minimum [θ*] is polished from the probe by four Newton
   steps, through the Hessian's pseudo-inverse, which drops its null
   direction, the biases' common shift. *)
let error f theta =
  let star = ref theta in
  for _ = 1 to 4 do
    let h = Rune.jacfwd' (Rune.grad' f) !star in
    let g = Nx.reshape [| -1; 1 |] (Rune.grad' f !star) in
    star :=
      Nx.sub !star
        (Nx.reshape [| -1 |] (Nx.matmul (Nx.pinv ~hermitian:true h) g))
  done;
  let sigma = Nx.add_s (Nx.mul_s (Nx.abs theta) 1e-6) 1e-6 in
  let r = Nx.div (Nx.sub theta !star) sigma in
  Float.sqrt (Nx.item [] (Nx.mean (Nx.square r)))

let minimum =
  prop "a probe is its objective's minimum" ~count:30 problems
    (fun (n, d, classes, seed) ->
      let x, y = labelled ~seed ~n ~d ~classes ~spread:1. in
      let c = 0.5 in
      let p = Eval.fit ~c x y in
      let f theta =
        match Nx.split_at ~axis:0 [ d * classes ] theta with
        | [ w; b ] -> objective ~c x y (Nx.reshape [| d; classes |] w, b)
        | _ -> assert false
      in
      let theta = Nx.concatenate ~axis:0 [ Nx.flatten p.w; p.b ] in
      at_most float_exact ~than:1. (error f theta);
      at_most float_exact ~than:1e-9 (Float.abs (Nx.item [] (Nx.sum p.b))))

let separable =
  test "a probe reads separated classes" (fun () ->
      let x, y = labelled ~seed:3 ~n:60 ~d:4 ~classes:3 ~spread:0.01 in
      let p = Eval.fit ~c:10. x y in
      let s = Eval.score ~predicted:(Eval.predict p x) y in
      equal float_exact 1. s.accuracy;
      equal float_exact 1. s.f1)

let scores =
  test "scores of a confusion" (fun () ->
      (* Labels 0 0 0 1 1 2, predicted 0 0 1 1 2 2: class 0 has t 2, m 1; class
         1 t 1, f 1, m 1; class 2 t 1, f 1. *)
      let y = Nx.create Nx.int64 [| 6 |] [| 0L; 0L; 0L; 1L; 1L; 2L |] in
      let predicted = Nx.create Nx.int64 [| 6 |] [| 0L; 0L; 1L; 1L; 2L; 2L |] in
      let s = Eval.score ~predicted y in
      equal (float 1e-15) (4. /. 6.) s.accuracy;
      let f1 = ((3. *. 0.8) +. (2. *. 0.5) +. (1. *. (2. /. 3.))) /. 6. in
      equal (float 1e-15) f1 s.f1)

let rankme =
  test "RankMe of equal and of single directions" (fun () ->
      (* Singular values all 2: shares 1/4, entropy log 4 up to the guards. *)
      let eye = Nx.mul_s (Nx.eye f64 4) 2. in
      let z = Nx.concatenate ~axis:0 [ eye; Nx.zeros f64 [| 3; 4 |] ] in
      equal (float 1e-6) 1. (Eval.rankme z);
      equal (float 1e-6) 1. (Eval.rankme (Nx.matrix_transpose z));
      (* One direction: one share near 1, the others 0. *)
      let col = Nx.reshape [| 6; 1 |] (Nx.arange f64 1 7 1) in
      let rank1 = Nx.matmul col (Nx.ones f64 [| 1; 5 |]) in
      less float_exact ~than:1e-5 (Eval.rankme rank1);
      (* Scale does not change the shares. *)
      let z = Nx.Rng.normal ~key:(Nx.Rng.key 2) f64 [| 9; 5 |] in
      equal (float 1e-6) (Eval.rankme z) (Eval.rankme (Nx.mul_s z 1e3)))

(* Splits *)

let fields =
  Gen.(
    list ~size:(int_range 0 12)
      (pair (of_list [ "a"; "b"; "c" ])
         (frequency [ (1, constant 0); (4, int_range 1 20) ])))

let split_law =
  prop "a split takes each class's smallest fields until its share is held"
    ~count:100
    Gen.(pair fields (float_range 0. 1.2))
    (fun (fs, share) ->
      let classes = Array.of_list (List.map fst fs)
      and areas = Array.of_list (List.map snd fs) in
      let train = Eval.split ~share ~classes ~areas in
      let n = Array.length areas in
      cover "a field left for testing"
        (List.exists
           (fun f -> areas.(f) > 0 && not train.(f))
           (List.init n Fun.id));
      cover "a share above 1" (share > 1.);
      for f = 0 to n - 1 do
        if areas.(f) = 0 then is_false ~msg:"area 0" train.(f)
      done;
      List.iter
        (fun c ->
          let of_c =
            List.filter
              (fun f -> classes.(f) = c && areas.(f) > 0)
              (List.init n Fun.id)
            |> List.sort (fun f g -> compare (areas.(f), f) (areas.(g), g))
          in
          let total = List.fold_left (fun a f -> a + areas.(f)) 0 of_c in
          let target = share *. Float.of_int total in
          let k = List.length (List.filter (fun f -> train.(f)) of_c) in
          let taken = List.filteri (fun i _ -> i < k) of_c in
          let area l =
            Float.of_int (List.fold_left (fun a f -> a + areas.(f)) 0 l)
          in
          (* The taken fields are the smallest, a prefix in (area, index). *)
          equal (list bool) (List.map (fun f -> train.(f)) of_c)
            (List.map (fun f -> List.mem f taken) of_c);
          (* Each was taken while those before it held less than the share,
             and the taking stopped only once they held it, or at the end. *)
          if k > 0 then
            less float_exact ~than:target
              (area (List.filteri (fun i _ -> i < k - 1) of_c));
          if k < List.length of_c then
            at_least float_exact ~than:target (area taken))
        [ "a"; "b"; "c" ])

let split_cases =
  let split ?(share = 0.3) classes areas =
    Array.to_list
      (Eval.split ~share ~classes:(Array.of_list classes)
         ~areas:(Array.of_list areas))
  in
  [
    test "a split breaks ties of area by index" (fun () ->
        (* Of 13, half: field 1 (3), then field 0 (5) before field 2 (5). *)
        equal (list bool) [ true; true; false ]
          (split ~share:0.5 [ "a"; "a"; "a" ] [ 5; 3; 5 ]));
    test "a split of share 0 takes none, of share 1 all" (fun () ->
        equal (list bool) [ false; false ]
          (split ~share:0. [ "a"; "b" ] [ 3; 4 ]);
        equal (list bool) [ true; true; false ]
          (split ~share:1. [ "a"; "a"; "b" ] [ 3; 4; 0 ]));
  ]

let () =
  exit
    (run "tessera.eval"
       [
         group "probe"
           [ minimum; separable; scores ];
         group "rankme" [ rankme ];
         group "split" (split_law :: split_cases);
       ])
