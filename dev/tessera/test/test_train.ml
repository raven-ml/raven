(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

let arch pre_ln =
  { Model.pre_ln; width = 16; heads = 2; layers = 2; ffn = 24; dims = 32 }

let post_ln = Model.init ~key:(Nx.Rng.key 1) (arch false)
let pre_ln = Model.init ~key:(Nx.Rng.key 2) (arch true)
let model pre = if pre then pre_ln else post_ln
let root = Nx.Rng.key 7

let shape ?(pixels = 4) ?(rows = 3) ?(slots = 3) ?(width = 256) () =
  { Batch.pixels; s2_rows = rows; s1_rows = rows; slots; width }

let alone = shape ~pixels:1 ~rows:1 ~slots:1 ()
let packed = shape ~pixels:7 ~rows:2 ~slots:5 ~width:300 ()

(* Each place's key: the key of its pixel's index, that of [0] where empty. *)
let keys_of (b : Nx.host Batch.t) =
  Nx.Rng.fold_in_tensor root (Nx.maximum b.pixel (Nx.scalar Nx.int64 0L))

(* The duplicate-filled pixels a model trains on. *)
let training spec =
  let px = Synth.pixels spec in
  Batch.expand px (Batch.v2_counts px)

(* [f] of each batch of [px] packed by [shape], with the pixel index of each of
   its places. *)
let batches shape px counts f =
  let n = Pixels.length px in
  let rec go from acc =
    let b, next = Batch.pack shape px counts ~from in
    let acc = f b :: acc in
    if next < n then go next acc else List.rev acc
  in
  go 0 []

let rows_of (b : Nx.host Batch.t) e =
  let dims = Nx.dim 1 e in
  let e = Nx.to_array e in
  Array.to_list (Nx.to_array b.pixel)
  |> List.mapi (fun i p -> (Int64.to_int p, Array.sub e (i * dims) dims))
  |> List.filter (fun (p, _) -> p >= 0)

let embed ~dropout m shape (px, counts) =
  batches shape px counts (fun b ->
      rows_of b (Model.train ~dropout ~remat:false ~keys:(keys_of b) m b))
  |> List.concat |> List.sort compare

let tol = 1e-5
let same a b = equal (list (pair int (array (float tol)))) a b
let cases_gen = Gen.(pair bool (Synth.spec ~long:true ()))
let half = { Model.layers = 0.5; reducer = 0.5 }
let bits x = Array.map Int32.bits_of_float (Nx.to_array x)

let rate0 =
  prop "train at rates 0 is apply" ~count:30 cases_gen (fun (pre, spec) ->
      let m = model pre in
      let px, counts = training spec in
      cover "the teacher's form" pre;
      List.iter
        (fun (a, t) -> equal (array int32) a t)
        (batches packed px counts (fun b ->
             ( bits (Model.apply m b),
               bits
                 (Model.train ~dropout:Model.no_dropout ~remat:false
                    ~keys:(keys_of b) m b) ))))

let layout =
  prop "a training embedding does not depend on the batch's shape" ~count:30
    cases_gen (fun (pre, spec) ->
      let m = model pre in
      let px = training spec in
      cover "pixels sharing a row" (List.length spec.s2 > 1);
      same (embed ~dropout:half m alone px) (embed ~dropout:half m packed px))

(* A pixel's part of each mask: its tokens' values, or its own row. *)
let pixel_masks (b : Nx.host Batch.t) masks =
  let slot (r : Nx.host Batch.rows) place =
    let s = Int64.to_int (Nx.item [ place ] r.at) in
    let slots = Nx.dim 1 r.slot_start in
    let row = s / slots and k = s mod slots in
    ( row,
      Int32.to_int (Nx.item [ row; k ] r.slot_start),
      Int32.to_int (Nx.item [ row; k ] r.slot_stop) )
  in
  let part name m place =
    let sensor = String.sub name 0 2 in
    let ends s = String.ends_with ~suffix:s name in
    if sensor = "s2" || sensor = "s1" then begin
      let row, lo, hi = slot (if sensor = "s2" then b.s2 else b.s1) place in
      if ends "_attn_p" then
        Nx.slice [ Nx.I row; Nx.A; Nx.R (lo, hi); Nx.R (lo, hi) ] m
      else Nx.slice [ Nx.I row; Nx.R (lo, hi); Nx.A ] m
    end
    else Nx.slice [ Nx.I place ] m
  in
  Array.to_list (Nx.to_array b.pixel)
  |> List.mapi (fun place p -> (place, Int64.to_int p))
  |> List.filter (fun (_, p) -> p >= 0)
  |> List.map (fun (place, p) ->
      ( p,
        List.map
          (fun (name, m) -> (name, Nx.to_array (part name m place)))
          masks ))

let masks_of m shape (px, counts) =
  batches shape px counts (fun b ->
      pixel_masks b (Model.masks ~dropout:half ~keys:(keys_of b) m b))
  |> List.concat |> List.sort compare

let masks_layout =
  prop "a pixel's masks do not depend on the batch's shape" ~count:30 cases_gen
    (fun (pre, spec) ->
      let m = model pre in
      let px = training spec in
      cover "pixels sharing a row" (List.length spec.s2 > 1);
      cover "the teacher's form" pre;
      equal
        (list (pair int (list (pair string (array bool)))))
        (masks_of m alone px) (masks_of m packed px))

(* Each site keeps a share of its values within six deviations of [1 − rate] of
   a binomial draw. *)
let kept =
  test "each site keeps 1 − rate of its values" (fun () ->
      List.iter
        (fun pre ->
          let m = model pre in
          let spec =
            { Synth.s2 = [ 40; 300; 17 ]; s1 = [ 64; 9; 1 ]; seed = 3 }
          in
          let px, counts = training spec in
          let b, _ = Batch.pack packed px counts ~from:0 in
          let masks = Model.masks ~dropout:half ~keys:(keys_of b) m b in
          let names = List.map fst masks in
          List.iter
            (fun s -> mem string s names)
            [
              "s2_layer0_attn_p";
              "s2_layer1_attn";
              "s1_layer0_ffn_hidden";
              "s1_layer1_ffn";
              "reducer";
            ];
          if pre then mem string "fusion_layer1_attn_p" names;
          List.iter
            (fun (name, mask) ->
              let a = Nx.to_array mask in
              let n = Float.of_int (Array.length a) in
              let k = Array.fold_left (fun k x -> if x then k + 1 else k) 0 a in
              let share = Float.of_int k /. n in
              equal ~msg:name (float (6. *. Float.sqrt (0.25 /. n))) 0.5 share)
            masks)
        [ false; true ])

let compiled_masks =
  test "compiled masks are eager's" (fun () ->
      List.iter
        (fun pre ->
          let m = model pre in
          let px, counts =
            training { Synth.s2 = [ 5; 30 ]; s1 = [ 9; 0 ]; seed = 2 }
          in
          let b, _ = Batch.pack packed px counts ~from:0 in
          let keys = keys_of b in
          let eager = Model.masks ~dropout:half ~keys m b in
          let f =
            Rune.jit
              Nx.Ptree.(
                instantiate (module Model)
                @-> v Batch.walk @-> Nx.Rng.ptree
                @-> returns (list tensor))
              (fun m b keys ->
                List.map snd (Model.masks ~dropout:half ~keys m b))
          in
          let bits x = Nx.to_array x in
          equal
            (list (array bool))
            (List.map (fun (_, x) -> bits x) eager)
            (List.map bits (f m b keys)))
        [ false; true ])

let refusals =
  test "train refuses a rate and keys" (fun () ->
      let px, counts = training { Synth.s2 = [ 3 ]; s1 = [ 2 ]; seed = 1 } in
      let b, _ = Batch.pack alone px counts ~from:0 in
      raises (Invalid_argument "Tessera.Model.train: a rate outside [0, 1)")
        (fun () ->
          Model.train
            ~dropout:{ layers = 1.; reducer = 0. }
            ~remat:false ~keys:(keys_of b) post_ln b);
      raises
        (Invalid_argument "Tessera.Model.train: keys of shape [1; 2] expected")
        (fun () ->
          Model.train ~dropout:half ~remat:false
            ~keys:(Nx.Rng.split_batch ~n:2 root)
            post_ln b))

let net = Nx.Ptree.instantiate (module Model.Net)

let remat =
  prop "remat changes neither the value nor the gradient" ~count:10 cases_gen
    (fun (pre, spec) ->
      let m = model pre in
      let px, counts = training spec in
      let b, _ = Batch.pack packed px counts ~from:0 in
      let keys = keys_of b in
      let f remat n =
        Nx.sum (Model.train ~dropout:half ~remat ~keys { m with net = n } b)
      in
      let v, g = Rune.value_and_grad net (f false) m.net in
      let v', g' = Rune.value_and_grad net (f true) m.net in
      cover "the teacher's form" pre;
      equal (array int32) (bits v) (bits v');
      let leaves g =
        Nx.Ptree.fold net
          (fun _ a acc -> bits (Nx.cast Nx.float32 a) :: acc)
          g []
      in
      equal (list (array int32)) (leaves g) (leaves g'))

let () =
  exit
    (run "tessera.train"
       [
         group "train"
           [
             rate0; layout; masks_layout; kept; compiled_masks; refusals; remat;
           ];
       ])
