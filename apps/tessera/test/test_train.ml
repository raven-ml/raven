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

(* Rows as wide as Batch.typical's, holding every pixel of a case. *)
let wide = shape ~pixels:7 ~rows:1 ~slots:7 ~width:1024 ()

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
      rows_of b (Model.train ~dropout ~keys:(keys_of b) m (Model.inputs m b)))
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
                 (Model.train ~dropout:Model.no_dropout ~keys:(keys_of b) m
                    (Model.inputs m b)) ))))

let layout =
  prop "a training embedding does not depend on the batch's shape" ~count:30
    cases_gen (fun (pre, spec) ->
      let m = model pre in
      let px = training spec in
      cover "pixels sharing a row" (List.length spec.s2 > 1);
      let one = embed ~dropout:half m alone px in
      same one (embed ~dropout:half m packed px);
      same one (embed ~dropout:half m wide px))

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

let masks_of ?(keys = keys_of) m shape (px, counts) =
  batches shape px counts (fun b ->
      pixel_masks b (Model.masks ~dropout:half ~keys:(keys b) m b))
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

let net = Nx.Ptree.instantiate (module Model.Net)

(* Tokens that keep no key *)

(* Each token of [r]'s rows that keeps no key of the ones it sees in any head of
   the mask [p] ([[rows; heads; width; width]]), with whether it sees one. *)
let blind (r : Nx.host Batch.rows) p =
  let rows = Nx.dim 0 p and heads = Nx.dim 1 p and width = Nx.dim 2 p in
  let p = Nx.to_array p in
  let start = Nx.to_array r.start and stop = Nx.to_array r.stop in
  let at row t = Int32.to_int start.((row * width) + t)
  and upto row t = Int32.to_int stop.((row * width) + t) in
  let kept row t =
    let any = ref false in
    for h = 0 to heads - 1 do
      for k = at row t to upto row t - 1 do
        if p.((((((row * heads) + h) * width) + t) * width) + k) then
          any := true
      done
    done;
    !any
  in
  List.concat_map
    (fun row ->
      List.filter_map
        (fun t ->
          if kept row t then None else Some (row, t, at row t < upto row t))
        (List.init width Fun.id))
    (List.init rows Fun.id)

let heavy = { Model.layers = 0.9; reducer = 0. }

(* Elements of [x] that are NaN or infinite. *)
let nonfinite x =
  Array.fold_left
    (fun n v -> if Float.is_finite v then n else n + 1)
    0
    (Nx.to_array (Nx.cast Nx.float32 x))

(* The gradient of a weighted sum of [Model.train]'s embeddings, eager and
   compiled. *)
let gradients ~dropout m (b : Nx.host Batch.t) keys =
  let w =
    Nx.Rng.normal ~key:(Nx.Rng.key 9) Nx.float32
      [| Nx.dim 0 b.pixel; Model.dims m |]
  in
  let f n =
    let m = { m with Model.net = n } in
    Nx.sum (Nx.mul w (Model.train ~dropout ~keys m (Model.inputs m b)))
  in
  let grad n = snd (Rune.value_and_grad net f n) in
  [ grad m.net; Rune.jit Nx.Ptree.(net @-> returns net) grad m.net ]

let finite g = Nx.Ptree.fold net (fun _ a () -> equal int 0 (nonfinite a)) g ()

(* A padding token sees no key, and at rate 0.9 a short slot's token often keeps
   none of its keys in either head: neither makes a value or a derivative
   NaN. *)
let no_key =
  prop "a token that keeps no key leaves the value and gradient finite"
    ~count:10 (Synth.spec ()) (fun spec ->
      let m = post_ln in
      let px, counts = training spec in
      let b, _ = Batch.pack packed px counts ~from:0 in
      let keys = keys_of b in
      let masks = Model.masks ~dropout:heavy ~keys m b in
      equal int 0
        (nonfinite (Model.train ~dropout:heavy ~keys m (Model.inputs m b)));
      let seen = ref [] in
      List.iter
        (fun (sensor, rows) ->
          for i = 0 to 1 do
            let name = Printf.sprintf "%s_layer%d_attn_p" sensor i in
            List.iter
              (fun (_, _, sees) -> seen := sees :: !seen)
              (blind rows (List.assoc name masks))
          done)
        [ ("s2", b.s2); ("s1", b.s1) ];
      cover "a padding token" (List.mem false !seen);
      cover "a token that drops every key it sees" (List.mem true !seen);
      List.iter finite (gradients ~dropout:heavy m b keys))

(* [m] with each attention's query and key weights times [f]: scores [f²] times
   as far apart, as logits grow without QK norm. *)
let sharp f (m : _ Model.t) =
  let scale (l : _ Model.layer) =
    let by (p : _ Kaun.Linear.t) = { p with w = Nx.mul_s p.w f } in
    { l with attn = { l.attn with q = by l.attn.q; k = by l.attn.k } }
  in
  let backbone (bb : _ Model.backbone) =
    { bb with layers = List.map scale bb.layers }
  in
  match m.net with
  | Post_ln n ->
      { m with net = Post_ln { n with s2 = backbone n.s2; s1 = backbone n.s1 } }
  | Pre_ln _ -> invalid_arg "sharp: a post-LN model expected"

(* float32's [exp] overflows past 88.7: a share of the row's weight computed
   through it would lose its derivative where a dropped key outscores the kept
   ones by more. *)
let far_key =
  prop "a dropped key far above the kept ones leaves the gradient finite"
    ~count:10 (Synth.spec ()) (fun spec ->
      let m = sharp 30. post_ln in
      let px, counts = training spec in
      let b, _ = Batch.pack packed px counts ~from:0 in
      let keys = keys_of b in
      equal int 0
        (nonfinite (Model.train ~dropout:half ~keys m (Model.inputs m b)));
      List.iter finite (gradients ~dropout:half m b keys))

(* Distillation *)

let heads =
  Distil.heads ~key:(Nx.Rng.key 4) ~prefixes:(Model.prefixes post_ln) 8

let cosines =
  test "cosines are each head's cosine to its target" (fun () ->
      let id k =
        let w = Nx.eye Nx.float32 k in
        (k, [ { Kaun.Linear.w; b = None } ])
      in
      let e = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 0.; 3.; 4.; 0.; 0. |] in
      let t = Nx.create Nx.float32 [| 3; 2 |] [| 2.; 0.; -4.; 3.; 1.; 1. |] in
      let c =
        Distil.cosines
          [
            ( 1,
              [
                {
                  Kaun.Linear.w = Nx.create Nx.float32 [| 2; 1 |] [| 1.; 0. |];
                  b = None;
                };
              ] );
            id 2;
          ]
          e ~targets:t
      in
      (* Row 2 is a zero vector: its norm is taken as 1e-8, its cosine 0. *)
      equal (array (float 1e-6)) [| 1.; 1.; -0.8; 0.; 0.; 0. |] (Nx.to_array c))

(* The groups of [px]'s pixels, [n] a group, each place's target the row of its
   pixel. *)
let groups ~per px counts targets =
  let shape = shape ~pixels:per ~rows:per ~slots:2 () in
  let gs =
    batches shape px counts (fun b ->
        let t =
          Nx.take ~axis:0 (Nx.maximum b.pixel (Nx.scalar Nx.int64 0L)) targets
        in
        { Distil.batch = b; keys = keys_of b; targets = t })
  in
  {
    Distil.batch =
      Batch.stack (List.map (fun (g : _ Distil.group) -> g.batch) gs);
    keys =
      Nx.Rng.of_tensor
        (Nx.stack
           (List.map (fun (g : _ Distil.group) -> Nx.Rng.to_tensor g.keys) gs));
    targets = Nx.stack (List.map (fun (g : _ Distil.group) -> g.targets) gs);
  }

let step =
  Distil.step ~compute:Nx.float32 ~dropout:half ~lr:(Vega.rate 1e-3)
    ~weight_decay:(Vega.rate 1e-4) ~clip:1e6

let targets n = Nx.Rng.normal ~key:(Nx.Rng.key 5) Nx.float32 [| n; 8 |]
let start = Distil.init post_ln heads

let leaves s st =
  Nx.Ptree.fold s
    (fun _ a acc -> Nx.to_array (Nx.cast Nx.float64 a) :: acc)
    st []

(* [a] and [b], arrays of one shape each, agree to [1e-5] of their largest
   magnitude: to rounding at their scale. *)
let close a b =
  let worst = ref 0. and scale = ref 0. in
  List.iter2
    (fun a b ->
      Array.iteri
        (fun i x ->
          worst := Float.max !worst (Float.abs (x -. b.(i)));
          scale := Float.max !scale (Float.abs x))
        a)
    a b;
  at_most float_exact ~than:(1e-5 *. !scale) !worst

let grouping =
  prop "a step's gradient does not depend on its groups" ~count:10
    (Synth.spec ~pixels:6 ()) (fun spec ->
      let px, counts = training spec in
      let t = targets (Pixels.length px) in
      let one = groups ~per:6 px counts t
      and three = groups ~per:2 px counts t in
      assume (Nx.dim 0 one.batch.pixel = 1);
      cover "several groups" (Nx.dim 0 three.batch.pixel > 1);
      assume
        (Nx.to_array (Nx.greater_equal_s one.batch.pixel 0L)
        |> Array.exists Fun.id);
      let l1, s1 = step (one, start) and l3, s3 = step (three, start) in
      let f64 x = [ Nx.to_array (Nx.cast Nx.float64 x) ] in
      close (f64 l1) (f64 l3);
      (* After one step AdamW's first moment is a tenth of the gradient. *)
      let mu (s : (_, _) Distil.state) = s.opt.mu in
      let p = Nx.Ptree.instantiate (module Distil.Params) in
      close (leaves p (mu s1)) (leaves p (mu s3)))

let resume =
  test "a resumed run takes the steps the run takes" (fun () ->
      let spec =
        {
          Synth.s2 = [ 5; 9; 0; 17; 3; 40 ];
          s1 = [ 2; 0; 7; 8; 12; 1 ];
          seed = 11;
        }
      in
      let px, counts = training spec in
      let t = targets (Pixels.length px) in
      let data i = groups ~per:(2 + (i mod 3)) px counts t in
      let run st from n =
        let rec go st i acc =
          if i = from + n then (st, List.rev acc)
          else
            let l, st = step (data i, st) in
            go st (i + 1) (bits l :: acc)
        in
        go st from []
      in
      let s = Nx.Ptree.instantiate (module Distil.State) in
      let straight, ls = run start 0 4 in
      let half_way, ls1 = run start 0 2 in
      let file = Filename.temp_file "tessera" ".safetensors" in
      Fun.protect ~finally:(fun () -> Sys.remove file) @@ fun () ->
      Checkpoint.save file (module Distil.State) half_way ~key:root ~position:2;
      let back, key, position =
        Checkpoint.load file (module Distil.State) ~like:start
      in
      equal int 2 position;
      equal (array int32)
        (Nx.to_array (Nx.Rng.to_tensor root))
        (Nx.to_array (Nx.Rng.to_tensor key));
      let resumed, ls2 = run back 2 2 in
      equal (list (array int32)) ls (ls1 @ ls2);
      equal (list (array float_exact)) (leaves s straight) (leaves s resumed))

(* A float16 targets file of three rows, replaced after its load by its rows in
   reverse, renamed over it: the load's mapping keeps the file it read, and a
   read by ranges reads the new one. A window of positions in the first epoch
   takes the load's rows, and any other reads the file, the one that straddles
   the first epoch's end included, each cast to float32. *)
let windows =
  test "a window's targets in the first epoch take the load, any others read"
    (fun () ->
      let rows = Nx.create Nx.float16 [| 3; 2 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
      let ids l = Nx.create Nx.int64 [| List.length l |] (Array.of_list l) in
      let file = Filename.temp_file "tessera" ".npy" in
      let next = Filename.temp_file "tessera" ".npy" in
      let remove f = if Sys.file_exists f then Sys.remove f in
      Fun.protect ~finally:(fun () -> List.iter remove [ file; next ])
      @@ fun () ->
      Nx_io.save_npy file rows;
      let t = Nx_io.load_npy file in
      let reversed = Nx.take ~axis:0 (ids [ 2L; 1L; 0L ]) rows in
      Nx_io.save_npy next reversed;
      Sys.rename next file;
      let at = ids [ 0L; 2L ] in
      let want x = Nx.to_array (Nx.cast Nx.float32 (Nx.take ~axis:0 at x)) in
      let got first = Nx.to_array (Distil.targets file t ~first at) in
      equal ~msg:"first epoch" (array float_exact) (want rows) (got 1);
      equal ~msg:"straddling" (array float_exact) (want reversed) (got 2);
      equal ~msg:"second epoch" (array float_exact) (want reversed) (got 4))

(* A step's groups of [px], each target the index of its pixel. *)
let step_groups ?(shape = shape ~pixels:3 ~slots:2 ()) ~groups ~first px =
  let n = Pixels.length px in
  let t = Nx.cast Nx.float32 (Nx.arange Nx.int32 0 n 1) in
  Distil.groups ~key:root ~shape ~groups ~first px (Nx.reshape [| n; 1 |] t)

let ints x = Array.to_list (Array.map Int64.to_int (Nx.to_array x))

let step_groups_law =
  prop "a step's groups hold its pixels in order, each with its target and key"
    ~count:40
    Gen.(triple (Synth.spec ~pixels:10 ()) (int_range 1 3) (int_range 0 1000))
    (fun (spec, groups, first) ->
      let px = Synth.pixels spec in
      let g, taken = step_groups ~groups ~first px in
      let pixel = ints (Nx.reshape [| -1 |] g.batch.pixel) in
      cover "several groups" (groups > 1);
      cover "pixels left for the next step" (taken < Pixels.length px);
      at_most int ~than:(Pixels.length px) taken;
      (* The pixels below [taken] with an observation, each once, in order. *)
      let l2 = ints (Nx_ragged.lengths px.s2)
      and l1 = ints (Nx_ragged.lengths px.s1) in
      let observed =
        List.filter
          (fun j -> List.nth l2 j > 0 || List.nth l1 j > 0)
          (List.init taken Fun.id)
      in
      equal (list int) observed (List.filter (fun p -> p >= 0) pixel);
      let target = Array.to_list (Nx.to_array g.targets) in
      List.iter2
        (fun p t -> if p >= 0 then equal float_exact (Float.of_int p) t)
        pixel target;
      let want =
        Nx.Rng.fold_in_tensor root
          (Nx.add_s
             (Nx.maximum g.batch.pixel (Nx.scalar Nx.int64 0L))
             (Int64.of_int first))
      in
      equal (array int32)
        (Nx.to_array (Nx.Rng.to_tensor want))
        (Nx.to_array (Nx.Rng.to_tensor g.keys)))

let step_groups_cases =
  [
    test "a step's group starts where the one before it stopped" (fun () ->
        let spec =
          {
            Synth.s2 = [ 5; 9; 0; 17; 3; 40; 2 ];
            s1 = [ 2; 0; 7; 8; 1; 1; 3 ];
            seed = 3;
          }
        in
        let px = Synth.pixels spec in
        let one, k = step_groups ~groups:1 ~first:10 px in
        let two, taken = step_groups ~groups:2 ~first:10 px in
        let rest = Pixels.sub px ~pos:k ~len:(Pixels.length px - k) in
        let next, k' = step_groups ~groups:1 ~first:(10 + k) rest in
        equal int (k + k') taken;
        let row i x = Nx.to_array (Nx.slice [ Nx.I i ] x) in
        let shift =
          Array.map (fun p -> if p >= 0L then Int64.(add p (of_int k)) else p)
        in
        equal (array int64) (row 0 one.batch.pixel) (row 0 two.batch.pixel);
        equal (array int64)
          (shift (row 0 next.batch.pixel))
          (row 1 two.batch.pixel);
        let keys g = Nx.Rng.to_tensor g.Distil.keys in
        equal (array int32) (row 0 (keys next)) (row 1 (keys two)));
    test "a step of one group of every pixel takes them all" (fun () ->
        let px =
          Synth.pixels { Synth.s2 = [ 5; 0; 17 ]; s1 = [ 2; 0; 1 ]; seed = 1 }
        in
        let shape = Distil.shape ~pixels:8 ~slots:4 px in
        equal int 3 (snd (step_groups ~shape ~groups:1 ~first:0 px)));
  ]

(* Pixels of [n] Sentinel-2 observations and none of Sentinel-1. *)
let only_s2 ~pixels n =
  Synth.pixels
    {
      Synth.s2 = List.init pixels (Fun.const n);
      s1 = List.init pixels (Fun.const 0);
      seed = 2;
    }

let cat parts =
  let cat f = Nx_ragged.concat (List.map f parts) in
  { Pixels.s2 = cat (fun p -> p.Pixels.s2); s1 = cat (fun p -> p.Pixels.s1) }

let fields (s : Batch.shape) =
  [ s.pixels; s.s2_rows; s.s1_rows; s.slots; s.width ]

let shapes =
  [
    test "a distillation's shape holds its sample's packed rows, 5 % more"
      (fun () ->
        (* 4096 pixels of 100 Sentinel-2 observations, filled to 104 tokens, two
           a row: 2048 rows, ⌈1.05 · 2048 · 64 / 4096⌉ + 1 = 35 at 64 pixels a
           group; their Sentinel-1 stand-ins, 4 a row: 1024 rows, 18. Only the
           first 4096 pixels are measured: 4096 more of 8 tokens, 4 a row,
           would make it ⌈1.05 · 3072 · 64 / 8192⌉ + 1 = 27. *)
        let px = cat [ only_s2 ~pixels:4096 100; only_s2 ~pixels:4096 1 ] in
        let shape = Distil.shape ~pixels:64 ~slots:4 px in
        equal (list int) [ 64; 35; 18; 4; 256 ] (fields shape);
        (* Its tokens alone, ⌈1.05 · 104 · 64 / 256⌉ + 1 = 29 rows and one
           of stand-ins, would hold 4 pixels. *)
        equal int 64 (snd (step_groups ~shape ~groups:1 ~first:0 px)));
    test "a distillation's shape needs a pixel" (fun () ->
        raises (Invalid_argument "Tessera.Distil.shape: a corpus of no pixel")
          (fun () -> Distil.shape ~pixels:64 ~slots:4 (only_s2 ~pixels:0 8)));
  ]

let v2_recipe =
  test "v2's recipe is its distillation's table" (fun () ->
      let r = Distil.v2_recipe in
      equal (list float_exact) [ 8e-4; 0.05; 1e-4; 1.0 ]
        [ r.lr; r.warmup; r.weight_decay; r.clip ];
      equal
        (pair float_exact float_exact)
        (0.1, 0.2)
        (Model.v2_dropout.layers, Model.v2_dropout.reducer))

(* Heads in PyTorch's layout *)

let with_npz entries f =
  let file = Filename.temp_file "tessera" ".npz" in
  Fun.protect ~finally:(fun () -> Sys.remove file) @@ fun () ->
  Nx_io.save_npz file (Nx_io.Archive.of_list entries);
  f file

let heads_round_trip =
  test "heads load from PyTorch's layout" (fun () ->
      let hs = Distil.heads ~key:(Nx.Rng.key 3) ~prefixes:[ 2; 4 ] 3 in
      let flat hs =
        List.concat_map
          (fun (k, layers) ->
            List.concat_map
              (fun (l : _ Kaun.Linear.t) ->
                [
                  (Printf.sprintf "%d.weight" k, l.w);
                  (Printf.sprintf "%d.bias" k, Option.get l.b);
                ])
              layers)
          hs
      in
      with_npz (List.map (fun (n, x) -> (n, Nx.P x)) (flat hs)) @@ fun file ->
      let back = Distil.load_heads file [ 2; 4 ] in
      let arrays hs =
        List.map (fun (n, x) -> (n, Nx.shape x, Nx.to_array x)) (flat hs)
      in
      equal
        (list (triple string (array int) (array float_exact)))
        (arrays hs) (arrays back))

let heads_missing =
  test "heads refuse an npz without a prefix's entry, naming it" (fun () ->
      let w = Nx.zeros Nx.float32 [| 3; 2 |] in
      with_npz [ ("2.weight", Nx.P w) ] @@ fun file ->
      raises (Failure "Nx_io.Archive.float: 2.bias: no entry in the archive")
        (fun () -> Distil.load_heads file [ 2 ]))

let positions =
  prop "a pixel's masks differ at each stream position" ~count:30
    Gen.(triple bool (Synth.spec ()) (int_range 0 1000))
    (fun (pre, spec, first) ->
      let m = model pre in
      let px = training spec in
      let n = Pixels.length (fst px) in
      let at ?(shape = packed) first =
        masks_of ~keys:(Distil.keys ~key:root ~first) m shape px
      in
      let here = at first in
      cover "pixels sharing a row" (n > 1);
      cover "the teacher's form" pre;
      let masks = list (pair string (array bool)) in
      equal (list (pair int masks)) here (at ~shape:alone first);
      List.iter
        (fun (later, name) ->
          List.iter2
            (fun (p, a) (p', b) ->
              equal int p p';
              not_equal ~msg:(Printf.sprintf "pixel %d, %s" p name) masks a b)
            here (at later))
        [
          (first + 1, "the next position");
          (first + n, "a corpus later");
          (first + (1 lsl 32), "2^32 positions later");
        ])

let host_jit =
  Nx.Ptree.(
    pair (instantiate (module Distil.Group)) (instantiate (module Distil.State))
    @-> returns (pair tensor (instantiate (module Distil.State))))

let compiled =
  test "a compiled step is the eager step" (fun () ->
      let spec = { Synth.s2 = [ 5; 9; 17 ]; s1 = [ 2; 7; 1 ]; seed = 4 } in
      let px, counts = training spec in
      let g = groups ~per:2 px counts (targets (Pixels.length px)) in
      let l, st = step (g, start) in
      let l', st' = Rune.jit host_jit step (g, start) in
      let f64 x = [ Nx.to_array (Nx.cast Nx.float64 x) ] in
      close (f64 l) (f64 l');
      let p = Nx.Ptree.instantiate (module Distil.Params) in
      close (leaves p st.opt.mu) (leaves p st'.opt.mu))

(* A norm's, a QK norm's or the fusion's fixed array, by its path. *)
let fixed path =
  List.exists
    (fun s ->
      let n = String.length s and m = String.length path in
      let rec go i = i + n <= m && (String.sub path i n = s || go (i + 1)) in
      go 0)
    [ "norm"; ".qk."; "modality" ]

let stepped =
  test "a step changes the norms' and the fusion's fixed arrays" (fun () ->
      List.iter
        (fun pre ->
          let m = model pre in
          let hs = Distil.heads ~key:(Nx.Rng.key 6) ~prefixes:[ 16 ] 8 in
          let st0 = Distil.init m hs in
          let spec = { Synth.s2 = [ 5; 9; 17 ]; s1 = [ 2; 7; 1 ]; seed = 8 } in
          let px, counts = training spec in
          let g = groups ~per:3 px counts (targets (Pixels.length px)) in
          (* No decay: what moves, the gradient moves. *)
          let _, st =
            Distil.step ~compute:Nx.float32 ~dropout:half ~lr:(Vega.rate 1e-3)
              ~weight_decay:(Vega.rate 0.) ~clip:1e6 (g, st0)
          in
          let p = Nx.Ptree.instantiate (module Distil.Params) in
          let by_path st =
            Nx.Ptree.fold p
              (fun path a acc ->
                ( Nx.Ptree.Path.to_string path,
                  Nx.to_array (Nx.cast Nx.float64 a) )
                :: acc)
              { Distil.net = st.Distil.model.net; heads = st.heads }
              []
          in
          let moved =
            List.filter_map
              (fun ((path, a), (_, b)) ->
                if fixed path then Some (path, a <> b) else None)
              (List.combine (by_path st0) (by_path st))
          in
          if pre then mem string "net.modality" (List.map fst moved);
          mem string "net.s2.layers.0.norm1.beta" (List.map fst moved);
          equal
            (list (pair string bool))
            (List.map (fun (p, _) -> (p, true)) moved)
            moved)
        [ false; true ])

let () =
  exit
    (run "tessera.train"
       [
         group "train"
           [
             rate0;
             layout;
             masks_layout;
             kept;
             compiled_masks;
             no_key;
             far_key;
           ];
         group "distil"
           [ cosines; grouping; resume; windows; positions; compiled; stepped ];
         group "groups" ((step_groups_law :: step_groups_cases) @ shapes);
         group "heads" [ heads_round_trip; heads_missing ];
         group "recipe" [ v2_recipe ];
       ])
