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
      rows_of b
        (Model.train ~dropout ~remat:false ~keys:(keys_of b) m
           (Model.inputs m b)))
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
                    ~keys:(keys_of b) m (Model.inputs m b)) ))))

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
            ~remat:false ~keys:(keys_of b) post_ln (Model.inputs post_ln b));
      raises
        (Invalid_argument "Tessera.Model.train: keys of shape [1; 2] expected")
        (fun () ->
          Model.train ~dropout:half ~remat:false
            ~keys:(Nx.Rng.split_batch ~n:2 root)
            post_ln (Model.inputs post_ln b)))

(* Mixup *)

(* Two batches of one layout: equal counts, other values, and the second's
   observations in reverse order, so its descending passes come first. *)
let pair_of_batches () =
  let spec seed = { Synth.s2 = [ 3; 5 ]; s1 = [ 6; 4 ]; seed } in
  let pack px =
    fst
      (Batch.pack
         (shape ~pixels:2 ~rows:2 ~slots:1 ~width:8 ())
         px (Batch.v2_counts px) ~from:0)
  in
  let px = Synth.pixels (spec 4) in
  let reversed, _ =
    Synth.rebuild px (Batch.v2_counts px) (fun _ cs ->
        Array.init (Array.length cs) (fun k -> Array.length cs - 1 - k))
  in
  (pack (Synth.pixels (spec 3)), pack reversed)

let mixup_alpha = 0.3

let mixed =
  test "a mixup mixes each orbit's standardised tokens" (fun () ->
      (* The students' statistics: the orbits' differ. *)
      let m = post_ln in
      let a, b = pair_of_batches () in
      let alpha = Nx.scalar Nx.float32 mixup_alpha in
      let i = Model.mixup ~alpha (Model.inputs m a) (Model.inputs m b) in
      let got = List.assoc "s1_in" (Model.blocks m i) in
      let mean = Nx.to_array (Nx.cast Nx.float64 m.norm.s1_mean)
      and std = Nx.to_array (Nx.cast Nx.float64 m.norm.s1_std) in
      let token (r : Nx.host Batch.rows) row j =
        Array.init 4 (fun c -> Nx.item [ row; j; c ] r.tokens)
      in
      let z t c =
        let o = Float.to_int t.(3) in
        (t.(c) -. mean.((2 * o) + c)) /. (std.((2 * o) + c) +. 1e-9)
      in
      let other_orbits = ref 0 and apart = ref 0. in
      for row = 0 to 1 do
        let stop = Int32.to_int (Nx.item [ row; 0 ] a.s1.slot_stop) in
        for j = 0 to stop - 1 do
          let ta = token a.s1 row j and tb = token b.s1 row j in
          let mix x y = (mixup_alpha *. x) +. ((1. -. mixup_alpha) *. y) in
          for c = 0 to 1 do
            let expected = mix (z ta c) (z tb c) in
            equal (float 1e-5) expected (Nx.item [ row; j; c ] got);
            (* Mixing raw values, standardised by the first token's orbit. *)
            let raw =
              z [| mix ta.(0) tb.(0); mix ta.(1) tb.(1); 0.; ta.(3) |] c
            in
            if ta.(3) <> tb.(3) then begin
              incr other_orbits;
              apart := Float.max !apart (Float.abs (expected -. raw))
            end
          done;
          equal (float 1e-3) (mix ta.(2) tb.(2)) (Nx.item [ row; j; 2 ] got)
        done
      done;
      (* Where the orbits differ, mixing raw values is another input. *)
      less int ~than:!other_orbits 0;
      less float_exact ~than:!apart 1e-2)

let mixup_refusals =
  test "a mixup refuses inputs of other layouts" (fun () ->
      let a, _ = pair_of_batches () in
      let px = Synth.pixels { Synth.s2 = [ 3; 5 ]; s1 = [ 6; 4 ]; seed = 4 } in
      let wide, _ =
        Batch.pack
          (shape ~pixels:2 ~rows:2 ~slots:1 ~width:9 ())
          px (Batch.v2_counts px) ~from:0
      in
      let i = Model.inputs post_ln a and i' = Model.inputs post_ln wide in
      raises
        (Invalid_argument
           "Tessera.Model.mixup: inputs of other layouts (s2.tokens)")
        (fun () -> Model.mixup ~alpha:(Nx.scalar Nx.float32 0.5) i i');
      raises (Invalid_argument "Tessera.Model.mixup: alpha is not a scalar")
        (fun () -> Model.mixup ~alpha:(Nx.ones Nx.float32 [| 1 |]) i i))

(* Distillation *)

let net = Nx.Ptree.instantiate (module Model.Net)

let remat =
  prop "remat changes neither the value nor the gradient" ~count:10 cases_gen
    (fun (pre, spec) ->
      let m = model pre in
      let px, counts = training spec in
      let b, _ = Batch.pack packed px counts ~from:0 in
      let keys = keys_of b in
      (* A weighted sum: a plain sum of standardised embeddings is 0. *)
      let w =
        Nx.Rng.normal ~key:(Nx.Rng.key 9) Nx.float32
          [| Nx.dim 0 b.pixel; Model.dims m |]
      in
      let f remat n =
        let m = { m with net = n } in
        Nx.sum
          (Nx.mul w
             (Model.train ~dropout:half ~remat ~keys m (Model.inputs m b)))
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

(* Tokens that keep no key *)

(* [m] with each attention's output bias 0, so a token that attends to 0 in
   every head records 0 as its attention's value. *)
let unbiased (m : _ Model.t) =
  let zero (l : _ Model.layer) =
    let out = l.attn.out in
    let out = { out with b = Option.map Nx.zeros_like out.b } in
    { l with attn = { l.attn with out } }
  in
  let backbone (bb : _ Model.backbone) =
    { bb with layers = List.map zero bb.layers }
  in
  match m.net with
  | Post_ln n ->
      { m with net = Post_ln { n with s2 = backbone n.s2; s1 = backbone n.s1 } }
  | Pre_ln _ -> invalid_arg "unbiased: a post-LN model expected"

(* Each token of [r]'s rows that keeps no key of the ones it sees in any head
   of the mask [p] ([[rows; heads; width; width]]), with whether it sees one. *)
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
    Nx.sum
      (Nx.mul w
         (Model.train ~dropout ~remat:false ~keys m (Model.inputs m b)))
  in
  let grad n = snd (Rune.value_and_grad net f n) in
  [ grad m.net; Rune.jit Nx.Ptree.(net @-> returns net) grad m.net ]

let finite g = Nx.Ptree.fold net (fun _ a () -> equal int 0 (nonfinite a)) g ()

(* A padding token sees no key, and at rate 0.9 a short slot's token often
   keeps none of its keys in either head. Both attend to 0, as a softmax over
   no key gives, and neither makes a value or a derivative NaN. *)
let no_key =
  prop "a token that keeps no key attends to 0" ~count:10 (Synth.spec ())
    (fun spec ->
      let m = unbiased post_ln in
      let px, counts = training spec in
      let b, _ = Batch.pack packed px counts ~from:0 in
      let keys = keys_of b in
      let masks = Model.masks ~dropout:heavy ~keys m b in
      let acc = ref [] in
      let record k v = acc := (k, v ()) :: !acc in
      let e =
        Model.train ~record ~dropout:heavy ~remat:false ~keys m
          (Model.inputs m b)
      in
      equal int 0 (nonfinite e);
      let seen = ref [] in
      List.iter
        (fun (sensor, rows) ->
          for i = 0 to 1 do
            let name = Printf.sprintf "%s_layer%d_attn" sensor i in
            let a = List.assoc name !acc in
            let d = Nx.dim 2 a in
            List.iter
              (fun (row, t, sees) ->
                seen := sees :: !seen;
                equal (array float_exact) (Array.make d 0.)
                  (Nx.to_array (Nx.slice [ Nx.I row; Nx.I t; Nx.A ] a)))
              (blind rows (List.assoc (name ^ "_p") masks))
          done)
        [ ("s2", b.s2); ("s1", b.s1) ];
      cover "a padding token" (List.mem false !seen);
      cover "a token that drops every key it sees" (List.mem true !seen);
      List.iter finite (gradients ~dropout:heavy m b keys))

(* [m] with each attention's query and key weights times [f]: scores [f²]
   times as far apart, as logits grow without QK norm. *)
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

(* The largest margin by which a key that S2's first layer drops outscores
   every key the token keeps, over [b]'s tokens and [m]'s heads: the scores
   restated from the layer's input and parameters. *)
let margin (m : _ Model.t) (b : Nx.host Batch.t) masks =
  let x = List.assoc "s2_encoder_in" (Model.blocks m (Model.inputs m b)) in
  let l =
    match m.net with
    | Post_ln n -> List.hd n.s2.layers
    | Pre_ln n -> List.hd n.s2.layers
  in
  let rows = Nx.dim 0 x and width = Nx.dim 1 x and heads = m.heads in
  let split (p : _ Kaun.Linear.t) =
    let y = Nx.add (Nx.matmul x p.w) (Option.get p.b) in
    let dh = Nx.dim 1 p.w / heads in
    Nx.swapaxes 1 2 (Nx.reshape [| rows; width; heads; dh |] y)
  in
  let q = split l.attn.q and k = split l.attn.k in
  let scale = 1. /. Float.sqrt (Float.of_int (Nx.dim 3 q)) in
  let s = Nx.mul_s (Nx.matmul q (Nx.swapaxes 2 3 k)) scale in
  let s = Nx.add s (Nx.reshape [| rows; 1; 1; width |] b.s2.log_count) in
  let s = Nx.to_array s
  and p = Nx.to_array (List.assoc "s2_layer0_attn_p" masks) in
  let start = Nx.to_array b.s2.start and stop = Nx.to_array b.s2.stop in
  let best = ref Float.neg_infinity in
  for row = 0 to rows - 1 do
    for h = 0 to heads - 1 do
      for t = 0 to width - 1 do
        let kept = ref Float.neg_infinity and lost = ref Float.neg_infinity in
        let i = (row * width) + t in
        for j = Int32.to_int start.(i) to Int32.to_int stop.(i) - 1 do
          let at = (((((row * heads) + h) * width) + t) * width) + j in
          if p.(at) then kept := Float.max !kept s.(at)
          else lost := Float.max !lost s.(at)
        done;
        if Float.is_finite !kept && Float.is_finite !lost then
          best := Float.max !best (!lost -. !kept)
      done
    done
  done;
  !best

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
      let masks = Model.masks ~dropout:half ~keys m b in
      cover "a dropped key 89 above the kept ones" (margin m b masks > 89.);
      equal int 0
        (nonfinite
           (Model.train ~dropout:half ~remat:false ~keys m (Model.inputs m b)));
      List.iter finite (gradients ~dropout:half m b keys))

(* Records *)

let leaf_bits g =
  Nx.Ptree.fold net (fun _ a acc -> bits (Nx.cast Nx.float32 a) :: acc) g []

(* A weighted sum of [Model.train]'s embeddings and its gradient, with each
   recorded value when [record] and none otherwise, eager or compiled. (A plain
   sum of standardised embeddings is 0 whatever the parameters.) *)
let differentiated ~record ~jit m b keys =
  let names = ref [] in
  let w =
    Nx.Rng.normal ~key:(Nx.Rng.key 9) Nx.float32
      [| Nx.dim 0 b.Batch.pixel; Model.dims m |]
  in
  let f n =
    let m = { m with Model.net = n } in
    let acc = ref [] in
    let record =
      if record then Some (fun k v -> acc := (k, v ()) :: !acc) else None
    in
    let y =
      Nx.sum
        (Nx.mul w
           (Model.train ?record ~dropout:half ~remat:false ~keys m
              (Model.inputs m b)))
    in
    names := List.rev_map fst !acc;
    (y, List.rev_map snd !acc)
  in
  let g n =
    let v, g, z = Rune.value_and_grad_aux net Nx.Ptree.(list tensor) f n in
    (v, (g, z))
  in
  let v, (g, z) =
    if jit then
      Rune.jit
        Nx.Ptree.(net @-> returns (pair tensor (pair net (list tensor))))
        g m.net
    else g m.net
  in
  (v, g, List.combine !names z)

(* With a record, the forward's values become outputs of the program the
   gradient is taken in. Eager, the value and the gradient stay the same, bit
   for bit. Compiled, the gradient does too; the value, a sum the compiler
   schedules beside the extra outputs, may round otherwise, and is not
   compared. *)
let transparent =
  prop "a record changes neither the value nor the gradient" ~count:10 cases_gen
    (fun (pre, spec) ->
      let m = model pre in
      let px, counts = training spec in
      let b, _ = Batch.pack packed px counts ~from:0 in
      let keys = keys_of b in
      cover "the teacher's form" pre;
      List.iter
        (fun jit ->
          let v, g, _ = differentiated ~record:false ~jit m b keys in
          let v', g', z = differentiated ~record:true ~jit m b keys in
          is_true (z <> []);
          if not jit then equal (array int32) (bits v) (bits v');
          equal (list (array int32)) (leaf_bits g) (leaf_bits g'))
        [ false; true ])

(* [m] with unit 0 of sensor S2's embedding and first feed-forward dead: their
   bias at -1000, so their ReLU's input is negative at every token. *)
let dead (m : _ Model.t) =
  let kill (l : _ Kaun.Linear.t) =
    let b = Option.get l.b in
    let rest = Nx.slice [ Nx.R (1, Nx.dim 0 b) ] b in
    {
      l with
      b =
        Some
          (Nx.concatenate ~axis:0
             [ Nx.create Nx.float32 [| 1 |] [| -1e3 |]; rest ]);
    }
  in
  let backbone (bb : _ Model.backbone) =
    let e1, e2 = bb.embed in
    match bb.layers with
    | l :: ls ->
        {
          bb with
          embed = (kill e1, e2);
          layers = { l with ffn1 = kill l.ffn1 } :: ls;
        }
    | [] -> bb
  in
  match m.net with
  | Post_ln r -> { m with net = Post_ln { r with s2 = backbone r.s2 } }
  | Pre_ln r -> { m with net = Pre_ln { r with s2 = backbone r.s2 } }

let s2 = function Model.Post_ln r -> r.s2 | Pre_ln r -> r.s2
let bias (l : _ Kaun.Linear.t) i = Nx.item [ i ] (Option.get l.b)

let relu_inputs =
  test "a record receives a ReLU's input, on the branch the gradient takes"
    (fun () ->
      List.iter
        (fun (pre, jit) ->
          let m = dead (model pre) in
          let px, counts =
            training { Synth.s2 = [ 5; 30 ]; s1 = [ 9; 3 ]; seed = 2 }
          in
          let b, _ = Batch.pack packed px counts ~from:0 in
          let _, g, z = differentiated ~record:true ~jit m b (keys_of b) in
          let unit0 name =
            let v = List.assoc name z in
            Nx.to_array (Nx.slice [ Nx.A; Nx.A; Nx.I 0 ] v)
          in
          let negative a = Array.for_all (fun x -> x < 0.) a in
          let bb = s2 g in
          is_true ~msg:"embedding's input" (negative (unit0 "s2_embed_hidden"));
          equal ~msg:"embedding's gradient" float_exact 0.
            (bias (fst bb.embed) 0);
          is_true ~msg:"feed-forward's input"
            (negative (unit0 "s2_layer0_linear1"));
          equal ~msg:"feed-forward's gradient" float_exact 0.
            (bias (List.hd bb.layers).ffn1 0);
          is_true ~msg:"a live unit's gradient"
            (bias (List.hd bb.layers).ffn1 1 <> 0.
            && bias (fst bb.embed) 1 <> 0.);
          let relu x = Nx.maximum x (Nx.scalar_like x 0.) in
          equal ~msg:"reducer's ReLU" (array int32)
            (bits (relu (List.assoc "reducer1_ln1" z)))
            (bits (List.assoc "reducer2_relu" z)))
        [ (false, false); (true, false); (false, true) ])

let heads = Distil.heads ~key:(Nx.Rng.key 4) ~prefixes:post_ln.prefixes 8

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
                  Kaun.Linear.w = Nx.create Nx.float32 [| 1; 2 |] [| 1.; 0. |];
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
      Checkpoint.save file
        (module Distil.State)
        half_way ~key:root ~position:2 ~run:"--steps 4";
      let back, key, position, run' =
        Checkpoint.load file (module Distil.State) ~like:start
      in
      equal int 2 position;
      equal string "--steps 4" run';
      equal (array int32)
        (Nx.to_array (Nx.Rng.to_tensor root))
        (Nx.to_array (Nx.Rng.to_tensor key));
      let resumed, ls2 = run back 2 2 in
      equal (list (array int32)) ls (ls1 @ ls2);
      equal (list (array float_exact)) (leaves s straight) (leaves s resumed))

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
             refusals;
             remat;
             transparent;
             no_key;
             far_key;
             relu_inputs;
             mixed;
             mixup_refusals;
           ];
         group "distil" [ cosines; grouping; resume; compiled; stepped ];
       ])
