(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Random-order generation on test_transformer's narrow upstream AION, whose
   decoder test_transformer checks at chosen positions against upstream's step.
   Draws are checked exactly, by replaying them from the keys the interface
   states, and orders by χ² over every subset at a small n; all keys are fixed,
   so that each run decides the same. *)

open Windtrap
open Aion

let ints x = Array.map Int64.to_int (Nx.to_array x)

(* An XP spectrum of 55 tokens from three modalities, and redshift, one token of
   a vocabulary of 1025, from two. *)
let xp =
  Modality.
    (Scalar Xp_rp, Fixture.inputs [ Scalar Xp_bp; Scalar Flux_g; Scalar Z ])

let z = Modality.(Scalar Z, Fixture.inputs [ Scalar Xp_bp; Scalar Flux_g ])

(* A key for each of the fixture's four objects, from [seed]. *)
let keys seed = Nx.Rng.split_batch ~n:4 (Nx.Rng.key seed)

let sample ~keys ?temperature ~reveal (target, inputs) =
  Roar.sample ~keys ?temperature ~reveal (Fixture.transformer ()) inputs target

(* The orders the interface states: permutations of [n] from [fold_in keys 0],
   one a key. *)
let order ~keys n =
  Nx.Rng.permutation_at Nx.Host.on ~key:(Nx.Rng.fold_in keys 0) n

(* Step [s] of a draw of [tokens] from [keys], as the interface states it: the
   next positions of each object's order, decoded with the tokens of the steps
   before present, each object drawing from its key of [fold_in keys (s + 1)] at
   its logits less their row's largest over the temperature, or their argmax at
   [0.]. Gives [tokens] at each step's positions, the tokens drawn there, and
   each step's logits. *)
let replay ~keys ~temperature ~reveal (target, inputs) tokens =
  let t = Fixture.transformer () in
  let b = Nx.dim 0 tokens and n = Modality.tokens target in
  let order = order ~keys n in
  let step (s, o, present, acc) k =
    let p = Nx.slice [ A; R (o, o + k) ] order in
    let l =
      List.hd
        (Transformer.forward t
           (inputs @ [ { Tokens.modality = target; tokens; present } ])
           [ { modality = target; positions = Some p } ])
    in
    let x =
      if temperature = 0. then Nx.argmax ~axis:(-1) l
      else
        let d = Nx.sub l (Nx.max ~axes:[ -1 ] ~keepdims:true l) in
        Nx.Rng.categorical
          ~key:(Nx.Rng.reshape [| b; 1; 1 |] (Nx.Rng.fold_in keys (s + 1)))
          (Nx.div_s d temperature)
    in
    let drawn = Nx.take_along_axis ~axis:1 p tokens in
    let present = Nx.scatter ~axis:1 p (Nx.ones Nx.bool (Nx.shape p)) present in
    (s + 1, o + k, present, (ints drawn, ints x, l) :: acc)
  in
  let _, _, _, acc =
    List.fold_left step (0, 0, Nx.zeros Nx.bool [| b; n |], []) reveal
  in
  let acc = List.rev acc in
  ( Array.concat (List.map (fun (d, _, _) -> d) acc),
    Array.concat (List.map (fun (_, x, _) -> x) acc),
    List.map (fun (_, _, l) -> l) acc )

let linear8 = [ 7; 7; 7; 7; 7; 7; 7; 6 ]

let schedules =
  [ ("one step", [ 55 ]); ("linear 8", linear8); ("uneven", [ 1; 30; 2; 22 ]) ]

(* Each schedule at each temperature, as (name, reveal, temperature). *)
let rows schedules temperatures =
  List.concat_map
    (fun (name, reveal) -> List.map (fun t -> (name, reveal, t)) temperatures)
    schedules

let row_id (name, _, temperature) = Printf.sprintf "%s at %g" name temperature

let replays =
  let one_by_one = ("one token per step", List.init 55 (fun _ -> 1)) in
  let xp_rows = rows (one_by_one :: schedules) [ 1.; 0.5; 0. ] in
  let z_rows = rows [ ("redshift", [ 1 ]) ] [ 1.; 0.5; 0. ] in
  group "replays"
    [
      cases "a draw is its stated keys' draw, [b; n]" ~id:row_id
        (xp_rows @ z_rows) (fun (name, reveal, temperature) ->
          let ((target, _) as x) = if name = "redshift" then z else xp in
          let keys = keys 11 in
          let tokens = sample ~keys ~temperature ~reveal x in
          equal (array int) [| 4; Modality.tokens target |] (Nx.shape tokens);
          let drawn, replayed, _ = replay ~keys ~temperature ~reveal x tokens in
          equal (array int) replayed drawn);
      test "redshift's class 1024 is drawn" (fun () ->
          (* 8192 copies of one object at a temperature that flattens the
             logits: each draws 1024 with probability near 1/1025. *)
          let copies = 8192 in
          let target, inputs = z in
          let copy (i : _ Tokens.t) : _ Tokens.t =
            let row x =
              let s = Array.copy (Nx.shape x) in
              s.(0) <- copies;
              Nx.broadcast_to s (Nx.slice [ R (0, 1) ] x)
            in
            { i with tokens = row i.tokens; present = row i.present }
          in
          let tokens =
            Roar.sample
              ~keys:(Nx.Rng.split_batch ~n:copies (Nx.Rng.key 18))
              ~temperature:1e6 ~reveal:[ 1 ] (Fixture.transformer ())
              (List.map copy inputs) target
          in
          at_least int ~than:1
            (Array.fold_left
               (fun a k -> if k = 1024 then a + 1 else a)
               0 (ints tokens)));
      cases "a tiny temperature takes the largest logit"
        ~id:(Printf.sprintf "%g") [ 1e-30; 1e-40; 1e-50; Float.min_float ]
        (fun temperature ->
          (* A logit divided by 1e-40 alone would overflow float32; less its
             row's largest it is -inf or 0. Below float32's least subnormal the
             temperature rounds to 0, and the largest logits keep their 0. *)
          let g = sample ~keys:(keys 17) ~temperature:0. ~reveal:linear8 xp in
          let s = sample ~keys:(keys 17) ~temperature ~reveal:linear8 xp in
          equal (array int) (ints g) (ints s));
    ]

(* The fixture's objects [ids] as a batch, object [i] keyed by [fold_in (key 26)
   i], as main.exe keys an object by its row. *)
let select ids inputs =
  let ids =
    Nx.create Nx.int64 [| Array.length ids |] (Array.map Int64.of_int ids)
  in
  let take x = Nx.take ~axis:0 ids x in
  ( Nx.Rng.fold_in_tensor (Nx.Rng.key 26) ids,
    List.map
      (fun (i : _ Tokens.t) : _ Tokens.t ->
        { i with tokens = take i.tokens; present = take i.present })
      inputs )

(* Some of the fixture's objects, then copies of the first, as a compiled device
   pads a batch. *)
let batch_law =
  let target, inputs = xp in
  prop "an object draws alone what it draws in a batch" ~count:20
    Gen.(
      let+ ids = list ~size:(int_range 1 4) (int_range 0 3)
      and+ pad = int_range 0 3 in
      (Array.of_list (ids @ List.init pad (fun _ -> List.hd ids)), pad))
    (fun (ids, pad) ->
      cover "two drawn objects" (Array.length ids - pad > 1);
      cover "a padded batch" (pad > 0);
      (* A draw of [ids], and the logits of each of its steps. *)
      let draw ids =
        let keys, inputs = select ids inputs in
        let tokens =
          Roar.sample ~keys ~reveal:linear8 (Fixture.transformer ()) inputs
            target
        in
        let _, _, logits =
          replay ~keys ~temperature:1. ~reveal:linear8 (target, inputs) tokens
        in
        (tokens, logits)
      in
      let tokens, logits = draw ids in
      Array.iteri
        (fun i id ->
          let row x = Nx.slice [ R (i, i + 1) ] x in
          let alone, alone_logits = draw [| id |] in
          (* The premise: at each step the object's logits are those it has
             alone, bit for bit. *)
          List.iter2
            (fun a l ->
              equal (array float_exact) (Nx.to_array a) (Nx.to_array (row l)))
            alone_logits logits;
          equal (array int) (ints alone) (ints (row tokens)))
        ids)

(* Orders *)

(* χ² of [counts] against equal expectations, and its quantile at 1 - 1e-3
   (scipy.stats.chi2.ppf(0.999, df)) for 19 and 29 degrees of freedom. *)
let chi2 counts =
  let total = Array.fold_left ( + ) 0 counts in
  let e = Float.of_int total /. Float.of_int (Array.length counts) in
  Array.fold_left
    (fun a c -> a +. (((Float.of_int c -. e) ** 2.) /. e))
    0. counts

let chi2_19 = 43.82019596451753
let chi2_29 = 58.301173489794905

(* The orders of [b] objects of [n] positions from [b] keys of [seed]. *)
let orders seed b n =
  let o = ints (order ~keys:(Nx.Rng.split_batch ~n:b (Nx.Rng.key seed)) n) in
  List.init b (fun i -> Array.to_list (Array.sub o (i * n) n))

(* The count of each order cut into steps of [cuts] positions, the sets each
   step reveals. *)
let tally cuts os =
  let rec cells cuts order =
    match cuts with
    | [] -> []
    | k :: rest ->
        List.sort compare (List.filteri (fun i _ -> i < k) order)
        :: cells rest (List.filteri (fun i _ -> i >= k) order)
  in
  let h = Hashtbl.create 64 in
  List.iter
    (fun o ->
      let c = cells cuts o in
      Hashtbl.replace h c (1 + Option.value ~default:0 (Hashtbl.find_opt h c)))
    os;
  Array.of_seq (Hashtbl.to_seq_values h)

let law ~cuts ~cells ~quantile seed =
  let counts = tally cuts (orders seed 3000 (List.fold_left ( + ) 0 cuts)) in
  (* Every cell is reached, often enough for χ²'s law. *)
  equal int cells (Array.length counts);
  at_least int ~than:50 (Array.fold_left min max_int counts);
  less float_exact ~than:quantile (chi2 counts)

let orders_law =
  group "orders"
    [
      cases "the first 3 of 6 positions are a uniform subset" ~id:string_of_int
        [ 21; 22 ]
        (law ~cuts:[ 3; 3 ] ~cells:20 ~quantile:chi2_19);
      cases "steps of 2, 2 and 1 of 5 positions are a uniform partition"
        ~id:string_of_int [ 23; 24 ]
        (law ~cuts:[ 2; 2; 1 ] ~cells:30 ~quantile:chi2_29);
      test "objects of distinct keys take their own orders" (fun () ->
          equal int 64 (List.length (List.sort_uniq compare (orders 25 64 55))));
    ]

(* Compiled draws: [start] and [step] compiled draw the eager tokens. Their
   logits differ in the last bits, which moves a draw only where its uniform
   lands that close to a boundary between two tokens' probabilities, about 10⁻⁶
   of a draw at float32's precision; on fixed keys each run decides the same. *)

module Soft = (val Nx.devices [ Result.get_ok (Rig.software "aion-roar") ])

(* [start] compiled, capturing the temperature and the target. *)
let start ?temperature target =
  Rune.jit
    Nx.Ptree.(
      Nx.Rng.ptree @-> list (v Tokens.walk) @-> returns (v Roar.walk_draw))
    (fun keys x -> Roar.start ~keys ?temperature x target)

(* The draws of the host and of the software set, each with its programs. *)
let on_host = lazy (Compiled.draw Nx.Host.on)
let on_soft = lazy (Compiled.draw Soft.on)

let draws =
  let target, inputs = xp in
  let soft = List.map (fun (n, r, t) -> ("soft " ^ n, r, t)) in
  group "compiled"
    [
      cases "compiled start and step draw the eager tokens, on any set"
        ~id:row_id
        (rows schedules [ 1.; 0.5; 2.; 0. ]
        @ soft (rows [ ("linear 8", linear8) ] [ 1.; 0. ]))
        (fun (name, reveal, temperature) ->
          let keys = keys 19 in
          let draw =
            Lazy.force
              (if String.starts_with ~prefix:"soft" name then on_soft
               else on_host)
          in
          equal (array int)
            (ints (sample ~keys ~temperature ~reveal xp))
            (ints (draw ~keys ~temperature ~reveal target inputs)));
      test "every positive temperature shares one step program" (fun () ->
          let traces = ref 0 in
          let draw = Compiled.draw ~traces Nx.Host.on in
          let run temperature =
            ignore
              (draw ~keys:(keys 21) ~temperature ~reveal:[ 55 ] target inputs)
          in
          List.iter run [ 1.; 0.5; 2. ];
          equal int 1 !traces;
          run 0.;
          equal int 2 !traces);
      test "start and step compute no float64" (fun () ->
          (* The programs' dtypes, each value's as pp_program prints it, at a
             positive temperature, whose step divides by it. *)
          let shown f x = Format.asprintf "%a" Rune.pp_program (f x) in
          let d = Roar.start ~keys:(keys 0) ~temperature:0.5 inputs target in
          let draw () = Nx.Ptree.v Roar.walk_draw in
          let start =
            shown
              (Rune.program
                 Nx.Ptree.(pair Nx.Rng.ptree (list (v Tokens.walk)))
                 (draw ())
                 (fun (keys, x) -> Roar.start ~keys ~temperature:0.5 x target))
              (keys 0, inputs)
          and step =
            shown
              (Rune.program (draw ()) (draw ())
                 (Roar.step (Fixture.transformer ()) 7))
              d
          in
          let has s =
            let n = String.length s in
            let rec go i =
              i + 7 <= n && (String.sub s i 7 = "float64" || go (i + 1))
            in
            go 0
          in
          equal bool ~msg:"start" false (has start);
          equal bool ~msg:"step" false (has step));
      test "a step on a set without eager kernels refuses at a later wait"
        (fun () ->
          (* Of 55 positions, the second step's last 5 are the first's. *)
          raises_match (Exn.invalid_arg ~sub:"already revealed") (fun () ->
              (Lazy.force on_soft) ~keys:(keys 0) ~reveal:[ 30; 30 ] target
                inputs));
    ]

(* Refusals, eagerly and by [start] and [step] compiled on the host, where
   [start] refuses as it traces and [step]'s checks raise at the call. *)

type mode = {
  start :
    keys:Nx.host Nx.Rng.key ->
    ?temperature:float ->
    Nx.host Tokens.t list ->
    Modality.t ->
    Nx.host Roar.draw;
  step : int -> Nx.host Roar.draw -> Nx.host Roar.draw;
}

let modes =
  let step =
    lazy
      (Rune.jit
         Nx.Ptree.(
           instantiate (module Transformer)
           @-> v Walk.int @-> v Roar.walk_draw
           @-> returns (v Roar.walk_draw))
         Roar.step)
  in
  [
    {
      start = Roar.start;
      step = (fun k d -> Roar.step (Fixture.transformer ()) k d);
    };
    {
      start =
        (fun ~keys ?temperature x target -> start ?temperature target keys x);
      step = (fun k d -> (Lazy.force step) (Fixture.transformer ()) k d);
    };
  ]

let refusals =
  let target, inputs = xp in
  let keys = keys 0 in
  let start ?temperature ?(keys = keys) ?(inputs = inputs) ?(target = target) m
      =
    ignore (m.start ~keys ?temperature inputs target)
  in
  let steps ?(inputs = inputs) counts m =
    let d = m.start ~keys inputs target in
    ignore (Roar.tokens (List.fold_left (fun d k -> m.step k d) d counts))
  in
  let bad (i : _ Tokens.t) =
    {
      i with
      tokens = Nx.full_like i.tokens 5000L;
      present = Nx.ones_like i.present;
    }
  in
  let temperature t =
    ( Printf.sprintf "temperature %g" t,
      "temperature",
      fun m -> start ~temperature:t m )
  in
  group "refusals"
    [
      cases "refuses, eagerly and compiled"
        ~id:(fun (name, _, _) -> name)
        [
          temperature (-1.);
          temperature (-1e-300);
          temperature Float.nan;
          temperature Float.infinity;
          temperature Float.neg_infinity;
          ( "a target among the inputs",
            "tok_z is an input",
            fun m -> start ~target:(Modality.Scalar Z) m );
          ( "one key",
            "keys of shape [2], [4; 2] expected",
            fun m -> start ~keys:(Nx.Rng.key 0) m );
          ( "three keys",
            "keys of shape [3; 2], [4; 2] expected",
            fun m -> start ~keys:(Nx.Rng.split_batch ~n:3 (Nx.Rng.key 0)) m );
          ( "no input",
            "no input",
            fun m -> start ~inputs:[] ~target:(Modality.Scalar Z) m );
          ("a count of 0", "a count of 0, not in [1, 55]", steps [ 0 ]);
          ("a count of -1", "a count of -1, not in [1, 55]", steps [ -1 ]);
          ("a count of 56", "a count of 56, not in [1, 55]", steps [ 56 ]);
          ("a revealed position", "already revealed", steps [ 30; 30 ]);
          ( "a token outside the vocabulary",
            "outside [0, 1024)",
            steps ~inputs:(List.map bad inputs) linear8 );
        ]
        (fun (_, sub, f) ->
          List.iter
            (fun m -> raises_match (Exn.invalid_arg ~sub) (fun () -> f m))
            modes);
      cases "sample refuses counts that are not a schedule, and as start does"
        ~id:(fun (name, _, _) -> name)
        [
          ( "a count of 0",
            "a count of 0",
            fun () -> sample ~keys ~reveal:[ 0; 55 ] xp );
          ( "a negative count",
            "a count of -1",
            fun () -> sample ~keys ~reveal:[ 56; -1 ] xp );
          ("too few", "sums to 54", fun () -> sample ~keys ~reveal:[ 54 ] xp);
          ("too many", "sums to 56", fun () -> sample ~keys ~reveal:[ 50; 6 ] xp);
          ("none", "sums to 0", fun () -> sample ~keys ~reveal:[] xp);
          ( "a negative temperature",
            "temperature",
            fun () -> sample ~keys ~temperature:(-1.) ~reveal:[ 55 ] xp );
        ]
        (fun (_, sub, f) -> raises_match (Exn.invalid_arg ~sub) f);
    ]

let () =
  exit
    (run "aion.roar"
       [ replays; group "batches" [ batch_law ]; orders_law; draws; refusals ])
