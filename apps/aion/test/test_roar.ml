(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Random-order generation on the narrow upstream AION of test_transformer
   (upstream/transformer.safetensors), whose decoder test_transformer checks at
   chosen positions against upstream's step. Draws are checked exactly, by
   replaying them from the keys the interface states, and the orders by χ² over
   every subset at a small n, from fixed keys, so that each run decides the
   same. *)

open Windtrap
open Aion

let weights = Nx_io.load_safetensors "upstream/transformer.safetensors"
let golden = Nx_io.load_safetensors "upstream/transformer_golden.safetensors"
let model = lazy (Transformer.of_archive weights)

let get dt name =
  match Nx_io.Archive.find name golden with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no entry " ^ name)

(* The fixture's inputs of modalities [ms], four objects with absent tokens. *)
let inputs ms =
  List.map
    (fun m ->
      let key = Modality.key m in
      let absent =
        Nx.not_equal_s (get Nx.uint8 ("input." ^ key ^ ".absent")) 0
      in
      Tokens.
        {
          modality = m;
          tokens = get Nx.int64 ("input." ^ key ^ ".tokens");
          present = Nx.logical_not absent;
        })
    ms

(* An XP spectrum of 55 tokens from three modalities, and redshift, one token,
   from two. *)
let xp =
  ( Modality.Scalar Xp_rp,
    inputs Modality.[ Scalar Xp_bp; Scalar Flux_g; Scalar Z ] )

let z = (Modality.Scalar Z, inputs Modality.[ Scalar Xp_bp; Scalar Flux_g ])
let ints x = Array.map Int64.to_int (Nx.to_array x)

(* A key for each of the fixture's four objects, from [seed]. *)
let keys seed = Nx.Rng.split_batch ~n:4 (Nx.Rng.key seed)

(* The orders the interface states for objects of [n] positions from [keys], a
   key each: argsort of float64 uniforms from [fold_in keys 0], [[|n|]] a
   key. *)
let order ~keys n =
  Nx.argsort ~axis:1
    (Nx.Rng.uniform_at Nx.Host.on ~key:(Nx.Rng.fold_in keys 0) Nx.float64
       [| n |])

(* Replaying a draw *)

(* Step [s] of a draw of [tokens] from [keys]: the next positions of each
   object's order, decoded with the tokens of the steps before present, and
   drawn as the interface states, each object from its key of
   [fold_in keys (s + 1)] at the logits less their row's largest over the
   temperature, or as their argmax at [0.]. Gives [tokens] at each step's
   positions, the tokens drawn there, and each step's logits. *)
let replay ~keys ~temperature ~reveal (target, inputs) tokens =
  let t = Lazy.force model in
  let b = Nx.dim 0 tokens and n = Modality.tokens target in
  let order = order ~keys n in
  let step (s, o, revealed, acc) k =
    let p = Nx.slice [ A; R (o, o + k) ] order in
    let target_input : _ Tokens.t =
      { modality = target; tokens; present = revealed }
    in
    let l =
      List.hd
        (Transformer.forward t
           (inputs @ [ target_input ])
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
    let revealed =
      Nx.scatter ~axis:1 p (Nx.ones Nx.bool (Nx.shape p)) revealed
    in
    (s + 1, o + k, revealed, (ints drawn, ints x, l) :: acc)
  in
  let _, _, _, acc =
    List.fold_left step (0, 0, Nx.zeros Nx.bool [| b; n |], []) reveal
  in
  let acc = List.rev acc in
  ( Array.concat (List.map (fun (d, _, _) -> d) acc),
    Array.concat (List.map (fun (_, x, _) -> x) acc),
    List.map (fun (_, _, l) -> l) acc )

let sample ~keys ?temperature ~reveal (target, inputs) =
  Roar.sample ~keys ?temperature ~reveal (Lazy.force model) inputs target

let linear8 = [ 7; 7; 7; 7; 7; 7; 7; 6 ]
let ones n = List.init n (fun _ -> 1)

let schedules =
  [
    ("one step", [ 55 ]);
    ("linear 8", linear8);
    ("one token per step", ones 55);
    ("uneven", [ 1; 30; 2; 22 ]);
  ]

let replays =
  let id (name, _, temperature) = Printf.sprintf "%s at %g" name temperature in
  let rows =
    List.concat_map
      (fun (name, reveal) ->
        List.map
          (fun temperature -> (name, reveal, temperature))
          [ 1.; 0.5; 0. ])
      schedules
  in
  group "replays"
    [
      cases "an XP spectrum's draws are its stated keys' draws" ~id rows
        (fun (_, reveal, temperature) ->
          let keys = keys 11 in
          let tokens = sample ~keys ~temperature ~reveal xp in
          let drawn, replayed, _ =
            replay ~keys ~temperature ~reveal xp tokens
          in
          equal (array int) replayed drawn);
      cases "a redshift's draw is one step's" ~id:(Printf.sprintf "%g")
        [ 1.; 0.5; 0. ] (fun temperature ->
          let keys = keys 12 in
          let tokens = sample ~keys ~temperature ~reveal:[ 1 ] z in
          let drawn, replayed, _ =
            replay ~keys ~temperature ~reveal:[ 1 ] z tokens
          in
          equal (array int) replayed drawn);
      test "a draw holds tokens of the vocabulary" (fun () ->
          let tokens = sample ~keys:(keys 13) ~reveal:linear8 xp in
          equal (array int) [| 4; 55 |] (Nx.shape tokens);
          let k = ints tokens in
          at_least int ~than:0 (Array.fold_left min max_int k);
          less int ~than:1024 (Array.fold_left max min_int k));
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
              ~temperature:1e6 ~reveal:[ 1 ]
              (Lazy.force model) (List.map copy inputs) target
          in
          let drawn =
            Array.fold_left
              (fun a k -> if k = 1024 then a + 1 else a)
              0 (ints tokens)
          in
          at_least int ~than:1 drawn);
      test "other keys draw other tokens" (fun () ->
          let a = sample ~keys:(keys 14) ~reveal:linear8 xp in
          let b = sample ~keys:(keys 15) ~reveal:linear8 xp in
          not_equal (array int) (ints a) (ints b));
    ]

(* Batches *)

(* The fixture's objects [ids] as a batch, with their keys: object [i]'s is
   [fold_in (key 26) i], as main.exe keys an object by its row. *)
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

(* A batch of the fixture's objects: some, then copies of the first, as a
   compiled device pads a batch. *)
let batches =
  Gen.(
    let+ ids = list ~size:(int_range 1 4) (int_range 0 3)
    and+ pad = int_range 0 3 in
    (Array.of_list (ids @ List.init pad (fun _ -> List.hd ids)), pad))

let batch_law =
  let target, inputs = xp in
  prop "an object draws alone what it draws in a batch" ~count:20 batches
    (fun (ids, pad) ->
      cover "two drawn objects" (Array.length ids - pad > 1);
      cover "a padded batch" (pad > 0);
      let t = Lazy.force model in
      (* A draw of [ids], and the logits of each of its steps. *)
      let draw ids =
        let keys, inputs = select ids inputs in
        let tokens = Roar.sample ~keys ~reveal:linear8 t inputs target in
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
          (* The premise: at each step, the object's logits are those it has
             alone, bit for bit. *)
          List.iter2
            (fun a l ->
              equal (array float_exact) (Nx.to_array a) (Nx.to_array (row l)))
            alone_logits logits;
          equal (array int) (ints alone) (ints (row tokens)))
        ids)

(* Temperature *)

let temperatures =
  group "temperature"
    [
      test "0 takes the largest logit" (fun () ->
          let t = Lazy.force model in
          let target, inputs = z in
          let l =
            List.hd
              (Transformer.forward t inputs
                 [ { modality = target; positions = None } ])
          in
          let tokens =
            sample ~keys:(keys 16) ~temperature:0. ~reveal:[ 1 ] z
          in
          equal (array int) (ints (Nx.argmax ~axis:(-1) l)) (ints tokens));
      cases "a tiny temperature takes the largest logit"
        ~id:(Printf.sprintf "%g") [ 1e-30; 1e-40; 1e-50; Float.min_float ]
        (fun temperature ->
          (* Each logit divided by 1e-40 alone would overflow float32; less its
             row's largest it is -inf or 0. Below float32's least subnormal the
             temperature rounds to 0, and the largest logits keep their 0. *)
          let g =
            sample ~keys:(keys 17) ~temperature:0. ~reveal:linear8 xp
          in
          let s = sample ~keys:(keys 17) ~temperature ~reveal:linear8 xp in
          equal (array int) (ints g) (ints s));
    ]

(* Orders *)

(* χ² of [counts] against equal expectations. *)
let chi2 counts =
  let total = Array.fold_left ( + ) 0 counts in
  let e = Float.of_int total /. Float.of_int (Array.length counts) in
  Array.fold_left
    (fun a c -> a +. (((Float.of_int c -. e) ** 2.) /. e))
    0. counts

(* χ²'s quantile at 1 - 1e-3 (scipy.stats.chi2.ppf(0.999, df)), for 19 and 29
   degrees of freedom. *)
let chi2_19 = 43.82019596451753
let chi2_29 = 58.301173489794905

(* The orders of [b] objects of [n] positions from [b] keys of [seed], as
   lists. *)
let orders seed b n =
  let o = ints (order ~keys:(Nx.Rng.split_batch ~n:b (Nx.Rng.key seed)) n) in
  List.init b (fun i -> Array.to_list (Array.sub o (i * n) n))

(* Each object's order cut into counts [cuts]: the sets revealed at each step,
   each sorted, as one cell. *)
let cells cuts order =
  let rec go cuts order =
    match cuts with
    | [] -> []
    | k :: rest ->
        List.sort compare (List.filteri (fun i _ -> i < k) order)
        :: go rest (List.filteri (fun i _ -> i >= k) order)
  in
  go cuts order

let tally cuts os =
  let h = Hashtbl.create 64 in
  List.iter
    (fun o ->
      let c = cells cuts o in
      Hashtbl.replace h c (1 + Option.value ~default:0 (Hashtbl.find_opt h c)))
    os;
  Array.of_seq (Hashtbl.to_seq_values h)

let law ~cuts ~cells:expected ~quantile seed =
  let n = List.fold_left ( + ) 0 cuts in
  let counts = tally cuts (orders seed 3000 n) in
  (* Every cell is reached, often enough for χ²'s law. *)
  equal int expected (Array.length counts);
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

(* Compiled draws *)

(* [start] and [step] compiled draw the eager tokens: the same keys, and logits
   that differ in their last bits, which move a draw only where its uniform
   lands that close to a boundary between two tokens' probabilities, about 10⁻⁶
   of a draw at float32's precision. The keys here are fixed, so each run
   decides the same. The temperatures share one step program, the temperature 0
   another. *)

module Soft = (val Nx.devices [ Result.get_ok (Rig.software "aion-roar") ])

(* [start] compiled, capturing the temperature and the target. *)
let start ?temperature target =
  Rune.jit
    Nx.Ptree.(
      Nx.Rng.ptree @-> list (v Tokens.walk) @-> returns (v Roar.walk_draw))
    (fun keys x -> Roar.start ~keys ?temperature x target)

(* The fold of a caller who compiles: [start], and [step], the count an
   argument, whose traces [traces] counts: one per program. *)
let compiled ?(traces = ref 0) () =
  let open Nx.Ptree in
  let draw () = v Roar.walk_draw in
  let step =
    Rune.jit
      (instantiate (module Transformer)
      @-> v Walk.int @-> draw () @-> returns (draw ()))
      (fun t k d ->
        incr traces;
        Roar.step t k d)
  in
  fun ~keys ?temperature ~reveal t inputs target ->
    let start = start ?temperature target in
    let d = start keys inputs in
    Roar.tokens (List.fold_left (fun d k -> step t k d) d reveal)

let draws =
  let target, inputs = xp in
  let draw = lazy (compiled ()) in
  let rows =
    List.concat_map
      (fun (name, reveal) ->
        List.map
          (fun temperature -> (name, reveal, temperature))
          [ 1.; 0.5; 2.; 0. ])
      [
        ("one step", [ 55 ]);
        ("linear 8", linear8);
        ("uneven", [ 1; 30; 2; 22 ]);
      ]
  in
  let id (name, _, temperature) = Printf.sprintf "%s at %g" name temperature in
  group "compiled"
    [
      cases "compiled start and step draw the eager tokens" ~id rows
        (fun (_, reveal, temperature) ->
          let keys = keys 19 in
          let eager = sample ~keys ~temperature ~reveal xp in
          let c =
            (Lazy.force draw) ~keys ~temperature ~reveal (Lazy.force model)
              inputs target
          in
          equal (array int) (ints eager) (ints c));
      cases "they draw on a set without eager kernels" ~id:(Printf.sprintf "%g")
        [ 1.; 0. ] (fun temperature ->
          let keys = keys 20 in
          let k =
            compiled () ~keys:(Nx.Rng.place Soft.on keys) ~temperature
              ~reveal:linear8
              (Nx.Ptree.place (module Transformer) Soft.on (Lazy.force model))
              (List.map (Nx.Ptree.place_fixed Tokens.walk Soft.on) inputs)
              target
          in
          let c = sample ~keys ~temperature ~reveal:linear8 xp in
          equal (array int) (ints c) (ints (Nx.place Nx.Host.on k)));
      test "every positive temperature shares one step program" (fun () ->
          let traces = ref 0 in
          let draw = compiled ~traces () in
          let run temperature =
            ignore
              (ints
                 (draw ~keys:(keys 21) ~temperature ~reveal:[ 55 ]
                    (Lazy.force model) inputs target))
          in
          List.iter run [ 1.; 0.5; 2. ];
          equal int 1 !traces;
          run 0.;
          equal int 2 !traces);
      test "a compiled start refuses at its trace" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"temperature -1") (fun () ->
              start ~temperature:(-1.) target (keys 0) inputs));
      test "a compiled step refuses a revealed position" (fun () ->
          (* Of 55 positions, the second step's last 5 are the first's. *)
          raises_match (Exn.invalid_arg ~sub:"already revealed") (fun () ->
              (Lazy.force draw) ~keys:(keys 0) ~reveal:[ 30; 30 ]
                (Lazy.force model) inputs target));
      test "so does one on a set without eager kernels" (fun () ->
          (* Its check raises at the wait that reads the tokens. *)
          raises_match (Exn.invalid_arg ~sub:"already revealed") (fun () ->
              Nx.place Nx.Host.on
                (compiled () ~keys:(Nx.Rng.place Soft.on (keys 0))
                   ~reveal:[ 30; 30 ]
                   (Nx.Ptree.place (module Transformer) Soft.on
                      (Lazy.force model))
                   (List.map (Nx.Ptree.place_fixed Tokens.walk Soft.on) inputs)
                   target)));
      test "a compiled step refuses a token outside the vocabulary" (fun () ->
          (* On the host its program's check raises at the call. *)
          let bad (i : _ Tokens.t) : _ Tokens.t =
            {
              i with
              tokens = Nx.full_like i.tokens 5000L;
              present = Nx.ones_like i.present;
            }
          in
          raises_match (Exn.invalid_arg ~sub:"outside [0, 1024)") (fun () ->
              (Lazy.force draw) ~keys:(keys 0) ~reveal:linear8
                (Lazy.force model) (List.map bad inputs) target));
    ]

(* Refusals *)

let refusals =
  let refuses sub f = raises_match (Exn.invalid_arg ~sub) f in
  let keys = keys 0 in
  group "refusals"
    [
      cases "a temperature outside [0, inf)" ~id:(Printf.sprintf "%g")
        [ -1.; -1e-300; Float.nan; Float.infinity; Float.neg_infinity ]
        (fun temperature ->
          refuses "temperature" (fun () ->
              sample ~keys ~temperature ~reveal:[ 55 ] xp));
      cases "counts that are not a schedule"
        ~id:(fun (name, _, _) -> name)
        [
          ("a count of 0", [ 0; 55 ], "a count of 0");
          ("a negative count", [ 56; -1 ], "a count of -1");
          ("too few", [ 54 ], "sums to 54");
          ("too many", [ 50; 6 ], "sums to 56");
          ("none", [], "sums to 0");
        ]
        (fun (_, reveal, sub) ->
          refuses sub (fun () -> sample ~keys ~reveal xp));
      test "a target among the inputs" (fun () ->
          let _, inputs = xp in
          refuses "tok_z is an input" (fun () ->
              sample ~keys ~reveal:[ 1 ] (Modality.Scalar Z, inputs)));
      cases "keys of another shape"
        ~id:(fun (name, _, _) -> name)
        [
          ("a key", Nx.Rng.key 0, "keys of shape [2], [4; 2] expected");
          ( "three keys",
            Nx.Rng.split_batch ~n:3 (Nx.Rng.key 0),
            "keys of shape [3; 2], [4; 2] expected" );
        ]
        (fun (_, keys, sub) ->
          refuses sub (fun () -> sample ~keys ~reveal:[ 55 ] xp));
      test "no input" (fun () ->
          refuses "no input" (fun () ->
              sample ~keys ~reveal:[ 1 ] (Modality.Scalar Z, [])));
      cases "a step's count outside [1, n]" ~id:string_of_int [ 0; -1; 56 ]
        (fun k ->
          let target, inputs = xp in
          let d = Roar.start ~keys inputs target in
          refuses (Printf.sprintf "a count of %d, not in [1, 55]" k) (fun () ->
              Roar.step (Lazy.force model) k d));
      test "a step at a revealed position" (fun () ->
          let target, inputs = xp and t = Lazy.force model in
          let d = Roar.step t 30 (Roar.start ~keys inputs target) in
          refuses "already revealed" (fun () -> Roar.step t 30 d));
    ]

let () =
  exit
    (run "aion.roar"
       [
         replays;
         group "batches" [ batch_law ];
         temperatures;
         orders_law;
         draws;
         refusals;
       ])
