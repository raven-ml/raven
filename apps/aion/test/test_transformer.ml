(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The transformer against a narrow upstream AION with random weights
   (upstream/transformer.safetensors): aion-base's structure at width 16, 2
   heads, 2 encoder and 2 decoder blocks, over tok_xp_bp, tok_flux_g, tok_z and
   tok_xp_rp. Its weights make each constant matter: embeddings and query-key
   projections near 1e-3, so the first layer norms and the query-key norms see a
   variance near their eps; a context bias; norm scales and shifts away from
   identity; decoder position tables unlike the encoder's. Its inputs leave
   tokens absent.

   upstream/transformer_golden.safetensors holds the inputs, upstream's float64
   context and logits (the decoder with training's mask, tok_xp_rp's for the
   second object alone), and the spread [ŝ] of their rounding ensemble at
   float32's scale; upstream/roar_golden.safetensors two random-order steps of
   tok_xp_rp at 7 chosen positions, [roar.0] with nothing revealed and [roar.1]
   with 28 tokens revealed per object. Each value must lie within Spread's τ ŝ
   of upstream's. *)

open Windtrap
open Aion

let weights = Fixture.load "transformer"
let golden = Fixture.load "transformer_golden"
let roar = Fixture.load "roar_golden"
let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let ints x = Array.map Int64.to_int (Nx.to_array x)
let ms = Modality.[ Scalar Xp_bp; Scalar Flux_g; Scalar Z ]

(* [x] against the float64 reference [f64.name] of [a], within τ ŝ. *)
let check a name x =
  Spread.values ~n:(Fixture.ensemble a)
    ~reference:(floats (Fixture.get a Nx.float64 ("f64." ^ name)))
    ~spread:(floats (Fixture.get a Nx.float32 ("s." ^ name)))
    (floats x)

let steps = [ "roar.0"; "roar.1" ]
let at name = Fixture.get roar Nx.int64 (name ^ ".positions")
let whole m : _ Transformer.target = { modality = m; positions = None }

let xp_rp positions : _ Transformer.target =
  { modality = Scalar Xp_rp; positions }

let decode_at name p =
  List.hd
    (Transformer.forward (Fixture.transformer ())
       (Fixture.inputs ~step:name ms)
       [ xp_rp p ])

(* Object 2's rows of logits of objects 1 and 2. *)
let second x = Nx.slice [ R (1, 2) ] x

(* [x]'s rows of each object at its positions [p], [b; k; v]. *)
let rows_at p x =
  Nx.take_along_axis ~axis:1 (Nx.reshape [| Nx.dim 0 p; Nx.dim 1 p; 1 |] p) x

let upstream =
  group "upstream's network"
    [
      test "encode gives upstream's context, lengths counting present tokens"
        (fun () ->
          let e =
            Transformer.encode (Fixture.transformer ()) (Fixture.inputs ms)
          in
          Spread.holds "context" (check golden "embed" e.context);
          equal (array int) [| 57; 2; 46; 56 |] (ints e.lengths));
      test "forward gives upstream's logits with training's mask" (fun () ->
          match
            Transformer.forward (Fixture.transformer ())
              (Fixture.inputs ~rows:[ 1; 2 ] ms)
              [ whole (Scalar Z); whole (Scalar Xp_rp) ]
          with
          | [ z; xp ] ->
              Spread.holds "tok_z" (check golden "logits.tok_z" z);
              Spread.holds "tok_xp_rp"
                (check golden "logits.tok_xp_rp" (second xp))
          | _ -> fail "two targets, two logits");
      cases "a target at chosen positions gives upstream's step" ~id:Fun.id
        steps (fun name ->
          Spread.holds name (check roar name (decode_at name (Some (at name))));
          (* Listing order does not change a prediction. *)
          let rev p = Nx.flip ~axes:[ 1 ] p in
          Spread.holds name
            (check roar name (rev (decode_at name (Some (rev (at name))))));
          (* A whole target's decoder holds 48 tokens more, which its rows
             see. *)
          let c = check roar name (rows_at (at name) (decode_at name None)) in
          greater float_exact ~than:1. c.worst);
      test "all positions in order are the whole target" (fun () ->
          let all = Nx.broadcast_to [| 4; 55 |] (Nx.arange Nx.int64 0 55 1) in
          equal (array float_exact)
            (floats (decode_at "roar.1" None))
            (floats (decode_at "roar.1" (Some all))));
    ]

let laws =
  group "laws"
    [
      test "an absent token's value is not read" (fun () ->
          let t = Fixture.transformer () in
          let scramble (i : _ Tokens.t) =
            let wild = Nx.full_like i.tokens (-7L) in
            { i with tokens = Nx.where i.present i.tokens wild }
          in
          let a = Transformer.encode t (Fixture.inputs ms) in
          let b =
            Transformer.encode t (List.map scramble (Fixture.inputs ms))
          in
          equal (array float_exact) (floats a.context) (floats b.context));
      test "an embedding is the mean of the context before the length"
        (fun () ->
          let e =
            Transformer.encode (Fixture.transformer ()) (Fixture.inputs ms)
          in
          let c = floats e.context and l = ints e.lengths in
          let n = Nx.dim 1 e.context and w = Nx.dim 2 e.context in
          if not (Array.exists (fun x -> x < n) l) then
            fail "no object with an absent token";
          let mean x =
            let i = x / w and s = ref 0. in
            for r = 0 to l.(i) - 1 do
              s := !s +. c.((((i * n) + r) * w) + (x mod w))
            done;
            !s /. float_of_int l.(i)
          in
          equal
            (array (float_rel ~rel:1e-5 ~abs:1e-6))
            (Array.init (Array.length l * w) mean)
            (floats (Transformer.embedding e)));
      test "a target's logits do not depend on the other targets" (fun () ->
          let t = Fixture.transformer ()
          and x = Fixture.inputs ~rows:[ 1; 2 ] ms in
          match
            ( Transformer.forward t x [ whole (Scalar Z); whole (Scalar Xp_rp) ],
              Transformer.forward t x [ whole (Scalar Xp_rp) ] )
          with
          | [ _; both ], [ alone ] ->
              Compiled.close "tok_xp_rp beside tok_z"
                ~spread:(Fixture.get golden Nx.float32 "s.logits.tok_xp_rp")
                (second both) (second alone)
          | _ -> fail "one logits per target");
    ]

(* Compiled, one set of entry points per device set. *)

let on_host = Compiled.transformer ()

(* A set over a software device, without eager kernels: it computes only under
   Rune.jit, on this machine's CPU. *)
module Soft =
  (val Nx.devices [ Result.get_ok (Rig.software "aion-transformer") ])

let on_soft = Compiled.transformer ()

(* Batches of one object and of four, drawn with repeats, so that each entry
   point traces two programs at each placement whatever the seed. A compiled
   transformer on the software set also states that place moves all of it. *)
let compiled_law =
  prop "compiled computes as eager, within 2 τ ŝ, on the host or another set"
    Gen.(
      let+ rows = list ~size:(of_list [ 1; 4 ]) (int_range 0 3)
      and+ step = of_list steps
      and+ perm = permutation (List.init 7 Fun.id)
      and+ zs = list ~size:(of_list [ 1; 2 ]) (int_range 1 2)
      and+ soft = bool in
      (rows, step, perm, zs, soft))
    (fun (rows, step, perm, zs, soft) ->
      cover "one object" (List.length rows = 1);
      cover "a repeated object"
        (List.length (List.sort_uniq compare rows) < List.length rows);
      cover "on the host" (not soft);
      cover "on a set without eager kernels" soft;
      if soft then Compiled.agrees Soft.on on_soft ~rows ~step ~perm ~zs
      else Compiled.agrees Nx.Host.on on_host ~rows ~step ~perm ~zs)

(* Refusals, each input failing one check. *)

let tokens ?(present = true) m shape k : _ Tokens.t =
  {
    modality = m;
    tokens = Nx.full Nx.int64 shape (Int64.of_int k);
    present = Nx.full Nx.bool shape present;
  }

let one m k = tokens m [| 1; 1 |] k

let later_wait =
  test "on a set without eager kernels a refusal raises at a later wait"
    (fun () ->
      let t =
        Nx.Ptree.place (module Transformer) Soft.on (Fixture.transformer ())
      in
      let bad =
        Nx.Ptree.place_fixed Tokens.walk Soft.on (one (Scalar Z) 1025)
      in
      let l = on_soft.forward t [ bad ] [ whole (Scalar Z) ] in
      raises_match (Exn.invalid_arg ~sub:"outside [0, 1025)") (fun () ->
          Nx.place Nx.Host.on (List.hd l)))

let matrix rows =
  Nx.create Nx.int64
    [| Array.length rows; Array.length rows.(0) |]
    (Array.concat (Array.to_list rows))

(* On the host a compiled call refuses at the call, as eagerly: a shape at the
   trace, a value by Nx.check. *)
let refusals =
  let t () = Fixture.transformer () in
  let encode x (c : _ Compiled.transformer) = ignore (c.encode (t ()) x) in
  let forward x ts (c : _ Compiled.transformer) =
    ignore (c.forward (t ()) x ts)
  in
  let decode shape lengths (c : _ Compiled.transformer) =
    let e : _ Transformer.encoding =
      {
        context = Nx.zeros Nx.float32 shape;
        lengths = Nx.create Nx.int64 [| Array.length lengths |] lengths;
      }
    in
    ignore (c.decode (t ()) e [ whole (Scalar Z) ])
  in
  let positions rows =
    forward (Fixture.inputs ~step:"roar.1" ms) [ xp_rp (Some (matrix rows)) ]
  in
  cases "refuses, eagerly and compiled"
    ~id:(fun (name, _, _) -> name)
    [
      ( "a token past the vocabulary",
        "outside [0, 1025)",
        encode [ one (Scalar Z) 1025 ] );
      ( "a negative token",
        "outside [0, 1024)",
        encode [ one (Scalar Flux_g) (-1) ] );
      ( "an object with no present token",
        "no present token",
        encode [ tokens ~present:false (Scalar Z) [| 1; 1 |] 3 ] );
      ( "a modality the weights lack",
        "no weights for tok_ebv",
        encode [ one (Scalar Ebv) 0 ] );
      ( "an input given twice",
        "given twice",
        encode [ one (Scalar Z) 0; one (Scalar Z) 1 ] );
      ( "XP tokens of [1; 54]",
        "[b; 55] expected",
        encode [ tokens (Scalar Xp_bp) [| 1; 54 |] 0 ] );
      ( "tokens of [1]",
        "[b; 1] expected",
        encode [ tokens (Scalar Z) [| 1 |] 0 ] );
      ( "tokens of [1; 1; 1]",
        "[b; 1] expected",
        encode [ tokens (Scalar Z) [| 1; 1; 1 |] 0 ] );
      ( "tokens of [1; 2]",
        "[b; 1] expected",
        encode [ tokens (Scalar Z) [| 1; 2 |] 0 ] );
      ( "presence of another shape",
        "presence of shape [1], [1; 1] expected",
        encode [ { (one (Scalar Z) 0) with present = Nx.ones Nx.bool [| 1 |] } ]
      );
      ( "inputs of different batches",
        "objects",
        encode [ one (Scalar Z) 0; tokens (Scalar Flux_g) [| 2; 1 |] 0 ] );
      ("no target", "no target", forward [ one (Scalar Z) 0 ] []);
      ("a context of another width", "decode:", decode [| 1; 3; 17 |] [| 2L |]);
      ("a context of two axes", "decode:", decode [| 3; 16 |] [| 2L |]);
      ("a context of no axis", "decode:", decode [||] [| 2L |]);
      ("lengths of another batch", "decode:", decode [| 1; 3; 16 |] [| 2L; 2L |]);
      ("a length of 0", "decode:", decode [| 2; 3; 16 |] [| 2L; 0L |]);
      ( "a length past the row",
        "a length outside [1, 3]",
        decode [| 2; 3; 16 |] [| 4L; 2L |] );
      ( "a repeated position",
        "tok_xp_rp position 9 at [2; 1] repeats within its row",
        positions
          [|
            [| 3L; 4L; 5L |];
            [| 3L; 4L; 5L |];
            [| 3L; 9L; 9L |];
            [| 3L; 4L; 5L |];
          |] );
      ( "a position past the target",
        "tok_xp_rp position 55 at [2; 0] outside [0, 55)",
        positions [| [| 0L |]; [| 1L |]; [| 55L |]; [| 2L |] |] );
      ( "a negative position",
        "tok_xp_rp position -1 at [2; 0] outside [0, 55)",
        positions [| [| 0L |]; [| 1L |]; [| -1L |]; [| 2L |] |] );
      ("no position", "k >= 1", positions [| [||]; [||]; [||]; [||] |]);
      ( "positions of another batch",
        "[4; k >= 1]",
        positions [| [| 0L |]; [| 1L |] |] );
    ]
    (fun (_, sub, f) ->
      raises_match (Exn.invalid_arg ~sub) (fun () ->
          f Transformer.{ encode; decode; forward; embedding });
      raises_match (Exn.invalid_arg ~sub) (fun () -> f on_host))

(* of_archive refuses a file it would read wrongly: the fixture with [name]
   dropped, or replaced by [p]. *)
let replace name p =
  Nx_io.Archive.of_list
    (List.filter_map
       (fun k ->
         if k = name then None
         else Option.map (fun v -> (k, v)) (Nx_io.Archive.find k weights))
       (Nx_io.Archive.names weights)
    @ Option.to_list (Option.map (fun p -> (name, p)) p))

let loading =
  let f64 name =
    let w = Fixture.get weights Nx.float32 name in
    (name ^ " as float64", name, name, Some (Nx.P (Nx.cast Nx.float64 w)))
  in
  group "of_archive"
    [
      test "reads the fixture's modalities" (fun () ->
          equal (list string)
            (List.map Modality.key
               Modality.[ Scalar Flux_g; Scalar Z; Scalar Xp_bp; Scalar Xp_rp ])
            (List.map Modality.key
               (Transformer.modalities (Fixture.transformer ()))));
      cases "refuses a file it would read wrongly"
        ~id:(fun (id, _, _, _) -> id)
        [
          ( "a tensor it does not read",
            "encoder_embeddings.tok_z.mod_emb",
            "encoder_embeddings.tok_z.mod_emb: an entry in the archive, never \
             read",
            Some (Nx.P (Nx.zeros Nx.float32 [| 1; 1; 16 |])) );
          ( "a missing weight",
            "decoder.1.cross_attn.kv.weight",
            "decoder.1.cross_attn.kv.weight",
            None );
          ( "a weight of another shape",
            "decoder_proj_context.bias",
            "decoder_proj_context.bias: shape [17] in the archive",
            Some (Nx.P (Nx.zeros Nx.float32 [| 17 |])) );
          f64 "encoder_norm.weight";
          f64 "encoder.0.attn.q_norm.weight";
          f64 "encoder.0.mlp.fc1.weight";
          f64 "decoder.1.norm2.weight";
        ]
        (fun (_, name, sub, p) ->
          raises_match (Exn.failure ~sub) (fun () ->
              Transformer.of_archive (replace name p)));
    ]

(* Transformers of [init], architectures drawn small, no block included. *)

let arch_gen =
  Gen.(
    let* heads = int_range 1 3 in
    let* head = int_range 1 3 in
    let* blocks = pair (int_range 0 2) (int_range 0 2) in
    let* hidden = int_range 1 8 in
    let+ modalities =
      list ~size:(int_range 1 3)
        (of_list Modality.[ Scalar Z; Scalar Flux_g; Scalar Xp_bp ])
    in
    let modalities = List.sort_uniq compare modalities in
    { Transformer.width = heads * head; heads; blocks; hidden; modalities })

let initialised =
  let a : Transformer.arch =
    {
      width = 4;
      heads = 1;
      blocks = (1, 1);
      hidden = 1;
      modalities = [ Scalar Z ];
    }
  in
  group "init"
    [
      prop "computes the shapes the interface states"
        Gen.(pair arch_gen (int_range 1 3))
        (fun (a, b) ->
          cover "no block" (a.blocks = (0, 0));
          cover "several modalities" (List.length a.modalities > 1);
          let t = Transformer.init ~key:(Nx.Rng.key 3) a in
          equal (list string)
            (List.map Modality.key a.modalities)
            (List.map Modality.key (Transformer.modalities t));
          let x =
            List.map
              (fun m -> tokens m [| b; Modality.tokens m |] 0)
              a.modalities
          in
          let n =
            List.fold_left (fun n m -> n + Modality.tokens m) 0 a.modalities
          in
          let e = Transformer.encode t x in
          equal (array int) [| b; n; a.width |] (Nx.shape e.context);
          equal (array int) [| b; a.width |]
            (Nx.shape (Transformer.embedding e));
          let m = List.hd a.modalities in
          equal
            (list (array int))
            [ [| b; Modality.tokens m; Modality.vocab m |] ]
            (List.map Nx.shape (Transformer.decode t e [ whole m ])));
      cases "refuses a malformed architecture"
        ~id:(fun (name, _, _) -> name)
        [
          ("no width", { a with width = 0 }, "not positive");
          ( "heads not dividing the width",
            { a with heads = 3 },
            "3 heads do not divide the width 4" );
          ("negative blocks", { a with blocks = (1, -1) }, "blocks negative");
          ("no modality", { a with modalities = [] }, "no modality");
          ( "a modality twice",
            { a with modalities = [ Scalar Z; Scalar Flux_g; Scalar Z ] },
            "modality tok_z given twice" );
        ]
        (fun (_, a, sub) ->
          raises_match (Exn.invalid_arg ~sub) (fun () ->
              Transformer.init ~key:(Nx.Rng.key 0) a));
    ]

let () =
  exit
    (run "aion.transformer"
       [
         upstream;
         laws;
         group "compiled" [ compiled_law; later_wait ];
         refusals;
         loading;
         initialised;
       ])
