(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The transformer against a narrow upstream AION with random weights
   (upstream/transformer.safetensors), of aion-base's structure at width 16, 2
   heads, 2 encoder and 2 decoder blocks, over tok_xp_bp, tok_flux_g, tok_z and
   tok_xp_rp. Its weights make each constant matter: embeddings and query-key
   projections near 1e-3, so the first layer norms and the query-key norms see a
   variance near their eps; a context projection with a bias; norm scales away
   from 1. The inputs leave tokens absent, so the order of the encoder's row and
   the lengths matter, and the forward pass has two targets, so the decoder's
   mask matters.

   upstream/transformer_golden.safetensors holds the inputs, upstream's float64
   encoding and logits (the decoder with the mask of training; tok_xp_rp's for
   the second object alone), and the spread [s] of their rounding ensemble at
   float32's scale. Agreement with upstream is reported as the largest [|x -
   reference| / s], and not yet checked within [Spread]'s τ. *)

open Windtrap
open Aion

let weights = Nx_io.load_safetensors "upstream/transformer.safetensors"
let golden = Nx_io.load_safetensors "upstream/transformer_golden.safetensors"
let model = lazy (Transformer.of_archive weights)

let get dt name =
  match Nx_io.Archive.find name golden with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no entry " ^ name)

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let reference name = floats (get Nx.float64 ("f64." ^ name))
let spread name = floats (get Nx.float32 ("s." ^ name))

(* The largest distance of [x] from [r] in units of [s]. *)
let worst ~r ~s x =
  let w = ref 0. in
  Array.iteri (fun i v -> w := Float.max !w (Float.abs (v -. r.(i)) /. s.(i))) x;
  !w

let report name x =
  Printf.printf "%s: largest |x - reference| / s %.3g\n%!" name
    (worst ~r:(reference name) ~s:(spread name) (floats x))

(* Object 2's rows of logits of objects 1 and 2. *)
let second x = Nx.slice [ R (1, 2) ] x
let inputs_of = Modality.[ Scalar Xp_bp; Scalar Flux_g; Scalar Z ]

(* The fixture's inputs, of the objects [rows]. *)
let inputs ?(rows = [ 0; 1; 2; 3 ]) () =
  let take x =
    Nx.take ~axis:0
      (Nx.create Nx.int64
         [| List.length rows |]
         (Array.of_list (List.map Int64.of_int rows)))
      x
  in
  List.map
    (fun m ->
      let key = Modality.key m in
      let absent =
        Nx.not_equal_s (get Nx.uint8 ("input." ^ key ^ ".absent")) 0
      in
      Transformer.
        {
          modality = m;
          tokens = take (get Nx.int64 ("input." ^ key ^ ".tokens"));
          present = Some (take (Nx.logical_not absent));
        })
    inputs_of

let whole m : _ Transformer.target = { modality = m; positions = None }
let targets = Modality.[ whole (Scalar Z); whole (Scalar Xp_rp) ]
let ints x = Array.map Int64.to_int (Nx.to_array x)

let upstream =
  group "upstream's network"
    [
      test "encode gives upstream's context" (fun () ->
          let e = Transformer.encode (Lazy.force model) (inputs ()) in
          report "embed" e.context);
      test "lengths count the present tokens" (fun () ->
          let e = Transformer.encode (Lazy.force model) (inputs ()) in
          equal (array int) [| 57; 2; 46; 56 |] (ints e.lengths));
      test "forward gives upstream's logits with training's mask" (fun () ->
          match
            Transformer.forward (Lazy.force model)
              (inputs ~rows:[ 1; 2 ] ())
              targets
          with
          | [ z; xp ] ->
              report "logits.tok_z" z;
              report "logits.tok_xp_rp" (second xp)
          | _ -> failf "two targets, two logits");
    ]

(* Laws the interface states. *)

let laws =
  group "laws"
    [
      test "an absent token's value is not read" (fun () ->
          let t = Lazy.force model in
          let scramble (i : _ Transformer.input) =
            match i.present with
            | None -> i
            | Some p ->
                let wild = Nx.full_like i.tokens (-7L) in
                { i with tokens = Nx.where p i.tokens wild }
          in
          let a = Transformer.encode t (inputs ()) in
          let b = Transformer.encode t (List.map scramble (inputs ())) in
          equal (array float_exact) (floats a.context) (floats b.context));
      test "an embedding is the mean of the context before the length"
        (fun () ->
          let e = Transformer.encode (Lazy.force model) (inputs ()) in
          let c = floats e.context and l = ints e.lengths in
          let n = Nx.dim 1 e.context and w = Nx.dim 2 e.context in
          let mean i k =
            let s = ref 0. in
            for r = 0 to l.(i) - 1 do
              s := !s +. c.((((i * n) + r) * w) + k)
            done;
            !s /. float_of_int l.(i)
          in
          let want =
            Array.init (Array.length l * w) (fun x -> mean (x / w) (x mod w))
          in
          if not (Array.exists (fun x -> x < n) l) then
            fail "no object with an absent token";
          equal
            (array (float_rel ~rel:1e-5 ~abs:1e-6))
            want
            (floats (Transformer.embedding e)));
      test "a target's logits do not depend on the other targets" (fun () ->
          let t = Lazy.force model in
          let x = inputs ~rows:[ 1; 2 ] () in
          match
            ( Transformer.forward t x targets,
              Transformer.forward t x [ whole (Scalar Xp_rp) ] )
          with
          | [ _; both ], [ alone ] ->
              (* Both are float32 computations of one exact value, each within τ
                 s of it, so they differ by at most 2 τ s; reported until τ is
                 in the test support, as above. *)
              let s = spread "logits.tok_xp_rp" in
              Printf.printf
                "tok_xp_rp with and without tok_z: largest difference / s %.3g\n\
                 %!"
                (worst ~r:(floats (second alone)) ~s (floats (second both)))
          | _ -> failf "one logits per target");
    ]

(* Loading refuses a file it would read wrongly. *)

let with_entry name p =
  Nx_io.Archive.union
    [
      Nx_io.Archive.of_list
        (List.filter_map
           (fun k ->
             if k = name then None
             else Option.map (fun v -> (k, v)) (Nx_io.Archive.find k weights))
           (Nx_io.Archive.names weights));
      Nx_io.Archive.of_list [ (name, p) ];
    ]

let without name =
  Nx_io.Archive.of_list
    (List.filter_map
       (fun k ->
         if k = name then None
         else Option.map (fun v -> (k, v)) (Nx_io.Archive.find k weights))
       (Nx_io.Archive.names weights))

let loading =
  let refuses sub a =
    raises_match (Exn.failure ~sub) (fun () -> Transformer.of_archive a)
  in
  group "loading"
    [
      test "reads the fixture's modalities" (fun () ->
          equal (list string)
            (List.map Modality.key
               Modality.[ Scalar Flux_g; Scalar Z; Scalar Xp_bp; Scalar Xp_rp ])
            (List.map Modality.key (Transformer.modalities (Lazy.force model))));
      test "refuses a tensor it does not read" (fun () ->
          refuses "encoder_embeddings.tok_z.mod_emb is not read"
            (with_entry "encoder_embeddings.tok_z.mod_emb"
               (Nx.P (Nx.zeros Nx.float32 [| 1; 1; 16 |]))));
      test "refuses a layer norm's nonzero shift" (fun () ->
          refuses "encoder.1.norm2.bias is not zero"
            (with_entry "encoder.1.norm2.bias"
               (Nx.P (Nx.full Nx.float32 [| 16 |] 1e-3))));
      test "refuses a missing weight" (fun () ->
          refuses "no tensor decoder.1.cross_attn.kv.weight"
            (without "decoder.1.cross_attn.kv.weight"));
      test "refuses position embeddings that differ" (fun () ->
          refuses "pos_emb differ"
            (with_entry "decoder_embeddings.tok_xp_bp.pos_emb"
               (Nx.P (Nx.zeros Nx.float32 [| 1; 55; 16 |]))));
      cases "refuses a weight that is not float32" ~id:Fun.id
        [
          "encoder_norm.weight";
          "encoder.0.attn.q_norm.weight";
          "encoder.0.mlp.fc1.weight";
          "decoder.1.norm2.weight";
        ] (fun name ->
          let w =
            Nx.unpack Nx.float32 (Option.get (Nx_io.Archive.find name weights))
          in
          refuses (name ^ " is not float32")
            (with_entry name (Nx.P (Nx.cast Nx.float64 w))));
      test "refuses a weight of another shape" (fun () ->
          refuses "decoder_proj_context.bias has shape"
            (with_entry "decoder_proj_context.bias"
               (Nx.P (Nx.zeros Nx.float32 [| 17 |]))));
    ]

let refusals =
  let t () = Lazy.force model in
  let one m k : _ Transformer.input =
    {
      modality = m;
      tokens = Nx.create Nx.int64 [| 1 |] [| Int64.of_int k |];
      present = None;
    }
  in
  let refuses sub f = raises_match (Exn.invalid_arg ~sub) f in
  group "refusals"
    [
      test "a present token outside the vocabulary" (fun () ->
          refuses "outside [0, 1025)" (fun () ->
              Transformer.encode (t ()) [ one (Scalar Z) 1025 ]));
      test "a negative present token" (fun () ->
          refuses "outside [0, 1024)" (fun () ->
              Transformer.encode (t ()) [ one (Scalar Flux_g) (-1) ]));
      test "an object with no present token" (fun () ->
          refuses "no present token" (fun () ->
              Transformer.encode (t ())
                [
                  {
                    (one (Scalar Z) 3) with
                    present = Some (Nx.create Nx.bool [| 1 |] [| false |]);
                  };
                ]));
      test "a modality the weights lack" (fun () ->
          refuses "no weights for tok_ebv" (fun () ->
              Transformer.encode (t ()) [ one (Scalar Ebv) 0 ]));
      test "an input given twice" (fun () ->
          refuses "given twice" (fun () ->
              Transformer.encode (t ()) [ one (Scalar Z) 0; one (Scalar Z) 1 ]));
      test "tokens of another shape" (fun () ->
          refuses "[b; 55] expected" (fun () ->
              Transformer.encode (t ())
                [
                  {
                    modality = Scalar Xp_bp;
                    tokens = Nx.zeros Nx.int64 [| 1; 54 |];
                    present = None;
                  };
                ]));
      test "inputs of different batches" (fun () ->
          refuses "objects" (fun () ->
              Transformer.encode (t ())
                [
                  one (Scalar Z) 0;
                  {
                    (one (Scalar Flux_g) 0) with
                    tokens = Nx.zeros Nx.int64 [| 2 |];
                  };
                ]));
      cases "decode: a malformed encoding"
        ~id:(fun (name, _, _) -> name)
        [
          ("context of another width", [| 1; 3; 17 |], [| 2L |]);
          ("context of two axes", [| 3; 16 |], [| 2L |]);
          ("lengths of another batch", [| 1; 3; 16 |], [| 2L; 2L |]);
          ("a length of 0", [| 2; 3; 16 |], [| 2L; 0L |]);
          ("a length past the row", [| 2; 3; 16 |], [| 4L; 2L |]);
        ]
        (fun (_, shape, lengths) ->
          let e : _ Transformer.encoding =
            {
              context = Nx.zeros Nx.float32 shape;
              lengths = Nx.create Nx.int64 [| Array.length lengths |] lengths;
            }
          in
          refuses "decode:" (fun () ->
              Transformer.decode (t ()) e [ whole (Scalar Z) ]));
      test "no target" (fun () ->
          refuses "no target" (fun () ->
              Transformer.forward (t ()) [ one (Scalar Z) 0 ] []));
    ]

(* Targets at chosen positions, against upstream's random-order step
   (upstream/roar_golden.safetensors): the fixture's inputs and tok_xp_rp as the
   target, revealed at some positions, with the decoder holding [k] = 7 of the
   others. [roar.0] is the first step of upstream's linear schedule, nothing
   revealed and one order for the batch; [roar.1] a later one, each object with
   its own 28 revealed tokens and 7 positions. Each holds upstream's float64
   logits and their rounding ensemble's spread at float32's scale. *)

let roar = Nx_io.load_safetensors "upstream/roar_golden.safetensors"

let rget dt name =
  match Nx_io.Archive.find name roar with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no entry " ^ name)

let ensemble () = Int64.to_int (Nx.item [ 0 ] (rget Nx.int64 "n"))
let steps = [ "roar.0"; "roar.1" ]
let at name = rget Nx.int64 (name ^ ".positions")

(* The fixture's inputs, and the target's tokens present where revealed. *)
let step_inputs name =
  let revealed = Nx.not_equal_s (rget Nx.uint8 (name ^ ".revealed")) 0 in
  inputs ()
  @ [
      Transformer.
        {
          modality = Scalar Xp_rp;
          tokens = rget Nx.int64 (name ^ ".tokens");
          present = Some revealed;
        };
    ]

let decode_at name p =
  let target : _ Transformer.target =
    { modality = Scalar Xp_rp; positions = p }
  in
  List.hd (Transformer.forward (Lazy.force model) (step_inputs name) [ target ])

let against name x =
  Spread.values ~n:(ensemble ())
    ~reference:(floats (rget Nx.float64 ("f64." ^ name)))
    ~spread:(floats (rget Nx.float32 ("s." ^ name)))
    (floats x)

let holds name (c : Spread.check) =
  if c.failures <> [] then failf "%a" (Spread.pp name) c

(* [x]'s rows of each object at its positions [p], [b; k; v]. *)
let rows_at p x =
  let b = Nx.dim 0 p and k = Nx.dim 1 p in
  Nx.take_along_axis ~axis:1 (Nx.reshape [| b; k; 1 |] p) x

let reversed p = Nx.flip ~axes:[ 1 ] p

let positions =
  let id = Fun.id in
  group "positions"
    [
      cases "a target at chosen positions gives upstream's step" ~id steps
        (fun name ->
          holds name (against name (decode_at name (Some (at name)))));
      cases "a whole target's rows are not the step" ~id steps (fun name ->
          (* The decoder then holds every position, so the step's rows see 48
             tokens more: their logits fall outside the step's spread. *)
          let c = against name (rows_at (at name) (decode_at name None)) in
          greater float_exact ~than:1. c.worst);
      cases "listing order does not change a prediction" ~id steps (fun name ->
          let p = reversed (at name) in
          holds name (against name (reversed (decode_at name (Some p)))));
      test "all positions in order are the whole target" (fun () ->
          let name = "roar.1" in
          let all = Nx.broadcast_to [| 4; 55 |] (Nx.arange Nx.int64 0 55 1) in
          equal (array float_exact)
            (floats (decode_at name None))
            (floats (decode_at name (Some all))));
      cases "refuses malformed positions"
        ~id:(fun (what, _, _) -> what)
        [
          (* Rows of one position but row 2's, which is [bad]. *)
          ( "a repeated position",
            "tok_xp_rp position 9 at [2; 1] repeats within its row",
            [|
              [| 3L; 4L; 5L |];
              [| 3L; 4L; 5L |];
              [| 3L; 9L; 9L |];
              [| 3L; 4L; 5L |];
            |] );
          ( "a position past the target",
            "tok_xp_rp position 55 at [2; 0] outside [0, 55)",
            [| [| 0L |]; [| 1L |]; [| 55L |]; [| 2L |] |] );
          ( "a negative position",
            "tok_xp_rp position -1 at [2; 0] outside [0, 55)",
            [| [| 0L |]; [| 1L |]; [| -1L |]; [| 2L |] |] );
          ("no position", "k >= 1", [| [||]; [||]; [||]; [||] |]);
          ("positions of another batch", "[4; k >= 1]", [| [| 0L |]; [| 1L |] |]);
        ]
        (fun (_, sub, rows) ->
          let k = Array.length rows.(0) in
          let p =
            Nx.create Nx.int64
              [| Array.length rows; k |]
              (Array.concat (Array.to_list rows))
          in
          raises_match (Exn.invalid_arg ~sub) (fun () ->
              decode_at "roar.1" (Some p)));
    ]

let () =
  exit (run "aion.transformer" [ upstream; laws; positions; loading; refusals ])
