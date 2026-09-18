(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Import validation for Llama 3.2 1B: the pretrained weights, loaded through
   [Llama.of_hf], must reproduce the logits of the reference implementation.

   [fixture.json] holds the reference values, recorded at float32 by
   [reference.py] from the transformers implementation on the same weight file
   (its sha256 is in the fixture). This program compares, on the fixture's token
   ids:

   - the argmax of every position and the eight largest logits of the last; -
   the residual stream after blocks 1 and 8 and after the final norm, so a
   disagreement is located and not only detected; - the cached path, fed in
   chunks, against the whole-sequence one: the decode contract's law, on real
   weights.

   Usage: validate.exe [--weights model.safetensors --config config.json]
   Without arguments the files are fetched from [Llama.default_repo] (2.5 GB,
   cached). Not part of the test suite: it needs the download. *)

open Kaun

let json_of_file path =
  let ic = open_in_bin path in
  let s =
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () -> In_channel.input_all ic)
  in
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok j -> j
  | Error e -> failwith (path ^ ": " ^ e)

let mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> failwith ("fixture: missing " ^ name))
  | _ -> failwith "fixture: not an object"

let floats = function
  | Jsont.Array (l, _) ->
      Array.of_list
        (List.map
           (function
             | Jsont.Number (f, _) -> f
             | _ -> failwith "fixture: number expected")
           l)
  | _ -> failwith "fixture: array expected"

let ints j = Array.map int_of_float (floats j)
let failures = ref 0

let check name ok detail =
  Printf.printf "%-52s %s%s\n%!" name (if ok then "ok" else "FAIL") detail;
  if not ok then incr failures

(* Float32 on both sides, different kernels: agreement to one part in ten
   thousand of the vector's largest magnitude. Observed: a few parts in a
   million. A NaN anywhere fails. *)
let close name expected actual =
  if Array.length expected <> Array.length actual then
    check name false
      (Printf.sprintf "  (%d values, expected %d)" (Array.length actual)
         (Array.length expected))
  else begin
    let scale =
      Array.fold_left (fun m e -> Float.max m (Float.abs e)) 1e-30 expected
    in
    let worst = ref 0.0 in
    Array.iteri
      (fun i e ->
        let d = Float.abs (e -. actual.(i)) /. scale in
        if not (d <= !worst) then worst := d)
      expected;
    check name (!worst < 1e-4)
      (Printf.sprintf "  (worst relative error %.1e)" !worst)
  end

let () =
  let weights = ref "" and config = ref "" and fixture = ref "fixture.json" in
  Arg.parse
    [
      ("--weights", Arg.Set_string weights, "model.safetensors");
      ("--config", Arg.Set_string config, "config.json");
      ( "--fixture",
        Arg.Set_string fixture,
        "fixture.json (default: ./fixture.json)" );
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "validate.exe [--weights FILE --config FILE] [--fixture FILE]";
  let fx = json_of_file !fixture in
  (match mem "weights_sha256" fx with
  | Jsont.String (h, _) ->
      Printf.printf "reference recorded from sha256 %s\n%!" h
  | _ -> ());
  let cfg, p =
    if !weights = "" then Llama.from_pretrained ()
    else
      let cfg = Llama.config_of_json (json_of_file !config) in
      (cfg, Llama.from_file cfg !weights)
  in
  let tokens = ints (mem "ids" fx) in
  let n = Array.length tokens in
  let ids = Nx.create Nx.int32 [| 1; n |] (Array.map Int32.of_int tokens) in
  let h = Llama.hidden cfg p ids in
  let logits = Llama.logits cfg p h in
  let argmax = Nx.to_array (Nx.argmax ~axis:2 logits) in
  check "argmax of every position"
    (Array.map Int32.to_int argmax = ints (mem "argmax_per_position" fx))
    "";
  let last = Nx.slice [ I 0; I (n - 1) ] logits in
  let at i = Nx.item [ i ] last in
  let top_ids = ints (mem "last_top_ids" fx) in
  close "the eight largest logits of the last position"
    (floats (mem "last_top_values" fx))
    (Array.map at top_ids);
  close "the first eight logits of the last position"
    (floats (mem "last_first_values" fx))
    (Array.init 8 at);
  (* The residual stream after [k] blocks is the model truncated to [k]. *)
  let stream k =
    let sub =
      { p with Llama.blocks = List.filteri (fun i _ -> i < k) p.blocks }
    in
    let h = Llama.hidden { cfg with n_layers = k } sub ids in
    (* The reference reports its last hidden state after the final norm. *)
    let h =
      if k = cfg.n_layers then Rms_norm.apply ~eps:cfg.norm_eps p.norm h else h
    in
    Nx.to_array (Nx.slice [ I 0; I (n - 1); R (0, 8) ] h)
  in
  List.iter
    (fun k ->
      close
        (Printf.sprintf "residual stream after block %d" k)
        (floats (mem (string_of_int k) (mem "hidden" fx)))
        (stream k))
    [ 1; 8; cfg.n_layers ];
  (* The decode contract on real weights: chunks through the caches. *)
  let slots = Nx.create Nx.int32 [| 1; n |] (Array.init n Int32.of_int) in
  let _, hs, _ =
    List.fold_left
      (fun (at, hs, caches) len ->
        let pos =
          Nx.create Nx.int32 [| 1; len |]
            (Array.init len (fun i -> Int32.of_int (at + i)))
        in
        let h, caches =
          Llama.cached cfg p caches
            (Attention.Span.make ~pos ~slots)
            (Nx.slice [ A; R (at, at + len) ] ids)
        in
        (at + len, h :: hs, caches))
      (0, [], Llama.cache cfg ~slots:n Nx.float32)
      [ 1; 7; n - 8 ]
  in
  let chunked = Llama.logits cfg p (Nx.concatenate ~axis:1 (List.rev hs)) in
  let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t)) in
  close "cached, in chunks of 1, 7 and the rest, every position" (flat logits)
    (flat chunked);
  if !failures > 0 then exit 1
