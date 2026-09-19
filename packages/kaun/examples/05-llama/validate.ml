(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Import validation: pretrained weights, loaded through [Llama.of_hf], must
   reproduce the reference implementation's logits.

   A fixture under [fixtures/] holds reference values recorded at float32 by
   [reference.py] from the transformers implementation, and names the sha256 of
   the weight file they were recorded from (printed here, not verified). On the
   fixture's token ids this program compares:

   - the argmax of every position and the largest logits of the last; - the
   residual stream after every block and after the final norm, so a disagreement
   is located and not only detected; - the cached path fed in chunks against the
   whole-sequence one: the decode contract's law, on real weights; - a ragged
   batch of two prompts through the caches against each prompt's own reference.

   Two fixtures ship: Llama 3.2 1B (tied head, the Llama 3 rotary schedule) and
   TinyLlama 1.1B (untied head, the standard schedule).

   [--dtype bfloat16] or [float16] runs the model at half precision against the
   same float32 reference, to the tolerance half precision allows. [--jit
   DEVICE] compiles the forward passes; the per-block streams are then checked
   at the last block only, each block being its own compilation.

   Usage: validate.exe [--fixture FILE] [--weights FILE --config FILE] [--dtype
   DT] [--jit DEVICE] Without [--weights] the files come from the fixture's
   repository (2 to 3 GB, cached). Not part of the test suite: it needs the
   download. *)

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

let string = function
  | Jsont.String (s, _) -> s
  | _ -> failwith "fixture: string expected"

let failures = ref 0

let check name ok detail =
  Printf.printf "%-58s %s%s\n%!" name (if ok then "ok" else "FAIL") detail;
  if not ok then incr failures

(* Agreement relative to the vector's largest magnitude. A NaN anywhere
   fails. *)
let close ~tol name expected actual =
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
    check name (!worst < tol)
      (Printf.sprintf "  (worst relative error %.1e)" !worst)
  end

let ids_tensor rows =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  Nx.create Nx.int32 [| batch; seq |]
    (Array.map Int32.of_int (Array.concat (Array.to_list rows)))

let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let validate (type b) ~device ~tol ~exact fx cfg
    (p : (float, b) Nx.t Llama.params) (dt : (float, b) Nx.dtype) =
  let compiled f x =
    match device with None -> f x | Some device -> Rune.jit' ~device f x
  in
  let to32 t = Nx.cast Nx.float32 t in
  let tokens = ints (mem "ids" fx) in
  let n = Array.length tokens in
  let ids = ids_tensor [| tokens |] in
  let logits =
    compiled (fun ids -> to32 (Llama.logits cfg p (Llama.hidden cfg p ids))) ids
  in
  let argmax =
    Array.map Int32.to_int (Nx.to_array (Nx.argmax ~axis:2 logits))
  in
  let reference = ints (mem "argmax_per_position" fx) in
  if exact then check "argmax of every position" (argmax = reference) ""
  else begin
    (* Half precision may swap near-ties: the chosen token must be one the
       reference scores within the tolerance of its maximum. *)
    let top_ids = ints (mem "last_top_ids" fx) in
    let top_values = floats (mem "last_top_values" fx) in
    let margin =
      match Array.find_index (fun id -> id = argmax.(n - 1)) top_ids with
      | None -> infinity
      | Some i -> (top_values.(0) -. top_values.(i)) /. Float.abs top_values.(0)
    in
    check "argmax of the last position, up to near-ties" (margin < tol)
      (Printf.sprintf "  (reference margin %.1e)" margin)
  end;
  let last = Nx.slice [ I 0; I (n - 1) ] logits in
  let at i = Nx.item [ i ] last in
  close ~tol "the largest logits of the last position"
    (floats (mem "last_top_values" fx))
    (Array.map at (ints (mem "last_top_ids" fx)));
  close ~tol "the first logits of the last position"
    (floats (mem "last_first_values" fx))
    (Array.init 8 at);
  (* The residual stream after [k] blocks is the model truncated to [k]; the
     reference reports its last one after the final norm. *)
  let stream k =
    let sub =
      { p with Llama.blocks = List.filteri (fun i _ -> i < k) p.blocks }
    in
    let cfg_k = { cfg with Llama.n_layers = k } in
    let f ids =
      let h = Llama.hidden cfg_k sub ids in
      let h =
        if k = cfg.n_layers then Rms_norm.apply ~eps:cfg.norm_eps p.norm h
        else h
      in
      to32 (Nx.slice [ I 0; I (n - 1); R (0, 8) ] h)
    in
    Nx.to_array (compiled f ids)
  in
  let blocks =
    if device = None then List.init cfg.n_layers (fun i -> i + 1)
    else [ cfg.n_layers ]
  in
  let worst_block = ref 0 and ok = ref true in
  List.iter
    (fun k ->
      let before = !failures in
      close ~tol
        (Printf.sprintf "residual stream after block %d" k)
        (floats (mem (string_of_int k) (mem "hidden" fx)))
        (stream k);
      if !failures > before && !ok then begin
        ok := false;
        worst_block := k
      end)
    blocks;
  if not !ok then Printf.printf "first disagreement: block %d\n%!" !worst_block;
  (* The decode contract on real weights: chunks through the caches. *)
  let slots = Nx.create Nx.int32 [| 1; n |] (Array.init n Int32.of_int) in
  let _, hs, _ =
    List.fold_left
      (fun (at, hs, caches) len ->
        let len = min len (n - at) in
        if len = 0 then (at, hs, caches)
        else
          let pos =
            Nx.create Nx.int32 [| 1; len |]
              (Array.init len (fun i -> Int32.of_int (at + i)))
          in
          let h, caches =
            Llama.cached cfg p caches
              (Cache_index.make ~pos ~table:slots ())
              (Nx.slice [ A; R (at, at + len) ] ids)
          in
          (at + len, h :: hs, caches))
      (0, [], Llama.cache cfg ~slots:n dt)
      [ 1; 7; n ]
  in
  let chunked =
    to32 (Llama.logits cfg p (Nx.concatenate ~axis:1 (List.rev hs)))
  in
  let eager_logits =
    if device = None then logits
    else to32 (Llama.logits cfg p (Llama.hidden cfg p ids))
  in
  close ~tol "cached, in chunks of 1, 7 and the rest, every position"
    (flat eager_logits) (flat chunked);
  (* Two prompts of different lengths in one left-padded batch. *)
  let short = ints (mem "short_ids" fx) in
  let m = Array.length short in
  let padded = Array.append (Array.make (n - m) 0) short in
  let context = n in
  let index = Cache_index.rows ~context [| n; m |] in
  let batch ids =
    let h, _ =
      Llama.cached cfg p (Llama.cache cfg ~slots:(2 * context) dt) index ids
    in
    to32 (Llama.logits cfg p (Nx.slice [ A; I (n - 1) ] h))
  in
  let both = compiled batch (ids_tensor [| tokens; padded |]) in
  let row r ids_key =
    Array.map (fun i -> Nx.item [ r; i ] both) (ints (mem ids_key fx))
  in
  close ~tol "ragged batch, the long row"
    (floats (mem "last_top_values" fx))
    (row 0 "last_top_ids");
  close ~tol "ragged batch, the short row"
    (floats (mem "short_top_values" fx))
    (row 1 "short_top_ids")

let () =
  let weights = ref "" and config = ref "" in
  let fixture = ref "fixtures/llama-3.2-1b.json" in
  let dtype = ref "float32" and jit = ref "" in
  Arg.parse
    [
      ( "--fixture",
        Arg.Set_string fixture,
        "Reference values (default: Llama 3.2 1B)" );
      ("--weights", Arg.Set_string weights, "model.safetensors");
      ("--config", Arg.Set_string config, "config.json");
      ("--dtype", Arg.Set_string dtype, "float32 (default), bfloat16 or float16");
      ("--jit", Arg.Set_string jit, "Compile the forward passes for this device");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "validate.exe [--fixture FILE] [--weights FILE --config FILE] [--dtype DT] \
     [--jit DEVICE]";
  let fx = json_of_file !fixture in
  let repo = string (mem "repo" fx) in
  Printf.printf "%s, reference recorded from sha256 %s\n%!" repo
    (string (mem "weights_sha256" fx));
  let device = if !jit = "" then None else Some !jit in
  let (Llama.Dtype dt) = Llama.dtype_of_string !dtype in
  let cfg, p =
    if !weights = "" then Llama.from_pretrained ?device ~repo_id:repo dt
    else
      let cfg = Llama.config_of_json (json_of_file !config) in
      (cfg, Llama.from_file ?device cfg dt !weights)
  in
  (* Float32 against float32 with different kernels agrees to a few parts in a
     million eagerly; half precision keeps about three digits. *)
  let exact = !dtype = "float32" in
  validate ~device ~tol:(if exact then 1e-4 else 5e-2) ~exact fx cfg p dt;
  if !failures > 0 then exit 1
