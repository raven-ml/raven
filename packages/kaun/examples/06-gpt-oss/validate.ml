(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Validation of the gpt-oss example against the transformers implementation.

   The fixtures under [fixtures/] hold reference values recorded at float32 on
   CPU by [reference.py] from two tiny random checkpoints, one with float
   experts and one with MXFP4 experts, and name the sha256 of the weight files
   they were recorded from (printed here, not verified). The packed checkpoint's
   scales are tiny, so its cases are also recorded with a constant added to
   every scale byte.

   The building blocks, at float32: [Mxfp4.dequant] against the reference
   dequantiser, bit for bit; the router's logits, experts and weights, and what
   it selects among equal logits; the MoE block in both formulations on a batch,
   on a ragged pair and on an input scaled until the activation clamps.

   The whole model, recorded with a sliding window of 4 so that the window binds
   on the 12-token prompt: the rotary tables and the sinks; each attention layer
   alone on its recorded input; the argmax of every position and the largest
   logits of the last; the residual stream after every block; the cached path
   fed in chunks of 1, 7 and the rest, the last boundary inside a window,
   against the whole-sequence one; a ragged batch of two prompts against each
   prompt's own reference.

   [--jit DEVICE] compiles every function under test; the per-block streams are
   then checked at the last block only. [--dtype bfloat16] runs the whole model
   at half precision against the same float32 reference and skips the building
   blocks.

   Usage: validate.exe [--fixtures DIR] [--float-weights FILE] [--mxfp4-weights
   FILE] [--jit DEVICE] [--dtype DT]. Without the weight files they come from
   the fixtures' repositories (14 MB each, cached), as the configurations always
   do. Not part of the test suite: it needs the download. *)

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

let members = function
  | Jsont.Object (mems, _) -> List.map (fun ((name, _), v) -> (name, v)) mems
  | _ -> failwith "fixture: not an object"

let mem name j =
  match List.assoc_opt name (members j) with
  | Some v -> v
  | None -> failwith ("fixture: missing " ^ name)

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
  Printf.printf "%-66s %s%s\n%!" name (if ok then "ok" else "FAIL") detail;
  if not ok then incr failures

let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

(* Agreement relative to the vector's largest magnitude. A NaN or an infinity
   anywhere fails. *)
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
    check name
      (!worst < tol && Array.for_all Float.is_finite actual)
      (Printf.sprintf "  (worst relative error %.1e)" !worst)
  end

(* Equality of every bit, the sign of zero included. [flushed i] says whether
   value [i] may come out as zero instead. *)
let identical ?(flushed = fun _ -> false) name expected actual =
  let differing = ref 0 and zeroed = ref 0 and first = ref "" in
  if Array.length expected <> Array.length actual then differing := -1
  else
    Array.iteri
      (fun i e ->
        if Int64.bits_of_float e <> Int64.bits_of_float actual.(i) then
          if flushed i && actual.(i) = 0.0 then incr zeroed
          else begin
            if !differing = 0 then
              first := Printf.sprintf ", first %h for %h" actual.(i) e;
            incr differing
          end)
      expected;
  let zeroed =
    if !zeroed = 0 then ""
    else Printf.sprintf ", %d subnormal flushed to zero" !zeroed
  in
  check name (!differing = 0)
    (Printf.sprintf "  (%d of %d values differ%s%s)" !differing
       (Array.length expected) !first zeroed)

let compiled device f x =
  match device with None -> f x | Some device -> Rune.jit' ~device f x

let number j =
  match j with
  | Jsont.Number (f, _) -> f
  | _ -> failwith "fixture: number expected"

let uint8 shape values = Nx.create Nx.uint8 shape values
let float32 shape values = Nx.create Nx.float32 shape values

(* Checkpoint tensors *)

let tensor ckpt name =
  match Checkpoint.get name ckpt with Nx.Ptree.P t -> Nx.cast Nx.float32 t

let bytes ckpt name =
  Nx.Ptree.unpack ~at:name Nx.uint8 (Checkpoint.get name ckpt)

let moe_params ckpt ~layer ~weight =
  let name leaf = Printf.sprintf "model.layers.%d.mlp.%s" layer leaf in
  let router =
    {
      Linear.w = Nx.transpose (tensor ckpt (name "router.weight"));
      b = Some (tensor ckpt (name "router.bias"));
    }
  in
  let experts =
    {
      Moe.gate_up = weight (name "experts.gate_up_proj");
      gate_up_bias = tensor ckpt (name "experts.gate_up_proj_bias");
      down = weight (name "experts.down_proj");
      down_bias = tensor ckpt (name "experts.down_proj_bias");
    }
  in
  (router, experts)

let float_weight ckpt name = Moe.Float (tensor ckpt name)

let packed_weight ckpt ~offset name =
  let scales = bytes ckpt (name ^ "_scales") in
  Moe.Mxfp4
    {
      blocks = bytes ckpt (name ^ "_blocks");
      scales = Nx.add scales (Nx.scalar Nx.uint8 offset);
    }

(* Checks *)

(* Metal flushes subnormal float32 numbers to zero, the scale 2^-127 of the byte
   0 included. Trained checkpoints hold no such scale; the tiny random one holds
   little else. *)
let min_normal = Float.ldexp 1.0 (-126)

let dequant ~device fx =
  List.iter
    (fun (name, case) ->
      let shape = ints (mem "blocks_shape" case) in
      let blocks = uint8 shape (ints (mem "blocks" case)) in
      let scale_bytes = ints (mem "scales" case) in
      let scales =
        uint8 (Array.sub shape 0 (Array.length shape - 1)) scale_bytes
      in
      let expected = floats (mem "values" case) in
      let flushed i =
        device = Some "METAL"
        && (scale_bytes.(i / 32) = 0 || Float.abs expected.(i) < min_normal)
      in
      identical ~flushed
        (Printf.sprintf "dequant %s, float32" name)
        expected
        (flat
           (compiled device (fun b -> Mxfp4.dequant b scales Nx.float32) blocks));
      identical ~flushed
        (Printf.sprintf "dequant %s, bfloat16" name)
        expected
        (flat
           (compiled device
              (fun b -> Nx.cast Nx.float32 (Mxfp4.dequant b scales Nx.bfloat16))
              blocks));
      let rows = Nx.dim 0 blocks in
      let per_row = Array.length expected / rows in
      let picks = Array.init (rows + 1) (fun i -> (rows - i) mod rows) in
      let ids =
        Nx.create Nx.int32 [| rows + 1 |] (Array.map Int32.of_int picks)
      in
      let source i = (picks.(i / per_row) * per_row) + (i mod per_row) in
      identical
        ~flushed:(fun i -> flushed (source i))
        (Printf.sprintf "dequant %s, selected rows" name)
        (Array.init ((rows + 1) * per_row) (fun i -> expected.(source i)))
        (flat
           (compiled device
              (fun b -> Mxfp4.dequant_rows b scales ids Nx.float32)
              blocks)))
    (members (mem "dequant" fx));
  let nan_scale = uint8 [| 1 |] [| 255 |] in
  let group =
    compiled device
      (fun b -> Mxfp4.dequant b nan_scale Nx.float32)
      (uint8 [| 1; 16 |] (Array.make 16 0x21))
  in
  check "dequant, the scale byte 255 is NaN"
    (Array.for_all Float.is_nan (flat group))
    ""

let ties ~device fx =
  let k = int_of_float (number (mem "num_experts_per_tok" (mem "config" fx))) in
  let case = mem "ties" fx in
  let logits = floats (mem "logits" case) in
  let experts = Array.length logits / 3 in
  let x = float32 [| 3; experts |] logits in
  let ids =
    Array.map Int32.to_int
      (flat (compiled device (fun x -> fst (Moe.route ~k x)) x))
  in
  let weights = flat (compiled device (fun x -> snd (Moe.route ~k x)) x) in
  let row a r = Array.sub a (r * k) k in
  let lowest_first =
    [| [| 0; 1; 2; 3 |]; [| 6; 13; 20; 27 |]; [| 5; 9; 20; 21 |] |]
  in
  check "equal logits: k experts, the lowest first"
    (Array.for_all (fun r -> row ids r = lowest_first.(r)) [| 0; 1; 2 |])
    "";
  let values = Array.mapi (fun i e -> logits.((i / k * experts) + e)) ids in
  identical "equal logits: the selected values are the reference's"
    (floats (mem "values" case))
    values;
  close ~tol:1e-6 "equal logits: expert weights"
    (floats (mem "expert_weights" case))
    weights;
  let reference = ints (mem "experts" case) in
  let same_sets =
    Array.for_all
      (fun r ->
        List.sort compare (Array.to_list (row ids r))
        = List.sort compare (Array.to_list (row reference r)))
      [| 0; 1; 2 |]
  in
  Printf.printf
    "  note: torch.topk orders equal logits arbitrarily; same sets as the \
     reference: %b\n\
     %!"
    same_sets

let block ~device ~tol ~k ~limit label (router, p) case =
  let shape = ints (mem "shape" case) in
  let width = shape.(Array.length shape - 1) in
  let x = float32 shape (floats (mem "hidden" case)) in
  let tokens = Nx.reshape [| -1; width |] x in
  let name what = Printf.sprintf "%s: %s" label what in
  close ~tol (name "router logits")
    (floats (mem "router_logits" case))
    (flat (compiled device (fun x -> Linear.apply router x) tokens));
  let route x = Moe.route ~k (Linear.apply router x) in
  let ids = compiled device (fun x -> fst (route x)) tokens in
  check (name "selected experts")
    (Array.map Int32.to_int (flat ids) = ints (mem "experts" case))
    "";
  close ~tol (name "expert weights")
    (floats (mem "expert_weights" case))
    (flat (compiled device (fun x -> snd (route x)) tokens));
  let last = Nx.dim 0 tokens - 1 in
  List.iter
    (fun (form, form_name) ->
      let apply x =
        compiled device (fun x -> Moe.apply form ~limit p (route x) x) x
      in
      let whole = apply tokens in
      close ~tol
        (name ("output, " ^ form_name))
        (floats (mem "output" case))
        (flat whole);
      close ~tol
        (name ("output, " ^ form_name ^ ", the last token alone"))
        (flat (Nx.slice [ I last ] whole))
        (flat (apply (Nx.slice [ R (last, last + 1) ] tokens))))
    [ (Moe.Gather, "gather"); (Moe.Dense, "dense") ]

let blocks ~device ~tol fx ~label ~weight ckpt =
  let config = mem "config" fx in
  let k = int_of_float (number (mem "num_experts_per_tok" config)) in
  let limit = number (mem "swiglu_limit" config) in
  let layer = int_of_float (number (mem "layer" fx)) in
  List.iter
    (fun (offset, cases) ->
      let weight = weight ckpt ~offset:(int_of_string offset) in
      let p = moe_params ckpt ~layer ~weight in
      List.iter
        (fun (case_name, case) ->
          let label =
            if offset = "0" then Printf.sprintf "%s %s" label case_name
            else Printf.sprintf "%s scales+%s %s" label offset case_name
          in
          block ~device ~tol ~k ~limit label p case)
        (members cases))
    (members (mem "cases" fx))

(* The whole model *)

let ids_tensor rows =
  let batch = Array.length rows and seq = Array.length rows.(0) in
  Nx.create Nx.int32 [| batch; seq |]
    (Array.map Int32.of_int (Array.concat (Array.to_list rows)))

let with_scale_offset offset (p : _ Gpt_oss.params) =
  let shift = function
    | Moe.Float w -> Moe.Float w
    | Moe.Mxfp4 { blocks; scales } ->
        Moe.Mxfp4 { blocks; scales = Nx.add scales (Nx.scalar Nx.uint8 offset) }
  in
  let block (b : _ Gpt_oss.block) =
    let moe =
      { b.moe with gate_up = shift b.moe.gate_up; down = shift b.moe.down }
    in
    { b with moe }
  in
  { p with blocks = List.map block p.blocks }

let rotary fx (cfg : Gpt_oss.config) =
  let r = mem "rotary" fx in
  close ~tol:1e-6 "rotary frequencies"
    (floats (mem "inv_freq" r))
    (Rope.frequencies cfg.rope);
  (* The reference scales its cosines and sines by a concentration, which the
     model moves to the scores as its square. *)
  let concentration = number (mem "attention_scaling" r) in
  close ~tol:1e-7 "attention scale"
    [| (concentration ** 2.0) /. sqrt (float_of_int cfg.head_dim) |]
    [| cfg.attention_scale |];
  (* Rotating [1 .. 1; 0 .. 0] reads the table: cosines then sines. The
     reference forms its frequencies in float32, a unit in the last place from
     ours, which the position multiplies: 2.5e-4 radians at 4095. *)
  let positions = ints (mem "positions" r) in
  let n = Array.length positions and half = cfg.head_dim / 2 in
  let x =
    Nx.concatenate ~axis:3
      [
        Nx.ones Nx.float32 [| 1; 1; n; half |];
        Nx.zeros Nx.float32 [| 1; 1; n; half |];
      ]
  in
  let table =
    Nx.mul_s
      (Rope.apply cfg.rope ~pos:(ids_tensor [| positions |]) x)
      concentration
  in
  let part lo = flat (Nx.slice [ A; A; A; R (lo, lo + half) ] table) in
  close ~tol:1e-3 "rotary cosines at positions up to 4095"
    (floats (mem "cos" r))
    (part 0);
  close ~tol:1e-3 "rotary sines at positions up to 4095"
    (floats (mem "sin" r))
    (part half)

let model (type b) ~device ~tol ~exact ~label fx case (cfg : Gpt_oss.config)
    (p : (float, b) Nx.t Gpt_oss.params) (dt : (float, b) Nx.dtype) =
  let name what = Printf.sprintf "%s: %s" label what in
  let to32 t = Nx.cast Nx.float32 t in
  let tokens = ints (mem "ids" fx) in
  let n = Array.length tokens in
  let ids = ids_tensor [| tokens |] in
  List.iteri
    (fun i (layer, (b : _ Gpt_oss.block)) ->
      let recorded = mem (string_of_int i) (mem "attention" case) in
      let kind =
        match layer with Gpt_oss.Sliding -> "sliding" | Gpt_oss.Full -> "full"
      in
      let attend layer x =
        let cache =
          Attention.Cache.make ~slots:0 ~kv_heads:cfg.n_kv_heads
            ~head_dim:cfg.head_dim dt
        in
        to32
          (fst
             (Gpt_oss.attention cfg layer b cache
                (Cache_index.whole ~batch:1 ~seq:n ())
                (Nx.cast dt x)))
      in
      let x = float32 [| 1; n; cfg.dim |] (floats (mem "input" recorded)) in
      let expected = floats (mem "output" recorded) in
      close ~tol
        (name (Printf.sprintf "attention of block %d (%s) alone" i kind))
        expected
        (flat (compiled device (attend layer) x));
      if layer = Gpt_oss.Sliding && exact then begin
        let unbound = flat (attend Gpt_oss.Full x) in
        let gap = ref 0.0 in
        Array.iteri
          (fun j e -> gap := Float.max !gap (Float.abs (e -. unbound.(j))))
          expected;
        check
          (name "the window binds on this prompt")
          (!gap > 1e-3)
          (Printf.sprintf "  (without it the output moves by %.1e)" !gap)
      end)
    (List.combine cfg.layers p.blocks);
  let logits =
    compiled device
      (fun ids -> to32 (Gpt_oss.logits cfg p (Gpt_oss.hidden cfg p ids)))
      ids
  in
  let argmax =
    Array.map Int32.to_int (Nx.to_array (Nx.argmax ~axis:2 logits))
  in
  let top_ids = ints (mem "last_top_ids" case) in
  let top_values = floats (mem "last_top_values" case) in
  if exact then
    check
      (name "argmax of every position")
      (argmax = ints (mem "argmax_per_position" case))
      ""
  else begin
    let margin =
      match Array.find_index (fun id -> id = argmax.(n - 1)) top_ids with
      | None -> infinity
      | Some i -> (top_values.(0) -. top_values.(i)) /. Float.abs top_values.(0)
    in
    check
      (name "argmax of the last position, up to near-ties")
      (margin < tol)
      (Printf.sprintf "  (reference margin %.1e)" margin)
  end;
  let last = Nx.slice [ I 0; I (n - 1) ] logits in
  let at i = Nx.item [ i ] last in
  close ~tol
    (name "the largest logits of the last position")
    top_values (Array.map at top_ids);
  close ~tol
    (name "the first logits of the last position")
    (floats (mem "last_first_values" case))
    (Array.init 8 at);
  Option.iter
    (fun device ->
      let as_inputs =
        Rune.jit ~device (Gpt_oss.ptree ())
          (fun p -> to32 (Gpt_oss.logits cfg p (Gpt_oss.hidden cfg p ids)))
          p
      in
      close ~tol
        (name "parameters as compiled inputs, packed ones included")
        (flat logits) (flat as_inputs))
    device;
  let n_layers = List.length cfg.layers in
  let stream k =
    let first l = List.filteri (fun i _ -> i < k) l in
    let sub = { p with Gpt_oss.blocks = first p.blocks } in
    let cfg_k = { cfg with Gpt_oss.layers = first cfg.layers } in
    let f ids =
      let h = Gpt_oss.hidden cfg_k sub ids in
      let h =
        if k = n_layers then Rms_norm.apply ~eps:cfg.norm_eps p.norm h else h
      in
      to32 (Nx.slice [ I 0; I (n - 1); R (0, 8) ] h)
    in
    Nx.to_array (compiled device f ids)
  in
  List.iter
    (fun k ->
      close ~tol
        (name (Printf.sprintf "residual stream after block %d" k))
        (floats (mem (string_of_int k) (mem "hidden" case)))
        (stream k))
    (if device = None then List.init n_layers (fun i -> i + 1) else [ n_layers ]);
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
            Gpt_oss.cached cfg p caches
              (Cache_index.make ~pos ~table:slots ())
              (Nx.slice [ A; R (at, at + len) ] ids)
          in
          (at + len, h :: hs, caches))
      (0, [], Gpt_oss.cache cfg ~slots:n dt)
      [ 1; 7; n ]
  in
  let chunked =
    to32 (Gpt_oss.logits cfg p (Nx.concatenate ~axis:1 (List.rev hs)))
  in
  let eager_logits =
    if device = None then logits
    else to32 (Gpt_oss.logits cfg p (Gpt_oss.hidden cfg p ids))
  in
  close ~tol
    (name "cached, in chunks of 1, 7 and the rest, every position")
    (flat eager_logits) (flat chunked);
  let short = ints (mem "short_ids" fx) in
  let m = Array.length short in
  let padded = Array.append (Array.make (n - m) 0) short in
  let index = Cache_index.rows ~context:n [| n; m |] in
  let batch ids =
    let h, _ =
      Gpt_oss.cached cfg p (Gpt_oss.cache cfg ~slots:(2 * n) dt) index ids
    in
    to32 (Gpt_oss.logits cfg p (Nx.slice [ A; I (n - 1) ] h))
  in
  let both = compiled device batch (ids_tensor [| tokens; padded |]) in
  let row r ids_key =
    Array.map (fun i -> Nx.item [ r; i ] both) (ints (mem ids_key case))
  in
  close ~tol
    (name "ragged batch, the long row")
    top_values (row 0 "last_top_ids");
  close ~tol
    (name "ragged batch, the short row")
    (floats (mem "short_top_values" case))
    (row 1 "short_top_ids")

let models ~device ~dtype ~label fx path =
  let repo = string (mem "repo" fx) in
  let cfg = Gpt_oss.config_of_json (Kaun_hf.load_config repo) in
  let cfg =
    { cfg with window = int_of_float (number (mem "sliding_window" fx)) }
  in
  let (Gpt_oss.Dtype dt) = Gpt_oss.dtype_of_string dtype in
  let p = Gpt_oss.from_file ?device cfg dt path in
  rotary fx cfg;
  List.iteri
    (fun i (b : _ Gpt_oss.block) ->
      identical
        (Printf.sprintf "%s: sinks of block %d" label i)
        (floats (mem (string_of_int i) (mem "sinks" fx)))
        (flat b.sinks))
    p.blocks;
  List.iter
    (fun (offset, case) ->
      let p = with_scale_offset (int_of_string offset) p in
      let label =
        if offset = "0" then label
        else Printf.sprintf "%s scales+%s" label offset
      in
      (* Float32 against float32 agrees to a few parts in a million. Half
         precision keeps about three digits: the reference's softmax also ran at
         float32 here, so the gap is that of the weights and activations. *)
      let exact = dtype = "float32" in
      model ~device
        ~tol:(if exact then 1e-5 else 5e-2)
        ~exact ~label fx case cfg p dt)
    (members (mem "cases" fx))

let () =
  let fixtures = ref "fixtures" and jit = ref "" and dtype = ref "float32" in
  let float_weights = ref "" and mxfp4_weights = ref "" in
  Arg.parse
    [
      ("--fixtures", Arg.Set_string fixtures, "Directory of reference values");
      ( "--float-weights",
        Arg.Set_string float_weights,
        "model.safetensors of the float checkpoint" );
      ( "--mxfp4-weights",
        Arg.Set_string mxfp4_weights,
        "model.safetensors of the MXFP4 checkpoint" );
      ( "--jit",
        Arg.Set_string jit,
        "Compile the functions under test for this device" );
      ("--dtype", Arg.Set_string dtype, "float32 (default) or bfloat16");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "validate.exe [--fixtures DIR] [--float-weights FILE] [--mxfp4-weights \
     FILE] [--jit DEVICE] [--dtype DT]";
  let device = if !jit = "" then None else Some !jit in
  let weights fx given =
    let repo = string (mem "repo" fx) in
    Printf.printf "%s, reference recorded from sha256 %s\n%!" repo
      (string (mem "weights_sha256" fx));
    if given <> "" then given
    else Kaun_hf.download_file ~file:"model.safetensors" repo
  in
  let fixture name =
    json_of_file (Filename.concat !fixtures (name ^ ".json"))
  in
  if !dtype = "float32" then begin
    (* Float32 against float32 with different kernels and a different order of
       accumulation over the experts. *)
    let tol = 1e-5 in
    let fx = fixture "gpt-oss-bf16" in
    let ckpt = Checkpoint.load (weights fx !float_weights) in
    blocks ~device ~tol fx ~label:"float"
      ~weight:(fun ckpt ~offset:_ name -> float_weight ckpt name)
      ckpt;
    ties ~device fx;
    let fx = fixture "gpt-oss-mxfp4" in
    let ckpt = Checkpoint.load (weights fx !mxfp4_weights) in
    dequant ~device fx;
    blocks ~device ~tol fx ~label:"mxfp4" ~weight:packed_weight ckpt
  end;
  let fx = fixture "gpt-oss-bf16-model" in
  models ~device ~dtype:!dtype ~label:"float model" fx
    (weights fx !float_weights);
  let fx = fixture "gpt-oss-mxfp4-model" in
  models ~device ~dtype:!dtype ~label:"mxfp4 model" fx
    (weights fx !mxfp4_weights);
  if !failures > 0 then exit 1
