(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Validation of the gpt-oss building blocks against the transformers
   implementation.

   The fixtures under [fixtures/] hold reference values recorded at float32 on
   CPU by [reference.py] from two tiny random checkpoints, one with float
   experts and one with MXFP4 experts, and name the sha256 of the weight files
   they were recorded from (printed here, not verified). This program compares:

   - [Mxfp4.dequant] with the reference dequantiser, bit for bit, on slices of
   the checkpoint and on a sweep over every scale byte; - the router: its
   logits, the experts it selects and their weights, and what it selects among
   equal logits; - the block's output in both formulations, on a batch, on a
   ragged pair and on an input scaled until the activation clamps, with float
   weights and with packed ones. The packed checkpoint's scales are tiny, so its
   cases are also recorded with a constant added to every scale byte.

   [--jit DEVICE] compiles every function under test.

   Usage: validate.exe [--fixtures DIR] [--float-weights FILE] [--mxfp4-weights
   FILE] [--jit DEVICE]. Without the weight files they come from the fixtures'
   repositories (14 MB each, cached). Not part of the test suite: it needs the
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
  {
    Moe.router =
      {
        Linear.w = Nx.transpose (tensor ckpt (name "router.weight"));
        b = Some (tensor ckpt (name "router.bias"));
      };
    gate_up = weight (name "experts.gate_up_proj");
    gate_up_bias = tensor ckpt (name "experts.gate_up_proj_bias");
    down = weight (name "experts.down_proj");
    down_bias = tensor ckpt (name "experts.down_proj_bias");
  }

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
  let identity =
    {
      Moe.router = { Linear.w = Nx.eye Nx.float32 experts; b = None };
      gate_up = Moe.Float (Nx.zeros Nx.float32 [| experts; experts; 2 |]);
      gate_up_bias = Nx.zeros Nx.float32 [| experts; 2 |];
      down = Moe.Float (Nx.zeros Nx.float32 [| experts; 1; experts |]);
      down_bias = Nx.zeros Nx.float32 [| experts; experts |];
    }
  in
  let x = float32 [| 3; experts |] logits in
  let ids =
    Array.map Int32.to_int
      (flat (compiled device (fun x -> fst (Moe.route ~k identity x)) x))
  in
  let weights =
    flat (compiled device (fun x -> snd (Moe.route ~k identity x)) x)
  in
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

let block ~device ~tol ~k ~limit label p case =
  let shape = ints (mem "shape" case) in
  let width = shape.(Array.length shape - 1) in
  let x = float32 shape (floats (mem "hidden" case)) in
  let tokens = Nx.reshape [| -1; width |] x in
  let name what = Printf.sprintf "%s: %s" label what in
  close ~tol (name "router logits")
    (floats (mem "router_logits" case))
    (flat (compiled device (fun x -> Linear.apply p.Moe.router x) tokens));
  let ids = compiled device (fun x -> fst (Moe.route ~k p x)) tokens in
  check (name "selected experts")
    (Array.map Int32.to_int (flat ids) = ints (mem "experts" case))
    "";
  close ~tol (name "expert weights")
    (floats (mem "expert_weights" case))
    (flat (compiled device (fun x -> snd (Moe.route ~k p x)) tokens));
  List.iter
    (fun (form, form_name) ->
      close ~tol
        (name ("output, " ^ form_name))
        (floats (mem "output" case))
        (flat (compiled device (fun x -> Moe.apply form ~k ~limit p x) x)))
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

let () =
  let fixtures = ref "fixtures" and jit = ref "" in
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
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "validate.exe [--fixtures DIR] [--float-weights FILE] [--mxfp4-weights \
     FILE] [--jit DEVICE]";
  let device = if !jit = "" then None else Some !jit in
  let load name given =
    let fx = json_of_file (Filename.concat !fixtures (name ^ ".json")) in
    let repo = string (mem "repo" fx) in
    Printf.printf "%s, reference recorded from sha256 %s\n%!" repo
      (string (mem "weights_sha256" fx));
    let path =
      if given <> "" then given
      else Kaun_hf.download_file ~file:"model.safetensors" repo
    in
    (fx, Checkpoint.load path)
  in
  (* Float32 against float32 with different kernels and a different order of
     accumulation over the experts. *)
  let tol = 1e-5 in
  let fx, ckpt = load "gpt-oss-bf16" !float_weights in
  blocks ~device ~tol fx ~label:"float"
    ~weight:(fun ckpt ~offset:_ name -> float_weight ckpt name)
    ckpt;
  ties ~device fx;
  let fx, ckpt = load "gpt-oss-mxfp4" !mxfp4_weights in
  dequant ~device fx;
  blocks ~device ~tol fx ~label:"mxfp4" ~weight:packed_weight ckpt;
  if !failures > 0 then exit 1
