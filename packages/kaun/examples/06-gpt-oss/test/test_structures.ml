(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What gpt-oss's maps visit: their leaf paths and reports, with float and
   packed expert weights. *)

open Windtrap
open Kaun

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

let leaf p = p ^ ": a leaf"
let linear p = [ leaf (p ^ ".w"); p ^ ".b: Some"; leaf (p ^ ".b") ]
let f shape = Nx.zeros Nx.float32 shape
let lin i o = { Linear.w = f [| i; o |]; b = Some (f [| o |]) }

let block weight =
  {
    Gpt_oss.attn_norm = { Rms_norm.gamma = f [| 4 |] };
    attn = { Attention.q = lin 4 4; k = lin 4 4; v = lin 4 4; out = lin 4 4 };
    sinks = f [| 2 |];
    ffn_norm = { Rms_norm.gamma = f [| 4 |] };
    router = lin 4 2;
    moe =
      {
        Moe.gate_up = weight ~outputs:8;
        gate_up_bias = f [| 2; 8 |];
        down = weight ~outputs:4;
        down_bias = f [| 2; 4 |];
      };
  }

let float ~outputs = Moe.Float (f [| 2; 32; outputs |])

let quant ~outputs =
  Moe.Quant
    (Nx_quant.mxfp4
       ~scales:(Nx.zeros Nx.uint8 [| 2; outputs; 1 |])
       (Nx.zeros Nx.uint8 [| 2; outputs; 16 |]))

let block_visits ~at weight =
  let at s = at ^ s in
  List.concat
    [
      [ leaf (at "attn_norm.gamma") ];
      linear (at "attn.q");
      linear (at "attn.k");
      linear (at "attn.v");
      linear (at "attn.out");
      [ leaf (at "sinks"); leaf (at "ffn_norm.gamma") ];
      linear (at "router");
      weight (at "moe.gate_up");
      [ leaf (at "moe.gate_up_bias") ];
      weight (at "moe.down");
      [ leaf (at "moe.down_bias") ];
    ]

let float_visits p = [ p ^ ": case \"float\""; leaf p ]

let quant_visits p =
  [
    p ^ ": case \"quant\"";
    p ^ ": case \"mxfp4\"";
    leaf (p ^ ".codes");
    leaf (p ^ ".scales");
  ]

let test_block () =
  let b = Nx.Ptree.instantiate (module Gpt_oss.Block) in
  equal ~msg:"float experts" (list string)
    (block_visits ~at:"" float_visits)
    (visits b (block float));
  equal ~msg:"packed experts" (list string)
    (block_visits ~at:"" quant_visits)
    (visits b (block quant));
  let packed = block quant in
  let cast = Nx.Ptree.cast (module Gpt_oss.Block) Nx.bfloat16 packed in
  is_true ~msg:"a cast keeps the packed weights"
    (match (cast.moe.gate_up, packed.moe.gate_up) with
    | Moe.Quant (Mxfp4 w), Moe.Quant (Mxfp4 w0) -> w.codes == w0.codes
    | _ -> false);
  is_true ~msg:"a cast casts the float leaves"
    (Nx.dtype cast.sinks = Nx.bfloat16)

let test_params () =
  let params = Nx.Ptree.instantiate (module Gpt_oss.Params) in
  let p =
    {
      Gpt_oss.tok = { Embedding.table = f [| 11; 4 |] };
      blocks = [ block float; block quant ];
      norm = { Rms_norm.gamma = f [| 4 |] };
      head = None;
    }
  in
  equal ~msg:"visits" (list string)
    (List.concat
       [
         [ leaf "tok.table"; "blocks: length 2" ];
         block_visits ~at:"blocks.0." float_visits;
         block_visits ~at:"blocks.1." quant_visits;
         [ leaf "norm.gamma"; "head: None" ];
       ])
    (visits params p)

let () =
  exit
    (run "gpt-oss structures"
       [
         group "structures"
           [
             test "a block's walk" test_block;
             test "the parameters' walk" test_params;
           ];
       ])
