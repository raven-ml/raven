(* Parity case: one gpt-oss MoE block in its decode form, end to end.

   Router, top-2 of 4 experts by selection rounds, softmax over the selected
   logits, MXFP4 rows of the selected experts decoded by arithmetic and made
   contiguous, the gate-up product with its gathered bias, the clamped SwiGLU,
   the down product, and the sum weighted by the routing weights. Each piece
   has a case of its own; this one holds their composition, where the kernels
   share operands across one another: 22 kernels for two tokens.

   Backends are limited to cpu and metal: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_frontend
module D = Tolk_uop.Dtype

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "metal")
    Helpers.all_backends

let ones shape = List.map (fun _ -> 1) shape

(* A scalar as the nx frontend hands it to a binary op: a constant broadcast to
   the shape of its peer. *)
let scalar value dtype shape =
  let const = Creation.full ~buffer:false ~dtype [] value in
  Movement.expand (Movement.reshape const (ones shape)) shape

(* table[codes], as [take] of a flat table. *)
let lookup table codes =
  let index = Movement.reshape (Dtype_ops.cast codes D.int32) [ -1 ] in
  Movement.reshape (Op.gather table ~dim:0 index) (Tensor.shape codes)

(* t[ids] along axis 0, as [take ~axis:0]. *)
let take_rows ids t =
  let n = Tensor.numel ids and rest = List.tl (Tensor.shape t) in
  let index =
    Movement.expand (Movement.reshape ids (n :: ones rest)) (n :: rest)
  in
  Movement.reshape (Op.gather t ~dim:0 index) (Tensor.shape ids @ rest)

(* Each byte holds two codes, low nibble first. *)
let nibbles blocks =
  let shape = Tensor.shape blocks in
  let low = Elementwise.bitwise_and blocks (scalar (Sint 15) D.uint8 shape) in
  let high = Elementwise.cdiv blocks (scalar (Sint 16) D.uint8 shape) in
  Op.cat ~dim:(-1) (Movement.unsqueeze low (-1)) [ Movement.unsqueeze high (-1) ]

(* The magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 are m / 2 up to 4, m - 2 for 5
   and 6, then 6; bit 3 is the sign. *)
let code_values codes =
  let shape = Tensor.shape codes in
  let f v = scalar (Sfloat v) D.float32 shape in
  let m =
    Dtype_ops.cast
      (Elementwise.bitwise_and codes (scalar (Sint 7) D.uint8 shape))
      D.float32
  in
  let magnitude =
    Elementwise.where
      (Elementwise.lt m (f 5.0))
      (Elementwise.mul m (f 0.5))
      (Elementwise.where
         (Elementwise.lt m (f 7.0))
         (Elementwise.sub m (f 2.0))
         (f 6.0))
  in
  let sign =
    Dtype_ops.cast
      (Elementwise.cdiv codes (scalar (Sint 8) D.uint8 shape))
      D.float32
  in
  Elementwise.mul magnitude
    (Elementwise.sub (f 1.0) (Elementwise.mul sign (f 2.0)))


let experts = 4
let k = 2
let tokens = 2
let width = 64
let hidden = 32
let limit = 7.0

let param idx ?dtype shape = Tensor.of_uop (Helpers.mk_param ~idx ?dtype shape)

let dequant_rows ids blocks scales scale_table outputs groups =
  let values = code_values (nibbles (take_rows ids blocks)) in
  let scale =
    Movement.reshape
      (lookup scale_table (take_rows ids scales))
      [ tokens; k; outputs; groups; 1; 1 ]
  in
  Movement.transpose ~dim0:(-1) ~dim1:(-2)
    (Elementwise.contiguous
       (Movement.reshape (Elementwise.mul values scale)
          [ tokens; k; outputs; groups * 32 ]))

let top_k x =
  let shape = [ tokens; experts ] and row = [ tokens; 1 ] in
  let position =
    Movement.broadcast_to
      (Movement.reshape (param 10 ~dtype:D.int32 [ experts ]) [ 1; experts ])
      shape
  in
  let low = scalar (Sfloat Float.neg_infinity) D.float32 shape in
  let real =
    Elementwise.bitwise_xor (Elementwise.ne x x) (scalar (Sbool true) D.bool shape)
  in
  let pick free =
    let live = Elementwise.bitwise_and free real in
    let greatest =
      Movement.reshape (Reduce.max ~axis:[ 1 ] (Elementwise.where live x low)) row
    in
    let best =
      Elementwise.bitwise_and live
        (Elementwise.eq x (Movement.broadcast_to greatest shape))
    in
    let some =
      Reduce.max ~axis:[ 1 ]
        (Elementwise.ne live (scalar (Sbool false) D.bool shape))
    in
    let chosen =
      Elementwise.where
        (Movement.broadcast_to (Movement.reshape some row) shape)
        best free
    in
    Dtype_ops.cast
      (Op.argmax ~axis:1 ~keepdim:true (Dtype_ops.cast chosen D.int32))
      D.int32
  in
  let rec rounds i free picks =
    if i = k then List.rev picks
    else
      let index = pick free in
      let taken = Elementwise.ne position (Movement.broadcast_to index shape) in
      rounds (i + 1) (Elementwise.bitwise_and free taken) (index :: picks)
  in
  let picks = rounds 0 (scalar (Sbool true) D.bool shape) [] in
  let indices = Op.cat ~dim:1 (List.hd picks) (List.tl picks) in
  let logits = Op.gather x ~dim:1 indices in
  let top =
    Movement.broadcast_to
      (Movement.reshape (Reduce.max ~axis:[ 1 ] logits) row)
      [ tokens; k ]
  in
  let e = Elementwise.exp (Elementwise.sub logits top) in
  let total = Movement.reshape (Reduce.sum ~axis:[ 1 ] ~dtype:D.float32 e) row in
  (indices, Elementwise.div e (Movement.broadcast_to total [ tokens; k ]))

let activation h =
  let out = [ tokens; k; 1; hidden ] in
  let f v = scalar (Sfloat v) D.float32 out in
  let pairs = Movement.reshape h [ tokens * k; hidden; 2 ] in
  let feature i =
    let column = Movement.shrink pairs [ (0, tokens * k); (0, hidden); (i, i + 1) ] in
    Movement.reshape (Movement.reshape column [ tokens * k; hidden ]) out
  in
  let gate = Elementwise.minimum (feature 0) (f limit) in
  let linear =
    Elementwise.minimum (Elementwise.maximum (feature 1) (f (-.limit))) (f limit)
  in
  let x = Elementwise.mul gate (f 1.702) in
  let sigmoid =
    Elementwise.reciprocal
      (Elementwise.add
         (Elementwise.exp2 (Elementwise.mul x (f (-1.0 /. Float.log 2.0))))
         (f 1.0))
  in
  Elementwise.mul (Elementwise.mul gate sigmoid) (Elementwise.add linear (f 1.0))

let build () =
  let x = param 0 [ tokens; width ] in
  let wr = param 1 [ width; experts ] in
  let br = param 2 [ experts ] in
  let gu_blocks = param 3 ~dtype:D.uint8 [ experts; 2 * hidden; width / 32; 16 ] in
  let gu_scales = param 4 ~dtype:D.uint8 [ experts; 2 * hidden; width / 32 ] in
  let gu_bias = param 5 [ experts; 2 * hidden ] in
  let dn_blocks = param 6 ~dtype:D.uint8 [ experts; width; hidden / 32; 16 ] in
  let dn_scales = param 7 ~dtype:D.uint8 [ experts; width; hidden / 32 ] in
  let dn_bias = param 8 [ experts; width ] in
  let scale_table = param 9 [ 256 ] in
  let logits =
    Elementwise.add (Op.matmul x wr)
      (Movement.broadcast_to (Movement.reshape br [ 1; experts ]) [ tokens; experts ])
  in
  let ids, weights = top_k logits in
  let gate_up = dequant_rows ids gu_blocks gu_scales scale_table (2 * hidden) (width / 32) in
  let down = dequant_rows ids dn_blocks dn_scales scale_table width (hidden / 32) in
  let h =
    Elementwise.add
      (Op.matmul (Movement.reshape x [ tokens; 1; 1; width ]) gate_up)
      (Movement.reshape (take_rows ids gu_bias) [ tokens; k; 1; 2 * hidden ])
  in
  let y =
    Elementwise.add
      (Op.matmul (Elementwise.contiguous (activation h)) down)
      (Movement.reshape (take_rows ids dn_bias) [ tokens; k; 1; width ])
  in
  let w =
    Movement.broadcast_to
      (Movement.reshape weights [ tokens; k; 1; 1 ])
      [ tokens; k; 1; width ]
  in
  Helpers.wrap_sink
    [ Tensor.uop (Reduce.sum ~axis:[ 1; 2 ] ~dtype:D.float32 (Elementwise.mul y w)) ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
