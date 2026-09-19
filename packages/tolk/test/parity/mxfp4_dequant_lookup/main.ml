(* Parity case: MXFP4 dequantisation of gathered experts by two lookups.

   The decode form: the ids of k = 2 experts gather their blocks
   [rows; groups; 16] and scales [rows; groups] from the tables of 4
   experts, then a byte splits into two 4-bit codes, a code reads a 16-entry
   value table, a scale reads a 256-entry table of powers of two, and the
   product is laid out as [tokens; k; rows; groups * 32]. Both row gathers
   collapse to gated loads, but each is a reduce whose result indexes a second
   gather, and a reduce is realised before the broadcast over the second
   table: the codes are written out as int32, four bytes for each weight, and
   so are the scale indices. Four kernels.

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

let experts = 4
let rows = 3
let groups = 2
let tokens = 1
let k = 2

let build () =
  let param idx ?dtype shape = Tensor.of_uop (Helpers.mk_param ~idx ?dtype shape) in
  let blocks = param 0 ~dtype:D.uint8 [ experts; rows; groups; 16 ] in
  let scales = param 1 ~dtype:D.uint8 [ experts; rows; groups ] in
  let ids = param 2 ~dtype:D.int32 [ tokens; k ] in
  let code_table = param 3 [ 16 ] in
  let scale_table = param 4 [ 256 ] in
  let values = lookup code_table (nibbles (take_rows ids blocks)) in
  let scale =
    Movement.reshape
      (lookup scale_table (take_rows ids scales))
      [ tokens; k; rows; groups; 1; 1 ]
  in
  Helpers.wrap_sink
    [
      Tensor.uop
        (Movement.reshape
           (Elementwise.mul values scale)
           [ tokens; k; rows; groups * 32 ]);
    ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
