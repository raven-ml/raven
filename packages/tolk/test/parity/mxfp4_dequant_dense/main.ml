(* Parity case: MXFP4 dequantisation of a whole table by two lookups.

   Blocks [rows; groups; 16] of packed bytes and one uint8 scale per group.
   A byte splits into two 4-bit codes, a code reads a 16-entry value table, a
   scale reads a 256-entry table of powers of two, and the product is laid out
   as [rows; groups * 32]. The code lookup fuses into the multiply: its
   index is elementwise over the blocks, and the one-hot sum collapses to an
   ungated load because a nibble cannot leave the table. The scale lookup is a
   kernel of its own, one float32 per group: a reduce is realised before the
   broadcast over the group's 32 values.

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

(* Each byte holds two codes, low nibble first. *)
let nibbles blocks =
  let shape = Tensor.shape blocks in
  let low = Elementwise.bitwise_and blocks (scalar (Sint 15) D.uint8 shape) in
  let high = Elementwise.cdiv blocks (scalar (Sint 16) D.uint8 shape) in
  Op.cat ~dim:(-1) (Movement.unsqueeze low (-1)) [ Movement.unsqueeze high (-1) ]

let rows = 3
let groups = 2

let build () =
  let param idx ?dtype shape = Tensor.of_uop (Helpers.mk_param ~idx ?dtype shape) in
  let blocks = param 0 ~dtype:D.uint8 [ rows; groups; 16 ] in
  let scales = param 1 ~dtype:D.uint8 [ rows; groups ] in
  let code_table = param 2 [ 16 ] in
  let scale_table = param 3 [ 256 ] in
  let values = lookup code_table (nibbles blocks) in
  let scale =
    Movement.reshape (lookup scale_table scales) [ rows; groups; 1; 1 ]
  in
  Helpers.wrap_sink
    [
      Tensor.uop
        (Movement.reshape (Elementwise.mul values scale) [ rows; groups * 32 ]);
    ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
