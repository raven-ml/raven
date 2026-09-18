(* Parity case: dst[idx[k], j] = src[k, j] with distinct row indices.

   No two updates share a row, so their order is free and the index range is a
   launch dimension next to the column range. A row index outside [0, 16) gates
   the store off. No range covers the 16 rows of dst.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_uop
module U = Uop

let rows = 16
let cols = 8
let updates = 5

let kernel () =
  let open U.O in
  let param slot dtype = U.param ~slot ~dtype ~shape:(U.const_int (-1)) () in
  let dst = param 0 Dtype.float32 in
  let idx = param 1 Dtype.int32 in
  let src = param 2 Dtype.float32 in
  let j = U.range ~size:(U.const_int cols) ~axis:0 ~kind:Axis_type.Weak () in
  let k =
    U.range ~size:(U.const_int updates) ~axis:1 ~kind:Axis_type.Weak ()
  in
  let row = U.load ~src:(U.index ~ptr:idx ~idxs:[ k ] ()) () in
  let i32 n = U.const (Const.int Dtype.int32 n) in
  let lt a b = U.alu_binary ~op:Ops.Cmplt ~lhs:a ~rhs:b in
  let in_bounds =
    U.alu_binary ~op:Ops.And
      ~lhs:
        (U.alu_binary ~op:Ops.Cmpne ~lhs:(lt row (i32 0))
           ~rhs:(U.const_bool true))
      ~rhs:(lt row (i32 rows))
  in
  let target =
    U.valid
      ~src:((U.cast ~src:row ~dtype:Dtype.weakint * U.const_int cols) + j)
      ~cond:in_bounds
  in
  let value =
    U.load ~src:(U.index ~ptr:src ~idxs:[ (k * U.const_int cols) + j ] ()) ()
  in
  let st = U.store ~dst:(U.index ~ptr:dst ~idxs:[ target ] ()) ~value () in
  let e = U.end_ ~value:st ~ranges:[ j; k ] in
  U.sink
    ~kernel_info:
      {
        U.name = "indexed_store_unique";
        axis_types = [ Axis_type.Weak; Axis_type.Weak ];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply = Some [];
        estimates = None;
        beam = 0;
      }
    [ e ]

let () =
  Helpers.dump
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
