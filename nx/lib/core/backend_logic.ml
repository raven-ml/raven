open Descriptor
open Views

module Make (B : Backend_intf.S) = struct
  let logical_and context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context uint8 out_shape in
    let a_desc' = broadcast_to a_desc out_shape in
    let b_desc' = broadcast_to b_desc out_shape in
    let a_op = B.view a_desc' a in
    let b_op = B.view b_desc' b in
    B.bit_and context a_op b_op c;
    c

  let logical_or context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context uint8 out_shape in
    let a_desc' = broadcast_to a_desc out_shape in
    let b_desc' = broadcast_to b_desc out_shape in
    let a_op = B.view a_desc' a in
    let b_op = B.view b_desc' b in
    B.bit_or context a_op b_op c;
    c

  let logical_not context a =
    let desc = B.descriptor a in
    let out = B.empty context uint8 (shape desc) in
    B.bit_not context a out;
    out

  let logical_xor context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context uint8 out_shape in
    let a_desc' = broadcast_to a_desc out_shape in
    let b_desc' = broadcast_to b_desc out_shape in
    let a_op = B.view a_desc' a in
    let b_op = B.view b_desc' b in
    B.bit_xor context a_op b_op c;
    c

  let isnan context a =
    let desc = B.descriptor a in
    let out = B.empty context uint8 (shape desc) in
    B.isnan context a out;
    out

  let isinf context a =
    let desc = B.descriptor a in
    let out = B.empty context uint8 (shape desc) in
    B.isinf context a out;
    out

  let isfinite context a =
    let desc = B.descriptor a in
    let out = B.empty context uint8 (shape desc) in
    B.isfinite context a out;
    out
end
