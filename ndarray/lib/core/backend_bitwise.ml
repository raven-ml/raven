open Descriptor
open Views

module Make (B : Backend_intf.S) = struct
  let bitwise_and context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_desc' = broadcast_to a_desc out_shape in
    let b_desc' = broadcast_to b_desc out_shape in
    let a_op = B.view a_desc' a in
    let b_op = B.view b_desc' b in
    B.bit_and context a_op b_op c;
    c

  let bitwise_or context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_desc' = broadcast_to a_desc out_shape in
    let b_desc' = broadcast_to b_desc out_shape in
    let a_op = B.view a_desc' a in
    let b_op = B.view b_desc' b in
    B.bit_or context a_op b_op c;
    c

  let bitwise_xor context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_desc' = broadcast_to a_desc out_shape in
    let b_desc' = broadcast_to b_desc out_shape in
    let a_op = B.view a_desc' a in
    let b_op = B.view b_desc' b in
    B.bit_xor context a_op b_op c;
    c

  let invert context a =
    let a_desc = B.descriptor a in
    let out = B.empty context (dtype a_desc) (shape a_desc) in
    B.bit_not context a out;
    out
end
