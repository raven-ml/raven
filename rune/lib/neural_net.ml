open Internal
open Tensor

(* one-hot encoding: converts integer labels to one-hot vectors *)
let one_hot (type a b c d) (dtype : (a, b) dtype) (labels : (c, d, [ `cpu ]) t)
    depth =
  let input_shape = shape labels in
  let n = size labels in
  let labels_flat = reshape labels [| n |] in
  let oh_flat = zeros dtype [| n; depth |] in
  let lbl_dtype = Dispatch.dtype labels.data in
  for i = 0 to n - 1 do
    let idx : int =
      match lbl_dtype with
      | Int8 -> get [| i |] labels_flat
      | UInt8 -> get [| i |] labels_flat
      | Int16 -> get [| i |] labels_flat
      | UInt16 -> get [| i |] labels_flat
      | Int32 -> Int32.to_int (get [| i |] labels_flat)
      | Int64 -> Int64.to_int (get [| i |] labels_flat)
      | _ -> failwith "one_hot: labels must have integer dtype"
    in
    let one : a = Ndarray_core.one dtype in
    set [| i; idx |] one oh_flat
  done;
  reshape oh_flat (Array.append input_shape [| depth |])
