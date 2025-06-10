(** Ready-to-use datasets for Kaun *)

val mnist :
  ?train:bool ->
  (* default: true *)
  ?flatten:bool ->
  (* default: false - keeps 28x28 shape *)
  ?normalize:bool ->
  (* default: true - scales to [0,1] *)
  ?data_format:[ `NCHW | `NHWC ] ->
  (* default: `NCHW *)
  unit ->
  ((Bigarray.float32_elt, 'dev) Kaun.tensor
  * (Bigarray.float32_elt, 'dev) Kaun.tensor)
  Kaun.Dataset.t
(** MNIST handwritten digits dataset *)
