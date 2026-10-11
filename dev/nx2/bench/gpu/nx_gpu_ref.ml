(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type view = { bytes : string; dtype : int; strides : int array }
type result = { worst : float; wrong : int; at : int }

external contract_c : view array -> int array -> int array -> result
  = "nx_gpu_ref_contract"

let contract ~a ~b ?init ~y ~batch ~m ~n ~k ~acc ?(flush = false) ~samples () =
  let views =
    match init with None -> [| a; b; y |] | Some i -> [| a; b; y; i |]
  in
  let r =
    contract_c views [| batch; m; n; k |] [| acc; Bool.to_int flush; samples |]
  in
  (* windtrap's at_most orders NaN below every number: a NaN worst would
     pass a bound. *)
  if Float.is_nan r.worst then failwith "Nx_gpu_ref.contract: a NaN ratio";
  r
