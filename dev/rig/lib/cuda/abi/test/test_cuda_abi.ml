(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The key finds a CUDA device's record among bindings under other keys. *)

open Windtrap
module Cuda = Rig_cuda_abi

type binding = B : 'a Type.Id.t * 'a -> binding

let find : type a. a Type.Id.t -> binding list -> a option =
 fun k bindings ->
  let found (B (k', v)) : a option =
    match Type.Id.provably_equal k k' with
    | Some Type.Equal -> Some v
    | None -> None
  in
  List.find_map found bindings

let launch_kernel = 0x7f00_1000n
let symbol = function "cuLaunchKernel" -> Some launch_kernel | _ -> None
let graph _ = Error "no graphs here"
let record = { Cuda.symbol; graph }

let test_found () =
  let other : int Type.Id.t = Type.Id.make () in
  let bindings = [ B (other, 1); B (Cuda.key, record) ] in
  match find Cuda.key bindings with
  | None -> fail "no record under the key"
  | Some cuda ->
      equal ~msg:"its lookup" (option nativeint) (Some launch_kernel)
        (cuda.symbol "cuLaunchKernel")

let test_alone () =
  let other : Cuda.t Type.Id.t = Type.Id.make () in
  is_none ~msg:"another key of the same type"
    (find other [ B (Cuda.key, record) ]);
  is_none ~msg:"the key under another key's binding"
    (find Cuda.key [ B (other, record) ])

let () =
  exit
  @@ run "rig_cuda_abi"
       [
         group ~timeout:10. "key"
           [
             test "a record declared under the key is found under it" test_found;
             test "no other key finds it, and it finds no other" test_alone;
           ];
       ]
