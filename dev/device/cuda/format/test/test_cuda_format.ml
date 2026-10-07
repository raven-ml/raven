(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The key, as a device's records are found: in a table of bindings, each under
   its own key. *)

open Windtrap
module Cuda = Device_cuda_format

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

let test_found () =
  let other : int Type.Id.t = Type.Id.make () in
  let bindings = [ B (other, 1); B (Cuda.key, { Cuda.symbol }) ] in
  match find Cuda.key bindings with
  | None -> fail "no record under the key"
  | Some cuda ->
      equal ~msg:"its lookup" (option nativeint) (Some launch_kernel)
        (cuda.symbol "cuLaunchKernel")

let test_alone () =
  let other : Cuda.t Type.Id.t = Type.Id.make () in
  is_none ~msg:"another key of the same type"
    (find other [ B (Cuda.key, { Cuda.symbol }) ]);
  is_none ~msg:"the key under another key's binding"
    (find Cuda.key [ B (other, { Cuda.symbol }) ])

let () =
  exit
  @@ run "device_cuda_format"
       [
         group ~timeout:10. "key"
           [
             test "a record declared under the key is found under it" test_found;
             test "no other key finds it, and it finds no other" test_alone;
           ];
       ]
