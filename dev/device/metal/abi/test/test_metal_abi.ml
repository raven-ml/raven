(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The key finds a Metal device's record among bindings under other keys. *)

open Windtrap
module Metal = Device_metal_abi

type binding = B : 'a Type.Id.t * 'a -> binding

let find : type a. a Type.Id.t -> binding list -> a option =
 fun k bindings ->
  let found (B (k', v)) : a option =
    match Type.Id.provably_equal k k' with
    | Some Type.Equal -> Some v
    | None -> None
  in
  List.find_map found bindings

let split = 0x7f00_2000n
let record = { Metal.align = 4; icb = (fun _ _ -> Error "no Metal"); split }

let test_found () =
  let other : int Type.Id.t = Type.Id.make () in
  match find Metal.key [ B (other, 1); B (Metal.key, record) ] with
  | None -> fail "no record under the key"
  | Some metal -> equal ~msg:"its split" nativeint split metal.split

let test_alone () =
  let other : Metal.t Type.Id.t = Type.Id.make () in
  is_none ~msg:"another key of the same type"
    (find other [ B (Metal.key, record) ]);
  is_none ~msg:"the key under another key's binding"
    (find Metal.key [ B (other, record) ])

let () =
  exit
  @@ run "device_metal_abi"
       [
         group ~timeout:10. "key"
           [
             test "a record declared under the key is found under it" test_found;
             test "no other key finds it, and it finds no other" test_alone;
           ];
       ]
