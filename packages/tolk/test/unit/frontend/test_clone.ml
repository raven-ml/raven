(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module U = Tolk_uop.Uop
module D = Tolk_uop.Dtype
module T = Tolk_frontend.Tensor
module Creation = Tolk_frontend.Creation
module Movement = Tolk_frontend.Movement
module Op = Tolk_frontend.Op
module Reduce = Tolk_frontend.Reduce
module Run = Tolk_frontend.Run

let devices = [ "CPU:1"; "CPU:2" ]
let cpu = U.Single "CPU"
let shape dims = T.symbolic_shape_uop (List.map U.const_int dims)

let allocation_size tensor =
  let base = U.storage_base (T.uop tensor) in
  match U.op base, U.arg base with
  | Tolk_uop.Ops.Alloc, U.Arg.Param_arg { size = Some size; _ } -> size
  | _ -> fail "clone does not own an allocation"

let gather tensor = Run.to_float_array (Creation.clone ~device:cpu tensor)
let check_values expected tensor = equal (array float_exact) expected (gather tensor)

let graph_clone () =
  List.iter (fun axis ->
      let source = T.of_uop (U.param ~slot:0 ~dtype:D.float32 ~axis
          ~device:(U.Multi (List.map Option.some devices)) ~shape:(shape [4; 6]) ()) in
      let clone = Creation.clone source in
      equal (list int) [4; 6] (T.shape clone);
      equal (option int) (Some axis) (U.axis (T.uop clone));
      equal (option (list string)) (Some devices)
        (match T.device clone with Some (U.Multi devices) -> Some (List.map Option.get devices) | _ -> None);
      equal int 12 (allocation_size clone);
      equal (list int) (if axis = 0 then [2; 6] else [4; 3])
        (U.max_shard_shape (T.uop clone));
      let gathered = Creation.clone ~device:cpu source in
      equal (option int) None (U.axis (T.uop gathered));
      equal int 24 (allocation_size gathered);
      let singleton = Creation.clone ~device:(U.Multi [Some "cpu:3"]) source in
      equal (list int) [4; 6] (T.shape singleton);
      equal (option int) None (U.axis (T.uop singleton));
      equal int 24 (allocation_size singleton);
      equal (option string) (Some "CPU:3")
        (match T.device singleton with Some (U.Single device) -> Some device | _ -> None)) [0; 1]

let clone_and_assign () =
  List.iter (fun axis ->
      let values = Array.init 16 (fun i -> float_of_int (i + 1)) in
      let input = Run.of_float_array ~shape:[4; 4] values in
      let source = Creation.shard ~axis ~devices input in
      let clone = Creation.clone source and sibling = Creation.clone source in
      equal int 8 (allocation_size clone);
      is_false ~msg:"separate clones own separate allocations"
        (U.equal (U.storage_base (T.uop clone)) (U.storage_base (T.uop sibling)));
      check_values values clone;
      check_values values sibling;
      let view = Movement.shrink clone
          (if axis = 0 then [0, 4; 0, 1] else [0, 1; 0, 4]) in
      ignore (Op.assign view (T.f 99.));
      let expected = Array.mapi (fun i value ->
          if (if axis = 0 then i mod 4 = 0 else i < 4) then 99. else value) values in
      check_values expected clone;
      check_values values source;
      check_values values sibling;
      equal (option int) (Some axis) (U.axis (T.uop clone))) [0; 1]

let cross_device_assignment () =
  let buffer tensor = match Run.buffer_of_node (T.uop tensor) with
    | Some buffer -> buffer
    | None -> fail "realized tensor has no owned buffer" in
  List.iter (fun partial ->
      List.iter (fun pending ->
          let initial = [|-1.; -2.; -3.; -4.|] in
          let source_values = if partial then [|7.; 8.|] else [|7.; 8.; 9.; 10.|] in
          let tensor device values = Creation.clone ~device:(U.Single device)
              (Run.of_float_array ~shape:[Array.length values] values) in
          let destination = tensor "CPU:1" initial in
          let source = tensor "CPU:2" source_values in
          ignore (Run.realize destination);
          ignore (Run.realize source);
          let allocation = buffer destination in
          equal string "CPU:1" (Tolk.Device.Buffer.device allocation);
          equal string "CPU:2" (Tolk.Device.Buffer.device (buffer source));
          let alias = Movement.reshape destination [2; 2] in
          let target = if partial then Movement.shrink destination [1, 3] else destination in
          let value = if pending then Tolk_frontend.Elementwise.add source (T.f 10.) else source in
          ignore (Op.assign target value);
          let values = Array.map (fun value -> value +. if pending then 10. else 0.) source_values in
          let expected = if partial then [|-1.; values.(0); values.(1); -4.|] else values in
          check_values expected destination;
          check_values expected alias;
          check_values source_values source;
          equal int (Tolk.Device.Buffer.id allocation)
            (Tolk.Device.Buffer.id (buffer destination))) [false; true]) [false; true]

let noncontiguous_cross_device_assignment () =
  let tensor device dims values =
    Creation.clone ~device:(U.Single device) (Run.of_float_array ~shape:dims values) in
  let destination = tensor "CPU:1" [2; 2] [|1.; 2.; 3.; 4.|] in
  let source = tensor "CPU:2" [2; 1] [|7.; 8.|] in
  ignore (Run.realize destination);
  ignore (Run.realize source);
  let target = Movement.shrink destination [0, 2; 0, 1] in
  ignore (Op.assign target source);
  raises_match
    (function Invalid_argument message ->
       String.starts_with ~prefix:"all buffers must be on the same device:" message
     | _ -> false)
    (fun () -> ignore (Run.realize destination));
  check_values [|7.; 8.|] source

let symbolic_clone () =
  List.iter (fun length ->
      let values = Array.init 16 (fun i -> float_of_int (i + 1)) in
      let input = Run.of_float_array ~shape:[4; 4] values in
      let source = Creation.shard ~axis:0 ~devices input in
      let var = U.variable ~name:"clone_columns" ~min_val:1 ~max_val:4 () in
      let bound = U.bind ~var ~value:(U.const_int length) in
      let source = Movement.symbolic_shrink source [None; Some (U.const_int 0, bound)] in
      let clone = Creation.clone source in
      is_true (List.equal U.equal (T.symbolic_shape source) (T.symbolic_shape clone));
      equal (option int) (Some 0) (U.axis (T.uop clone));
      equal int 8 (allocation_size clone);
      let expected = Array.fold_left ( +. ) 0.
          (Array.mapi (fun i value -> if i mod 4 < length then value else 0.) values) in
      check_values [|expected|] (Reduce.sum clone)) [1; 3; 4]

let disk_clone () =
  let path = Filename.temp_file "tolk_clone" ".bin" in
  Sys.remove path;
  Fun.protect ~finally:(fun () -> if Sys.file_exists path then Sys.remove path) (fun () ->
      let device = U.Single ("DISK:" ^ path) in
      let source = Creation.empty ~device [2] in
      List.iter (fun clone ->
          raises_match (function Invalid_argument message -> String.starts_with
              ~prefix:"Creation.clone: cannot clone DISK storage" message | _ -> false) clone;
          equal bool false (Sys.file_exists path))
        [ (fun () -> ignore (Creation.clone source));
          (fun () -> ignore (Creation.clone ~device (T.f 1.)));
          (fun () -> ignore (Creation.clone ~device:(U.Single ("disk:" ^ path)) (T.f 1.))) ];
      let host = Creation.clone ~device:cpu source in
      equal (option string) (Some "CPU")
        (match T.device host with Some (U.Single device) -> Some device | _ -> None);
      is_true (List.exists (fun u -> U.op u = Tolk_uop.Ops.Copy) (U.toposort (T.uop host)));
      equal bool false (Sys.file_exists path))

let () =
  exit (Tolk.Helpers.Context_var.with_context
    [B (Tolk.Helpers.dev, [Tolk_uop.Target.of_string "CPU"])] (fun () ->
      run "Clone"
        [test "sharded clones allocate per shard and preserve their axis" graph_clone;
         test "sharded clone views write independently of their source and sibling" clone_and_assign;
         test "cross-device assignment preserves destination aliases and source storage" cross_device_assignment;
         test "cross-device assignment rejects noncontiguous destinations" noncontiguous_cross_device_assignment;
         test "sharded clones preserve symbolic nonsharded dimensions" symbolic_clone;
         test "DISK destinations reject before allocation and host gathers remain available" disk_clone]))
