(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AQL packets over integers, against hsa.h's kernel dispatch packet and ROCr's
   vendor packet of PM4 commands. *)

open Windtrap
open Device_amd_abi

let timeout = Device_amd_abi_support.timeout
let words = Device_amd_abi_support.encode

let kernel : Code_object.kernel =
  {
    descriptor = 0x1000;
    entry = 0x1100;
    group_segment = 1024;
    private_segment = 16;
    kernarg_size = 24;
    rsrc1 = 0;
    rsrc2 = 0;
    rsrc3 = 0;
    wave32 = true;
    dispatch_ptr = true;
    private_segment_buffer = false;
  }

(* A barrier, and system scope for both fences. *)
let header = (1 lsl 8) lor (2 lsl 9) lor (2 lsl 11)

let dispatches =
  let open Gen in
  let side = int_range 1 0xffff and u32 = int_range 0 0xffff_ffff in
  let addr = int_range 0 ((1 lsl 48) - 1) in
  with_pp
    (fun ppf ((k : Code_object.kernel), (x, y, z), (gx, gy, gz), d, a) ->
      Format.fprintf ppf
        "private %d, group %d, threads (%d, %d, %d), grid (%d, %d, %d), \
         descriptor 0x%x, args 0x%x"
        k.private_segment k.group_segment x y z gx gy gz d a)
    (let+ private_segment = u32
     and+ group_segment = u32
     and+ threads = triple side side side
     and+ grid = triple u32 u32 u32
     and+ descriptor = addr
     and+ args = addr in
     ( { kernel with private_segment; group_segment },
       threads,
       grid,
       descriptor,
       args ))

let dispatch =
  group ~timeout "dispatch"
    [
      prop "a dispatch packet lays out hsa.h's fields" dispatches
        (fun (k, (x, y, z), (gx, gy, gz), descriptor, args) ->
          let lo n = n land 0xffff_ffff and hi n = n lsr 32 in
          equal (list int)
            [
              header lor 2 lor (3 lsl 16);
              x lor (y lsl 16);
              z;
              gx;
              gy;
              gz;
              k.Code_object.private_segment;
              k.group_segment;
              lo descriptor;
              hi descriptor;
              lo args;
              hi args;
              0;
              0;
              0;
              0;
            ]
            (words
               (Aql.dispatch k ~descriptor ~args ~threads:(x, y, z)
                  ~grid:(gx, gy, gz))));
      cases ~name:string_of_int "a workgroup side of 16 bits is taken"
        [ 1; 0xffff ] (fun t ->
          equal int
            (t lor (t lsl 16))
            (List.nth
               (words
                  (Aql.dispatch kernel ~descriptor:0 ~args:0 ~threads:(t, t, t)
                     ~grid:(1, 1, 1)))
               1));
      cases ~name:string_of_int "a workgroup side outside 16 bits is refused"
        [ 0; 0x10000 ] (fun t ->
          raises_match (Exn.invalid_arg ~substring:"Aql.dispatch") (fun () ->
              Aql.dispatch kernel ~descriptor:0 ~args:0 ~threads:(1, t, 1)
                ~grid:(1, 1, 1)));
    ]

let indirect =
  group ~timeout "indirect_buffer"
    [
      test "PM4 words in a vendor packet of 16 words" (fun () ->
          equal (list int)
            ([ header lor (1 lsl 16) ]
            @ words (Pm4.indirect_buffer 0x1_0000_0100 ~dwords:16)
            @ [ 10 ]
            @ List.init 10 (fun _ -> 0))
            (words (Aql.indirect_buffer 0x1_0000_0100 ~dwords:16)));
    ]

let () = exit (run "device_amd_abi.aql" [ dispatch; indirect ])
