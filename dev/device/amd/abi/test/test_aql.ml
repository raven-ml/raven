(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AQL packets over integers, against hsa.h's kernel dispatch packet and ROCr's
   vendor packet of PM4 commands. *)

open Windtrap
open Device_amd_abi

let words p =
  let s = Packet.encode Int64.of_int p in
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

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

let dispatch =
  group "dispatch"
    [
      test "a kernel dispatch packet" (fun () ->
          equal (list int)
            [
              header lor 2 lor (3 lsl 16);
              64 lor (2 lsl 16);
              1;
              128;
              4;
              1;
              16;
              1024;
              0x1000;
              0x1;
              0x2000;
              0x2;
              0;
              0;
              0;
              0;
            ]
            (words
               (Aql.dispatch kernel ~descriptor:0x1_0000_1000
                  ~args:0x2_0000_2000 ~threads:(64, 2, 1) ~grid:(128, 4, 1))));
      cases ~name:string_of_int "a workgroup side outside 16 bits is refused"
        [ 0; 0x10000 ] (fun t ->
          raises_match (Exn.invalid_arg ~substring:"Aql.dispatch") (fun () ->
              Aql.dispatch kernel ~descriptor:0 ~args:0 ~threads:(1, t, 1)
                ~grid:(1, 1, 1)));
    ]

let indirect =
  group "indirect_buffer"
    [
      test "PM4 words in a vendor packet of 16 words" (fun () ->
          equal (list int)
            ([ header lor (1 lsl 16) ]
            @ words (Pm4.indirect_buffer 0x1_0000_0100 ~dwords:16)
            @ [ 10 ]
            @ List.init 10 (fun _ -> 0))
            (words (Aql.indirect_buffer 0x1_0000_0100 ~dwords:16)));
      test "an indirect buffer past IB_SIZE is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Aql.indirect_buffer")
            (fun () -> Aql.indirect_buffer 0 ~dwords:(1 lsl 20)));
    ]

let () = exit (run "device_amd_abi.aql" [ dispatch; indirect ])
