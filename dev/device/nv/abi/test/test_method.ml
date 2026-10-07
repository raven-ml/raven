(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Channel methods against NVIDIA's class headers: clc56f.h (host), clc7b5.h
   (copy engine). *)

open Windtrap
open Device_nv_abi

let words p =
  let s = Packet.encode Int64.of_int p in
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

(* The runs of [ws] as (subchannel, method, arguments), each header an
   incrementing method (SEC_OP 1) counting the words after it. *)
let rec runs = function
  | [] -> []
  | h :: rest ->
      equal ~msg:"an incrementing method" int 1 (h lsr 29);
      let n = (h lsr 16) land 0x1fff in
      ( (h lsr 13) land 7,
        (h land 0xfff) lsl 2,
        List.filteri (fun i _ -> i < n) rest )
      :: runs (List.filteri (fun i _ -> i >= n) rest)

let semaphores =
  group "semaphores"
    [
      cases
        ~name:(function Packet.Agent -> "Agent" | System -> "System")
        "a release waits for idle, then writes 64 bits" [ Agent; System ]
        (fun s ->
          match
            runs (words (Method.release s 0x12_3456_7890 ((1 lsl 33) + 5)))
          with
          | [ (0, 0x5c, [ alo; ahi; vlo; vhi; exe ]) ] ->
              equal ~msg:"address" int 0x12_3456_7890 (alo lor (ahi lsl 32));
              equal ~msg:"value" int ((1 lsl 33) + 5) (vlo lor (vhi lsl 32));
              equal ~msg:"RELEASE, RELEASE_WFI_EN, PAYLOAD_SIZE_64BIT" int
                (1 lor (1 lsl 20) lor (1 lsl 24))
                exe
          | _ -> fail "a release is one run of SEM_ADDR_LO to SEM_EXECUTE");
    ]

let copies =
  group "copies"
    [
      test "a copy release writes a 64-bit payload" (fun () ->
          match
            runs
              (words
                 (Method.copy_release System 0x1_0000_0010 ((1 lsl 32) + 7)))
          with
          | [ (4, 0x240, [ hi; lo; plo; phi ]); (4, 0x300, [ launch ]) ] ->
              equal ~msg:"address, high word first" (pair int int) (1, 0x10)
                (hi, lo);
              equal ~msg:"payload" (pair int int) (7, 1) (plo, phi);
              equal ~msg:"PAYLOAD_SIZE_TWO_WORD" int 1 ((launch lsr 27) land 1)
          | _ ->
              fail
                "a copy release is SET_SEMAPHORE_A to _PAYLOAD_UPPER and a \
                 launch");
    ]

let () = exit (run "device_nv_abi.method" [ semaphores; copies ])
