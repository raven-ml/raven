(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Completions, against rdma-core's mlx5dv.h: the queue pair's number in the low
   24 bits of the big-endian word at byte 56, the entry's index at byte 60, the
   opcode in the high 4 bits of byte 63 and the owner in its bit 0; an error's
   syndrome at byte 55 and the vendor's at byte 54. The owner rule is
   rdma-core's cq.c: a completion is the process's when its opcode is not 15
   (invalid) and its owner bit is the count's bit log2 entries. *)

open Windtrap
open Rig_mlx5_abi

let strf = Printf.sprintf
let size = 64

let buffer () =
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout (2 * size) in
  Bigarray.Array1.fill b '\x00';
  b

let set b at v = Bigarray.Array1.set b at (Char.chr v)

let set_be b at n v =
  for i = 0 to n - 1 do
    set b (at + i) ((v lsr (8 * (n - 1 - i))) land 0xff)
  done

(* A completion the NIC writes at [at]: [op] and [owner] in byte 63. *)
let completion b at ~qp ~index ~op ~owner ?(syndrome = 0) ?(vendor = 0) () =
  set_be b (at + 56) 4 ((0xab lsl 24) lor qp);
  set_be b (at + 60) 2 index;
  set b (at + 55) syndrome;
  set b (at + 54) vendor;
  set b (at + 63) ((op lsl 4) lor owner)

let pp_status ppf = function
  | Completion.Done -> Format.pp_print_string ppf "Done"
  | Failed { error; vendor } ->
      Format.fprintf ppf "Failed (%s, vendor 0x%x)"
        (Completion.message
           { qp = 0; index = 0; status = Failed { error; vendor } })
        vendor
  | Unexpected op -> Format.fprintf ppf "Unexpected 0x%x" op

let completion_t =
  Testable.make
    ~pp:(fun ppf (c : Completion.t) ->
      Format.fprintf ppf "{qp 0x%x; index %d; %a}" c.qp c.index pp_status
        c.status)
    ~equal:( = )

let ownership =
  let gen =
    Gen.(
      let+ log = int_range 0 22
      and+ count = frequency [ (3, int_range 0 (1 lsl 30)); (1, int_range 0 8) ]
      and+ op_own = int_range 0 255 in
      (1 lsl log, count, op_own))
  in
  group "ownership"
    [
      prop "a completion is owned iff valid and its owner is the count's bit"
        (Gen.with_pp
           (fun ppf (n, c, o) ->
             Format.fprintf ppf "%d entries, count %d, byte 0x%02x" n c o)
           gen)
        (fun (entries, count, op_own) ->
          let b = buffer () in
          set b (size + 63) op_own;
          let expected =
            op_own lsr 4 <> 15 && op_own land 1 = count / entries land 1
          in
          cover "owned" expected;
          cover "invalid opcode" (op_own lsr 4 = 15);
          equal bool expected (Completion.owned b size ~count ~entries));
      prop "an invalidated completion is owned at no count of its first pass"
        Gen.(pair (int_range 0 22) (int_range 0 (1 lsl 22)))
        (fun (log, count) ->
          let entries = 1 lsl log in
          let b = buffer () in
          completion b 0 ~qp:1 ~index:0 ~op:0 ~owner:0 ();
          Completion.invalidate b 0;
          is_false (Completion.owned b 0 ~count:(count mod entries) ~entries));
      test "the first pass's completions are owner 0, the second's owner 1"
        (fun () ->
          let b = buffer () in
          completion b 0 ~qp:1 ~index:0 ~op:0 ~owner:0 ();
          equal bool true (Completion.owned b 0 ~count:3 ~entries:4);
          equal bool false (Completion.owned b 0 ~count:7 ~entries:4);
          completion b 0 ~qp:1 ~index:0 ~op:0 ~owner:1 ();
          equal bool true (Completion.owned b 0 ~count:7 ~entries:4));
      test "entries that are not a power of two are refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"power of two") (fun () ->
              Completion.owned (buffer ()) 0 ~count:0 ~entries:3));
      test "a negative count is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
              Completion.owned (buffer ()) 0 ~count:(-1) ~entries:4));
      test "a completion past the buffer is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"outside") (fun () ->
              Completion.owned (buffer ()) (size + 1) ~count:0 ~entries:4));
    ]

(* The syndromes of mlx5dv.h, each with its error. *)
let syndromes =
  Completion.
    [
      (0x01, Local_length);
      (0x02, Local_qp_operation);
      (0x04, Local_protection);
      (0x05, Flushed);
      (0x06, Memory_window_bind);
      (0x10, Bad_response);
      (0x11, Local_access);
      (0x12, Remote_invalid_request);
      (0x13, Remote_access);
      (0x14, Remote_operation);
      (0x15, Retry_exceeded);
      (0x16, Receiver_not_ready_retry_exceeded);
      (0x22, Remote_aborted);
      (0x03, Other_error 0x03);
      (0xff, Other_error 0xff);
    ]

let reading =
  group "reading"
    [
      prop "a completed entry reads back its queue pair and index"
        Gen.(pair (int_range 0 0xff_ffff) (int_range 0 0xffff))
        (fun (qp, index) ->
          let b = buffer () in
          completion b 0 ~qp ~index ~op:0 ~owner:1 ();
          equal completion_t { qp; index; status = Done } (Completion.read b 0));
      cases
        ~name:(fun (s, _) -> strf "syndrome 0x%02x" s)
        "a requester's error reads its syndrome" syndromes
        (fun (syndrome, error) ->
          let b = buffer () in
          completion b size ~qp:0x1c3 ~index:7 ~op:13 ~owner:0 ~syndrome
            ~vendor:0x81 ();
          equal completion_t
            { qp = 0x1c3; index = 7; status = Failed { error; vendor = 0x81 } }
            (Completion.read b size));
      test "a responder's error is a failure too" (fun () ->
          let b = buffer () in
          completion b 0 ~qp:2 ~index:1 ~op:14 ~owner:0 ~syndrome:0x13
            ~vendor:0x88 ();
          equal completion_t
            {
              qp = 2;
              index = 1;
              status = Failed { error = Remote_access; vendor = 0x88 };
            }
            (Completion.read b 0));
      cases ~name:(strf "opcode %d") "another opcode is unexpected"
        [ 1; 2; 3; 4; 12; 15 ] (fun op ->
          let b = buffer () in
          completion b 0 ~qp:2 ~index:1 ~op ~owner:0 ();
          equal completion_t
            { qp = 2; index = 1; status = Unexpected op }
            (Completion.read b 0));
      test "messages" (fun () ->
          let b = buffer () in
          completion b 0 ~qp:0x1c3 ~index:7 ~op:13 ~owner:0 ~syndrome:0x15
            ~vendor:0x81 ();
          print_endline (Completion.message (Completion.read b 0));
          completion b 0 ~qp:0x1c3 ~index:8 ~op:13 ~owner:0 ~syndrome:0x05
            ~vendor:0x68 ();
          print_endline (Completion.message (Completion.read b 0));
          completion b 0 ~qp:0x1c3 ~index:9 ~op:0 ~owner:0 ();
          print_endline (Completion.message (Completion.read b 0));
          expect (output ())
          @@ __POS_OF__
               {|
            queue pair 0x1c3, entry 7: retry counter exceeded (vendor syndrome 0x81)
            queue pair 0x1c3, entry 8: flushed (vendor syndrome 0x68)
            queue pair 0x1c3, entry 9: completed
          |});
    ]

let () = exit (run "rig_mlx5_abi.completion" [ ownership; reading ])
