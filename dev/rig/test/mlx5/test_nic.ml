(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A ConnectX NIC of this machine, opened through the kernel: two queue pairs of
   one NIC connected to each other, the transfers between their regions, and
   what the library refuses. Every test skips where the machine has no NIC the
   mlx5 driver holds. *)

open Windtrap
module M = Rig_mlx5
module E = Rig_mlx5_abi.Entry
module C = Rig_mlx5_abi.Completion

external address : Rig_mlx5_abi.buffer -> int = "rig_mlx5_test_address"

let nic () =
  match Rig_mlx5_uverbs.names () with
  | [] -> skip ~reason:"no NIC the mlx5 driver holds" ()
  | name :: _ -> (
      match Rig_mlx5_uverbs.open_ name with
      | Ok nic -> nic
      | Error e -> failf "opening %s: %s" name e)

(* Host memory for a region, kept for the run. *)
let kept = ref []

let memory n =
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill b '\000';
  kept := b :: !kept;
  b

let register nic access b =
  match
    M.Region.register nic access
      (Host { address = address b; bytes = Bigarray.Array1.dim b })
  with
  | Ok r -> r
  | Error e -> failf "registering: %s" e

(* Two queue pairs of [nic] connected to each other, on one queue. *)
let pair nic =
  let cq = Result.get_ok (M.Cq.make nic 64) in
  let a = Result.get_ok (M.Qp.make cq 16) in
  let b = Result.get_ok (M.Qp.make cq 16) in
  (match M.Qp.connect a (M.Qp.endpoint b) with
  | Ok () -> ()
  | Error e -> failf "connecting: %s" e);
  (match M.Qp.connect b (M.Qp.endpoint a) with
  | Ok () -> ()
  | Error e -> failf "connecting: %s" e);
  (cq, a)

(* The next completion of [cq], within 5 s. *)
let completion cq =
  let deadline = Unix.gettimeofday () +. 5. in
  let rec go () =
    match M.Cq.poll cq with
    | Some c -> c
    | None when Unix.gettimeofday () < deadline -> go ()
    | None -> fail "no completion within 5 s"
  in
  go ()

let pp_status ppf (s : C.status) =
  match s with
  | Done -> Format.pp_print_string ppf "done"
  | Failed _ | Unexpected _ ->
      Format.pp_print_string ppf (C.message { qp = 0; index = 0; status = s })

let status = Testable.make ~pp:pp_status ~equal:( = )
let string_of b = String.init (Bigarray.Array1.dim b) (Bigarray.Array1.get b)

let transfers =
  group "transfers"
    [
      test "a write lands its bytes in the peer's region" (fun () ->
          let nic = nic () in
          let cq, qp = pair nic in
          let src = memory 4096 and dst = memory 4096 in
          String.iteri
            (fun i c -> Bigarray.Array1.set src i c)
            (String.init 4096 (fun i -> Char.chr (i land 0xff)));
          let s = register nic Local src
          and d = register nic Remote_write dst in
          M.Qp.post qp
            {
              op =
                Write
                  { src = M.Region.local s 0 4096; dst = M.Region.remote d 0 };
              signal = true;
            };
          M.Qp.ring qp;
          equal status Done (completion cq).status;
          equal string (string_of src) (string_of dst);
          M.close nic);
      test "a read brings the peer's bytes back" (fun () ->
          let nic = nic () in
          let cq, qp = pair nic in
          let src = memory 64 and dst = memory 64 in
          Bigarray.Array1.fill src 'r';
          let s = register nic Remote src and d = register nic Local dst in
          M.Qp.post qp
            {
              op =
                Read { src = M.Region.remote s 0; dst = M.Region.local d 0 64 };
              signal = true;
            };
          M.Qp.ring qp;
          equal status Done (completion cq).status;
          equal string (String.make 64 'r') (string_of dst);
          M.close nic);
      test "a write of no bytes completes and writes nothing" (fun () ->
          let nic = nic () in
          let cq, qp = pair nic in
          let src = memory 64 and dst = memory 64 in
          Bigarray.Array1.fill src 's';
          let s = register nic Local src
          and d = register nic Remote_write dst in
          M.Qp.post qp
            {
              op =
                Write { src = M.Region.local s 0 0; dst = M.Region.remote d 0 };
              signal = true;
            };
          M.Qp.ring qp;
          equal status Done (completion cq).status;
          equal string (String.make 64 '\000') (string_of dst);
          M.close nic);
      test "an inline write lands the bytes the entry holds" (fun () ->
          let nic = nic () in
          let cq, qp = pair nic in
          let dst = memory 64 in
          let d = register nic Remote_write dst in
          M.Qp.post qp
            {
              op =
                Write_inline
                  {
                    data = "\007\000\000\000\000\000\000\000";
                    dst = M.Region.remote d 8;
                  };
              signal = true;
            };
          M.Qp.ring qp;
          equal status Done (completion cq).status;
          equal string "\007\000\000\000\000\000\000\000"
            (String.sub (string_of dst) 8 8);
          M.close nic);
    ]

let refusals =
  group "refusals"
    [
      test "queue pairs that would hold more than their queue are refused"
        (fun () ->
          let nic = nic () in
          let cq = Result.get_ok (M.Cq.make nic 16) in
          ignore (Result.get_ok (M.Qp.make cq 8) : M.Qp.t);
          ignore (Result.get_ok (M.Qp.make cq 8) : M.Qp.t);
          raises_match (Exn.invalid_arg ~substring:"Qp.make") (fun () ->
              M.Qp.make cq 1);
          M.close nic);
      test "a piece whose end overflows is refused" (fun () ->
          let nic = nic () in
          let r = register nic Remote (memory 64) in
          raises_match (Exn.invalid_arg ~substring:"Region") (fun () ->
              M.Region.local r max_int 1);
          raises_match (Exn.invalid_arg ~substring:"Region") (fun () ->
              M.Region.local r 1 max_int);
          M.close nic);
      test "calls after close raise, and a second close does nothing" (fun () ->
          let nic = nic () in
          let cq = Result.get_ok (M.Cq.make nic 4) in
          M.close nic;
          M.close nic;
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              M.Cq.poll cq));
    ]

let events =
  test "wait with no event waits its whole time" (fun () ->
      let nic = nic () in
      let start = Unix.gettimeofday () in
      equal int 0 (List.length (M.wait nic ~ms:50));
      (* The library times on a monotonic clock, this test on the wall clock: a
         millisecond covers their difference. *)
      at_least ~msg:"seconds waited" float_exact ~than:0.049
        (Unix.gettimeofday () -. start);
      M.close nic)

let () = exit (run "rig_mlx5" [ transfers; refusals; events ])
