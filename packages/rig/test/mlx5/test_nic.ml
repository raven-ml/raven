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

let read path =
  match In_channel.with_open_bin path In_channel.input_all with
  | s -> Some (String.trim s)
  | exception Sys_error _ -> None

(* Whether [name]'s port 1 carries RDMA, from /sys/class/infiniband: the type
   file of an unused identifier cannot be read, and a port whose link layer
   cannot be read is left to opening. *)
let carries_rdma name =
  let port = "/sys/class/infiniband/" ^ name ^ "/ports/1" in
  match read (port ^ "/link_layer") with
  | Some "Ethernet" ->
      let types = port ^ "/gid_attrs/types" in
      let entries = try Sys.readdir types with Sys_error _ -> [||] in
      Array.exists (fun e -> read (types ^ "/" ^ e) = Some "RoCE v2") entries
  | _ -> true

(* The NIC these tests drive: the first the mlx5 driver holds whose port 1
   carries RDMA, as the kernel's files say before anything opens it. A port
   carries RDMA on InfiniBand, or on Ethernet with a RoCE v2 global identifier.
   A ConnectX function whose RoCE is off, as a cloud machine's virtual function,
   lists an Ethernet port with none, which Rig_mlx5.open_ refuses: these tests
   skip there. A NIC that carries RDMA and fails to open fails the test. *)
let nic_name () =
  match Rig_mlx5.names () with
  | Error e -> failf "listing the NICs: %s" e
  | Ok [] -> skip ~reason:"no NIC the mlx5 driver holds" ()
  | Ok names -> (
      match List.find_opt carries_rdma names with
      | Some name -> name
      | None ->
          skip
            ~reason:
              (Printf.sprintf
                 "no RDMA on %s: port 1 is Ethernet with no RoCE v2 identifier"
                 (String.concat ", " names))
            ())

let nic () =
  let name = nic_name () in
  match Rig_mlx5.open_ name with
  | Ok nic -> nic
  | Error e -> failf "opening %s: %s" name e

let memory n =
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill b '\000';
  b

let register nic access order b =
  match
    M.Region.register nic access order (Host b)
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
          let s = register nic Local In_order src
          and d = register nic Remote_write Any_order dst in
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
          let s = register nic Remote In_order src
          and d = register nic Local In_order dst in
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
          let s = register nic Local In_order src
          and d = register nic Remote_write Any_order dst in
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
          let d = register nic Remote_write Any_order dst in
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
          let r = register nic Remote In_order (memory 64) in
          raises_match (Exn.invalid_arg ~substring:"Region") (fun () ->
              M.Region.local r max_int 1);
          raises_match (Exn.invalid_arg ~substring:"Region") (fun () ->
              M.Region.local r 1 max_int);
          M.close nic);
      test
        "a call inside a lock of its thread's raises naming it, and one \
         outside runs" (fun () ->
          let nic = nic () in
          let refused fn f =
            match Rig_support.inside_lock f with
            | () -> failf "%s ran inside a lock" fn
            | exception Invalid_argument why ->
                starts_with ~msg:fn
                  ~affix:("Rig_mlx5." ^ fn ^ ": called from")
                  why
          in
          refused "Cq.make" (fun () -> ignore (M.Cq.make nic 4));
          refused "wait" (fun () -> ignore (M.wait nic ~ms:0));
          refused "close" (fun () -> M.close nic);
          ignore (Result.get_ok (M.Cq.make nic 4) : M.Cq.t);
          M.close nic);
      test "calls after close raise, and a second close does nothing" (fun () ->
          let nic = nic () in
          let cq = Result.get_ok (M.Cq.make nic 4) in
          M.close nic;
          M.close nic;
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              M.Cq.poll cq));
    ]

(* Where the mlx5 driver's answer to the context holds the most entries a
   queue pair's send ring holds: mlx5_ib_alloc_ucontext_resp's
   max_send_wqebb, a u32, in Linux's include/uapi/rdma/mlx5-abi.h. *)
let max_send_wqebb_at = 20

(* The rings' own bounds: a queue's consumer count is 24 bits, and a queue
   pair's context gives its ring's size as a 4-bit log. *)
let most_completions = 1 lsl 24
let most_entries = 1 lsl 15
let u32 s at = Int32.to_int (String.get_int32_le s at) land 0xffff_ffff

let with_u32 s at v =
  let b = Bytes.of_string s in
  Bytes.set_int32_le b at (Int32.of_int v);
  Bytes.to_string b

(* The NIC through its path, whose context [with_context] may change: the
   bounds a queue is checked against are those the context reads. *)
let through ?(with_context = Fun.id) () =
  let name = nic_name () in
  let p =
    match Rig_mlx5.path name with
    | Ok p -> p
    | Error e -> failf "opening %s: %s" name e
  in
  let context d n = Result.map with_context (p.context d n) in
  require_ok (M.make { p with context })

let bounds =
  group "bounds"
    [
      test "a queue past the NIC's most completions is refused" (fun () ->
          let cqes = ref 0 in
          let nic =
            through
              ~with_context:(fun (c : M.context) ->
                cqes := c.cqes;
                c)
              ()
          in
          let most = min !cqes most_completions in
          let why = require_error (M.Cq.make nic (most + 1)) in
          contains ~sub:(Printf.sprintf "at most %d" most) why;
          raises_match (Exn.invalid_arg ~substring:"Cq.make") (fun () ->
              M.Cq.make nic 0);
          M.close nic);
      test "a queue's bound is the one the NIC's context reads" (fun () ->
          let nic =
            through
              ~with_context:(fun (c : M.context) -> { c with cqes = 64 })
              ()
          in
          is_error ~msg:"128 completions" (M.Cq.make nic 128);
          is_error ~msg:"65 completions" (M.Cq.make nic 65);
          ignore (require_ok (M.Cq.make nic 64) : M.Cq.t);
          M.close nic);
      test "a queue pair holds the NIC's most entries, and no more" (fun () ->
          let sends = ref 0 in
          let nic =
            through
              ~with_context:(fun (c : M.context) ->
                sends := u32 c.answer max_send_wqebb_at;
                c)
              ()
          in
          let most = min !sends most_entries in
          let cq = require_ok (M.Cq.make nic most) in
          ignore (require_ok (M.Qp.make cq most) : M.Qp.t);
          let why = require_error (M.Qp.make cq (most + 1)) in
          contains ~sub:(Printf.sprintf "at most %d" most) why;
          raises_match (Exn.invalid_arg ~substring:"Qp.make") (fun () ->
              M.Qp.make cq 0);
          M.close nic);
      test "a queue pair's bound is the one the driver's answer states"
        (fun () ->
          let nic =
            through
              ~with_context:(fun (c : M.context) ->
                { c with answer = with_u32 c.answer max_send_wqebb_at 64 })
              ()
          in
          let cq = require_ok (M.Cq.make nic 256) in
          is_error ~msg:"128 entries" (M.Qp.make cq 128);
          is_error ~msg:"65 entries" (M.Qp.make cq 65);
          ignore (require_ok (M.Qp.make cq 64) : M.Qp.t);
          M.close nic);
    ]

(* The NIC writes a host region until it is deregistered, so it keeps the
   bigarray: here only the NIC reaches it. *)
let lifetime =
  test "a registered bigarray lives until its region is deregistered"
    (fun () ->
      let nic = nic () in
      let freed = Atomic.make false in
      let[@inline never] region () =
        let b = memory 64 in
        Gc.finalise (fun _ -> Atomic.set freed true) b;
        register nic Local In_order b
      in
      let r = region () in
      Gc.full_major ();
      equal bool ~msg:"while registered" false (Atomic.get freed);
      require_ok (M.Region.deregister r);
      Gc.full_major ();
      equal bool ~msg:"once deregistered" true (Atomic.get freed);
      M.close nic)

(* The kernel's object stays when its destroy is refused, which this path does
   by keeping its queue pairs and regions: the object is gone here all the same,
   a second destroy answers as the first, a queue pair's entries keep their
   room, and the kernel refuses its queue's destroy in turn. *)
let kept =
  test "an object the kernel keeps is destroyed here" (fun () ->
      let p =
        let name = nic_name () in
        match Rig_mlx5.path name with
        | Ok p -> p
        | Error e -> failf "opening %s: %s" name e
      in
      let destroy kind h =
        match kind with
        | `Qp | `Region -> Error "kept"
        | `Cq -> p.destroy kind h
      in
      let nic = require_ok (M.make { p with destroy }) in
      let cq = require_ok (M.Cq.make nic 16) in
      let qp = require_ok (M.Qp.make cq 16) in
      is_error ~msg:"the queue pair" (M.Qp.destroy qp);
      is_error ~msg:"a second destroy" (M.Qp.destroy qp);
      raises_match (Exn.invalid_arg ~substring:"destroyed") (fun () ->
          M.Qp.ring qp);
      raises_match (Exn.invalid_arg ~substring:"Qp.make") (fun () ->
          M.Qp.make cq 1);
      is_error ~msg:"its queue" (M.Cq.destroy cq);
      raises_match (Exn.invalid_arg ~substring:"destroyed") (fun () ->
          M.Cq.poll cq);
      let r = register nic Remote In_order (memory 64) in
      is_error ~msg:"a region" (M.Region.deregister r);
      is_error ~msg:"a second deregister" (M.Region.deregister r);
      M.close nic)

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

(* A wait that takes a completion's event is walked: its [n]-th allocation and
   every later one raise, as an asynchronous exception does. The event comes
   once, from that wait or the next, and the queue's next arm still raises
   one: the NIC ignores an arm whose count of events is short. *)
let interrupted =
  test "a wait interrupted at any allocation loses no event" (fun () ->
      let nic = nic () in
      let cq, qp = pair nic in
      let d = register nic Remote_write Any_order (memory 64) in
      let events got =
        List.length
          (List.filter (function M.Completion c -> c == cq | _ -> false) got)
      in
      let rec walk n =
        M.Cq.arm cq;
        M.Qp.post qp
          {
            op =
              Write_inline
                { data = String.make 8 'w'; dst = M.Region.remote d 0 };
            signal = true;
          };
        M.Qp.ring qp;
        equal status Done (completion cq).status;
        let got = ref [] in
        let reached =
          Rig_support.interrupted n (fun () -> got := M.wait nic ~ms:5000)
        in
        let first = events !got in
        let got = if first > 0 then first else events (M.wait nic ~ms:5000) in
        equal ~msg:(Printf.sprintf "events, from allocation %d" n) int 1 got;
        if reached then walk (n + 1)
      in
      walk 1;
      M.close nic)

let () =
  exit
    (run "rig_mlx5"
       [ transfers; refusals; bounds; lifetime; kept; events; interrupted ])
