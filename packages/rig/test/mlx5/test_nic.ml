(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A ConnectX NIC of this machine, opened through the kernel: two queue pairs of
   one NIC connected to each other, or one connected to itself, the transfers
   between their regions, sends into posted receives, and what the library
   refuses. Every test skips where the machine has no NIC the mlx5 driver
   holds. *)

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

(* Connects [a] to [b], letting [b] do [access] through [a]. *)
let connect a access b =
  match M.Qp.connect a access (M.Qp.endpoint b) with
  | Ok () -> ()
  | Error e -> failf "connecting: %s" e

(* Two queue pairs of [nic] connected to each other, on one queue: [a], which
   lets its peer do nothing, and [b], with [receives] receives, which lets its
   peer do [access]. *)
let pairs ?(receives = 0) ~access nic =
  let cq = Result.get_ok (M.Cq.make nic 128) in
  let a = Result.get_ok (M.Qp.make cq ~receives:0 16) in
  let b = Result.get_ok (M.Qp.make cq ~receives 16) in
  connect a Local b;
  connect b access a;
  (cq, a, b)

let pair nic =
  let cq, a, _ = pairs ~access:Remote nic in
  (cq, a)

(* The next completion of [cq], within [s] seconds. *)
let completion ?(s = 5.) cq =
  let deadline = Unix.gettimeofday () +. s in
  let rec go () =
    match M.Cq.poll cq with
    | Some c -> c
    | None when Unix.gettimeofday () < deadline -> go ()
    | None -> failf "no completion within %g s" s
  in
  go ()

let completions cq n = List.init n (fun _ -> completion cq)

(* The statuses of queue pair [qp]'s completions of [cs], in their order. *)
let statuses cs qp =
  List.filter_map
    (fun (c : C.t) -> if c.qp = M.Qp.number qp then Some c.status else None)
    cs

let pp_status ppf (s : C.status) =
  match s with
  | Done -> Format.pp_print_string ppf "done"
  | Received n -> Format.fprintf ppf "received %d" n
  | Failed _ | Unexpected _ ->
      Format.pp_print_string ppf (C.message { qp = 0; index = 0; status = s })

let status = Testable.make ~pp:pp_status ~equal:( = )

(* A status's ring and error, for failures whose vendor syndrome the spec
   leaves to the NIC. *)
let failure (s : C.status) =
  match s with
  | Failed { ring; error; _ } -> Some (ring, error)
  | Done | Received _ | Unexpected _ -> None

let ring_name : Rig_mlx5_abi.ring -> string = function
  | Send -> "send"
  | Receive -> "receive"

let ring_t =
  Testable.make
    ~pp:(fun ppf r -> Format.pp_print_string ppf (ring_name r))
    ~equal:( = )

let pp_failure ppf = function
  | None -> Format.pp_print_string ppf "no failure"
  | Some (ring, error) ->
      Format.fprintf ppf "%s ring: %s" (ring_name ring)
        (C.message
           {
             qp = 0;
             index = 0;
             status = Failed { ring; error; vendor = 0 };
           })

let failure_t = Testable.make ~pp:pp_failure ~equal:( = )
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

let fill b s = String.iteri (fun i c -> Bigarray.Array1.set b i c) s

(* Sends of queue pair [a] into the receives of its peer [b], both of one
   NIC. *)
let sends =
  group "sends"
    [
      test "sends land in the receives posted, in order" (fun () ->
          let nic = nic () in
          let cq, a, b = pairs ~receives:4 ~access:Local nic in
          let src = memory 256 and dst = memory 256 in
          fill src (String.init 256 (fun i -> Char.chr (i land 0xff)));
          let s = register nic Local In_order src
          and d = register nic Local In_order dst in
          List.iter
            (fun at -> M.Qp.receive b (M.Region.local d at 64))
            [ 0; 64; 128 ];
          List.iter
            (fun (at, n) ->
              M.Qp.post a
                { op = Send { src = M.Region.local s at n }; signal = true })
            [ (0, 10); (10, 0); (16, 64) ];
          M.Qp.ring a;
          let got = completions cq 6 in
          equal (list status) ~msg:"the sender's" [ Done; Done; Done ]
            (statuses got a);
          equal (list status) ~msg:"the receiver's"
            [ Received 10; Received 0; Received 64 ]
            (statuses got b);
          let dst = string_of dst and src = string_of src in
          equal string ~msg:"the first" (String.sub src 0 10)
            (String.sub dst 0 10);
          equal string ~msg:"the empty one's receive" (String.make 64 '\000')
            (String.sub dst 64 64);
          equal string ~msg:"the third" (String.sub src 16 64)
            (String.sub dst 128 64);
          equal int ~msg:"receive slots" 4 (M.Qp.room b Receive);
          M.close nic);
      test "a send posted before its receive lands once the receive is posted"
        (fun () ->
          let nic = nic () in
          let cq, a, b = pairs ~receives:4 ~access:Local nic in
          let src = memory 64 and dst = memory 64 in
          Bigarray.Array1.fill src 'p';
          let s = register nic Local In_order src
          and d = register nic Local In_order dst in
          M.Qp.post a
            { op = Send { src = M.Region.local s 0 32 }; signal = true };
          M.Qp.ring a;
          let deadline = Unix.gettimeofday () +. 0.1 in
          while Unix.gettimeofday () < deadline do
            match M.Cq.poll cq with
            | None -> ()
            | Some c ->
                failf "a completion before the receive: %s" (C.message c)
          done;
          M.Qp.receive b (M.Region.local d 0 64);
          equal (list status) ~msg:"the receiver's" [ Received 32 ]
            (statuses (completions cq 2) b);
          equal string (String.make 32 'p' ^ String.make 32 '\000')
            (string_of dst);
          M.close nic);
      test "a message longer than its receive fails the receive and the send"
        (fun () ->
          let nic = nic () in
          let cq, a, b = pairs ~receives:4 ~access:Local nic in
          let s = register nic Local In_order (memory 64)
          and d = register nic Local In_order (memory 64) in
          M.Qp.receive b (M.Region.local d 0 8);
          M.Qp.post a
            { op = Send { src = M.Region.local s 0 16 }; signal = true };
          M.Qp.ring a;
          let got = completions cq 2 in
          equal (list failure_t) ~msg:"the receive"
            [ Some (Receive, Local_length) ]
            (List.map failure (statuses got b));
          equal (list (option ring_t)) ~msg:"the send" [ Some Send ]
            (List.map (fun s -> Option.map fst (failure s)) (statuses got a));
          M.close nic);
      test "a queue pair connected to itself reads and writes its own regions"
        (fun () ->
          let nic = nic () in
          let cq = Result.get_ok (M.Cq.make nic 16) in
          let qp = Result.get_ok (M.Qp.make cq ~receives:0 4) in
          connect qp Remote qp;
          let src = memory 64 and dst = memory 64 in
          Bigarray.Array1.fill src 'f';
          let s = register nic Remote In_order src
          and d = register nic Remote_write In_order dst in
          M.Qp.post qp
            {
              op =
                Read { src = M.Region.remote s 0; dst = M.Region.local d 0 4 };
              signal = true;
            };
          M.Qp.ring qp;
          equal status ~msg:"the read" Done (completion cq).status;
          equal string ~msg:"read" "ffff" (String.sub (string_of dst) 0 4);
          M.Qp.post qp
            {
              op =
                Write { src = M.Region.local s 0 8; dst = M.Region.remote d 8 };
              signal = true;
            };
          M.Qp.ring qp;
          equal status ~msg:"the write" Done (completion cq).status;
          equal string ~msg:"written" "ffffffff"
            (String.sub (string_of dst) 8 8);
          M.close nic);
      test "a queue pair that lets its peer do nothing refuses a write"
        (fun () ->
          let nic = nic () in
          let cq, a, _ = pairs ~access:Local nic in
          let src = memory 64 and dst = memory 64 in
          Bigarray.Array1.fill src 'w';
          let s = register nic Local In_order src
          and d = register nic Remote In_order dst in
          M.Qp.post a
            {
              op =
                Write
                  { src = M.Region.local s 0 64; dst = M.Region.remote d 0 };
              signal = true;
            };
          M.Qp.ring a;
          equal failure_t ~msg:"the write"
            (Some (Send, Remote_access))
            (failure (completion cq).status);
          equal string (String.make 64 '\000') (string_of dst);
          M.close nic);
      test "a queue pair that lets its peer write takes a write and refuses a \
            read"
        (fun () ->
          let nic = nic () in
          let cq, a, _ = pairs ~access:Remote_write nic in
          let src = memory 64 and dst = memory 64 in
          Bigarray.Array1.fill src 'w';
          let s = register nic Remote In_order src
          and d = register nic Remote In_order dst in
          M.Qp.post a
            {
              op =
                Write { src = M.Region.local s 0 8; dst = M.Region.remote d 0 };
              signal = true;
            };
          M.Qp.ring a;
          equal status ~msg:"the write" Done (completion cq).status;
          equal string ~msg:"written" "wwwwwwww"
            (String.sub (string_of dst) 0 8);
          M.Qp.post a
            {
              op =
                Read { src = M.Region.remote d 0; dst = M.Region.local s 8 8 };
              signal = true;
            };
          M.Qp.ring a;
          equal failure_t ~msg:"the read"
            (Some (Send, Remote_access))
            (failure (completion cq).status);
          M.close nic);
      test "a peer's sends land nowhere once its queue pair is destroyed"
        (fun () ->
          let nic = nic () in
          let cq, a, b = pairs ~receives:4 ~access:Local nic in
          let src = memory 64 and dst = memory 64 in
          Bigarray.Array1.fill src 'x';
          let s = register nic Local In_order src
          and d = register nic Local In_order dst in
          M.Qp.receive b (M.Region.local d 0 64);
          require_ok (M.Qp.destroy b);
          M.Qp.post a
            { op = Send { src = M.Region.local s 0 64 }; signal = true };
          M.Qp.ring a;
          (* Retries end after 7 timeouts of 1.07 s on InfiniBand. *)
          let c = completion ~s:15. cq in
          equal int ~msg:"the sender's" (M.Qp.number a) c.qp;
          equal (option ring_t) ~msg:"the send failed" (Some Send)
            (Option.map fst (failure c.status));
          equal string (String.make 64 '\000') (string_of dst);
          M.close nic);
    ]

let refusals =
  group "refusals"
    [
      test "queue pairs that would hold more than their queue are refused"
        (fun () ->
          let nic = nic () in
          let cq = Result.get_ok (M.Cq.make nic 16) in
          ignore (Result.get_ok (M.Qp.make cq ~receives:0 8) : M.Qp.t);
          ignore (Result.get_ok (M.Qp.make cq ~receives:0 8) : M.Qp.t);
          raises_match (Exn.invalid_arg ~substring:"Qp.make") (fun () ->
              M.Qp.make cq ~receives:0 1);
          M.close nic);
      test "receives count against their queue too" (fun () ->
          let nic = nic () in
          let cq = Result.get_ok (M.Cq.make nic 16) in
          ignore (Result.get_ok (M.Qp.make cq ~receives:8 8) : M.Qp.t);
          raises_match (Exn.invalid_arg ~substring:"Qp.make") (fun () ->
              M.Qp.make cq ~receives:0 1);
          M.close nic);
      test "a receive needs a connected queue pair with a receive ring"
        (fun () ->
          let nic = nic () in
          let cq = Result.get_ok (M.Cq.make nic 64) in
          let d = register nic Local In_order (memory 64) in
          let none = Result.get_ok (M.Qp.make cq ~receives:0 4) in
          connect none Local none;
          raises_match (Exn.invalid_arg ~substring:"no receive ring") (fun () ->
              M.Qp.receive none (M.Region.local d 0 8));
          let unconnected = Result.get_ok (M.Qp.make cq ~receives:4 4) in
          raises_match (Exn.invalid_arg ~substring:"not connected") (fun () ->
              M.Qp.receive unconnected (M.Region.local d 0 8));
          M.close nic);
      test "a full receive ring refuses a receive until a completion frees one"
        (fun () ->
          let nic = nic () in
          let cq, a, b = pairs ~receives:1 ~access:Local nic in
          let s = register nic Local In_order (memory 64)
          and d = register nic Local In_order (memory 64) in
          equal int ~msg:"rounded up" 4 (M.Qp.room b Receive);
          for _ = 1 to 4 do
            M.Qp.receive b (M.Region.local d 0 8)
          done;
          equal int ~msg:"posted" 0 (M.Qp.room b Receive);
          raises_match (Exn.invalid_arg ~substring:"full") (fun () ->
              M.Qp.receive b (M.Region.local d 0 8));
          M.Qp.post a
            { op = Send { src = M.Region.local s 0 8 }; signal = true };
          M.Qp.ring a;
          ignore (completions cq 2 : C.t list);
          equal int ~msg:"once polled" 1 (M.Qp.room b Receive);
          M.close nic);
      test "an entry longer than the port's largest message is refused"
        (fun () ->
          let nic = nic () in
          let _, qp = pair nic in
          let n = M.max_message nic + 1 in
          raises_match (Exn.invalid_arg ~substring:"Qp.post") (fun () ->
              M.Qp.post qp
                {
                  op = Send { src = { address = 0; bytes = n; key = 0 } };
                  signal = true;
                });
          equal int ~msg:"nothing posted" 16 (M.Qp.room qp Send);
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
   queue pair's send ring holds, and its receive ring:
   mlx5_ib_alloc_ucontext_resp's max_send_wqebb and max_recv_wr, u32s, in
   Linux's include/uapi/rdma/mlx5-abi.h. *)
let max_send_wqebb_at = 20
let max_recv_wr_at = 24

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
          ignore (require_ok (M.Qp.make cq ~receives:0 most) : M.Qp.t);
          let why = require_error (M.Qp.make cq ~receives:0 (most + 1)) in
          contains ~sub:(Printf.sprintf "at most %d" most) why;
          raises_match (Exn.invalid_arg ~substring:"Qp.make") (fun () ->
              M.Qp.make cq ~receives:0 0);
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
          is_error ~msg:"128 entries" (M.Qp.make cq ~receives:0 128);
          is_error ~msg:"65 entries" (M.Qp.make cq ~receives:0 65);
          ignore (require_ok (M.Qp.make cq ~receives:0 64) : M.Qp.t);
          M.close nic);
      test "a receive ring holds the NIC's most receives, and no more"
        (fun () ->
          let receives = ref 0 in
          let nic =
            through
              ~with_context:(fun (c : M.context) ->
                receives := u32 c.answer max_recv_wr_at;
                c)
              ()
          in
          let most = min !receives most_entries in
          let cq = require_ok (M.Cq.make nic (most + 1)) in
          ignore (require_ok (M.Qp.make cq ~receives:most 1) : M.Qp.t);
          let why = require_error (M.Qp.make cq ~receives:(most + 1) 1) in
          contains ~sub:(Printf.sprintf "at most %d receives" most) why;
          raises_match (Exn.invalid_arg ~substring:"Qp.make") (fun () ->
              M.Qp.make cq ~receives:(-1) 1);
          M.close nic);
      test "a receive ring's bound is the one the driver's answer states"
        (fun () ->
          let nic =
            through
              ~with_context:(fun (c : M.context) ->
                { c with answer = with_u32 c.answer max_recv_wr_at 64 })
              ()
          in
          let cq = require_ok (M.Cq.make nic 256) in
          is_error ~msg:"128 receives" (M.Qp.make cq ~receives:128 1);
          is_error ~msg:"65 receives" (M.Qp.make cq ~receives:65 1);
          ignore (require_ok (M.Qp.make cq ~receives:64 1) : M.Qp.t);
          M.close nic);
      test "the largest message is the one the port reports" (fun () ->
          let nic =
            through
              ~with_context:(fun (c : M.context) -> { c with message = 4096 })
              ()
          in
          equal int 4096 (M.max_message nic);
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
      let qp = require_ok (M.Qp.make cq ~receives:0 16) in
      is_error ~msg:"the queue pair" (M.Qp.destroy qp);
      is_error ~msg:"a second destroy" (M.Qp.destroy qp);
      raises_match (Exn.invalid_arg ~substring:"destroyed") (fun () ->
          M.Qp.ring qp);
      raises_match (Exn.invalid_arg ~substring:"Qp.make") (fun () ->
          M.Qp.make cq ~receives:0 1);
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
       [
         transfers;
         sends;
         refusals;
         bounds;
         lifetime;
         kept;
         events;
         interrupted;
       ])
