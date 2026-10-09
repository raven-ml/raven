(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a NIC does between the calls it makes of its path, with no NIC: the path
   here answers each call as its record's contract states, with objects numbered
   in order and access regions in this process's memory. No NIC reads a ring, so
   nothing here completes an entry; these tests hold what Rig_mlx5 decides
   itself: the access regions it maps, the bounds of a region's pieces, the room
   a completion queue gives its queue pairs, and the events it reports.

   The driver's answers are laid out as Linux v6.12's mlx5-abi.h lays them out:
   a context's at bf_reg_size 4, tot_bfregs 8, log_uar_size 56 and
   num_uars_per_page 60, in 72 bytes; a completion queue's cqn at 0, in 8; a
   queue pair's bfreg_index at 0, in 40. *)

open Windtrap
module M = Rig_mlx5

let strf = Printf.sprintf

external address : Rig_mlx5_abi.buffer -> int = "rig_mlx5_test_address"

(* A page every host's page size divides. *)
let page = 65536

(* Memory the path maps, kept for the run. *)
let mapped = ref []

let map _ n =
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill b '\000';
  mapped := b :: !mapped;
  Ok (address b)

let le fields size =
  let b = Bytes.make size '\000' in
  List.iter (fun (at, v) -> Bytes.set_int32_le b at (Int32.of_int v)) fields;
  Bytes.to_string b

let context_answer ?(log_uar_size = 12) ?(per_page = 1) ?(registers = 2) () =
  le [ (4, 512); (8, registers); (56, log_uar_size); (60, per_page) ] 72

type calls = { mutable closed : int; mutable made : int }

(* A path over this process. [qp] runs before each queue pair is made; [wait]
   answers the NIC's events. *)
let path ?(context = context_answer ()) ?(qp = fun () -> ())
    ?(wait = fun _ -> []) () =
  let calls = { closed = 0; made = 0 } in
  let next () =
    calls.made <- calls.made + 1;
    calls.made
  in
  let p =
    {
      M.name = "mlx5_test";
      bus = "0000:00:00.0";
      page;
      context =
        (fun _ _ ->
          Ok
            {
              M.answer = context;
              port = `Infiniband 1;
              mtu = 4096;
              reads = 16;
              served = 16;
            });
      map;
      register =
        (fun _ _ ->
          let h = next () in
          Ok { M.handle = h; local = h; remote = h + 0x1000 });
      cq = (fun ~entries:_ ~tag:_ _ _ -> Ok (next (), le [ (0, 0x40) ] 8));
      qp =
        (fun ~cq:_ ~entries:_ ~tag:_ _ _ ->
          qp ();
          let h = next () in
          Ok (h, 0x100 + h, le [ (0, 1) ] 40));
      modify = (fun _ _ -> Ok ());
      destroy = (fun _ _ -> ());
      wait;
      close = (fun () -> calls.closed <- calls.closed + 1);
    }
  in
  (p, calls)

let open_nic ?context ?qp ?wait () =
  let p, calls = path ?context ?qp ?wait () in
  match M.make p with Ok nic -> (nic, calls) | Error e -> failf "make: %s" e

let opening =
  group "opening"
    [
      cases
        ~name:(fun (log, per_page) ->
          strf "log_uar_size %d, num_uars_per_page %d" log per_page)
        "a context that counts no regions per page has one per page"
        [ (0, 0); (12, 0); (0, 4) ]
        (fun (log_uar_size, per_page) ->
          let p, calls =
            path ~context:(context_answer ~log_uar_size ~per_page ()) ()
          in
          (match M.make p with
          | Ok nic -> M.close nic
          | Error e -> failf "make: %s" e);
          equal ~msg:"the path is closed once" int 1 calls.closed);
      test "a refused context closes the path" (fun () ->
          let p, calls = path () in
          let p = { p with context = (fun _ _ -> Error "refused") } in
          is_error
            ~pp:(fun ppf _ -> Format.pp_print_string ppf "a NIC")
            (M.make p);
          equal int 1 calls.closed);
    ]

let regions =
  let region () =
    let nic, _ = open_nic () in
    match
      M.Region.register nic Remote (Host { address = 0x10000; bytes = 64 })
    with
    | Ok r -> r
    | Error e -> failf "register: %s" e
  in
  let refused name f =
    test name (fun () ->
        let r = region () in
        raises_match (Exn.invalid_arg ~substring:"Region") (fun () -> f r))
  in
  group "regions"
    [
      test "a piece at the region's end is in it" (fun () ->
          let r = region () in
          equal int (0x10000 + 63) (M.Region.local r 63 1).address);
      refused "a piece past the end is refused" (fun r -> M.Region.local r 63 2);
      refused "a piece whose end overflows is refused" (fun r ->
          M.Region.local r max_int 1);
      refused "a piece of an overflowing length is refused" (fun r ->
          M.Region.local r 1 max_int);
      refused "an offset past the end is refused" (fun r ->
          M.Region.remote r 64);
      refused "the largest offset is refused" (fun r ->
          M.Region.remote r max_int);
    ]

(* Two queue pairs made at once on a completion queue with room for one: the
   first holds in its path call until the second's make returned, so both check
   the queue's room before either is made. *)
let room =
  test "two queue pairs made at once share their queue's room" (fun () ->
      let inside = Semaphore.Binary.make false in
      let release = Semaphore.Binary.make false in
      let first = Atomic.make true in
      let qp () =
        if Atomic.exchange first false then begin
          Semaphore.Binary.release inside;
          Semaphore.Binary.acquire release
        end
      in
      let nic, _ = open_nic ~qp () in
      let cq = Result.get_ok (M.Cq.make nic 2) in
      let a = Domain.spawn (fun () -> M.Qp.make cq 2) in
      Semaphore.Binary.acquire inside;
      let b =
        match M.Qp.make cq 2 with
        | _ -> `Made
        | exception Invalid_argument _ -> `Refused
      in
      Semaphore.Binary.release release;
      is_ok ~pp:Format.pp_print_string (Domain.join a);
      equal ~msg:"the second make"
        (Testable.make
           ~pp:(fun ppf -> function
             | `Made -> Format.pp_print_string ppf "made"
             | `Refused -> Format.pp_print_string ppf "refused")
           ~equal:( = ))
        `Refused b)

(* A kernel that reports, at once and every time, a completion of a queue this
   NIC no longer has: no event of the NIC's. *)
let events =
  group "events"
    [
      test "wait waits its time when the kernel reports nothing of the NIC's"
        (fun () ->
          let nic, _ = open_nic ~wait:(fun _ -> [ M.Completed 0x7fff ]) () in
          let start = Unix.gettimeofday () in
          let e = M.wait nic ~ms:50 in
          let elapsed = Unix.gettimeofday () -. start in
          equal int 0 (List.length e);
          at_least ~msg:"seconds waited" float_exact ~than:0.05 elapsed);
      test "wait answers an event at once" (fun () ->
          let nic, _ = open_nic ~wait:(fun _ -> [ M.Nic_failed ]) () in
          let start = Unix.gettimeofday () in
          let e = M.wait nic ~ms:10_000 in
          equal int 1 (List.length e);
          less ~msg:"seconds waited" float_exact ~than:1.
            (Unix.gettimeofday () -. start));
    ]

let () = exit (run "rig_mlx5" [ opening; regions; room; events ])
