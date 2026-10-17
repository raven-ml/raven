(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module M = Rig_mlx5

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

(* The NIC the test drives: the first the mlx5 driver holds whose port 1 carries
   RDMA, as the kernel's files say before anything opens it. A port carries RDMA
   on InfiniBand, or on Ethernet with a RoCE v2 global identifier. A ConnectX
   function whose RoCE is off, as a cloud machine's virtual function, lists an
   Ethernet port with none, which Rig_mlx5.open_ refuses: the test skips there.
   A NIC that carries RDMA and fails to open fails the test. *)
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

let open_path () =
  let name = nic_name () in
  match Rig_mlx5.path name with
  | Ok p -> p
  | Error e -> failf "opening %s: %s" name e

let collect () =
  for _ = 1 to 3 do
    Gc.full_major ()
  done

let reused_handle =
  test "deregister keeps memory of a region reusing its kernel handle"
    (fun () ->
      let p = open_path () in
      let entered = Semaphore.Binary.make false in
      let resume = Semaphore.Binary.make false in
      let paused = Atomic.make None in
      let pause = Atomic.make false in
      let last_handle = ref (-1) in
      let register access order memory =
        let r = p.register access order memory in
        Result.iter (fun (r : M.registered) -> last_handle := r.handle) r;
        r
      in
      (* Hold the real kernel's completed deregistration before Region removes
         its keeper. The other domain registers while the handle is free. *)
      let destroy kind handle =
        let destroyed = p.destroy kind handle in
        if kind = `Region && Atomic.exchange pause false then begin
          Atomic.set paused (Some handle);
          Semaphore.Binary.release entered;
          Semaphore.Binary.acquire resume
        end;
        destroyed
      in
      let nic = Result.get_ok (M.make { p with register; destroy }) in
      Fun.protect
        ~finally:(fun () -> M.close nic)
        (fun () ->
          let[@inline never] region freed =
            let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 64 in
            Gc.finalise (fun _ -> Atomic.set freed true) b;
            Result.get_ok (M.Region.register nic Local In_order (Host b))
          in
          let warm = region (Atomic.make false) in
          require_ok (M.Region.deregister warm);
          collect ();
          let first_freed = Atomic.make false in
          let second_freed = Atomic.make false in
          let first = region first_freed in
          Atomic.set pause true;
          let worker =
            Domain.spawn (fun () ->
                Fun.protect
                  ~finally:(fun () -> Semaphore.Binary.release entered)
                  (fun () -> require_ok (M.Region.deregister first)))
          in
          Semaphore.Binary.acquire entered;
          let second =
            Fun.protect
              ~finally:(fun () ->
                Semaphore.Binary.release resume;
                Domain.join worker)
              (fun () ->
                let handle = Option.get (Atomic.get paused) in
                let second = region second_freed in
                if !last_handle <> handle then
                  skip ~reason:"kernel did not reuse the released region handle"
                    ();
                second)
          in
          collect ();
          equal bool ~msg:"the replacement is still registered" false
            (Atomic.get second_freed);
          equal bool ~msg:"the first region is deregistered" true
            (Atomic.get first_freed);
          require_ok (M.Region.deregister second);
          collect ();
          equal bool ~msg:"the replacement is deregistered" true
            (Atomic.get second_freed)))

let () = exit (run "rig_mlx5.region" [ reused_handle ])
