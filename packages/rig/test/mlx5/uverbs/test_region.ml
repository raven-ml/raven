(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module M = Rig_mlx5

let open_path () =
  match Rig_mlx5_uverbs.names () with
  | [] -> skip ~reason:"no NIC the mlx5 driver holds" ()
  | name :: _ -> (
      match Rig_mlx5_uverbs.path name with
      | Ok p -> p
      | Error e -> failf "opening %s: %s" name e)

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

let () = exit (run "rig_mlx5_uverbs.region" [ reused_handle ])
