(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPUs over a function of a machine reached through a transport. The testers
   own the suite. *)

open Windtrap
open Device_pci

external far : int -> int -> int = "test_far"

(* A function whose configuration space is a buffer and whose BAR 0 is the
   transport's bytes. *)
let fake tr =
  let config = Bytes.make 64 '\000' in
  {
    Machine.addressing = Physical;
    config = (fun off _ -> Bytes.get_uint8 config off);
    set_config = (fun off _ x -> Bytes.set_uint8 config off (x land 0xff));
    bar = (fun i -> if i = 0 then Some (0, 4096) else None);
    map = (fun _ off n -> Window.through tr off n);
    unmap = (fun _ -> ());
    interrupt = (fun _ -> false);
    reset = (fun () -> ());
    alloc_dma =
      (fun ~contiguous:_ ~va:_ n -> (Window.through tr 0 n, [ (0, n) ]));
    free_dma = (fun _ -> ());
    pin = (fun a n -> [ (a, n) ]);
    unpin = (fun _ _ -> ());
    release = (fun () -> ());
  }

let test_gpus () =
  let tr = Window.transport (far 0 4096) in
  let m =
    Machine.make ~name:"far:3"
      {
        transport = tr;
        page = 4096;
        functions =
          (fun () ->
            [
              { bus = "0000:01:00.0"; vendor = 0x1002; device = 1; class_ = 3 };
            ]);
        take = (fun ~lock:_ _ -> Ok (fake tr));
        reserve = (fun ~base:_ _ -> ());
      }
  in
  let g =
    Gpus.make ~name:"AMD" ~lock:"smoke" ~memory_bar:0 (fun id ->
        id.vendor = 0x1002)
  in
  equal ~msg:"buses" (list string) [ "0000:01:00.0" ] (Gpus.buses g m);
  let opened () = Gpus.open_pci g m 0 (fun h _ -> Ok h) in
  let h = Result.get_ok (opened ()) in
  equal ~msg:"held" bool true (Result.is_error (opened ()));
  equal ~msg:"another machine's through its kernel driver" bool true
    (Result.is_error (Gpus.open_kernel g m 0 (fun h -> Ok h)));
  Gpus.lose h;
  raises_match (Exn.invalid_arg ~substring:"given back") (fun () ->
      Gpus.release h);
  equal ~msg:"lost until a reset" bool true (Result.is_error (opened ()));
  equal ~msg:"reset" (result unit string) (Ok ()) (Gpus.reset g m 0 ignore);
  Gpus.release (Result.get_ok (opened ()));
  equal ~msg:"no GPU 1" (result unit string)
    (Error "no GPU 1; there are 1 AMD GPUs") (Gpus.reset g m 1 ignore)

let () = exit @@ run "device_pci smoke" [ test "gpus" test_gpus ]
