(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Machines and functions: this machine's refusal, and a function of a machine
   reached through a transport. The tester owns the suite. *)

open Windtrap
open Device_pci

external far : int -> int -> int = "test_far"
external break : int -> unit = "test_far_break"

let test_this () =
  equal ~msg:"bus order is numeric" int (-1)
    (Machine.compare_address "ffff:00:00.0" "10000:00:00.0");
  match Function.take Machine.this ~lock:"smoke" "ffff:ff:1f.7" with
  | Ok _ -> fail "a function no machine has is taken"
  | Error why ->
      equal ~msg:"refusal" string
        "ffff:ff:1f.7 is no PCI function of this machine" why

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

let test_transport () =
  let p = far 0 4096 in
  let tr = Window.transport p in
  let m =
    Machine.make ~name:"far:1"
      {
        transport = tr;
        page = 4096;
        functions = (fun () -> []);
        take = (fun ~lock:_ _ -> Ok (fake tr));
        reserve = (fun ~base:_ _ -> ());
      }
  in
  let f = Result.get_ok (Function.take m ~lock:"smoke" "0000:01:00.0") in
  Function.set_config f 4 1 7;
  equal ~msg:"config" int 7 (Function.config f 4 1);
  let w = Function.map f 0 ~off:256 ~length:256 in
  Window.set32 w 0 42;
  equal ~msg:"a BAR window" int 42 (Window.get32 w 0);
  Function.unmap f w;
  raises_match (Exn.invalid_arg ~substring:"no such window") (fun () ->
      Function.unmap f w);
  raises_match (Exn.invalid_arg ~substring:"not pinned") (fun () ->
      Function.unpin f 0 4096);
  equal ~msg:"a wait" bool true (Machine.wait m ~ms:10 (fun () -> true));
  break p;
  equal ~msg:"failed" bool true (Option.is_some (Machine.failed m));
  raises_match (Exn.failure ~substring:"") (fun () ->
      Machine.wait m ~ms:10 (fun () -> false));
  Function.release f;
  equal ~msg:"released" bool true (Function.released f)

let () =
  exit
  @@ run "device_pci smoke"
       [
         test "this machine" test_this;
         test "a machine through a transport" test_transport;
       ]
