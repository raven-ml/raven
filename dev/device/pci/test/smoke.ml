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

(* Memory over the fake function, with page tables kept in a table that counts
   its writes. Entries: the address, bit 0 valid, bit 1 a table. *)
let format writes =
  let entries = Hashtbl.create 64 in
  let key table i = table + (8 * i) in
  {
    Page_table.levels = [ 12; 21; 30; 39 ];
    bits = 48;
    first = 0;
    get =
      (fun ~level:_ ~table i ->
        Option.value ~default:0L (Hashtbl.find_opt entries (key table i)));
    set =
      (fun ~level:_ ~table i e ->
        incr writes;
        Hashtbl.replace entries (key table i) e);
    encode =
      (fun ~level:_ ~table _ ~uncached:_ ~snooped:_ ~fragment:_ ~valid pa ->
        Int64.of_int (pa lor (if valid then 1 else 0) lor if table then 2 else 0));
    valid = (fun e -> Int64.logand e 1L = 1L);
    leaf = (fun ~level e -> level = 3 || Int64.logand e 2L = 0L);
    address = (fun e -> Int64.to_int e land lnot 0xfff);
    large = (fun ~level -> level >= 2);
    zero = (fun _ _ -> ());
    flush = (fun () -> ());
  }

let test_memory () =
  let p = far 0 4096 in
  let tr = Window.transport p in
  let m =
    Machine.make ~name:"far:2"
      {
        transport = tr;
        page = 4096;
        functions = (fun () -> []);
        take = (fun ~lock:_ _ -> Ok (fake tr));
        reserve = (fun ~base:_ _ -> ());
      }
  in
  let f = Result.get_ok (Function.take m ~lock:"smoke" "0000:01:00.0") in
  let writes = ref 0 in
  let space = Space.create ~base:(1 lsl 40) (1 lsl 30) in
  let tables =
    Page_table.create (format writes) space ~memory:(64 lsl 20) ~boot:(1 lsl 20)
      ~tables:Pool
      ~pages:[ (0x1000, 0x1000) ]
  in
  Page_table.booted tables;
  let mem = Memory.create f tables ~bar:0 in
  let host = Option.get (Memory.alloc mem Host 4096) in
  equal ~msg:"system memory has a window" bool true (Option.is_some host.host);
  Memory.free mem host;
  raises_match (Exn.invalid_arg ~substring:"does not hold") (fun () ->
      Memory.free mem host);
  let gpu = Option.get (Memory.alloc mem Gpu (64 lsl 10)) in
  Function.release f;
  let before = !writes in
  Memory.free mem gpu;
  equal ~msg:"a released GPU's memory is not written" int before !writes

let test_firmware () =
  equal ~msg:"sha256" string
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    (Firmware.sha256 "abc");
  let dir = Filename.temp_dir "smoke" "firmware" in
  Out_channel.with_open_bin (Filename.concat dir "image.bin") (fun oc ->
      output_string oc "abc");
  equal ~msg:"from a directory"
    (result (option string) string)
    (Ok (Some "abc"))
    (Firmware.find ~dir "image.bin" ~sha256:(Firmware.sha256 "abc"));
  equal ~msg:"another digest" bool true
    (Result.is_error (Firmware.find ~dir "image.bin" ~sha256:"00"))

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

let () =
  exit
  @@ run "device_pci smoke"
       [
         test "this machine" test_this;
         test "a machine through a transport" test_transport;
         test "memory" test_memory;
         test "firmware" test_firmware;
         test "gpus" test_gpus;
       ]
