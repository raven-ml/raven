(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Another machine's host, served by this process on the loopback: the
   connection, copies to, from and between machines, host programs there
   (compiled with clang, skipped without it), and the machine going away. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar
module P = Nx_device.Program
module Server = Nx_device_support.Remote_server

let key = "the key of this test, 32 bytes."

let serve () =
  Nx_remote_device.listen ~key (Unix.ADDR_INET (Unix.inet_addr_loopback, 0))

let port s =
  match Server.address s with
  | Unix.ADDR_INET (_, p) -> p
  | Unix.ADDR_UNIX _ -> assert false

let connect s =
  match Nx_remote_device.connect ~port:(port s) ~key "127.0.0.1" with
  | Ok d -> d
  | Error why -> fail why

let of_string s =
  B.of_bigarray
    (Bigarray.Array1.init Bigarray.char Bigarray.c_layout (String.length s)
       (String.get s))

let read b =
  let ba =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (B.nbytes b)
  in
  B.copy ~src:b ~dst:(B.of_bigarray ba);
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

let pattern n =
  String.init n (fun i -> Char.chr (((i * 13) + (i lsr 12)) land 0xff))

(* One machine for the tests that keep it. *)
let server = lazy (serve ())
let far = lazy (connect (Lazy.force server))

let test_connect () =
  let s = Lazy.force server in
  let d = Lazy.force far in
  equal ~msg:"the name" string
    (Printf.sprintf "CPU@127.0.0.1:%d" (port s))
    (Nx_device.name d);
  equal ~msg:"the machine's instruction set" string
    (Nx_device.arch Nx_device.host)
    (Nx_device.arch d);
  is_true ~msg:"its own host" (Nx_device.host_of d == d);
  is_true ~msg:"one device per machine" (connect s == d);
  is_some ~msg:"its connection" (Nx_remote_device.remote d);
  is_none ~msg:"this machine's host" (Nx_remote_device.remote Nx_device.host);
  (match
     Nx_remote_device.connect ~port:(port s) ~key:"a key the server lacks"
       "localhost"
   with
  | Error why -> contains ~msg:"one client at a time" ~sub:"busy" why
  | Ok _ -> fail "a second client was served");
  let other = serve () in
  Fun.protect
    ~finally:(fun () -> Server.stop other)
    (fun () ->
      match
        Nx_remote_device.connect ~port:(port other)
          ~key:"a key the server lacks" "localhost"
      with
      | Error why -> contains ~msg:"refused" ~sub:"key" why
      | Ok _ -> fail "a wrong key was accepted");
  raises_match (Exn.invalid_arg ~substring:"no host") (fun () ->
      Nx_remote_device.remote
        (Nx_device.make ~name:"X" ~arch:"x" ~budget:0
           ~memory:{ alloc = (fun _ -> None); free = ignore }
           ()))

let test_copies () =
  let d = Lazy.force far in
  let n = (64 lsl 20) + 4097 in
  let bytes = pattern n in
  let b = B.create d S.UInt8 n and c = B.create d S.UInt8 n in
  B.copy ~src:(of_string bytes) ~dst:b;
  B.copy ~src:b ~dst:c;
  is_true ~msg:"to the machine, within it and back"
    (String.equal bytes (read c));
  let small = B.create d S.UInt8 5 in
  B.copy ~src:(B.view c ~offset:1000 S.UInt8 5) ~dst:small;
  equal ~msg:"a view" string (String.sub bytes 1000 5) (read small);
  let st = Nx_device.stats d in
  is_true ~msg:"bytes in" (Nx_device.Stats.bytes_in st >= n);
  Nx_device.synchronize d

let clang =
  lazy
    (Sys.command (Printf.sprintf "clang --version > %s 2>&1" Filename.null) = 0)

let compile src =
  if not (Lazy.force clang) then skip ~reason:"no clang" ();
  let c = Filename.temp_file "nx_remote" ".c" in
  let o = Filename.temp_file "nx_remote" ".o" in
  Fun.protect
    ~finally:(fun () -> List.iter Sys.remove [ c; o ])
    (fun () ->
      Out_channel.with_open_bin c (fun oc -> output_string oc src);
      let arch = Nx_device.arch Nx_device.host in
      let cmd =
        Printf.sprintf
          "clang -c -x c -O2 -fPIC -ffreestanding -nostdlib -fno-ident \
           --target=%s-none-unknown-elf %s %s -o %s"
          arch
          (if arch = "arm64" then "-ffixed-x18" else "")
          (Filename.quote c) (Filename.quote o)
      in
      if Sys.command cmd <> 0 then failf "clang failed: %s" cmd;
      In_channel.with_open_bin o In_channel.input_all)

let test_programs () =
  if Sys.win32 then skip ~reason:"programs use the System V ABI here" ();
  let d = Lazy.force far in
  let binary =
    compile
      {|void fill(void **b, const long long *v) {
  char *out = b[0];
  for (long long i = 0; i < v[0]; i++) out[i] = (char)(v[1] + i);
}|}
  in
  let p = P.load d ~binary ~name:"fill" in
  is_true ~msg:"the same program" (P.load d ~binary ~name:"fill" == p);
  let b = B.create d S.UInt8 8 in
  P.call p [| B.view b ~offset:2 S.UInt8 4 |] [| 4; 65 |];
  Nx_device.synchronize d;
  equal ~msg:"it ran there" string "ABCD" (String.sub (read b) 2 4);
  raises_match (Exn.invalid_arg ~substring:"does not address") (fun () ->
      P.call p [| B.create Nx_device.host S.UInt8 8 |] [| 0; 0 |])

let test_gone () =
  let s = serve () in
  let d = connect s in
  let b = B.create d S.UInt8 16 in
  let here = B.create Nx_device.host S.UInt8 16 in
  Server.stop s;
  raises_match (Exn.failure ~substring:"") (fun () -> B.copy ~src:here ~dst:b);
  raises_match (Exn.failure ~substring:"") (fun () -> Nx_device.synchronize d);
  is_true ~msg:"the machine's host failed"
    (match Nx_remote_device.remote d with
    | Some r -> Nx_device_support.Remote.failed r <> None
    | None -> false);
  B.copy ~src:(of_string (String.make 16 'x')) ~dst:here;
  equal ~msg:"this machine goes on" string (String.make 16 'x') (read here)

let () =
  exit
    (run "nx.remote.device"
       [
         group "a remote host"
           [
             test "connecting" test_connect;
             test "copies" test_copies;
             test "programs" test_programs;
             test "the machine goes away" test_gone;
           ];
       ])
