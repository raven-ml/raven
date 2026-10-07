(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Host programs compiled at test time with clang: their arguments and writes,
   their links to their own data and functions and to the math library, the
   cache and the release of their code, the runtime released while they run, and
   the binaries the host refuses. Every test skips without clang. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar
module P = Nx_device.Program

external mapped : nativeint -> bool = "test_host_mapped"

let host = Nx_device.host
let arch = Nx_device.arch host

(* The function [name] of [binary], which the host loads. *)
let load ~binary ~name =
  match P.load host ~binary ~name with Ok p -> p | Error why -> failwith why

let clang =
  lazy
    (Sys.command (Printf.sprintf "clang --version > %s 2>&1" Filename.null) = 0)

(* [src] compiled for [target] as the host loads it: a relocatable ELF object,
   position-independent, with the platform's calling convention. *)
let compile ?(target = arch) src =
  if not (Lazy.force clang) then skip ~reason:"no clang" ();
  let c = Filename.temp_file "nx_host" ".c" in
  let o = Filename.temp_file "nx_host" ".o" in
  Fun.protect
    ~finally:(fun () -> List.iter Sys.remove [ c; o ])
    (fun () ->
      let abi = if Sys.win32 then "__attribute__((ms_abi))" else "" in
      Out_channel.with_open_bin c (fun oc ->
          Printf.fprintf oc "#define ABI %s\n%s" abi src);
      let cmd =
        Printf.sprintf
          "clang -c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib \
           -fno-ident --target=%s-none-unknown-elf %s %s -o %s"
          target
          (if target = "arm64" then "-ffixed-x18" else "")
          (Filename.quote c) (Filename.quote o)
      in
      if Sys.command cmd <> 0 then failf "clang failed: %s" cmd;
      In_channel.with_open_bin o In_channel.input_all)

let int32s l =
  B.of_bigarray (Bigarray.Array1.of_array Bigarray.int32 Bigarray.c_layout l)

let int32s_of b =
  Array.to_list
    (Array.init (B.length b)
       (Bigarray.Array1.get (B.bigarray Bigarray.int32 b)))

(* Arguments *)

let affine =
  lazy
    (compile
       {|ABI void affine(void **b, const long long *v) {
  int *out = b[0];
  const int *in = b[1];
  for (long long i = 0; i < v[0]; i++) out[i] = in[i] * (int)v[1] + (int)v[2];
}|})

let test_arguments () =
  let p = load ~binary:(Lazy.force affine) ~name:"affine" in
  let out = B.create host S.Int32 4 in
  let input = int32s [| 1l; 2l; 3l; 4l |] in
  P.call p [| out; input |] [| 4; 3; -5 |];
  equal ~msg:"the writes to a buffer" (list int32) [ -2l; 1l; 4l; 7l ]
    (int32s_of out);
  P.call p
    [| B.view out ~offset:8 S.Int32 2; B.view input ~offset:4 S.Int32 2 |]
    [| 2; 1; 100 |];
  equal ~msg:"a view is passed at its first byte" (list int32)
    [ -2l; 1l; 102l; 103l ] (int32s_of out)

(* Test devices of the host's memory, whose buffers the host's programs take,
   and the host's programs by address. *)
let test_devices () =
  let module Driver = Nx_device.Driver in
  let cpu i =
    Driver.device
      ~name:(Printf.sprintf "CPU:%d" i)
      ~arch ~budget:max_int
      (Host_visible { memory = Driver.host_memory; mapping = Some Identity })
  in
  let d = cpu 1 in
  is_true ~msg:"it shares the host's memory" (Nx_device.shares_host_memory d);
  let p = load ~binary:(Lazy.force affine) ~name:"affine" in
  let out = B.create d S.Int32 4 in
  let input = B.create d S.Int32 4 in
  B.copy ~src:(int32s [| 1l; 2l; 3l; 4l |]) ~dst:input;
  P.call p [| out; input |] [| 4; 3; -5 |];
  let back = B.create host S.Int32 4 in
  B.copy ~src:out ~dst:back;
  equal ~msg:"a host program on a test device's buffers" (list int32)
    [ -2l; 1l; 4l; 7l ] (int32s_of back);
  let programs = Driver.host_programs in
  match programs.load ~binary:(Lazy.force affine) ~entry:"affine" with
  | Error why -> fail why
  | Ok (entry, unload) ->
      let at b = (B.address b, B.nbytes b) in
      programs.call entry [| at back; at back |] [| 4; 2; 0 |];
      equal ~msg:"a program by address" (list int32) [ -4l; 2l; 8l; 14l ]
        (int32s_of back);
      unload ()

(* Linking *)

let linked =
  lazy
    (compile
       {|double cbrt(double);
static const volatile unsigned char bytes[4] = {1, 2, 3, 4};
static const double scale[2] = {0.5, 4.0};
__attribute__((noinline)) double twice(double x) { return 2.0 * x; }
ABI void linked(void **b, const long long *v) {
  double *out = b[0];
  const double *in = b[1];
  out[0] = cbrt(in[0]);
  out[1] = scale[v[0] & 1] * bytes[2];
  out[2] = twice(in[1]);
}|})

let test_links () =
  let p = load ~binary:(Lazy.force linked) ~name:"linked" in
  let out = B.create host S.Float64 3 in
  let input =
    B.of_bigarray
      (Bigarray.Array1.of_array Bigarray.float64 Bigarray.c_layout
         [| 27.; 1.5 |])
  in
  P.call p [| out; input |] [| 1 |];
  let r = B.bigarray Bigarray.float64 out in
  equal ~msg:"a call to the math library" (float 1e-12) 3. r.{0};
  equal ~msg:"reads of its constants" float_exact 12. r.{1};
  equal ~msg:"a call to its own function" float_exact 3. r.{2}

(* Compiler builtins *)

let builtins =
  lazy
    (compile
       {|__bf16 __truncsfbf2(float);
_Float16 __truncsfhf2(float);
float __extendhfsf2(_Float16);
ABI void builtins(void **b, const long long *v) {
  const float *in = b[0];
  unsigned short *bf16 = b[1], *f16 = b[2];
  const unsigned short *halves = b[3];
  float *out = b[4];
  for (long long i = 0; i < v[0]; i++) {
    __bf16 x = __truncsfbf2(in[i]);
    _Float16 y = __truncsfhf2(in[i]);
    __builtin_memcpy(&bf16[i], &x, 2);
    __builtin_memcpy(&f16[i], &y, 2);
  }
  for (long long i = 0; i < v[1]; i++) {
    _Float16 h;
    __builtin_memcpy(&h, &halves[i], 2);
    out[i] = __extendhfsf2(h);
  }
}|})

(* Float32 bits, and their bfloat16 and float16 roundings: ties to even,
   overflow, NaN payloads, subnormals, zeros. *)
let narrowed =
  [
    (0x3f800000, 0x3f80, 0x3c00);
    (0x3f808000, 0x3f80, 0x3c04);
    (0x3f818000, 0x3f82, 0x3c0c);
    (0x3f801000, 0x3f80, 0x3c00);
    (0x3f803000, 0x3f80, 0x3c02);
    (0xbf80ffff, 0xbf81, 0xbc08);
    (0x477fefff, 0x4780, 0x7bff);
    (0x477ff000, 0x4780, 0x7c00);
    (0x7f7fffff, 0x7f80, 0x7c00);
    (0x7f800000, 0x7f80, 0x7c00);
    (0xff800000, 0xff80, 0xfc00);
    (0x7fc00001, 0x7fc1, 0x7e01);
    (0x7f800001, 0x7f81, 0x7c01);
    (0x7f802000, 0x7f81, 0x7c01);
    (0x33800000, 0x3380, 0x0001);
    (0x33000000, 0x3300, 0x0000);
    (0x33c00000, 0x33c0, 0x0002);
    (0x387fe000, 0x3880, 0x0400);
    (0x00000001, 0x0000, 0x0000);
    (0x80000000, 0x8000, 0x8000);
  ]

(* Float16 bits and their float32 widening. *)
let widened =
  [
    (0x3c00, 0x3f800000);
    (0x0001, 0x33800000);
    (0x03ff, 0x387fc000);
    (0x0400, 0x38800000);
    (0x7bff, 0x477fe000);
    (0x7c00, 0x7f800000);
    (0xfc01, 0xff802000);
    (0x7e00, 0x7fc00000);
    (0x8000, 0x80000000);
  ]

let test_builtins () =
  let p = load ~binary:(Lazy.force builtins) ~name:"builtins" in
  let n = List.length narrowed and m = List.length widened in
  let input = B.create host S.Float32 n in
  List.iteri
    (fun i (f, _, _) -> (B.bigarray Bigarray.int32 input).{i} <- Int32.of_int f)
    narrowed;
  let bf16 = B.create host S.UInt16 n and f16 = B.create host S.UInt16 n in
  let halves = B.create host S.UInt16 m and out = B.create host S.Float32 m in
  List.iteri
    (fun i (h, _) -> (B.bigarray Bigarray.int16_unsigned halves).{i} <- h)
    widened;
  P.call p [| input; bf16; f16; halves; out |] [| n; m |];
  let u16 b i = (B.bigarray Bigarray.int16_unsigned b).{i} in
  let u32 b i =
    Int32.to_int (B.bigarray Bigarray.int32 b).{i} land 0xffff_ffff
  in
  equal ~msg:"__truncsfbf2" (list int)
    (List.map (fun (_, b, _) -> b) narrowed)
    (List.init n (u16 bf16));
  equal ~msg:"__truncsfhf2" (list int)
    (List.map (fun (_, _, h) -> h) narrowed)
    (List.init n (u16 f16));
  equal ~msg:"__extendhfsf2" (list int) (List.map snd widened)
    (List.init m (u32 out))

(* Lifetime *)

let test_cache () =
  let binary = Lazy.force affine in
  let p = load ~binary ~name:"affine" in
  equal ~msg:"loaded again" nativeint (P.handle p)
    (P.handle (load ~binary ~name:"affine"));
  equal (triple string bool bool) ("affine", true, true)
    (P.name p, Nx_device.equal host (P.device p), P.handle p <> 0n)

let test_release () =
  let binary = Lazy.force affine in
  let code = P.handle (load ~binary ~name:"affine") in
  is_true ~msg:"the code is mapped while the program is reachable" (mapped code);
  Gc.full_major ();
  Nx_device.synchronize host;
  is_false ~msg:"and freed once it is not" (mapped code);
  let p = load ~binary ~name:"affine" in
  let out = B.create host S.Int32 1 in
  P.call p [| out; int32s [| 2l |] |] [| 1; 2; 1 |];
  equal ~msg:"a new load of it runs" (list int32) [ 5l ] (int32s_of out)

(* The runtime *)

let waiting =
  lazy
    (compile
       {|ABI void waiting(void **b, const long long *v) {
  volatile long long *w = b[0];
  w[0] = 1;
  for (long long i = 0; i < v[0] && w[1] == 0; i++) {}
  w[2] = w[1];
}|})

(* The collector needs every domain to reach it: it would wait for the program
   to give up if the calling domain held the runtime while it runs. *)
let test_released () =
  let p = load ~binary:(Lazy.force waiting) ~name:"waiting" in
  let words = B.create host S.Int64 3 in
  let w = B.bigarray Bigarray.int64 words in
  Bigarray.Array1.fill w 0L;
  let d = Domain.spawn (fun () -> P.call p [| words |] [| 2_000_000_000 |]) in
  while w.{0} = 0L do
    Domain.cpu_relax ()
  done;
  for _ = 1 to 3 do
    Gc.minor ()
  done;
  w.{1} <- 1L;
  Domain.join d;
  equal ~msg:"the program saw the flag set after collections" int64 1L w.{2}

(* Refusals *)

let other_arch = if arch = "arm64" then "x86_64" else "arm64"

let test_refusals () =
  let refused ?(name = "f") sub binary =
    match P.load host ~binary ~name with
    | Ok _ -> failf "loaded, not refused for %s" sub
    | Error why ->
        contains ~msg:"the host's name" ~sub:"CPU: " why;
        contains ~msg:sub ~sub why
  in
  refused "not an ELF object" "not an object";
  refused ~name:"g" "no function g"
    (compile "ABI void f(void **b, const long long *v) {}");
  refused "is for"
    (compile ~target:other_arch "ABI void f(void **b, const long long *v) {}");
  refused "nx_no_such_symbol"
    (compile
       {|void nx_no_such_symbol(void);
ABI void f(void **b, const long long *v) { nx_no_such_symbol(); }|});
  refused "writable data"
    (compile
       {|static int calls;
ABI void f(void **b, const long long *v) { *(int *)b[0] = ++calls; }|});
  let fake =
    Nx_device.Driver.device ~name:"FAKE" ~arch:"fake" ~budget:0
      ~load:(fun ~binary:_ ->
        Ok
          {
            Nx_device.Driver.code = None;
            entry = (fun _ -> Ok 1n);
            unload = ignore;
          })
      (Host_visible
         { memory = { alloc = (fun _ -> None); free = ignore }; mapping = None })
  in
  let on_fake = Result.get_ok (P.load fake ~binary:"" ~name:"f") in
  raises_match ~msg:"a program of another device"
    (Exn.invalid_arg ~substring:"FAKE") (fun () -> P.call on_fake [||] [||]);
  let p = load ~binary:(Lazy.force affine) ~name:"affine" in
  let far = Nx_device.Driver.Region.v 0n 4 in
  raises_match ~msg:"memory the host does not address"
    (Exn.invalid_arg ~substring:"does not address") (fun () ->
      P.call p [| Nx_device.Driver.buffer fake far S.UInt8 4 |] [| 0; 0; 0 |])

let test_profile () =
  let p = load ~binary:(Lazy.force affine) ~name:"affine" in
  let out = B.create host S.Int32 4 and input = int32s [| 1l; 2l; 3l; 4l |] in
  let profile = Nx_device.Profile.start () in
  let before = Nx_device.Profile.now () in
  P.call p [| out; input |] [| 4; 3; -5 |];
  let after = Nx_device.Profile.now () in
  let lane = Printf.sprintf "domain %d" (Domain.self () :> int) in
  match Nx_device.Profile.stop profile with
  | [ Span s ] ->
      equal
        (triple string string string)
        ("CPU", lane, "affine")
        (Nx_device.name s.device, s.lane, s.name);
      is_true (before <= s.start && s.start <= s.stop && s.stop <= after)
  | events -> failf "%d events" (List.length events)

(* Splits *)

(* [block] marks each iteration of its block with the block's first and counts
   how often each iteration runs. *)
let block =
  lazy
    (compile
       {|ABI void block(void **b, const long long *v) {
  long long *first = b[0], *runs = b[1];
  for (long long i = v[1]; i < v[2]; i++) { first[i] = v[1]; runs[i] += 1; }
}|})

let int64s_of b =
  Array.to_list
    (Array.init (B.length b)
       (Bigarray.Array1.get (B.bigarray Bigarray.int64 b)))

let split_run ~extent ~blocks =
  let p = load ~binary:(Lazy.force block) ~name:"block" in
  let first = B.create host S.Int64 (max 1 extent)
  and runs = B.create host S.Int64 (max 1 extent) in
  Bigarray.Array1.fill (B.bigarray Bigarray.int64 runs) 0L;
  P.call
    ~split:{ extent; blocks; lo = 1; hi = 2 }
    p [| first; runs |] [| 99; -1; -1 |];
  ( List.filteri (fun i _ -> i < extent) (int64s_of first),
    List.filteri (fun i _ -> i < extent) (int64s_of runs) )

let test_split (extent, blocks) =
  let first, runs = split_run ~extent ~blocks in
  equal ~msg:"every iteration runs once" (list int64)
    (List.init extent (fun _ -> 1L))
    runs;
  let start = Array.make extent 0L in
  for b = 0 to blocks - 1 do
    for i = b * extent / blocks to ((b + 1) * extent / blocks) - 1 do
      start.(i) <- Int64.of_int (b * extent / blocks)
    done
  done;
  equal ~msg:"in the block of its iterations" (list int64) (Array.to_list start)
    first

let test_split_refusals () =
  let p = load ~binary:(Lazy.force block) ~name:"block" in
  let b = B.create host S.Int64 1 in
  let call split values = P.call ~split p [| b; b |] values in
  List.iter
    (fun (why, split, values) ->
      raises_match ~msg:why (Exn.invalid_arg ?substring:None) (fun () ->
          call split values))
    [
      ("no block", { extent = 1; blocks = 0; lo = 1; hi = 2 }, [| 0; 0; 0 |]);
      ( "fewer than no iteration",
        { extent = -1; blocks = 1; lo = 1; hi = 2 },
        [| 0; 0; 0 |] );
      ( "a slot past the values",
        { extent = 1; blocks = 1; lo = 1; hi = 3 },
        [| 0; 0; 0 |] );
      ( "one slot for both bounds",
        { extent = 4; blocks = 2; lo = 1; hi = 1 },
        [| 0; 0; 0 |] );
      ( "iterations times blocks past max_int",
        { extent = max_int / 2; blocks = 3; lo = 1; hi = 2 },
        [| 0; 0; 0 |] );
    ]

(* The entry *)

(* [via] calls a host program through the entry, from its own code: [v.(0)] is
   the entry's address, [v.(1)] the program's, [v.(2)] whether it splits,
   [v.(3)] to [v.(6)] the split, and the program's values follow. *)
let via =
  lazy
    (compile
       {|typedef ABI void (*prog)(void **, const long long *);
typedef void (*entry)(prog, void **, const long long *, long long,
                      const long long *);
ABI void via(void **b, const long long *v) {
  long long n = v[7];
  ((entry)v[0])((prog)v[1], b, v + 8, n, v[2] ? v + 3 : 0);
}|})

(* [entered p buffers values ?split] calls [p] through the entry, from [via]. *)
let entered ?split p buffers values =
  let v = load ~binary:(Lazy.force via) ~name:"via" in
  let split =
    match split with
    | None -> [| 0; 0; 0; 0; 0 |]
    | Some (s : P.split) -> [| 1; s.extent; s.blocks; s.lo; s.hi |]
  in
  P.call v buffers
    (Array.concat
       [
         [| Nativeint.to_int P.entry; Nativeint.to_int (P.handle p) |];
         split;
         [| Array.length values |];
         values;
       ])

let entry_split_run ~extent ~blocks =
  let p = load ~binary:(Lazy.force block) ~name:"block" in
  let first = B.create host S.Int64 (max 1 extent)
  and runs = B.create host S.Int64 (max 1 extent) in
  Bigarray.Array1.fill (B.bigarray Bigarray.int64 runs) 0L;
  entered
    ~split:{ extent; blocks; lo = 1; hi = 2 }
    p [| first; runs |] [| 99; -1; -1 |];
  ( List.filteri (fun i _ -> i < extent) (int64s_of first),
    List.filteri (fun i _ -> i < extent) (int64s_of runs) )

(* Splits of no iteration, one, fewer than their blocks, and more blocks than
   the host's threads. *)
let entry_splits =
  let workers = P.workers () in
  Gen.with_pp
    (fun ppf (extent, blocks) -> Format.fprintf ppf "(%d, %d)" extent blocks)
    (Gen.one_of
       [
         Gen.pair (Gen.int_range 0 300) (Gen.int_range 1 16);
         Gen.pair (Gen.int_range 0 1) (Gen.int_range 1 4);
         Gen.pair (Gen.int_range 1 8) (Gen.int_range 9 16);
         Gen.pair (Gen.int_range 64 300)
           (Gen.int_range (workers + 1) ((4 * workers) + 1));
       ])

let test_entry_split =
  prop "a split through the entry runs the blocks a split call runs"
    entry_splits (fun (extent, blocks) ->
      cover "no iteration" (extent = 0);
      cover "fewer iterations than blocks" (extent > 0 && extent < blocks);
      cover "more blocks than threads" (blocks > P.workers ());
      equal
        (pair (list int64) (list int64))
        (split_run ~extent ~blocks)
        (entry_split_run ~extent ~blocks))

let test_entry_call () =
  let p = load ~binary:(Lazy.force affine) ~name:"affine" in
  let run call =
    let out = B.create host S.Int32 4 in
    call p [| out; int32s [| 1l; 2l; 3l; 4l |] |] [| 4; 3; -5 |];
    int32s_of out
  in
  equal (list int32)
    (run (fun p b v -> P.call p b v))
    (run (fun p b v -> entered p b v))

(* The spans of [f ()] as [(device, lane, name)], [f] run while a profile is
   taken. *)
let spans f =
  let profile = Nx_device.Profile.start () in
  f ();
  List.filter_map
    (function
      | Nx_device.Profile.Span s ->
          Some (Nx_device.name s.device, s.lane, s.name)
      | _ -> None)
    (Nx_device.Profile.stop profile)

let test_entry_spans () =
  let p = load ~binary:(Lazy.force affine) ~name:"affine" in
  let out = B.create host S.Int32 4 and input = int32s [| 1l; 2l; 3l; 4l |] in
  let lane = Printf.sprintf "domain %d" (Domain.self () :> int) in
  let direct = spans (fun () -> P.call p [| out; input |] [| 4; 3; -5 |]) in
  let entered = spans (fun () -> entered p [| out; input |] [| 4; 3; -5 |]) in
  equal ~msg:"a call"
    (list (triple string string string))
    [ ("CPU", lane, "affine") ]
    direct;
  equal ~msg:"a call through the entry, inside the program that calls it"
    (list (triple string string string))
    [ ("CPU", lane, "via"); ("CPU", lane, "affine") ]
    entered

(* Calls through the entry under nested profiles: each profile sees the calls
   made while it is taken. *)
let test_entry_nested () =
  let p = load ~binary:(Lazy.force affine) ~name:"affine" in
  let out = B.create host S.Int32 4 and input = int32s [| 1l; 2l; 3l; 4l |] in
  let call () = entered p [| out; input |] [| 4; 3; -5 |] in
  let names =
    List.filter_map (function
      | Nx_device.Profile.Span s -> Some s.name
      | _ -> None)
  in
  let outer = Nx_device.Profile.start () in
  call ();
  let (), inner = Nx_device.Profile.take call in
  call ();
  let outer = Nx_device.Profile.stop outer in
  equal ~msg:"inner" (list string) [ "via"; "affine" ] (names inner);
  equal ~msg:"outer" (list string)
    [ "via"; "affine"; "via"; "affine"; "via"; "affine" ]
    (names outer)

let test_entry_no_profile () =
  let p = load ~binary:(Lazy.force affine) ~name:"affine" in
  let out = B.create host S.Int32 4 in
  entered p [| out; int32s [| 1l; 2l; 3l; 4l |] |] [| 4; 3; -5 |];
  equal (list (triple string string string)) [] (spans ignore)

(* [held] is [block] whose first block says it started, in [flags.(0)], then
   runs once [flags.(1)] is set. *)
let held =
  lazy
    (compile
       {|ABI void held(void **b, const long long *v) {
  long long *flags = b[0], *first = b[1], *runs = b[2];
  if (v[1] == 0) {
    __atomic_store_n(&flags[0], 1, __ATOMIC_SEQ_CST);
    while (__atomic_load_n(&flags[1], __ATOMIC_SEQ_CST) == 0) {}
  }
  for (long long i = v[1]; i < v[2]; i++) { first[i] = v[1]; runs[i] += 1; }
}|})

(* A domain's split call while another's holds the host's threads: the second
   call runs, whether it waits for the first or starts before the first holds
   the threads, and each runs every iteration once. *)
let test_split_domains () =
  let p = load ~binary:(Lazy.force held) ~name:"held" in
  let extent = 64 and blocks = 8 in
  let flags = B.create host S.Int64 2 in
  let first = B.create host S.Int64 extent
  and runs = B.create host S.Int64 extent in
  List.iter
    (fun b -> Bigarray.Array1.fill (B.bigarray Bigarray.int64 b) 0L)
    [ flags; runs ];
  let flag = B.bigarray Bigarray.int64 flags in
  let holding =
    Domain.spawn (fun () ->
        P.call
          ~split:{ extent; blocks; lo = 1; hi = 2 }
          p [| flags; first; runs |] [| 0; -1; -1 |])
  in
  while Bigarray.Array1.get flag 0 = 0L do
    Domain.cpu_relax ()
  done;
  let waiting = Domain.spawn (fun () -> split_run ~extent ~blocks) in
  Bigarray.Array1.set flag 1 1L;
  Domain.join holding;
  let _, second = Domain.join waiting in
  let ones = List.init extent (fun _ -> 1L) in
  equal ~msg:"the holding call" (list int64) ones (int64s_of runs);
  equal ~msg:"the second call" (list int64) ones second

(* The host's threads, shared *)

(* Split calls, and nx.cpu's kernels, which run on the same threads: [sqrt] of
   the host's compute-bound class over more than 2^17 floats runs on two threads
   or more, in chunks fewer than its elements, unlike a split call's one chunk
   per block. The squares of 0 to [max_root - 1] are exact, and so are their
   square roots. *)
let min_root = 1 lsl 17
let max_root = min_root + (1 lsl 15)

let squares =
  let r = Nx.arange_f Nx.float64 0. (Float.of_int max_root) 1. in
  Nx.mul r r

(* The first indices, at most four, whose square root is not the index. *)
let sqrt_misses n =
  let roots =
    Bigarray.array1_of_genarray
      (Nx.to_bigarray (Nx.sqrt (Nx.slice [ Nx.R (0, n) ] squares)))
  in
  let misses = ref [] in
  for i = n - 1 downto 0 do
    if Bigarray.Array1.unsafe_get roots i <> Float.of_int i then
      misses := i :: !misses
  done;
  List.filteri (fun i _ -> i < 4) !misses

let shared_commands =
  let split_gen =
    Gen.with_pp
      (fun ppf (extent, blocks) -> Format.fprintf ppf "(%d, %d)" extent blocks)
      (Gen.pair (Gen.int_range 0 300) (Gen.int_range 1 16))
  in
  [
    command "split"
      (split_gen @-> returns (list int64))
      (fun (extent, _) -> List.init extent (fun _ -> 1L))
      (fun (extent, blocks) -> snd (split_run ~extent ~blocks));
    command "sqrt"
      (Gen.int_range min_root max_root @-> returns (list int))
      (fun _ -> [])
      sqrt_misses;
  ]

(* Parking *)

external running_threads : unit -> int = "test_host_running_threads"

let needs_thread_states () =
  if running_threads () < 0 then
    skip ~reason:"the system does not report its threads' states" ()

(* Polls until at most [n] threads other than this one run, for at most [within]
   seconds, and is the last count. *)
let settle_to n ~within =
  let deadline = Unix.gettimeofday () +. within in
  let rec poll () =
    let k = running_threads () in
    if k <= n || Unix.gettimeofday () > deadline then k
    else (
      Unix.sleepf 0.001;
      poll ())
  in
  poll ()

(* Polls [ready] for at most [within] seconds, and fails with [why] if it never
   holds: a thread that waits for a wakeup that never comes fails the test
   instead of hanging it. *)
let within seconds why ready =
  let deadline = Unix.gettimeofday () +. seconds in
  while not (ready ()) do
    if Unix.gettimeofday () > deadline then failf "after %gs, %s" seconds why;
    Unix.sleepf 0.001
  done

(* [hold] is a split call of two blocks, each given its own copy of the values,
   the caller's at the lowest address. Each block waits for the other to start,
   so a worker must run one. Then the block with the higher copy, a worker's,
   waits for [state.(1)] to be set while the caller finishes. *)
let hold =
  lazy
    (compile
       {|ABI void hold(void **b, const long long *v) {
  unsigned long long *state = b[0];
  long long *runs = b[1];
  unsigned long long n = __atomic_fetch_add(&state[0], 1, __ATOMIC_SEQ_CST);
  unsigned long long me = (unsigned long long)v, other;
  __atomic_store_n(&state[2 + n], me, __ATOMIC_SEQ_CST);
  while ((other = __atomic_load_n(&state[3 - n], __ATOMIC_SEQ_CST)) == 0) {}
  if (me > other)
    while (__atomic_load_n(&state[1], __ATOMIC_SEQ_CST) == 0) {}
  for (long long i = v[1]; i < v[2]; i++) runs[i] += 1;
}|})

(* An idle pool's threads park. A job then wakes a parked worker, and a caller
   whose job a worker holds parks until the worker counts down. *)
let test_parks () =
  needs_thread_states ();
  if P.workers () < 2 then skip ~reason:"the host splits on one thread" ();
  let p = load ~binary:(Lazy.force hold) ~name:"hold" in
  ignore (split_run ~extent:64 ~blocks:8);
  equal ~msg:"no thread runs once the pool is idle" int 0
    (settle_to 0 ~within:5.);
  let state = B.create host S.Int64 4 and runs = B.create host S.Int64 2 in
  List.iter
    (fun b -> Bigarray.Array1.fill (B.bigarray Bigarray.int64 b) 0L)
    [ state; runs ];
  let s = B.bigarray Bigarray.int64 state in
  let finished = Atomic.make false in
  let caller =
    Domain.spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Atomic.set finished true)
          (fun () ->
            P.call
              ~split:{ extent = 2; blocks = 2; lo = 1; hi = 2 }
              p [| state; runs |] [| 0; -1; -1 |]))
  in
  within 10. "a parked worker was not woken for the job" (fun () ->
      Bigarray.Array1.get s 0 = 2L);
  equal ~msg:"only the holding worker runs: the caller parked" int 1
    (settle_to 1 ~within:5.);
  Bigarray.Array1.set s 1 1L;
  within 10. "the parked caller was not woken at the job's end" (fun () ->
      Atomic.get finished);
  Domain.join caller;
  equal ~msg:"each iteration ran once" (list int64) [ 1L; 1L ] (int64s_of runs)

(* A burst of split calls of two blocks takes the caller and one worker. The
   other workers, which the call before the burst kept spinning and which take
   part in none of its calls, park while it runs: once at most two threads run
   beside this one, the count read every millisecond stays there. A worker that
   a broadcast wakes runs for a moment, so the median is what counts. *)
let test_narrow_burst () =
  needs_thread_states ();
  if P.workers () < 3 then skip ~reason:"every worker takes part" ();
  let p = load ~binary:(Lazy.force block) ~name:"block" in
  let buffers =
    Array.init 2 (fun _ -> B.create host S.Int64 (Int.max 2 (P.workers ())))
  in
  let split = { P.extent = 2; blocks = 2; lo = 1; hi = 2 }
  and values = [| 0; -1; -1 |] in
  let wide = { split with extent = P.workers (); blocks = P.workers () } in
  let stop = Atomic.make false in
  let burst =
    Domain.spawn (fun () ->
        P.call ~split:wide p buffers values;
        while not (Atomic.get stop) do
          P.call ~split p buffers values
        done)
  in
  let samples =
    Fun.protect
      ~finally:(fun () ->
        Atomic.set stop true;
        Domain.join burst)
      (fun () ->
        ignore (settle_to 2 ~within:5.);
        List.init 51 (fun _ ->
            Unix.sleepf 0.001;
            running_threads ()))
  in
  let median = List.nth (List.sort Int.compare samples) 25 in
  at_most
    ~msg:
      (Printf.sprintf
         "threads running beside the burst's caller and its worker, of %s"
         (String.concat " " (List.map string_of_int samples)))
    int ~than:2 median

let () =
  exit
    (run "nx.device host programs"
       [
         test "a program reads its arguments and writes its buffers"
           test_arguments;
         test
           "a program runs on the buffers of test devices of the host's \
            memory, and by address"
           test_devices;
         test
           "a program links its constants, its own functions and the math \
            library"
           test_links;
         test
           "a program's calls to the 16-bit float conversions of the \
            compiler's runtime link and round to nearest even"
           test_builtins;
         test "a function of a binary loads once while it is reachable"
           test_cache;
         test "an unreachable program's code is freed" test_release;
         test "the runtime is released while a program runs" test_released;
         test "a call is a span of its program while a profile is taken"
           test_profile;
         test "the host refuses what it cannot load or call" test_refusals;
         cases
           ~name:(fun (n, b) -> Printf.sprintf "%d iterations, %d blocks" n b)
           "a split call runs each iteration once, in its block,"
           [ (1000, 7); (5, 8); (64, 1); (0, 3); (3, 3) ]
           test_split;
         test "a split call refuses a split it cannot run" test_split_refusals;
         test "a call through the entry is a call" test_entry_call;
         test_entry_split;
         test
           "a call through the entry is a span of its program while a profile \
            is taken"
           test_entry_spans;
         test
           "a call through the entry is a span of each profile taken, nested \
            ones included"
           test_entry_nested;
         test "a call through the entry records no span while none is taken"
           test_entry_no_profile;
         test "a split call runs while another domain's holds the threads"
           test_split_domains;
         stateful "split calls and nx.cpu kernels each run every iteration once"
           shared_commands;
         stateful ~domains:2 ~count:20
           "split calls and nx.cpu kernels from two domains each run every \
            iteration once"
           shared_commands;
         test "the host's threads park when idle, and a job wakes them"
           test_parks;
         test "a worker that a burst of narrow calls leaves out parks"
           test_narrow_burst;
       ])
