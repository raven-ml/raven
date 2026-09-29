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
  let p = P.load host ~binary:(Lazy.force affine) ~name:"affine" in
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
  let p = P.load host ~binary:(Lazy.force linked) ~name:"linked" in
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
  let p = P.load host ~binary:(Lazy.force builtins) ~name:"builtins" in
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
  let p = P.load host ~binary ~name:"affine" in
  is_true ~msg:"loaded again" (P.load host ~binary ~name:"affine" == p);
  equal (triple string bool bool) ("affine", true, true)
    (P.name p, Nx_device.equal host (P.device p), P.handle p <> 0n)

let test_release () =
  let binary = Lazy.force affine in
  let load () = P.handle (P.load host ~binary ~name:"affine") in
  let code = load () in
  is_true ~msg:"the code is mapped while the program is reachable" (mapped code);
  Gc.full_major ();
  Nx_device.synchronize host;
  is_false ~msg:"and freed once it is not" (mapped code);
  let p = P.load host ~binary ~name:"affine" in
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
  let p = P.load host ~binary:(Lazy.force waiting) ~name:"waiting" in
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
  let load ?(name = "f") binary () = ignore (P.load host ~binary ~name) in
  let refused substring binary =
    raises_match ~msg:substring (Exn.failure ~substring) (load binary)
  in
  refused "not an ELF object" "not an object";
  raises_match ~msg:"no function"
    (Exn.failure ~substring:"no function g")
    (load ~name:"g" (compile "ABI void f(void **b, const long long *v) {}"));
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
    Nx_device.make ~name:"FAKE" ~arch:"fake" ~budget:0
      ~memory:{ alloc = (fun _ -> None); free = ignore }
      ~load:(fun ~binary:_ ~name:_ -> 1n)
      ()
  in
  raises_match ~msg:"a program of another device"
    (Exn.invalid_arg ~substring:"FAKE") (fun () ->
      P.call (P.load fake ~binary:"" ~name:"f") [||] [||]);
  let p = P.load host ~binary:(Lazy.force affine) ~name:"affine" in
  let far = { Nx_device.host = None; device = 0n; handle = 0n } in
  raises_match ~msg:"memory the host does not address"
    (Exn.invalid_arg ~substring:"does not address") (fun () ->
      P.call p [| Nx_device.external_buffer fake far S.UInt8 4 |] [| 0; 0; 0 |])

let () =
  exit
    (run "nx.device host programs"
       [
         test "a program reads its arguments and writes its buffers"
           test_arguments;
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
         test "the host refuses what it cannot load or call" test_refusals;
       ])
