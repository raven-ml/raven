open Windtrap
open Tolk
module Compiler = Renderer.Compiler

let kernel =
  {|extern "C" __global__ void add(float *a, const float *b) {
  a[threadIdx.x] += b[threadIdx.x];
}|}

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

let rejection f =
  match f () with _ -> None | exception Compiler.Compile_error msg -> Some msg

(* NVRTC's absence is a Compile_error that names the variable to set. *)
let absent msg = has_infix ~affix:"NVRTC_PATH" msg

(* [with_nvrtc f] is [f ()], or skips the test if NVRTC is not on the
   machine. *)
let with_nvrtc f =
  match f () with
  | v -> v
  | exception Compiler.Compile_error msg when absent msg -> skip ~reason:msg ()

(* The cache *)

let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f

(* [named ~prefix table] checks that [table] is [prefix] then a digest. *)
let named ~prefix table =
  let n = Int.min (String.length prefix) (String.length table) in
  equal string ~msg:"the name" prefix (String.sub table 0 n);
  equal int ~msg:"the digest's length" 32 (String.length table - n)

let table c = Option.get (Compiler.cachekey c)

let cache =
  group "cache"
    [
      test
        "binaries are cached in the table of cuda, the architecture and a \
         digest" (fun () ->
          named ~prefix:"compile_cuda_sm_89_"
            (table (Compiler_cuda.nvrtc "sm_89")));
      test "cubins and PTX are cached in different tables" (fun () ->
          not_equal string
            (table (Compiler_cuda.nvrtc "sm_89"))
            (table (Compiler_cuda.nvrtc ~ptx:false "sm_89")));
      test "cache_key names the table" (fun () ->
          named ~prefix:"compile_nv_sm_120_"
            (table (Compiler_cuda.nvrtc ~ptx:false ~cache_key:"nv" "sm_120")));
      test "the table is named with ccache off" (fun () ->
          let table () = Compiler.cachekey (Compiler_cuda.nvrtc "sm_89") in
          equal (option string) (table ()) (ccache_off table));
    ]

(* Without NVRTC on the machine *)

let without_nvrtc =
  group "without NVRTC on the machine"
    [
      test "a compile raises Compile_error naming nvrtc and NVRTC_PATH"
        (fun () ->
          match
            rejection (fun () ->
                Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel)
          with
          | None -> skip ~reason:"NVRTC is on the machine" ()
          | Some msg -> in_order ~subs:[ "nvrtc"; "NVRTC_PATH" ] msg);
    ]

(* A library that does not load: these tests run in a process of their own,
   where NVRTC_PATH names a file that is no library. *)

let load_failure () =
  rejection (fun () -> Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel)

let not_a_library =
  group ~tags:[ "no-library" ] "a library that does not load"
    [
      test "making a compiler loads nothing, and a compile raises Compile_error"
        (fun () ->
          let msg = require_some (load_failure ()) in
          in_order ~subs:[ "nvrtc"; "not_a_library" ] msg);
      test "every compile raises the same Compile_error" (fun () ->
          let first = load_failure () in
          equal (option string) first (load_failure ());
          is_some first);
      test "compiles from several domains at once raise it" (fun () ->
          let domains = List.init 4 (fun _ -> Domain.spawn load_failure) in
          let msgs = List.map Domain.join domains in
          is_some (List.hd msgs);
          List.iter (equal (option string) (List.hd msgs)) msgs);
      test "a cached binary is served without loading NVRTC" (fun () ->
          let nvrtc = Compiler_cuda.nvrtc "sm_89" in
          let table = Option.get (Compiler.cachekey nvrtc) in
          Helpers.Diskcache.put ~table kernel "cached binary";
          equal string "cached binary" (Compiler.compile_cached nvrtc kernel));
    ]

(* Disassembly *)

let disassembly =
  group "disassembly"
    [
      test "a PTX that ptxas cannot assemble prints why" (fun () ->
          Compiler.disassemble (Compiler_cuda.nvrtc "sm_89") "not ptx";
          flush stdout;
          contains ~sub:"ptxas -arch=sm_89 " (output ()));
      test "a cubin that nvdisasm cannot read prints why" (fun () ->
          Compiler.disassemble
            (Compiler_cuda.nvrtc ~ptx:false "sm_89")
            "not a cubin";
          flush stdout;
          let text = output () in
          contains ~sub:"nvdisasm " text;
          not_contains ~sub:"ptxas -arch" text);
    ]

(* NVRTC *)

let smoke =
  group "NVRTC"
    [
      slow "compiles a kernel to PTX" (fun () ->
          let ptx =
            with_nvrtc (fun () ->
                Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel)
          in
          in_order ~subs:[ ".target sm_89"; ".entry add" ] ptx);
      slow "compiles a kernel to a cubin" (fun () ->
          let cubin =
            with_nvrtc (fun () ->
                Compiler.compile (Compiler_cuda.nvrtc ~ptx:false "sm_89") kernel)
          in
          starts_with ~affix:"\x7fELF" cubin);
      slow "a rejected source raises Compile_error with NVRTC's log" (fun () ->
          match
            Compiler.compile
              (Compiler_cuda.nvrtc "sm_89")
              "extern \"C\" __global__ void f() { undeclared_name = 1; }"
          with
          | _ -> fail "NVRTC accepted an undeclared name"
          | exception Compiler.Compile_error msg when absent msg ->
              skip ~reason:msg ()
          | exception Compiler.Compile_error msg ->
              in_order
                ~subs:[ "NVRTC_ERROR_COMPILATION"; "undeclared_name" ]
                msg);
      slow "compiles from several domains at once" (fun () ->
          let compile arch () =
            Compiler.compile (Compiler_cuda.nvrtc arch) kernel
          in
          ignore (with_nvrtc (compile "sm_80"));
          let domains =
            List.map
              (fun arch -> Domain.spawn (compile arch))
              [ "sm_75"; "sm_80"; "sm_86"; "sm_89" ]
          in
          List.iter2
            (fun arch ptx -> contains ~sub:(".target " ^ arch) ptx)
            [ "sm_75"; "sm_80"; "sm_86"; "sm_89" ]
            (List.map Domain.join domains));
    ]

let () =
  exit
    (run "Tolk.Compiler_cuda"
       [ cache; without_nvrtc; not_a_library; disassembly; smoke ])
