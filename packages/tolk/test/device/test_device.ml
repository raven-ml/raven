open Windtrap
open Tolk
module Elf = Device.Tiny_elf

(* Parameters *)

let pp_kind ppf (addrspace, dtype) =
  Format.fprintf ppf "%a %a" Dtype.pp_addr_space addrspace Dtype.pp dtype

(* A parameter's address space and type: a scalar by value, or a buffer in any
   other space, passed by its address. *)
let gen_kind =
  let open Gen in
  let+ addrspace =
    frequency
      [
        (3, constant Dtype.Alu);
        (3, constant Dtype.Global);
        (1, constant Dtype.Local);
        (1, constant Dtype.Reg);
      ]
  and+ dtype = Dtypes.stored in
  (addrspace, dtype)

let gen_kinds =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_kind)
    (Gen.list ~size:(Gen.int_range 0 8) gen_kind)

let is_buffer (addrspace, _) = addrspace <> Dtype.Alu

(* The parameter of [kind] in [slot]: a buffer of [slot + 1] elements or a
   scalar, named after its slot. *)
let elf_param slot ((addrspace, dtype) as kind) : Elf.param =
  {
    name = Some (Printf.sprintf "p%d" slot);
    addrspace;
    dtype;
    shape = (if is_buffer kind then [ slot + 1 ] else []);
  }

let signature kinds = List.mapi elf_param kinds
let layout_w = list (pair int Dtypes.dtype)

(* Compiled programs *)

let program_of case = Golden.sink (case ^ "_program.golden")

(* The signature tinygrad's [to_elf] gives a program, which the golden's
   generator packs: each parameter's name, address space, type and shape, in
   slot order. *)
let to_elf_signature prg : Elf.param list =
  List.map
    (fun u : Elf.param ->
      let p = Ops.param_arg_of u in
      {
        name = p.name;
        addrspace = p.addrspace;
        dtype = Ops.dtype u;
        shape = Shape.max_shape u;
      })
    (Shape.kernel_params prg)

let programs =
  [
    "elementwise";
    "elementwise_metal";
    "sparse";
    "scalars";
    "interleaved";
    "interleaved_metal";
    "fill";
  ]

(* Tiny_elf.iter_sig *)

let packed_as_golden cell =
  let signature = to_elf_signature (program_of (cell "program")) in
  let offset = int_of_string (cell "offset") in
  equal (pair int Dtypes.dtype)
    (int_of_string (cell "at"), Dtypes.dtype_of_cell (cell "dtype"))
    (List.nth
       (Elf.iter_sig ~offset signature)
       (int_of_string (cell "position")))

let gen_offset =
  Gen.frequency
    [
      (6, Gen.int_range 0 40);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 7; 8; 9 ]);
    ]

(* Each value lies at the first offset aligned to its size past the previous
   one, in the signature's order, a buffer as its address. *)
let packed (offset, kinds) =
  let layout = Elf.iter_sig ~offset (signature kinds) in
  cover "a buffer is passed" (List.exists is_buffer kinds);
  cover "a buffer lies outside global memory"
    (List.exists (fun (a, _) -> a = Dtype.Local || a = Dtype.Reg) kinds);
  equal (list Dtypes.dtype)
    (List.map
       (fun ((_, dtype) as kind) ->
         if is_buffer kind then Dtype.Uint64 else dtype)
       kinds)
    (List.map snd layout);
  ignore
    (List.fold_left
       (fun past (at, dtype) ->
         let size = Dtype.itemsize dtype in
         cover "a value is padded" (at > past);
         equal
           ~msg:(Printf.sprintf "at %d, modulo its size" at)
           int 0 (at mod size);
         at_least int ~than:past at;
         less int ~than:(past + size) at;
         at + size)
       offset layout)

let iter_sig =
  group "Tiny_elf.iter_sig"
    [
      group
        "packs each program's signature from each offset as its golden states"
        [
          Golden.cases
            ~key:[ "program"; "offset"; "position" ]
            "layouts.golden" packed_as_golden;
        ];
      prop "packs each value at the next offset aligned to its size"
        ~examples:
          [
            (0, Dtype.[ (Alu, Int8); (Global, Int8); (Alu, Int16) ]);
            (5, []);
            (3, Dtype.[ (Local, Float16); (Alu, Float64) ]);
          ]
        (Gen.pair gen_offset gen_kinds)
        packed;
      prop "starts at byte 0 by default" gen_kinds (fun kinds ->
          equal layout_w
            (Elf.iter_sig ~offset:0 (signature kinds))
            (Elf.iter_sig (signature kinds)));
    ]

(* Tiny_elf.layout *)

(* The parameter node of [kind] in [slot]. *)
let param_node slot (addrspace, dtype) = Shape.param ~addrspace slot dtype

let rec scalar_after_buffer = function
  | b :: (s :: _ as rest) ->
      (is_buffer b && not (is_buffer s)) || scalar_after_buffer rest
  | _ -> false

let packed_as_iter_sig kinds =
  cover "a scalar follows a buffer" (scalar_after_buffer kinds);
  equal layout_w
    (Elf.iter_sig (signature kinds))
    (Elf.layout (List.mapi param_node kinds))

let layout =
  group "Tiny_elf.layout"
    [
      prop "is iter_sig of the parameters, in their order"
        ~examples:
          [ []; Dtype.[ (Global, Int8); (Alu, Int64); (Local, Float32) ] ]
        gen_kinds packed_as_iter_sig;
      cases
        "packs a compiled program's parameters as iter_sig packs its signature"
        ~id:Fun.id programs (fun case ->
          let prg = program_of case in
          equal layout_w
            (Elf.iter_sig (to_elf_signature prg))
            (Elf.layout (Shape.kernel_params prg)));
      test "raises on a node that is no parameter" (fun () ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              Elf.layout
                [ param_node 0 (Global, Float32); Ops.int ~dtype:Int32 1 ]));
    ]

(* Targets *)

let target_of_string s =
  require_ok ~pp:Format.pp_print_string (Helpers.Target.of_string s)

let pp_target = Helpers.Target.pp

(* The target of [device], with no other field: [device] is taken as it is. *)
let on device =
  {
    Helpers.Target.device;
    renderer = "";
    arch = "";
    interface = "";
    indices = "";
  }

(* Renderers *)

let every_device = [ "CPU"; "METAL"; "CUDA"; "NV"; "AMD" ]

(* Each device's renderer, as device.mli lists them. *)
let own_renderer = function
  | "CPU" -> "CLANG"
  | "METAL" -> "METAL"
  | "CUDA" | "NV" -> "CUDA"
  | _ -> "HIP"

let renderer_named = function
  | "CLANG" -> Cstyle.clang
  | "METAL" -> Cstyle.metal
  | "CUDA" -> Cstyle.cuda
  | "HIP" -> Cstyle.hip
  | name -> failf "no renderer is named %s" name

let pp_renderer ppf (r : Renderer.t) =
  Format.fprintf ppf "%s for %a" r.name pp_target r.target

let renderer_w =
  Testable.make ~pp:pp_renderer ~equal:(fun (r0 : Renderer.t) r1 ->
      r0.name = r1.name && r0.target = r1.target)

let renderer_or_error = result renderer_w string

(* One renderer, made once: physical identity. *)
let same = Testable.make ~pp:pp_renderer ~equal:( == )

(* A renderer that fails to make fails with its own message, which tinygrad
   writes as Python's exception text and tolk as the renderer's
   [Invalid_argument]. *)
let message_of_renderer name target =
  match renderer_named name target with
  | exception Invalid_argument m -> m
  | _ -> failf "%s makes a renderer for %a" name pp_target target

(* tinygrad's error ends with a renderer the name may misspell, which tolk
   leaves out: each device has one renderer. *)
let without_hint error =
  let hint = ", did you mean: " in
  let n = String.length hint in
  let rec cut i =
    if i + n > String.length error then error
    else if String.sub error i n = hint then String.sub error 0 i
    else cut (i + 1)
  in
  cut 0

(* The golden's target is the one a DEV setting gives the device. *)
let selected_as_golden cell =
  let target = target_of_string (cell "target") in
  let expected =
    match cell "outcome" with
    | "ok" -> Ok (renderer_named (cell "renderer") target)
    | "no renderer" -> Error (without_hint (cell "error"))
    | "renderer fails" -> Error (message_of_renderer (cell "renderer") target)
    | outcome -> failf "no outcome %s" outcome
  in
  equal renderer_or_error expected (Device.renderer target)

(* Architectures each device's renderer takes, then one it refuses. *)
let arches = function
  | "CPU" -> [ "arm64,apple-m1"; "x86_64,x86-64"; "sparc,v9" ]
  | "METAL" -> [ "Apple9"; "Apple6"; "AppleX" ]
  | "CUDA" | "NV" -> [ "sm_89"; "sm_75"; "sm_x" ]
  | _ -> [ "gfx1100"; "gfx942"; "gfx1201" ]

(* Targets that often have a renderer, and often differ by one field. *)
let gen_target =
  let open Gen in
  let* device = of_list ~pp:Format.pp_print_string every_device in
  let+ renderer =
    frequency
      [
        (3, constant "");
        (3, constant (own_renderer device));
        ( 1,
          of_list ~pp:Format.pp_print_string [ "CLANG"; "METAL"; "CUDA"; "HIP" ]
        );
      ]
  and+ arch = of_list ~pp:Format.pp_print_string (arches device) in
  { (on device) with renderer; arch }

let gen_placed_target =
  Gen.with_pp pp_target
    (let open Gen in
     let+ t = gen_target
     and+ interface = of_list ~pp:Format.pp_print_string [ ""; "PCI"; "USB" ]
     and+ indices = of_list ~pp:Format.pp_print_string [ ""; "0"; "1,2" ] in
     { t with interface; indices })

let made_for (t : Helpers.Target.t) =
  match Device.renderer t with
  | Error _ -> ()
  | Ok r ->
      cover "a target with an interface has a renderer" (t.interface <> "");
      equal renderer_w (renderer_named (own_renderer t.device) t) r

let refused =
  [
    "QCOM";
    "CL";
    "PYTHON";
    "NULL";
    "";
    "CPU:0";
    "CPU:1";
    "AMD:0";
    "NV:1";
    "cpu";
    "metal";
    "cuda";
    "nv";
    "amd";
    "Cuda";
    "DISK";
    "DISK:/tmp/weights";
    "CUDA:CUDA:sm_89";
    "CPU::arm64,apple-m1";
  ]

(* comgr is the library a test can hide: once COMGR_PATH names a missing file,
   LD_LIBRARY_PATH is unset and ROCM_PATH is an empty directory, only a system
   directory still holds it. No generator here draws the target's interface, so
   no other test has asked for its renderer. *)
let missing_library () =
  let target = target_of_string "MISSING:1+AMD:HIP:gfx1030" in
  setenv "COMGR_PATH" (Some (Filename.concat (temp_dir ()) "comgr"));
  setenv "LD_LIBRARY_PATH" None;
  let missing f = Setting.context [ B (Setting.rocm_path, temp_dir ()) ] f in
  match missing (fun () -> Cstyle.hip target) with
  | _ -> skip ~reason:"this host has comgr in a system directory" ()
  | exception Failure why ->
      missing (fun () ->
          raises (Failure why) (fun () -> Device.renderer target));
      setenv "COMGR_PATH" (Some (temp_file ()));
      equal renderer_or_error (Ok (Cstyle.hip target)) (Device.renderer target)

(* Calls for a few targets, so that some ask for one target twice. *)
let gen_calls =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_target)
    (let open Gen in
     let* pool = list ~size:(int_range 1 4) gen_target in
     list ~size:(int_range 2 8) (of_list pool))

let rec pairs = function
  | [] -> []
  | x :: rest -> List.map (fun y -> (x, y)) rest @ pairs rest

let answer t = Result.map (fun r -> (t, r)) (Device.renderer t)

let one_renderer_per_target calls =
  let pairs =
    pairs (List.filter_map (fun t -> Result.to_option (answer t)) calls)
  in
  cover "two calls for one target return renderers"
    (List.exists (fun ((t0, _), (t1, _)) -> t0 = t1) pairs);
  cover "calls for two targets return renderers"
    (List.exists (fun ((t0, _), (t1, _)) -> t0 <> t1) pairs);
  List.iter
    (fun ((t0, r0), (t1, r1)) ->
      let msg = Format.asprintf "%a and %a" pp_target t0 pp_target t1 in
      if t0 = t1 then equal ~msg same r0 r1 else not_equal ~msg same r0 r1)
    pairs

(* A CPU target no call has asked for, with an interface no other test draws and
   indices of a count, and the calls asking for it. *)
type fresh = { target : Helpers.Target.t; asking : int Atomic.t }

let fresh =
  let count = Atomic.make 0 in
  fun () ->
    {
      target =
        {
          (on "CPU") with
          arch = "x86_64,x86-64";
          interface = "FRESH";
          indices = string_of_int (Atomic.fetch_and_add count 1);
        };
      asking = Atomic.make 0;
    }

(* The renderer for [f], and whether another call asked for it meanwhile. A verb
   runs on the test's domain only, so the judge reads both. *)
let ask f =
  let before = Atomic.fetch_and_add f.asking 1 in
  let r = Device.renderer f.target in
  let after = Atomic.fetch_and_add f.asking (-1) - 1 in
  (r, before > 0 || after > 0)

let asked = pair renderer_or_error bool
let fresh_target = abstract "t"

(* Of two renderers that two domains made for one target, the table keeps one,
   and a later call returns it. *)
let kept_by_the_table () = function
  | Ok (r, concurrently) ->
      (* A label, not a cover: calls overlap only when each domain has a
         processor, which a loaded or small machine does not give. On 14 loaded
         cores they overlap in 55-70% of cases. *)
      classify "two domains ask for one target concurrently" concurrently;
      let r : Renderer.t = require_ok ~pp:Format.pp_print_string r in
      equal same
        (require_ok ~pp:Format.pp_print_string (Device.renderer r.target))
        r
  | Error e -> raise e

let renderer =
  group "renderer"
    [
      group "is each target's renderer or error as its golden states"
        [
          Golden.cases
            ~key:[ "dev"; "device"; "arch" ]
            "renderers.golden" selected_as_golden;
        ];
      prop "is made for the target it is given"
        ~examples:[ target_of_string "PCI:1+AMD:HIP:gfx942" ]
        gen_placed_target made_for;
      cases "raises on a name that is no device, naming it"
        ~id:(Printf.sprintf "%S") refused (fun device ->
          raises_match
            (Exn.invalid_arg ~sub:(Printf.sprintf "'%s'" device))
            (fun () -> Device.renderer (on device)));
      test
        "raises Failure as the renderer does when its compiler's library is \
         missing, keeping no renderer for the target"
        missing_library;
      prop "is made once for each target, whatever the order of the calls"
        ~examples:
          (List.map
             (List.map target_of_string)
             [
               [ "CPU::arm64,apple-m1"; "CPU::arm64,apple-m1" ];
               [ "CPU::arm64,apple-m1"; "CPU::x86_64,x86-64" ];
               [ "CPU:CLANG:arm64,apple-m1"; "CPU::arm64,apple-m1" ];
               [ "CUDA::sm_89"; "NV::sm_89" ];
             ])
        gen_calls one_renderer_per_target;
      stateful ~domains:2 ~steps:4 ~count:20
        "is made once for a target that two domains ask for concurrently"
        [
          command "fresh" (Gen.unit @-> makes fresh_target) (fun () -> ()) fresh;
          command "renderer"
            (fresh_target ^-> judges asked)
            kept_by_the_table ask;
        ];
    ]

let () = exit (run "Tolk.Device" [ iter_sig; layout; renderer ])
