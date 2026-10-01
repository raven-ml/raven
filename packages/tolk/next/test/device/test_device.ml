open Windtrap
open Tolk_next
module Elf = Device.Tiny_elf

let refuses ?substring f = raises_match (Exn.invalid_arg ?substring) f

(* Targets *)

let target_of_string s =
  require_ok ~pp:Format.pp_print_string (Helpers.Target.of_string s)

let targets_of_cell s = List.map target_of_string (String.split_on_char ';' s)
let under dev f = Helpers.context [ B (Helpers.dev, targets_of_cell dev) ] f

let renderer_named = function
  | "CLANG" -> Cstyle.clang
  | "METAL" -> Cstyle.metal
  | "CUDA" -> Cstyle.cuda
  | "HIP" -> Cstyle.hip
  | name -> failf "no renderer is named %s" name

let renderer_w =
  Testable.make
    ~pp:(fun ppf (r : Renderer.t) ->
      Format.fprintf ppf "%s for %s" r.name
        (Format.asprintf "%a" Helpers.Target.pp r.target))
    ~equal:(fun (r0 : Renderer.t) r1 ->
      r0.name = r1.name && r0.target = r1.target)

let renderer_or_error = result renderer_w string

(* Renderer selection *)

(* A renderer that tinygrad's fails to make fails with its own message, which
   tinygrad writes as Python's exception text and tolk.next as the renderer's
   [Invalid_argument]. *)
let message_of_renderer name target =
  match renderer_named name target with
  | exception Invalid_argument m -> m
  | _ ->
      failf "%s makes a renderer for %s" name
        (Format.asprintf "%a" Helpers.Target.pp target)

let selected_like_tinygrad cell =
  under (cell "dev") (fun () ->
      let target = target_of_string (cell "target") in
      let expected =
        match cell "outcome" with
        | "ok" -> Ok (renderer_named (cell "renderer") target)
        | "no renderer" -> Error (cell "error")
        | "renderer fails" ->
            Error (message_of_renderer (cell "renderer") target)
        | outcome -> failf "no outcome %s" outcome
      in
      equal renderer_or_error expected
        (Device.renderer ~arch:(cell "arch") (cell "device")))

let chosen ?arch ?(dev = "") device =
  under dev (fun () ->
      require_ok ~pp:Format.pp_print_string (Device.renderer ?arch device))

let every_device = [ "CPU"; "METAL"; "CUDA"; "NV"; "AMD" ]

let selection =
  group "renderer"
    [
      group "picks a device's renderer as tinygrad does"
        [
          Golden.cases
            ~key:[ "dev"; "device"; "arch" ]
            "renderers.golden" selected_like_tinygrad;
        ];
      test "renders for the setting's target of the device" (fun () ->
          equal string "PCI:1+AMD:HIP:gfx942"
            (Format.asprintf "%a" Helpers.Target.pp
               (chosen ~dev:"CPU::x86_64,x86-64;PCI:1+AMD:HIP:gfx942"
                  ~arch:"gfx1100" "AMD")
                 .target));
      test "takes the architecture the caller gives when the setting names none"
        (fun () ->
          equal string "sm_75"
            (chosen ~dev:"CUDA" ~arch:"sm_75" "CUDA").target.arch);
      test "defaults the architecture to none" (fun () ->
          equal string "" (chosen ~dev:"METAL" "METAL").target.arch);
      test "names the renderer a target misspells" (fun () ->
          under "AMD:HIPP" (fun () ->
              equal renderer_or_error
                (Error "AMD has no renderer 'HIPP', did you mean: 'HIP'?")
                (Device.renderer ~arch:"gfx1100" "AMD")));
      test "fails with the renderer's own message on an architecture it refuses"
        (fun () ->
          under "CPU::sparc,v9" (fun () ->
              equal renderer_or_error
                (Error
                   (message_of_renderer "CLANG"
                      (target_of_string "CPU::sparc,v9")))
                (Device.renderer "CPU")));
      test "raises on a name that is no device" (fun () ->
          List.iter
            (fun device ->
              refuses ~substring:device (fun () -> Device.renderer device))
            [ "QCOM"; "CL"; "PYTHON"; "NULL"; "" ]);
    ]

(* Devices are named exactly as the caller gives them, with no index, no case
   folding and no architecture read from the name. *)
let named_never_parsed =
  group "renderer takes a device's name as it is (D6)"
    [
      test "refuses a device name with an index" (fun () ->
          List.iter
            (fun device ->
              refuses ~substring:device (fun () -> Device.renderer device))
            [ "CPU:0"; "CPU:1"; "AMD:0"; "NV:1" ]);
      test "refuses a device name in lower case" (fun () ->
          List.iter
            (fun device ->
              refuses ~substring:device (fun () -> Device.renderer device))
            [ "cpu"; "metal"; "cuda"; "nv"; "amd"; "Cuda" ]);
      test "refuses a disk, which renders nothing" (fun () ->
          List.iter
            (fun device ->
              refuses ~substring:device (fun () -> Device.renderer device))
            [ "DISK"; "DISK:/tmp/weights" ]);
      test "reads no renderer or architecture from the name" (fun () ->
          refuses (fun () -> Device.renderer "CUDA:CUDA:sm_89");
          refuses (fun () -> Device.renderer "CPU::arm64,apple-m1"));
    ]

(* Memoisation *)

let pp_target = Helpers.Target.pp

(* The settings and architectures a query draws from, so that queries often
   share a target and often differ by one field. *)
let gen_query =
  let open Gen in
  let+ dev =
    of_list ~pp:Format.pp_print_string
      [
        "";
        "CPU";
        "CPU:CLANG";
        ":CLANG";
        "AMD";
        "NV::sm_89";
        "CUDA::sm_80;AMD::gfx942";
      ]
  and+ device = of_list ~pp:Format.pp_print_string every_device
  and+ arch =
    of_list ~pp:Format.pp_print_string
      [
        "arm64,apple-m1"; "x86_64,x86-64"; "Apple9"; "sm_89"; "sm_75"; "gfx1100";
      ]
  in
  (dev, device, arch)

let pp_query ppf (dev, device, arch) =
  Format.fprintf ppf "DEV=%S %s ~arch:%S" dev device arch

let gen_queries =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_query)
    (Gen.list ~size:(Gen.int_range 2 12) gen_query)

(* Each query's target, and its renderer when it has one. *)
let answer (dev, device, arch) =
  under dev (fun () ->
      ( Helpers.target ~arch device,
        Result.to_option (Device.renderer ~arch device) ))

let one_renderer_per_target queries =
  let answers = List.map answer queries in
  cover "two queries share a target"
    (List.exists
       (fun (t0, _) ->
         List.length (List.filter (fun (t1, _) -> t0 = t1) answers) > 1)
       answers);
  List.iter
    (fun (t0, r0) ->
      List.iter
        (fun (t1, r1) ->
          match (r0, r1) with
          | Some r0, Some r1 ->
              equal
                ~msg:(Format.asprintf "%a and %a" pp_target t0 pp_target t1)
                bool (t0 = t1) (r0 == r1)
          | _ -> ())
        answers)
    answers

let memoisation =
  group "renderer's memory"
    [
      test "returns the same renderer to a second call" (fun () ->
          is_true
            (chosen ~arch:"arm64,apple-m1" "CPU"
            == chosen ~arch:"arm64,apple-m1" "CPU"));
      test "returns one renderer to two settings that give one target"
        (fun () ->
          is_true
            (chosen ~dev:"" ~arch:"gfx1100" "AMD"
            == chosen ~dev:"CPU:CLANG;AMD" ~arch:"gfx1100" "AMD"));
      test "makes a renderer for each architecture" (fun () ->
          let arm = chosen ~arch:"arm64,apple-m1" "CPU"
          and x86 = chosen ~arch:"x86_64,x86-64" "CPU" in
          is_true (arm != x86);
          equal (pair string string)
            ("arm64,apple-m1", "x86_64,x86-64")
            (arm.target.arch, x86.target.arch));
      test "makes a renderer for a target that names its renderer" (fun () ->
          is_true
            (chosen ~dev:"CPU:CLANG" ~arch:"arm64,apple-m1" "CPU"
            != chosen ~dev:"CPU" ~arch:"arm64,apple-m1" "CPU"));
      test "makes renderers of their own for CUDA and NV" (fun () ->
          let cuda = chosen ~arch:"sm_89" "CUDA"
          and nv = chosen ~arch:"sm_89" "NV" in
          is_true (cuda != nv);
          equal (pair string string) ("CUDA", "NV")
            (cuda.target.device, nv.target.device));
      test "returns one renderer to two domains asking at once" (fun () ->
          under "" (fun () ->
              (* No other test asks for this target, so the two domains race to
                 make its renderer. *)
              let ask () =
                List.init 50 (fun _ ->
                    require_ok ~pp:Format.pp_print_string
                      (Device.renderer ~arch:"sm_87" "NV"))
              in
              let other = Domain.spawn ask in
              let mine = ask () in
              match mine @ Domain.join other with
              | first :: rest -> List.iter (fun r -> is_true (r == first)) rest
              | [] -> fail "no renderer"));
      prop "returns one renderer per target, whatever the order of the calls"
        gen_queries one_renderer_per_target;
    ]

(* Compiled programs *)

let program_of case = Golden.sink (case ^ "_program.golden")
let elf_of case = Elf.of_program (program_of case)

let binary_of prg =
  match List.rev (Ops.src prg) with
  | binary :: _ -> (
      match Ops.arg binary with
      | Bytes lib -> lib
      | _ -> failf "the program's last source is no binary")
  | [] -> failf "the program has no source"

let int_of_cell = int_of_string

(* tinygrad writes a name as [None] or as a quoted string. *)
let name_of_cell = function
  | "None" -> None
  | cell -> Some (String.sub cell 1 (String.length cell - 2))

(* tinygrad writes a shape as a tuple: [()] or [(16,)]. *)
let shape_of_cell cell =
  String.sub cell 1 (String.length cell - 2)
  |> String.split_on_char ','
  |> List.filter_map (fun s ->
      match String.trim s with "" -> None | n -> Some (int_of_string n))

let compiled_like_tinygrad cell =
  let elf = elf_of (cell "program") in
  equal string (cell "name") elf.name;
  equal string (cell "target")
    (Format.asprintf "%a" Helpers.Target.pp elf.target);
  equal int (int_of_cell (cell "params")) (List.length elf.signature);
  equal int (int_of_cell (cell "lib_bytes")) (String.length elf.lib)

let param_like_tinygrad cell =
  let elf = elf_of (cell "program") in
  let p = List.nth elf.signature (int_of_cell (cell "position")) in
  equal (option string) (name_of_cell (cell "name")) p.name;
  equal int (int_of_cell (cell "slot")) p.slot;
  equal Dtypes.dtype (Dtypes.dtype_of_cell (cell "dtype")) p.dtype;
  equal (list int) (shape_of_cell (cell "shape")) p.shape

let param_printed_like_tinygrad cell =
  let elf = elf_of (cell "program") in
  let p = List.nth elf.signature (int_of_cell (cell "position")) in
  equal string (cell "repr") (Format.asprintf "%a" Elf.pp_param p)

let packed_like_tinygrad cell =
  let signature = (elf_of (cell "program")).signature in
  let offset = int_of_cell (cell "offset") in
  equal (pair int Dtypes.dtype)
    (int_of_cell (cell "at"), Dtypes.dtype_of_cell (cell "dtype"))
    (List.nth (Elf.iter_sig ~offset signature) (int_of_cell (cell "position")))

let cases = [ "elementwise"; "elementwise_metal"; "sparse"; "scalars"; "fill" ]

(* Hand-made programs *)

let hip = target_of_string "PCI:1+AMD:HIP:gfx1100"

let program ?(name = "k") ~globals ~vars linear =
  let info : Ops.program_info =
    {
      global_size = [ Int 1; Int 1; Int 1 ];
      local_size = [ Int 1; Int 1; Int 1 ];
      vars;
      globals;
      outs = globals;
      ins = globals;
      target = hip;
    }
  in
  Ops.v Program ~arg:(Program info)
    ~src:
      [
        Ops.sink ~kernel:(Ops.kernel_info ~name ()) [];
        Ops.v Linear ~src:linear;
        Ops.v Source ~arg:(String "source");
        Ops.v Binary ~arg:(Bytes "\x7fELF\x00lib");
      ]

let buffer_dtypes = [| Dtype.Float32; Dtype.Int8; Dtype.Float16; Dtype.Int64 |]

(* Buffer [slot] of [slot + 1] elements, named after its slot. *)
let buffer slot =
  Ops.param
    ~shape:[ Int (slot + 1) ]
    ~name:(Printf.sprintf "b%d" slot)
    slot
    buffer_dtypes.(slot mod Array.length buffer_dtypes)

let scalar (name, dtype) =
  Ops.variable ~dtype name (`Int Bigint.zero) (`Int (Bigint.of_int 9))

let scalars =
  [
    ("i", Dtype.Int32);
    ("w", Dtype.Int64);
    ("c", Dtype.Int8);
    ("u", Dtype.Uint16);
  ]

(* A program's buffers, by slot, in its linear order and in its globals, and its
   variables. *)
type layout = {
  linear : int list;
  globals : int list;
  vars : (string * Dtype.t) list;
}

let pp_ints =
  Format.(
    pp_print_list ~pp_sep:(fun ppf () -> pp_print_string ppf ", ") pp_print_int)

let pp_layout ppf l =
  Format.fprintf ppf "linear [%a], globals [%a], vars [%s]" pp_ints l.linear
    pp_ints l.globals
    (String.concat ", " (List.map fst l.vars))

let gen_layout =
  (let open Gen in
   let* slots = subsequence ~pp:Format.pp_print_int (List.init 12 Fun.id) in
   let* linear = permutation ~pp:Format.pp_print_int slots in
   let* globals = permutation ~pp:Format.pp_print_int slots in
   let* chosen = subsequence scalars in
   let+ vars = permutation chosen in
   { linear; globals; vars })
  |> Gen.with_pp pp_layout

let rec position x = function
  | [] -> invalid_arg "position"
  | y :: ys -> if x = y then 0 else 1 + position x ys

let program_of_layout l =
  let vars = List.map scalar l.vars in
  (* The linear order holds the variables too, ahead of the buffers. *)
  program ~globals:l.globals ~vars (vars @ List.map buffer l.linear)

let param_w =
  Testable.make ~pp:Elf.pp_param ~equal:(fun (p0 : Elf.param) p1 ->
      p0.name = p1.name && p0.slot = p1.slot
      && Dtype.equal p0.dtype p1.dtype
      && p0.shape = p1.shape)

let signature_of_layout l =
  cover "a buffer is not in its slot's place" (l.linear <> l.globals);
  cover "a variable follows the buffers" (l.vars <> [] && l.globals <> []);
  let buffers =
    List.map
      (fun slot : Elf.param ->
        {
          name = Some (Printf.sprintf "b%d" slot);
          slot = position slot l.globals;
          dtype = buffer_dtypes.(slot mod Array.length buffer_dtypes);
          shape = [ slot + 1 ];
        })
      l.linear
  and vars =
    List.mapi
      (fun j (name, dtype) : Elf.param ->
        {
          name = Some name;
          slot = List.length l.globals + j;
          dtype;
          shape = [];
        })
      l.vars
  in
  equal (list param_w) (buffers @ vars)
    (Elf.of_program (program_of_layout l)).signature

let refuses_incomplete () =
  let sink = Ops.sink ~kernel:(Ops.kernel_info ()) [] in
  let info = Ops.program_info_of_sink ~target:hip sink in
  refuses (fun () -> Elf.of_program sink);
  refuses (fun () ->
      Elf.of_program (Ops.v Program ~arg:(Program info) ~src:[ sink ]))

(* [prg] with its source [i] replaced by [u]. *)
let with_source i u prg =
  Ops.replace
    ~src:(List.mapi (fun j s -> if i = j then u else s) (Ops.src prg))
    prg

let refuses_nameless () =
  let prg = program ~globals:[] ~vars:[] [] in
  refuses (fun () -> Elf.of_program (with_source 0 (Ops.sink []) prg))

let refuses_textual_binary () =
  let prg = program ~globals:[] ~vars:[] [] in
  refuses (fun () ->
      Elf.of_program (with_source 3 (Ops.v Binary ~arg:(String "lib")) prg))

let of_program =
  group "Tiny_elf.of_program"
    [
      group "compiles as tinygrad's to_elf does"
        [ Golden.cases "elfs.golden" compiled_like_tinygrad ];
      group "lays out the signature as tinygrad's to_elf does"
        [
          Golden.cases ~key:[ "program"; "position" ] "signatures.golden"
            param_like_tinygrad;
        ];
      test "takes the program's binary as its lib" (fun () ->
          List.iter
            (fun case ->
              let prg = program_of case in
              equal ~msg:case string (binary_of prg) (Elf.of_program prg).lib)
            cases);
      test "keys the program's profile with its key" (fun () ->
          List.iter
            (fun case ->
              let prg = program_of case in
              equal ~msg:case (option string)
                (Some (Ops.key prg))
                (Elf.of_program prg).profile_key)
            cases);
      test "names the program after its kernel, as an identifier" (fun () ->
          equal string "r3Ascalars202" (elf_of "scalars").name;
          equal string "a_b2E"
            (Elf.of_program (program ~name:"a_b." ~globals:[] ~vars:[] [])).name);
      test "takes the target of the program, whatever rendered it" (fun () ->
          equal string "PCI:1+AMD:HIP:gfx1100"
            (Format.asprintf "%a" Helpers.Target.pp
               (Elf.of_program (program ~globals:[ 0 ] ~vars:[] [ buffer 0 ]))
                 .target));
      test "compiles a program of no parameter to an empty signature" (fun () ->
          equal (list param_w) []
            (Elf.of_program (program ~globals:[] ~vars:[] [])).signature);
      prop "numbers buffers by their place among the globals, then variables"
        ~examples:
          [
            {
              linear = [ 7; 2 ];
              globals = [ 2; 7 ];
              vars = [ ("i", Dtype.Int32) ];
            };
            { linear = [ 3; 11; 0 ]; globals = [ 11; 0; 3 ]; vars = [] };
          ]
        gen_layout signature_of_layout;
      test "refuses a node that is no program, and a program not compiled"
        refuses_incomplete;
      test "refuses a program whose kernel has no name" refuses_nameless;
      test "refuses a program whose binary holds no bytes"
        refuses_textual_binary;
      test "refuses a program whose variable is no parameter" (fun () ->
          refuses (fun () ->
              Elf.of_program (program ~globals:[] ~vars:[ Ops.int 3 ] [])));
      test "refuses a buffer of the linear order that is not among the globals"
        (fun () ->
          refuses (fun () ->
              Elf.of_program
                (program ~globals:[ 2 ] ~vars:[] [ buffer 2; buffer 7 ])));
    ]

(* Printing *)

let elfs_to_print =
  let sixteen_floats : Elf.param =
    { name = None; slot = 0; dtype = Dtype.Float32; shape = [ 16 ] }
  and n : Elf.param =
    { name = Some "n"; slot = 1; dtype = Dtype.Int32; shape = [] }
  in
  let no_target = target_of_string "" in
  [
    ( "some",
      Elf.
        {
          lib = "\x7fELF\x00'\"";
          name = "k";
          target = target_of_string "CPU:CLANG:x86_64,x86-64";
          signature = [ sixteen_floats; n ];
          profile_key = Some "key\n";
        } );
    ( "none",
      Elf.
        {
          lib = "";
          name = "E_4";
          target = hip;
          signature = [ sixteen_floats ];
          profile_key = None;
        } );
    ( "empty",
      Elf.
        {
          lib = "lib";
          name = "k";
          target = no_target;
          signature = [];
          profile_key = None;
        } );
  ]

let printed_like_tinygrad cell =
  equal string (cell "repr")
    (Format.asprintf "%a" Elf.pp (List.assoc (cell "case") elfs_to_print))

let printing =
  group "Tiny_elf printers"
    [
      group "pp_param prints a parameter as tinygrad does"
        [
          Golden.cases ~key:[ "program"; "position" ] "signatures.golden"
            param_printed_like_tinygrad;
        ];
      group "pp prints a program as tinygrad does"
        [ Golden.cases "reprs.golden" printed_like_tinygrad ];
    ]

(* Packing *)

let gen_offset =
  Gen.frequency
    [
      (6, Gen.int_range 0 40);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 7; 8; 9 ]);
    ]

let param_of_dtype dtype : Elf.param =
  { name = None; slot = 0; dtype; shape = [] }

let gen_signature =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space Dtype.pp)
    (Gen.list ~size:(Gen.int_range 0 8) Dtypes.stored)

(* Each value lies at the first offset aligned to its size past the previous
   one, in the signature's order. *)
let packed (offset, dtypes) =
  let layout = Elf.iter_sig ~offset (List.map param_of_dtype dtypes) in
  equal (list Dtypes.dtype) dtypes (List.map snd layout);
  ignore
    (List.fold_left
       (fun past (at, dtype) ->
         let size = Dtype.itemsize dtype in
         cover "a value is padded" (at > past);
         equal ~msg:(Printf.sprintf "at %d" at) int 0 (at mod size);
         at_least int ~than:past at;
         less int ~than:(past + size) at;
         at + size)
       offset layout)

let iter_sig =
  group "Tiny_elf.iter_sig"
    [
      group "packs a signature as tinygrad's iter_sig does"
        [
          Golden.cases
            ~key:[ "program"; "offset"; "position" ]
            "layouts.golden" packed_like_tinygrad;
        ];
      prop "packs each value at the next offset aligned to its size"
        (Gen.pair gen_offset gen_signature)
        packed;
      test "starts at byte 0 by default" (fun () ->
          let signature = (elf_of "scalars").signature in
          equal
            (list (pair int Dtypes.dtype))
            (Elf.iter_sig ~offset:0 signature)
            (Elf.iter_sig signature));
      test "packs an empty signature to nothing" (fun () ->
          equal (list (pair int Dtypes.dtype)) [] (Elf.iter_sig ~offset:5 []));
    ]

let () =
  exit
    (run "Tolk_next.Device"
       [
         selection;
         named_never_parsed;
         memoisation;
         of_program;
         printing;
         iter_sig;
       ])
