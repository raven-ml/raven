open Windtrap
open Tolk
module Elf = Device.Tiny_elf

let refuses ?substring f = raises_match (Exn.invalid_arg ?sub:substring) f

(* Targets *)

let target_of_string s =
  require_ok ~pp:Format.pp_print_string (Helpers.Target.of_string s)

(* The target of [device], with no other field: [device] is taken as it is. *)
let on device =
  {
    Helpers.Target.device;
    renderer = "";
    arch = "";
    interface = "";
    indices = "";
  }

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

(* A GPU's renderer is made with its compiler, which finds its vendor's library
   or raises Failure: [made f] is [f ()], or skips the test on a machine without
   the library. *)
let made f =
  match f () with v -> v | exception Failure why -> skip ~reason:why ()

(* Renderer selection *)

(* A renderer that tinygrad's fails to make fails with its own message, which
   tinygrad writes as Python's exception text and tolk as the renderer's
   [Invalid_argument]. *)
let message_of_renderer name target =
  match renderer_named name target with
  | exception Invalid_argument m -> m
  | _ ->
      failf "%s makes a renderer for %s" name
        (Format.asprintf "%a" Helpers.Target.pp target)

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

(* The golden's target is the one tinygrad's DEV setting gives the device. *)
let selected_like_tinygrad cell =
  let target = target_of_string (cell "target") in
  let expected =
    match cell "outcome" with
    | "ok" -> Ok (made (fun () -> renderer_named (cell "renderer") target))
    | "no renderer" -> Error (without_hint (cell "error"))
    | "renderer fails" -> Error (message_of_renderer (cell "renderer") target)
    | outcome -> failf "no outcome %s" outcome
  in
  equal renderer_or_error expected (Device.renderer target)

let chosen s =
  require_ok ~pp:Format.pp_print_string
    (made (fun () -> Device.renderer (target_of_string s)))

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
      test "renders for the target it is given" (fun () ->
          equal string "PCI:1+AMD:HIP:gfx942"
            (Format.asprintf "%a" Helpers.Target.pp
               (chosen "PCI:1+AMD:HIP:gfx942").target));
      test "refuses a renderer the device lacks" (fun () ->
          equal renderer_or_error (Error "AMD has no renderer 'HIPP'")
            (Device.renderer (target_of_string "AMD:HIPP:gfx1100")));
      test "fails with the renderer's own message on an architecture it refuses"
        (fun () ->
          let target = target_of_string "CPU::sparc,v9" in
          equal renderer_or_error
            (Error (message_of_renderer "CLANG" target))
            (Device.renderer target));
      test "raises on a name that is no device" (fun () ->
          List.iter
            (fun device ->
              refuses ~substring:device (fun () -> Device.renderer (on device)))
            [ "QCOM"; "CL"; "PYTHON"; "NULL"; "" ]);
    ]

(* Devices are named exactly as the caller gives them, with no index, no case
   folding and no architecture read from the name. *)
let named_never_parsed =
  group "renderer takes a device's name as it is"
    [
      test "refuses a device name with an index" (fun () ->
          List.iter
            (fun device ->
              refuses ~substring:device (fun () -> Device.renderer (on device)))
            [ "CPU:0"; "CPU:1"; "AMD:0"; "NV:1" ]);
      test "refuses a device name in lower case" (fun () ->
          List.iter
            (fun device ->
              refuses ~substring:device (fun () -> Device.renderer (on device)))
            [ "cpu"; "metal"; "cuda"; "nv"; "amd"; "Cuda" ]);
      test "refuses a disk, which renders nothing" (fun () ->
          List.iter
            (fun device ->
              refuses ~substring:device (fun () -> Device.renderer (on device)))
            [ "DISK"; "DISK:/tmp/weights" ]);
      test "reads no renderer or architecture from the name" (fun () ->
          refuses (fun () -> Device.renderer (on "CUDA:CUDA:sm_89"));
          refuses (fun () -> Device.renderer (on "CPU::arm64,apple-m1")));
    ]

(* Memoisation *)

let pp_target = Helpers.Target.pp

(* Targets drawn so that they often coincide and often differ by one field. *)
let gen_target =
  let open Gen in
  let+ device = of_list ~pp:Format.pp_print_string every_device
  and+ renderer =
    of_list ~pp:Format.pp_print_string [ ""; "CLANG"; "METAL"; "CUDA"; "HIP" ]
  and+ arch =
    of_list ~pp:Format.pp_print_string
      [
        "arm64,apple-m1"; "x86_64,x86-64"; "Apple9"; "sm_89"; "sm_75"; "gfx1100";
      ]
  in
  { (on device) with renderer; arch }

let gen_targets =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_target)
    (Gen.list ~size:(Gen.int_range 2 12) gen_target)

(* Each target, and its renderer when it has one and the machine makes it. *)
let answer t =
  match Device.renderer t with
  | r -> (t, Result.to_option r)
  | exception Failure _ -> (t, None)

let one_renderer_per_target targets =
  let answers = List.map answer targets in
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
          is_true (chosen "CPU::arm64,apple-m1" == chosen "CPU::arm64,apple-m1"));
      test "returns one renderer to two equal targets" (fun () ->
          is_true
            (chosen "CPU::x86_64,x86-64"
            == require_ok ~pp:Format.pp_print_string
                 (Device.renderer { (on "CPU") with arch = "x86_64,x86-64" })));
      test "makes a renderer for each architecture" (fun () ->
          let arm = chosen "CPU::arm64,apple-m1"
          and x86 = chosen "CPU::x86_64,x86-64" in
          is_true (arm != x86);
          equal (pair string string)
            ("arm64,apple-m1", "x86_64,x86-64")
            (arm.target.arch, x86.target.arch));
      test "makes a renderer for a target that names its renderer" (fun () ->
          is_true
            (chosen "CPU:CLANG:arm64,apple-m1" != chosen "CPU::arm64,apple-m1"));
      test "makes renderers of their own for CUDA and NV" (fun () ->
          let cuda = chosen "CUDA::sm_89" and nv = chosen "NV::sm_89" in
          is_true (cuda != nv);
          equal (pair string string) ("CUDA", "NV")
            (cuda.target.device, nv.target.device));
      test "returns one renderer to two domains asking at once" (fun () ->
          (* No other test asks for this target, so the two domains race to make
             its renderer. *)
          let ask () = List.init 50 (fun _ -> chosen "CPU::arm64,cortex-a72") in
          let other = Domain.spawn ask in
          let mine = ask () in
          match mine @ Domain.join other with
          | first :: rest -> List.iter (fun r -> is_true (r == first)) rest
          | [] -> fail "no renderer");
      prop "returns one renderer per target, whatever the order of the calls"
        gen_targets one_renderer_per_target;
    ]

(* Compiled programs *)

let program_of case = Golden.sink (case ^ "_program.golden")

(* The signature a runtime packs: the program's parameters, buffers and
   scalars, in slot order. *)
let signature_of prg : Elf.param list =
  List.map
    (fun u ->
      match Ops.arg u with
      | Param p ->
          ({
             name = p.name;
             addrspace = p.addrspace;
             dtype = Ops.dtype u;
             shape = Shape.max_shape u;
           }
            : Elf.param)
      | _ -> failf "not a parameter")
    (Shape.kernel_params prg)

let int_of_cell = int_of_string

let packed_like_tinygrad cell =
  let signature = signature_of (program_of (cell "program")) in
  let offset = int_of_cell (cell "offset") in
  equal (pair int Dtypes.dtype)
    (int_of_cell (cell "at"), Dtypes.dtype_of_cell (cell "dtype"))
    (List.nth (Elf.iter_sig ~offset signature) (int_of_cell (cell "position")))

(* Hand-made programs *)

(* A compiled program whose parameters are [params], in slot order, and whose
   linear order holds [linear]. *)
let program ~params linear =
  let slot u = match Ops.arg u with Param p -> p.slot | _ -> -1 in
  let globals =
    List.filter_map
      (fun u -> if Ops.addrspace u = Some Alu then None else Some (slot u))
      params
  in
  let info : Ops.program_info =
    {
      global_size = [ Int 1; Int 1; Int 1 ];
      local_size = [ Int 1; Int 1; Int 1 ];
      params;
      globals;
      outs = globals;
      ins = globals;
      target = target_of_string "PCI:1+AMD:HIP:gfx1100";
    }
  in
  Ops.v Program ~arg:(Program info)
    ~src:
      [
        Ops.sink ~kernel:(Ops.kernel_info ()) [];
        Ops.v Linear ~src:linear;
        Ops.v Source ~arg:(String "source");
        Ops.v Binary ~arg:(Bytes "\x7fELF\x00lib");
      ]

let buffer_dtypes = [| Dtype.Float32; Dtype.Int8; Dtype.Float16; Dtype.Int64 |]

(* Buffer [slot] of [slot + 1] elements, named after its slot. *)
let buffer slot =
  Shape.param
    ~shape:[ Int (slot + 1) ]
    ~name:(Printf.sprintf "b%d" slot)
    slot
    buffer_dtypes.(slot mod Array.length buffer_dtypes)

let scalar_dtypes = [| Dtype.Int32; Dtype.Int64; Dtype.Int8; Dtype.Uint16 |]

(* Scalar [slot], named after its slot. *)
let scalar slot =
  let v =
    Ops.variable
      ~dtype:scalar_dtypes.(slot mod Array.length scalar_dtypes)
      (Printf.sprintf "v%d" slot)
      (`Int Bigint.zero)
      (`Int (Bigint.of_int 9))
  in
  match Ops.arg v with
  | Param p -> Ops.replace v ~arg:(Param { p with slot })
  | _ -> v

(* A program's parameters, each a buffer or a scalar of its slot, in its linear
   order, and those its linear order leaves out, as a calling convention leaves
   out what it passes on the stack. *)
type layout = { linear : (int * bool) list; dropped : (int * bool) list }

let pp_slots ppf l =
  Format.(
    pp_print_list
      ~pp_sep:(fun ppf () -> pp_print_string ppf ", ")
      (fun ppf (slot, b) -> fprintf ppf "%s%d" (if b then "b" else "v") slot)
      ppf l)

let pp_layout ppf l =
  Format.fprintf ppf "linear [%a], dropped [%a]" pp_slots l.linear pp_slots
    l.dropped

let gen_layout =
  (let open Gen in
   let* slots = subsequence ~pp:Format.pp_print_int (List.init 12 Fun.id) in
   let* kinds = list ~size:(constant (List.length slots)) bool in
   let params = List.combine slots kinds in
   let* kept = subsequence params in
   let+ linear = permutation kept in
   {
     linear;
     dropped = List.filter (fun p -> not (List.mem p kept)) params;
   })
  |> Gen.with_pp pp_layout

let node (slot, is_buffer) = if is_buffer then buffer slot else scalar slot

let program_of_layout l =
  let params =
    List.sort (fun (a, _) (b, _) -> Int.compare a b) (l.linear @ l.dropped)
  in
  program ~params:(List.map node params) (List.map node l.linear)

let pp_param ppf (p : Elf.param) =
  Format.fprintf ppf "%s %a %a [%s]"
    (Option.value p.name ~default:"-")
    Dtype.pp_addr_space p.addrspace Dtype.pp p.dtype
    (String.concat "; " (List.map string_of_int p.shape))

let param_w =
  Testable.make ~pp:pp_param ~equal:(fun (p0 : Elf.param) p1 ->
      p0.name = p1.name && p0.addrspace = p1.addrspace
      && Dtype.equal p0.dtype p1.dtype
      && p0.shape = p1.shape)

let signature_of_layout l =
  let all = List.sort compare (l.linear @ l.dropped) in
  let rec interleaved = function
    | (_, true) :: (_, false) :: (_, true) :: _ -> true
    | _ :: rest -> interleaved rest
    | [] -> false
  in
  cover "a scalar lies between buffers" (interleaved all);
  cover "the linear order is not slot order"
    (List.map fst l.linear <> List.sort compare (List.map fst l.linear));
  cover "a parameter is left out of the linear order" (l.dropped <> []);
  let expected =
    List.map
      (fun (slot, is_buffer) : Elf.param ->
        if is_buffer then
          {
            name = Some (Printf.sprintf "b%d" slot);
            addrspace = Global;
            dtype = buffer_dtypes.(slot mod Array.length buffer_dtypes);
            shape = [ slot + 1 ];
          }
        else
          {
            name = Some (Printf.sprintf "v%d" slot);
            addrspace = Alu;
            dtype = scalar_dtypes.(slot mod Array.length scalar_dtypes);
            shape = [];
          })
      all
  in
  equal (list param_w) expected (signature_of (program_of_layout l))

let params_in_order =
  group "Shape.kernel_params"
    [
      prop "takes every parameter in slot order, buffers and scalars"
        ~examples:
          [
            { linear = [ (7, true); (2, true); (4, false) ]; dropped = [] };
            { linear = [ (0, true); (3, true) ]; dropped = [ (1, false) ] };
          ]
        gen_layout signature_of_layout;
    ]

(* Packing *)

let gen_offset =
  Gen.frequency
    [
      (6, Gen.int_range 0 40);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 7; 8; 9 ]);
    ]

let param_of_dtype dtype : Elf.param =
  { name = None; addrspace = Alu; dtype; shape = [] }

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
          let signature = signature_of (program_of "scalars") in
          equal
            (list (pair int Dtypes.dtype))
            (Elf.iter_sig ~offset:0 signature)
            (Elf.iter_sig signature));
      test "packs a buffer as its 64-bit address" (fun () ->
          let buffer : Elf.param =
            { name = None; addrspace = Global; dtype = Int8; shape = [ 4 ] }
          in
          equal
            (list (pair int Dtypes.dtype))
            [ (0, Int8); (8, Uint64); (16, Int16) ]
            (Elf.iter_sig
               [ param_of_dtype Int8; buffer; param_of_dtype Int16 ]));
      test "packs an empty signature to nothing" (fun () ->
          equal (list (pair int Dtypes.dtype)) [] (Elf.iter_sig ~offset:5 []));
    ]

let () =
  exit
    (run "Tolk.Device"
       [
         selection;
         named_never_parsed;
         memoisation;
         params_in_order;
         iter_sig;
       ])
