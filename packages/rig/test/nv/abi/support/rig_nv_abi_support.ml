(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Rig_nv_abi

let strf = Printf.sprintf

(* GPUs and kernels *)

let ampere = 0xc7c0
let ada = 0xc9c0
let blackwell = 0xcec0
let classes = [ ampere; ada; blackwell ]

let class_name c =
  if c = ampere then "ampere"
  else if c = ada then "ada"
  else if c = blackwell then "blackwell"
  else strf "0x%x" c

let gpu ?(compute_class = ada) ?(sass_version = 0x89)
    ?(shared_window = 0x7294_0000_0000) ?(local_window = 0x7293_0000_0000) () =
  {
    Gpu.compute_class;
    sass_version;
    gpcs = 11;
    tpcs_per_gpc = 6;
    sms_per_tpc = 2;
    warps_per_sm = 48;
    shared_window;
    local_window;
    local = (fun _ -> Ok ());
  }

let kernel ?(code_bytes = 0x100) ?(registers = 32) ?(shared_bytes = 0)
    ?(stack_bytes = 0x20) ?(params_offset = 0) ?(banks = []) () =
  {
    Cubin.code = 0x80;
    code_bytes;
    registers;
    shared_bytes;
    stack_bytes;
    params_offset;
    params_bytes = 0;
    banks;
  }

let launch g k = match Launch.make g k with Ok l -> l | Error e -> failwith e

let pp_bank ppf (b : Cubin.bank) =
  Format.fprintf ppf "{ index = %d; offset = 0x%x; bytes = %d }" b.index
    b.offset b.bytes

let bank = Testable.make ~pp:pp_bank ~equal:( = )

let compute_class =
  Gen.of_list
    ~pp:(fun ppf c -> Format.pp_print_string ppf (class_name c))
    classes

let pp_kernel ppf (k : Cubin.kernel) =
  Format.fprintf ppf
    "{ code = 0x%x; code_bytes = %d; registers = %d; shared_bytes = %d; \
     stack_bytes = %d; params_offset = 0x%x; params_bytes = %d; banks = [%a] }"
    k.code k.code_bytes k.registers k.shared_bytes k.stack_bytes k.params_offset
    k.params_bytes
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       pp_bank)
    k.banks

(* Words *)

let words s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let u64 =
  Gen.frequency
    [
      (3, Gen.int64);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf -> Format.fprintf ppf "0x%Lx")
          [
            0L;
            1L;
            -1L;
            0xffff_ffffL;
            0x1_0000_0000L;
            0x8000_0000L;
            Int64.min_int;
            Int64.max_int;
          ] );
    ]

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 n;
  Bytes.to_string b

let round_up n a = (n + a - 1) / a * a

(* Templates *)

let pp_operand ppf : Template.operand -> unit = function
  | Int n -> Format.fprintf ppf "Int 0x%x" n
  | Slot i -> Format.fprintf ppf "Slot %d" i

let pp_patch ppf (p : Template.patch) =
  Format.fprintf ppf
    "{ at = %d; bits = %d; slot = %d; add = 0x%Lx; shift = %d }" p.at p.bits
    p.slot p.add p.shift

(* An address below 2^bits, a multiple of 2^align. *)
let address ~bits ~align =
  Gen.map
    (fun n -> Int64.of_int (n lsl align))
    (Gen.int_range 0 ((1 lsl (bits - align)) - 1))

(* Descriptor fields *)

let field b (hi, lo) =
  let n = ref 0 in
  for bit = hi downto lo do
    n := (!n lsl 1) lor ((Char.code b.[bit / 8] lsr (bit mod 8)) land 1)
  done;
  !n

(* Descriptors *)

let dims = Qmd.[ Grid X; Grid Y; Grid Z; Block X; Block Y; Block Z ]

let dim_name (d : Qmd.dim) =
  let axis : Qmd.axis -> string = function X -> "X" | Y -> "Y" | Z -> "Z" in
  match d with Grid a -> "Grid " ^ axis a | Block a -> "Block " ^ axis a

(* The slots of a drawn descriptor's values. *)
let grid_slot = 0
let block_slot = 3
let program_slot = 6
let local_slot = 7
let shared_slot = 8
let chain_slot = 9
let bank_slot i = 10 + i
let release_slot j = 18 + (2 * j)
let slots = 22

type drawn = {
  gpu : Gpu.t;
  kernel : Cubin.kernel;
  values : int array;
  desc : Qmd.t;
}

let descriptor d = Qmd.make (launch d.gpu d.kernel) d.desc
let value d i = Int64.of_int d.values.(i)
let filled d = Template.fill (value d) (descriptor d)

let pp_drawn ppf d =
  let o = pp_operand in
  let triple ppf (x, y, z) = Format.fprintf ppf "(%a, %a, %a)" o x o y o z in
  let option ppf = function
    | None -> Format.fprintf ppf "None"
    | Some v -> Format.fprintf ppf "Some (%a)" o v
  in
  Format.fprintf ppf
    "@[<v>%s, %a@,\
     program %a, local %a, grid %a, block %a, shared %a, chain %a@,\
     banks [%a]@,\
     releases [%a]@,\
     values [%a]@]"
    (class_name d.gpu.compute_class)
    pp_kernel d.kernel o d.desc.program o d.desc.local triple d.desc.grid triple
    d.desc.block option d.desc.shared option d.desc.chain
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       (fun ppf (i, v) -> Format.fprintf ppf "%d, %a" i o v))
    d.desc.banks
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       (fun ppf -> function
         | Qmd.Release (a, v) -> Format.fprintf ppf "Release (%a, %a)" o a o v
         | Stamped (a, v) -> Format.fprintf ppf "Stamped (%a, %a)" o a o v))
    d.desc.releases
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       (fun ppf v -> Format.fprintf ppf "0x%x" v))
    (Array.to_list d.values)

(* Banks 0 to 7, the descriptor's, each of at most 0xffff bytes. *)
let banks =
  let open Gen in
  let* indices = subsequence [ 0; 1; 2; 3; 4; 5; 6; 7 ] in
  let+ sizes =
    list ~size:(constant (List.length indices)) (int_range 0 0xffff)
  in
  List.map2
    (fun index bytes -> { Cubin.index; offset = 0x100 * index; bytes })
    indices sizes

let kernels =
  let open Gen in
  let+ registers = int_range 0 255
  and+ shared_bytes = int_range 0 ((99 * 1024) - 1)
  and+ stack_bytes = int_range 0 0x1000
  and+ banks = banks in
  kernel ~registers ~shared_bytes ~stack_bytes ~banks ()

let dim =
  Gen.of_list ~pp:(fun ppf d -> Format.pp_print_string ppf (dim_name d)) dims

(* Some of the banks [indices], each with a value [gen i] draws. *)
let list_of_banks indices gen =
  let open Gen in
  let* indices = subsequence indices in
  List.fold_right
    (fun i rest ->
      let+ v = gen i and+ rest = rest in
      (i, v) :: rest)
    indices (constant [])

(* A value drawn by [gen] for [slot]: known, or left to the slot. *)
let operand slot gen =
  let open Gen in
  let+ v = gen and+ known = bool in
  (slot, v, known)

let use (slot, v, known) = if known then Template.Int v else Slot slot

let drawn =
  let open Gen in
  let gen =
    let* compute_class, kernel = pair compute_class kernels in
    let gpu = gpu ~compute_class () in
    let l = launch gpu kernel in
    let address ~bits ~align = map Int64.to_int (address ~bits ~align) in
    let size d = int_range 0 (Qmd.max_size d) in
    let three at = function
      | [ x; y; z ] ->
          let+ x = operand at (size x)
          and+ y = operand (at + 1) (size y)
          and+ z = operand (at + 2) (size z) in
          (x, y, z)
      | _ -> assert false
    in
    let local_bytes = Launch.local_bytes l in
    let+ grid = three grid_slot Qmd.[ Grid X; Grid Y; Grid Z ]
    and+ block = three block_slot Qmd.[ Block X; Block Y; Block Z ]
    and+ program = operand program_slot (address ~bits:49 ~align:8)
    and+ local =
      operand local_slot
        (map
           (fun n -> n * 16)
           (int_range ((local_bytes + 15) / 16) ((1 lsl 16) - 1)))
    and+ shared =
      option
        (operand shared_slot
           (map
              (fun n -> n * 128)
              (int_range 0 (Launch.dynamic_shared l / 128))))
    and+ chain = option (operand chain_slot (address ~bits:40 ~align:8))
    and+ releases =
      let release j =
        let+ stamped = bool
        and+ a = operand (release_slot j) (address ~bits:40 ~align:4)
        and+ v = operand (release_slot j + 1) (int_range 0 max_int) in
        (stamped, a, v)
      in
      let* n = int_range 0 2 in
      List.fold_right
        (fun j rest ->
          let+ r = release j and+ rest = rest in
          r :: rest)
        (List.init n Fun.id) (constant [])
    and+ banks =
      list_of_banks
        (List.map (fun (b : Cubin.bank) -> b.index) (Launch.banks l))
        (fun i -> operand (bank_slot i) (address ~bits:49 ~align:6))
    in
    let values = Array.make slots 0 in
    let all =
      [ program; local ]
      @ (let x, y, z = grid in
         [ x; y; z ])
      @ (let x, y, z = block in
         [ x; y; z ])
      @ Option.to_list shared @ Option.to_list chain @ List.map snd banks
      @ List.concat_map (fun (_, a, v) -> [ a; v ]) releases
    in
    List.iter (fun (s, v, _) -> values.(s) <- v) all;
    let three (x, y, z) = (use x, use y, use z) in
    {
      gpu;
      kernel;
      values;
      desc =
        {
          program = use program;
          local = use local;
          banks = List.map (fun (i, v) -> (i, use v)) banks;
          grid = three grid;
          block = three block;
          shared = Option.map use shared;
          chain = Option.map use chain;
          releases =
            List.map
              (fun (stamped, a, v) ->
                if stamped then Qmd.Stamped (use a, use v)
                else Release (use a, use v))
              releases;
        };
    }
  in
  with_pp pp_drawn gen
