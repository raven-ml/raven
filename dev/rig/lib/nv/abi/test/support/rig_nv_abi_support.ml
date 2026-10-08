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
     stack_bytes = %d; params_offset = 0x%x; banks = [%a] }"
    k.code k.code_bytes k.registers k.shared_bytes k.stack_bytes k.params_offset
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

(* Terms *)

let rec eval value : _ Packet.term -> int64 = function
  | Value v -> value v
  | Add (t, n) -> Int64.add (eval value t) n
  | Shift (t, n) -> Int64.shift_right_logical (eval value t) n

let rec pp_term pp_v ppf : _ Packet.term -> unit = function
  | Value v -> pp_v ppf v
  | Add (t, n) -> Format.fprintf ppf "Add (%a, 0x%Lx)" (pp_term pp_v) t n
  | Shift (t, n) -> Format.fprintf ppf "Shift (%a, %d)" (pp_term pp_v) t n

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

type op =
  | Set_dim of Qmd.dim * int
  | Patch_dim of Qmd.dim * int64
  | Set_program of int64
  | Set_bank of int * int64
  | Set_local_memory of int64
  | Release of Packet.scope * int64 * int64
  | Release_stamp of Packet.scope * int64 * int64
  | Chain of int64

let dims = Qmd.[ Grid X; Grid Y; Grid Z; Block X; Block Y; Block Z ]

let dim_name (d : Qmd.dim) =
  let axis : Qmd.axis -> string = function X -> "X" | Y -> "Y" | Z -> "Z" in
  match d with Grid a -> "Grid " ^ axis a | Block a -> "Block " ^ axis a

let scope_name : Packet.scope -> string = function
  | Agent -> "Agent"
  | System -> "System"

let pp_op ppf = function
  | Set_dim (d, n) -> Format.fprintf ppf "set_dim (%s) %d" (dim_name d) n
  | Patch_dim (d, v) -> Format.fprintf ppf "patch_dim (%s) %Ld" (dim_name d) v
  | Set_program a -> Format.fprintf ppf "set_program 0x%Lx" a
  | Set_bank (i, a) -> Format.fprintf ppf "set_bank %d 0x%Lx" i a
  | Set_local_memory n -> Format.fprintf ppf "set_local_memory %Ld" n
  | Release (s, a, v) ->
      Format.fprintf ppf "release %s 0x%Lx 0x%Lx" (scope_name s) a v
  | Release_stamp (s, a, v) ->
      Format.fprintf ppf "release_stamp %s 0x%Lx 0x%Lx" (scope_name s) a v
  | Chain a -> Format.fprintf ppf "chain 0x%Lx" a

let apply op q =
  let full r = Option.value r ~default:q in
  match op with
  | Set_dim (d, n) -> Qmd.set_dim d n q
  | Patch_dim (d, v) -> Qmd.patch_dim d v q
  | Set_program a -> Qmd.set_program a q
  | Set_bank (i, a) -> Qmd.set_bank i a q
  | Set_local_memory n -> Qmd.set_local_memory n q
  | Release (s, a, v) -> full (Qmd.release s a v q)
  | Release_stamp (s, a, v) -> full (Qmd.release_stamp s a v q)
  | Chain a -> Qmd.chain a q

type drawn = { gpu : Gpu.t; kernel : Cubin.kernel; ops : op list }

let descriptor d =
  List.fold_left
    (fun q op -> apply op q)
    (Qmd.make (launch d.gpu d.kernel))
    d.ops

let pp_drawn ppf d =
  Format.fprintf ppf "@[<v>%s, %a@,%a@]"
    (class_name d.gpu.compute_class)
    pp_kernel d.kernel
    (Format.pp_print_list pp_op)
    d.ops

let scope =
  Gen.of_list
    ~pp:(fun ppf s -> Format.pp_print_string ppf (scope_name s))
    [ Packet.Agent; System ]

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

let op (k : Cubin.kernel) (banks : Cubin.bank list) =
  let open Gen in
  let size d = int_range 0 (Qmd.max_size d) in
  let local_bytes = k.stack_bytes + 576 in
  one_of
    [
      (let* d = dim in
       map (fun n -> Set_dim (d, n)) (size d));
      (let* d = dim in
       map (fun n -> Patch_dim (d, Int64.of_int n)) (size d));
      map (fun a -> Set_program a) (address ~bits:49 ~align:8);
      (let* i = of_list (List.map (fun (b : Cubin.bank) -> b.index) banks) in
       map (fun a -> Set_bank (i, a)) (address ~bits:49 ~align:6));
      map
        (fun n -> Set_local_memory (Int64.of_int (n * 16)))
        (int_range ((local_bytes + 15) / 16) ((1 lsl 16) - 1));
      (let+ s = scope and+ a = address ~bits:40 ~align:3 and+ v = u64 in
       Release (s, a, v));
      (let+ s = scope and+ a = address ~bits:40 ~align:4 and+ v = u64 in
       Release_stamp (s, a, v));
      map (fun a -> Chain a) (address ~bits:40 ~align:8);
    ]

let drawn =
  let open Gen in
  let gen =
    let* compute_class, kernel = pair compute_class kernels in
    let gpu = gpu ~compute_class () in
    let banks = Launch.banks (launch gpu kernel) in
    let+ ops = list ~size:(int_range 0 12) (op kernel banks) in
    { gpu; kernel; ops }
  in
  with_pp pp_drawn gen
