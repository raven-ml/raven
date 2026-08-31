(* Typed access to the generated NVIDIA driver tables. *)

module Defs = Nv_defs
module Versions = Nv_defs_versions
module Reg_defs = Nv_reg_defs
module Gsp_defs = Nv_gsp_defs

let defs_for_driver ~major : Nv_defs_versions.t =
  if major >= 610 then Nv_defs_versions.v610
  else if major >= 580 then Nv_defs_versions.v580
  else Nv_defs_versions.v570

type blob =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let create_blob size =
  let b = Bigarray.Array1.create Bigarray.Char Bigarray.c_layout size in
  Bigarray.Array1.fill b '\000';
  b

let field_pos (b : blob) ~base (off, size) =
  (match size with
  | 1 | 2 | 4 | 8 -> ()
  | _ -> invalid_arg (Printf.sprintf "blob field: unsupported size %d" size));
  let pos = base + off in
  if pos < 0 || pos + size > Bigarray.Array1.dim b then
    invalid_arg
      (Printf.sprintf "blob field: bytes [%d, %d) outside blob of %d bytes" pos
         (pos + size) (Bigarray.Array1.dim b));
  pos

let get_field ?(base = 0) (b : blob) field =
  let _, size = field in
  let pos = field_pos b ~base field in
  let v = ref 0L in
  for i = size - 1 downto 0 do
    v :=
      Int64.logor (Int64.shift_left !v 8)
        (Int64.of_int (Char.code (Bigarray.Array1.unsafe_get b (pos + i))))
  done;
  match Int64.unsigned_to_int !v with
  | Some v -> v
  | None ->
      invalid_arg (Printf.sprintf "blob field: value 0x%Lx exceeds max_int" !v)

let set_field ?(base = 0) (b : blob) field v =
  let _, size = field in
  let pos = field_pos b ~base field in
  let v = ref (Int64.of_int v) in
  for i = 0 to size - 1 do
    Bigarray.Array1.unsafe_set b (pos + i)
      (Char.unsafe_chr (Int64.to_int (Int64.logand !v 0xffL)));
    v := Int64.shift_right_logical !v 8
  done

let escape_code ~nr ~size =
  (3 lsl 30) lor ((size land 0x1fff) lsl 16) lor (0x46 lsl 8) lor (nr land 0xff)

external blob_addr : blob -> nativeint = "caml_tolk_nv_blob_addr"
external ioctl : fd:int -> request:int -> blob -> int = "caml_tolk_nv_ioctl"
