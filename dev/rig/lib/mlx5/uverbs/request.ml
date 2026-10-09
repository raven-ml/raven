(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external address : params -> int = "caml_rig_mlx5_uverbs_address"
external get16 : params -> int -> int = "%caml_bigstring_get16"
external get32 : params -> int -> int32 = "%caml_bigstring_get32"
external get64 : params -> int -> int64 = "%caml_bigstring_get64"
external set16 : params -> int -> int -> unit = "%caml_bigstring_set16"
external set32 : params -> int -> int32 -> unit = "%caml_bigstring_set32"
external set64 : params -> int -> int64 -> unit = "%caml_bigstring_set64"

let strf = Printf.sprintf

let params n =
  let p = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill p '\000';
  p

let of_string s =
  let p = params (String.length s) in
  String.iteri (fun i c -> Bigarray.Array1.unsafe_set p i c) s;
  p

let to_string p = String.init (Bigarray.Array1.dim p) (Bigarray.Array1.get p)

(* The kernel's fields are the host's order: little-endian on every host this
   library runs on. *)
let field p (at, n) =
  match n with
  | 1 -> Char.code (Bigarray.Array1.get p at)
  | 2 -> get16 p at
  | 4 -> Int32.to_int (get32 p at) land 0xffff_ffff
  | 8 -> Int64.to_int (get64 p at)
  | _ -> invalid_arg (strf "Request.field: %d bytes" n)

let set p (at, n) v =
  match n with
  | 1 -> Bigarray.Array1.set p at (Char.unsafe_chr (v land 0xff))
  | 2 -> set16 p at (v land 0xffff)
  | 4 -> set32 p at (Int32.of_int v)
  | 8 -> set64 p at (Int64.of_int v)
  | _ -> invalid_arg (strf "Request.set: %d bytes" n)

type attr =
  | Value of int * int
  | Word of int * int
  | In of int * params
  | Handle of int * int
  | Made of int
  | Out of int * params

type t = { bytes : params; attrs : attr list }

(* Linux's _IOWR(type, nr, size) (asm-generic/ioctl.h): read and write in bits
   31:30, the size in 29:16, the type in 15:8, the number in 7:0. The verbs
   ioctl's size is its header's. *)
let number =
  (3 lsl 30)
  lor (D.Ioctl_hdr.sizeof lsl 16)
  lor (D.rdma_ioctl_magic lsl 8) lor 1

(* The kernel reads at most a page of header and attributes. *)
let max_header = 4096
let max_attr = 0xffff

let id = function
  | Value (i, _) | Word (i, _) | In (i, _) | Handle (i, _) | Made i | Out (i, _)
    ->
      i

let call ~driver ~obj ~meth attrs =
  let n = List.length attrs in
  let size = D.Ioctl_hdr.sizeof + (n * D.Attr.sizeof) in
  if size > max_header then
    invalid_arg (strf "Request.call: %d attributes, past a page" n);
  let b = params size in
  set b D.Ioctl_hdr.length size;
  set b D.Ioctl_hdr.object_id obj;
  set b D.Ioctl_hdr.method_id meth;
  set b D.Ioctl_hdr.num_attrs n;
  set b D.Ioctl_hdr.driver_id driver;
  List.iteri
    (fun i a ->
      let at = D.Ioctl_hdr.sizeof + (i * D.Attr.sizeof) in
      let f (o, w) = (at + o, w) in
      set b (f D.Attr.attr_id) (id a);
      set b (f D.Attr.flags) D.uverbs_attr_f_mandatory;
      let len, data =
        match a with
        | Value (_, v) -> (8, v)
        | Word (_, v) -> (4, v land 0xffff_ffff)
        | In (_, p) when Bigarray.Array1.dim p <= 8 ->
            let d = Bigarray.Array1.dim p in
            for k = 0 to d - 1 do
              Bigarray.Array1.set b
                (fst (f D.Attr.data) + k)
                (Bigarray.Array1.get p k)
            done;
            (d, -1)
        | In (_, p) | Out (_, p) -> (Bigarray.Array1.dim p, address p)
        | Handle (_, h) -> (0, h)
        | Made _ -> (0, 0)
      in
      if len > max_attr then
        invalid_arg (strf "Request.call: attribute 0x%x of %d bytes" (id a) len);
      set b (f D.Attr.len) len;
      if data >= 0 then set b (f D.Attr.data) data)
    attrs;
  { bytes = b; attrs }

let bytes r = r.bytes

let made r want =
  let rec find i = function
    | [] -> invalid_arg (strf "Request.made: no attribute 0x%x" want)
    | Made id :: _ when id = want ->
        field r.bytes
          (D.Ioctl_hdr.sizeof + (i * D.Attr.sizeof) + fst D.Attr.data, 8)
    | _ :: rest -> find (i + 1) rest
  in
  find 0 r.attrs

let write ~driver ~cmd req ~out ~uhw ~uhw_out =
  let nonempty p = Bigarray.Array1.dim p > 0 in
  let attrs =
    [ Value (D.uverbs_attr_write_cmd, cmd); In (D.uverbs_attr_core_in, req) ]
    @ (if nonempty out then [ Out (D.uverbs_attr_core_out, out) ] else [])
    @ (if uhw <> "" then [ In (D.uverbs_attr_uhw_in, of_string uhw) ] else [])
    @ if nonempty uhw_out then [ Out (D.uverbs_attr_uhw_out, uhw_out) ] else []
  in
  call ~driver ~obj:D.uverbs_object_device ~meth:D.uverbs_method_invoke_write
    attrs
