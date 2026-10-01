(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next

type column = { leaf : Leaf.t; ty : Type.any }
type format = column array

(* [with_bytes b f] is [f] applied to the bytes of [b], mapped on the host, with
   the errors of the private modules as [Error]. *)
let with_bytes b f =
  match Nx_device.Buffer.borrow Nx_device.host b with
  | Error why -> Error (Error.v why)
  | Ok host ->
      let bytes = Nx_device.Buffer.bigarray Bigarray.int8_unsigned host in
      let r =
        try Ok (f bytes)
        with Meta.Error { row_group; bytes; text; msg } ->
          Error (Error.v ?row_group ?bytes ?text msg)
      in
      (* The bigarray of a borrowed file does not keep its mapping alive. *)
      ignore (Sys.opaque_identity host);
      r

let open_ bytes =
  let m = Meta.footer bytes in
  let leaves = Leaf.of_schema m.schema in
  Array.iteri
    (fun g (rg : Meta.row_group) ->
      if Array.length rg.chunks <> Array.length leaves then
        Meta.fail ~row_group:g
          "the row group has %d column chunks for %d columns"
          (Array.length rg.chunks) (Array.length leaves);
      Array.iteri (fun i l -> Chunk.check m ~row_group:g i l) leaves)
    m.row_groups;
  (m, leaves)

let sniff b =
  with_bytes b @@ fun bytes ->
  Array.map (fun leaf -> { leaf; ty = Leaf.default leaf }) (snd (open_ bytes))

let index fn f name =
  let rec loop i =
    if i = Array.length f then
      invalid_arg
        (Printf.sprintf "Talon_next_parquet.%s: no column %S in the format." fn
           name)
    else if String.equal f.(i).leaf.name name then i
    else loop (i + 1)
  in
  loop 0

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t

let with_type name t f =
  let i = index "with_type" f name in
  let leaf = f.(i).leaf in
  if not (Leaf.reads_as leaf t) then
    invalid_arg
      (Format.asprintf
         "Talon_next_parquet.with_type: column %S (%a) reads as %a, not %s. \
          Cast it once read (Expr.cast)."
         name Leaf.pp leaf Leaf.pp_reads leaf (type_name t));
  let f = Array.copy f in
  f.(i) <- { leaf; ty = t };
  f

(* [scalars s] is the number of Unicode scalar values of the UTF-8 [s]. *)
let scalars s =
  let n = ref 0 in
  String.iter (fun c -> if Char.code c land 0xC0 <> 0x80 then incr n) s;
  !n

let pp_format ppf f =
  let left =
    Array.map
      (fun c ->
        Format.asprintf "%a" Schema.pp (Schema.v [ (c.leaf.name, c.ty) ]))
      f
  in
  let width = Array.fold_left (fun w s -> max w (scalars s)) 0 left in
  let n = Array.length f in
  Format.fprintf ppf "@[<v>parquet (%d column%s)" n (if n = 1 then "" else "s");
  Array.iteri
    (fun i c ->
      Format.fprintf ppf "@,  %s%s ← %a" left.(i)
        (String.make (width - scalars left.(i)) ' ')
        Leaf.pp c.leaf)
    f;
  Format.fprintf ppf "@]"

module Private = struct
  type column = Chunk.t =
    | Fixed of { valid : Nx.bool_t option; values : Nx.packed }
    | Varsize of {
        valid : Nx.bool_t option;
        offsets : Nx.int64_t;
        data : Nx.uint8_t;
      }

  let read_column f b ~row_group name =
    let c = f.(index "Private.read_column" f name) in
    with_bytes b @@ fun bytes ->
    let m, leaves = open_ bytes in
    if row_group < 0 || row_group >= Array.length m.row_groups then
      Meta.fail "the file has no row group %d" row_group;
    let rec find i =
      if i = Array.length leaves then Meta.fail "the file has no column %S" name
      else if String.equal leaves.(i).Leaf.name name then i
      else find (i + 1)
    in
    let i = find 0 in
    if not (Leaf.reads_as leaves.(i) c.ty) then
      Meta.fail "column %S reads as %s, not as the format's %s" name
        (Format.asprintf "%a" Leaf.pp_reads leaves.(i))
        (type_name c.ty);
    Chunk.read bytes m ~row_group i leaves.(i) c.ty
end
