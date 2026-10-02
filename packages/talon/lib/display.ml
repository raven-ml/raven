(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type limits = { head : int; tail : int; columns : int; width : int }

let limits = { head = 5; tail = 5; columns = 12; width = 32 }
let err fmt = Format.kasprintf invalid_arg fmt

(* Cells *)

(* A cell's text and its width in scalar values. *)
type cell = { text : string; width : int }

let null = { text = "∅"; width = 1 }

(* [to_string pp] is the text [pp] writes, on one line. *)
let to_string pp =
  let b = Buffer.create 16 in
  let ppf = Format.formatter_of_buffer b in
  Format.pp_set_geometry ppf ~max_indent:(Format.pp_infinity - 2)
    ~margin:(Format.pp_infinity - 1);
  pp ppf;
  Format.pp_print_flush ppf ();
  Buffer.contents b

(* [fit width s] is the cell of [s], its control bytes and bytes that are not
   UTF-8 escaped as [Type.pp_quoted] escapes them, cut with […] past [width]
   scalar values. An escape is cut whole. *)
let fit width s =
  let rec pieces i acc =
    if i = String.length s then List.rev acc
    else
      let d = String.get_utf_8_uchar s i and c = Char.code s.[i] in
      if (not (Uchar.utf_decode_is_valid d)) || c < 0x20 || c = 0x7f then
        pieces (i + 1) ((Printf.sprintf "\\x%02x" c, 4) :: acc)
      else
        let n = Uchar.utf_decode_length d in
        pieces (i + n) ((String.sub s i n, 1) :: acc)
  in
  let pieces = pieces 0 [] in
  let total = List.fold_left (fun n (_, w) -> n + w) 0 pieces in
  let rec keep room = function
    | (p, w) :: ps when w <= room ->
        let kept, n = keep (room - w) ps in
        (p :: kept, n + w)
    | _ -> ([], 0)
  in
  if total <= width then
    { text = String.concat "" (List.map fst pieces); width = total }
  else
    let kept, n = keep (width - 1) pieces in
    { text = String.concat "" kept ^ "…"; width = n + 1 }

let is_number : type a. a Type.t -> bool =
 fun ty -> match Type.kind ty with Int | Float -> true | _ -> false

(* [cells width c] is the cells of [c]'s rows. Text shows unquoted, and the
   floats of a column share their decimals. *)
let cells width c =
  let n = Column.length c in
  let valid =
    match Column.valid c with
    | None -> Fun.const true
    | Some m -> Array.get (Nx.to_array m)
  in
  let cells text = Array.init n (fun i -> if valid i then text i else null) in
  let (Any ty) = Column.type_ c in
  let (Any storage) = Type.storage ty in
  match (storage, Column.data c) with
  | (String | Binary), Bytes r ->
      let o = Nx.to_array (Nx_ragged.offsets r) in
      let b = Nx.to_array (Nx_ragged.values r) in
      cells (fun i ->
          let pos = Int64.to_int o.(i) in
          let len = Int64.to_int o.(i + 1) - pos in
          fit width (String.init len (fun k -> Char.chr b.(pos + k))))
  | Categorical d, Fixed (P x) ->
      let codes = Nx.to_array (Nx.cast Nx.int32 x) in
      cells (fun i -> fit width (Iarray.get d (Int32.to_int codes.(i))))
  | (Float16 | Float32 | Float64), Fixed (P x) ->
      let xs = Nx.to_array (Nx.cast Nx.float64 x) in
      let shown = List.filter valid (List.init n Fun.id) in
      let pp = Form.pp_floats (Array.of_list (List.map (Array.get xs) shown)) in
      cells (fun i -> fit width (to_string (Format.dprintf "%a" pp xs.(i))))
  | _ -> Array.init n (fun i -> fit width (to_string (Form.pp_cell c i)))

(* Rows *)

(* [part t j ~offset ~length] is column [j]'s rows [offset] to [offset + length
   - 1], as views of the batches that hold them. *)
let part t j ~offset ~length =
  let stop = offset + length in
  let rec loop start = function
    | [] -> []
    | b :: bs ->
        let c = (Table.columns b).(j) in
        let n = Column.length c in
        let lo = Int.max offset start and hi = Int.min stop (start + n) in
        let rest = loop (start + n) bs in
        if lo < hi then
          Column.sub c ~offset:(lo - start) ~length:(hi - lo) :: rest
        else rest
  in
  loop 0 (Table.batches t)

(* Layout *)

let plural n what = Printf.sprintf "%d %s%s" n what (if n = 1 then "" else "s")

let check l =
  let negative name v =
    if v < 0 then err "Talon.pp_with: %s is %d, negative" name v
  in
  negative "head" l.head;
  negative "tail" l.tail;
  negative "columns" l.columns;
  if l.width < 1 then err "Talon.pp_with: width is %d, not positive" l.width

(* A shown column: its name, type and rows, which align right when numbers. *)
type column = {
  name : cell;
  type_ : cell;
  body : cell array;
  right : bool;
  width : int;
}

(* [pad ~last ~right width c] is [c] padded to [width] on the left when [right],
   else on the right unless [last], so that lines end without spaces. *)
let pad ~last ~right width (c : cell) =
  let fill = String.make (width - c.width) ' ' in
  if right then fill ^ c.text else if last then c.text else c.text ^ fill

let pp l ppf t =
  check l;
  let rows = Table.rows t and columns = Schema.columns (Table.schema t) in
  let elided = rows - l.head > l.tail in
  let ranges =
    if elided then [ (0, l.head); (rows - l.tail, l.tail) ] else [ (0, rows) ]
  in
  let column j (name, Type.Any ty) =
    let part (offset, length) = part t j ~offset ~length in
    let body =
      match List.concat_map part ranges with
      | [] -> [||]
      | parts -> cells l.width (Column.concat parts)
    in
    let name =
      fit l.width (to_string (Format.dprintf "%a" Type.pp_name name))
    in
    let type_ = fit l.width (to_string (Format.dprintf "%a" Type.pp ty)) in
    let widest w (c : cell) = Int.max w c.width in
    let width = Array.fold_left widest (widest name.width type_) body in
    { name; type_; body; right = is_number ty; width }
  in
  let shown = List.mapi column (List.filteri (fun j _ -> j < l.columns) columns)
  and hidden = List.filteri (fun j _ -> j >= l.columns) columns in
  let line ~right cell =
    let last = List.length shown - 1 in
    let pad k col =
      pad ~last:(k = last) ~right:(right && col.right) col.width (cell col)
    in
    " " ^ String.concat "  " (List.mapi pad shown)
  in
  let row i = line ~right:true (fun col -> col.body.(i)) in
  let body =
    if not elided then List.init rows row
    else
      let not_shown = rows - l.head - l.tail in
      List.init l.head row @ [ " ⋮" ]
      @ List.init l.tail (fun i -> row (l.head + i))
      @ [ Printf.sprintf " %s not shown" (plural not_shown "row") ]
  in
  let table =
    if shown = [] then []
    else
      line ~right:false (fun col -> col.name)
      :: line ~right:false (fun col -> col.type_)
      :: body
  in
  let hidden =
    if hidden = [] then []
    else
      let names =
        List.map
          (fun (n, _) -> to_string (Format.dprintf "%a" Type.pp_name n))
          hidden
      in
      [
        Printf.sprintf " %s not shown: %s"
          (plural (List.length hidden) "column")
          (String.concat ", " names);
      ]
  in
  let header =
    Printf.sprintf "table %s × %s" (plural rows "row")
      (plural (List.length columns) "column")
  in
  Format.pp_open_vbox ppf 0;
  Format.pp_print_list ~pp_sep:Format.pp_print_cut Format.pp_print_string ppf
    ((header :: table) @ hidden);
  Format.pp_close_box ppf ()
