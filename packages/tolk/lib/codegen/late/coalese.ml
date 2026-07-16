(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

module U = Uop

let ceil_div a b = (a + b - 1) / b
let lane src i = U.index ~ptr:src ~idxs:[ U.const_int i ] ()

(* An image buffer is a param/buffer whose shape is [(height, width, 4)].
   Pointer-ness and image-ness are structural now: there is no image dtype. *)
let is_image_shape u =
  match (try U.max_shape u with Invalid_argument _ -> []) with
  | [ _; _; 4 ] -> true
  | _ -> false

let image_valid_dims ?(osx = false) ~image_pitch_alignment ~base ~size () =
  match image_pitch_alignment with
  | None | Some 0 -> []
  | Some align ->
      let max_width = 16384 in
      let pxls = size / 4 in
      let supported_base =
        Dtype.equal base Dtype.float16 || Dtype.equal base Dtype.float32
      in
      if (not supported_base) || size > (4 * max_width * max_width) then []
      else if size mod (align * 4) <> 0 then
        let byte_align = if osx then 64 else align in
        if (Dtype.itemsize base * size) mod byte_align <> 0
           || pxls > max_width
        then []
        else [ (1, pxls) ]
      else
        let units = pxls / align in
        let k_min = ceil_div units max_width in
        let k_max = min units (max_width / align) in
        let rec loop k acc =
          if k > k_max then List.rev acc
          else
            let acc =
              if units mod k = 0 then (units / k, align * k) :: acc
              else acc
            in
            loop (k + 1) acc
        in
        loop k_min []

let is_invalid_const u =
  match U.op u, U.arg u with
  | Ops.Const, U.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let invalid_where_index idx =
  match U.op idx, U.src idx with
  | Ops.Where, [| valid; value; invalid |] when is_invalid_const invalid ->
      Some (valid, value)
  | _ -> None

let fake_var ~index ~lo ~hi ~(dtype : Dtype.t) () =
  U.variable ~name:(Printf.sprintf "fake%d" index) ~min_val:lo
    ~max_val:hi ~dtype ()

let drop_valid_stmts valid idx height width =
  let coord_before_zero coord =
    match U.const_int_value coord with
    | Some n -> n < 0
    | None -> U.vmax coord < 0
  in
  let coord_after_bound coord bound =
    match U.const_int_value coord with
    | Some n -> n >= bound
    | None -> U.vmin coord >= bound
  in
  let dropped = ref [] in
  List.iteri
    (fun i stmt ->
      match Symbolic.parse_valid stmt with
      | None -> ()
      | Some (x, is_upper_bound, c) ->
          let terms = U.split_uop x Ops.Add in
          let can_drop_simplex =
            (not is_upper_bound) && c = 1
            && List.for_all
                 (fun u ->
                   List.mem (U.op u) Ops.Group.irreducible && U.vmin u = 0)
                 terms
          in
          if can_drop_simplex then begin
            let testidx =
              List.fold_left
                (fun nowidx u ->
                  U.substitute [ (u, U.const_like u 0) ] nowidx)
                idx terms
            in
            let xcoord = lane testidx 0 in
            let ycoord = lane testidx 1 in
            if coord_before_zero xcoord || coord_before_zero ycoord then
              dropped := stmt :: !dropped
          end;
          if not (List.exists (U.equal stmt) !dropped) then begin
            let lo, hi =
              if is_upper_bound then (c + 1, U.vmax x)
              else (U.vmin x, c - 1)
            in
            if lo <= hi then begin
              let fake = fake_var ~index:i ~lo ~hi ~dtype:(U.dtype x) () in
              let coord_out_of_bounds coord bound =
                let rw = U.substitute [ (x, fake) ] coord |> U.simplify in
                coord_after_bound rw bound || coord_before_zero rw
              in
              let xcoord = lane idx 0 in
              let ycoord = lane idx 1 in
              if coord_out_of_bounds xcoord width
                 || coord_out_of_bounds ycoord height
              then dropped := stmt :: !dropped
            end
          end)
    (U.split_uop valid Ops.And);
  List.rev !dropped

let simplify_valid_load ptr start_idx valid =
  let idx = Symbolic.uop_given_valid valid start_idx in
  if U.equal idx start_idx || U.equal idx (U.simplify start_idx) then None
  else Some (U.index ~ptr ~idxs:[ U.valid ~src:idx ~cond:valid ] ())

(* Image load/store is always float-typed regardless of the buffer's scalar
   dtype; the half conversion is handled by the image-float rules. *)
let image_index ptr idxs = U.replace (U.index ~ptr ~idxs ()) ~dtype:Dtype.float32 ()

let simplify_valid_image_coords buf y x =
  match invalid_where_index y, invalid_where_index x with
  | Some (valid_y, y_value), Some (valid_x, x_value)
    when U.equal valid_y valid_x -> (
      match (try U.max_shape buf with Invalid_argument _ -> []) with
      | [ height; width; 4 ] ->
          let y_value, x_value =
            if Dtype.equal (U.dtype x_value) (U.dtype y_value) then
              (y_value, x_value)
            else
              ( U.cast ~src:y_value ~dtype:Dtype.int32,
                U.cast ~src:x_value ~dtype:Dtype.int32 )
          in
          let start_idx =
            U.stack ~dtype:(U.dtype x_value) [ x_value; y_value ]
          in
          let idx = Symbolic.uop_given_valid valid_y start_idx in
          let drop_stmt = drop_valid_stmts valid_y idx height width in
          if drop_stmt = [] && U.equal idx start_idx then None
          else
            let kept =
              U.split_uop valid_y Ops.And
              |> List.filter (fun stmt ->
                     not (List.exists (fun d -> U.equal d stmt) drop_stmt))
            in
            let y' = lane idx 1 in
            let x' = lane idx 0 in
            (match kept with
             | [] -> Some (y', x')
             | _ ->
                 let new_valid = U.uprod kept in
                 Some
                   ( U.valid ~src:y' ~cond:new_valid,
                     U.valid ~src:x' ~cond:new_valid ))
      | _ -> None)
  | _ -> None

let simplify_valid_image_load ptr idx =
  match U.op idx, U.src idx with
  | Ops.Stack, [| y; x |] -> (
      match simplify_valid_image_coords ptr y x with
      | Some (y, x) ->
          Some (image_index ptr [ U.stack ~dtype:(U.dtype y) [ y; x ] ])
      | None -> None)
  | _ -> None

let indexing_simplify_rule node =
  match U.as_index node with
  | Some { ptr; idxs = [ idx ] } -> (
      match invalid_where_index idx with
      | Some (valid, value) -> simplify_valid_load ptr value valid
      | None -> simplify_valid_image_load ptr idx)
  | Some { ptr; idxs = [ y; x ] } -> (
      match simplify_valid_image_coords ptr y x with
      | Some (y, x) -> Some (image_index ptr [ y; x ])
      | None -> None)
  | Some _ | None -> None

let indexing_simplify : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    op ~name:"idx" Ops.Index
    => fun bs -> indexing_simplify_rule (bs $ "idx");
  ]

let host_is_osx () =
  try
    let ic = Unix.open_process_in "uname -s" in
    let value = input_line ic in
    let _ = Unix.close_process_in ic in
    String.equal (String.trim value) "Darwin"
  with _ -> false

let image_target ren =
  List.mem (Renderer.device ren) [ "QCOM"; "CL"; "PYTHON"; "NULL" ]

let transform_to_image shapes ren buf offset =
  if (not (Helpers.getenv "IMAGE" 0 <> 0)) || not (image_target ren) then None
  else
    let valid, offset =
      match invalid_where_index offset with
      | None -> (None, offset)
      | Some (valid, offset) -> (Some valid, offset)
    in
    match U.as_param buf with
    | Some { param; _ } ->
        let base = U.dtype buf in
        let size =
          List.fold_left ( * ) 1
            (try U.max_shape buf with Invalid_argument _ -> [])
        in
        let candidates =
          match Hashtbl.find_opt shapes param.slot with
          | Some dims -> [ dims ]
          | None ->
              image_valid_dims
                ~osx:(host_is_osx ())
                ~image_pitch_alignment:(Renderer.image_pitch_alignment ren)
                ~base ~size ()
        in
        let valid_u = Option.value valid ~default:(U.const_bool true) in
        let c4 = U.const_int 4 in
        let candidates =
          List.map
            (fun (height, width) ->
              let width_u = U.const_int width in
              let row_stride = U.const_int (4 * width) in
              let x = U.O.((offset // c4) mod width_u) in
              let y = U.O.(offset // row_stride) in
              let cidx =
                Symbolic.uop_given_valid valid_u
                  (U.stack ~dtype:(U.dtype x) [ x; y ])
              in
              let dropped =
                drop_valid_stmts valid_u cidx height width |> List.length
              in
              (dropped, height, width, cidx))
            candidates
        in
        let best_drop =
          List.fold_left (fun acc (d, _, _, _) -> max acc d) (-1) candidates
        in
        let candidates =
          List.filter (fun (d, _, _, _) -> d = best_drop) candidates
        in
        let pick =
          match candidates with
          | [] -> None
          | [ cand ] -> Some cand
          | cand :: cands ->
              let score (_, _, _, idx) =
                lane idx 1 |> U.simplify |> U.backward_slice |> List.length
              in
              Some
                (List.fold_left
                   (fun best cand ->
                     if score cand < score best then cand else best)
                   cand cands)
        in
        (match pick with
         | None -> None
         | Some (_, height, width, cidx) ->
             let shape_arg =
               U.stack
                 [ U.const_int height; U.const_int width; U.const_int 4 ]
             in
             let buf = U.replace buf ~src:[| shape_arg |] () in
             Hashtbl.replace shapes param.slot (height, width);
             let x = lane cidx 0 in
             let y = lane cidx 1 in
             let coord =
               match valid with
               | None -> U.stack ~dtype:(U.dtype x) [ y; x ]
               | Some valid ->
                   U.stack ~dtype:(U.dtype x)
                     [ U.valid ~src:y ~cond:valid; U.valid ~src:x ~cond:valid ]
             in
             Some (image_index buf [ coord ]))
    | None -> None

let transform_to_image_rule shapes ren node =
  match U.op node, U.src node with
  | Ops.Shrink, [| buf; offset; size |] -> (
      match U.const_int_value size with
      | Some 4 -> transform_to_image shapes ren buf offset
      | _ -> None)
  | _ -> None

let strip_float_half_float node =
  match U.op node, U.src node, U.dtype node with
  | Ops.Cast, [| half |], dst when Dtype.equal dst Dtype.float32 -> (
      match U.op half, U.src half, U.dtype half with
      | Ops.Cast, [| src |], mid
        when Dtype.equal mid Dtype.float16
             && Dtype.equal (U.dtype src) Dtype.float32 ->
          Some src
      | _ -> None)
  | _ -> None

let image_float_rule node =
  match strip_float_half_float node with
  | Some _ as r -> r
  | None -> (
      match U.as_load node with
      | Some { src; alt = None; gate = None }
        when U.op src = Ops.Index
             && Dtype.equal (U.dtype src) Dtype.float32
             && Dtype.equal (U.dtype node) Dtype.float16 ->
          Some (U.cast ~src:(U.load ~src ()) ~dtype:Dtype.float16)
      | _ -> (
          match U.as_store node with
          | Some { dst; value; gate = None }
            when U.op dst = Ops.Index
                 && Dtype.equal (U.dtype dst) Dtype.float32
                 && Dtype.equal (U.dtype value) Dtype.float16 ->
              Some
                (U.store ~dst ~value:(U.cast ~src:value ~dtype:Dtype.float32) ())
          | _ -> None))

let pm_simplify_add_image ren =
  let shapes = Hashtbl.create 8 in
  let open Upat in
  Pattern_matcher.make
    [
      ops ~name:"node" Ops.Group.all
      => fun bs ->
           let node = bs $ "node" in
           (match transform_to_image_rule shapes ren node with
            | Some _ as r -> r
            | None -> image_float_rule node);
    ]

type memory_op = Mem_load | Mem_store

type base =
  | Base_uop of U.t
  | Base_const
  | Base_invalid

type memory_entry = {
  op : memory_op;
  node : U.t;
  index : U.t;
  buf : U.t;
  base : base;
  valid : U.t option;
  offset : int;
  value : U.t option;
}

let coalesced_stack = function
  | [] -> invalid_arg "coalesced_stack: empty"
  | values -> U.stack values

let index_base_offset idx =
  match U.op idx, U.src idx, U.const_int_value idx with
  | Ops.Const, _, _ when is_invalid_const idx -> Some (Base_invalid, 0)
  | Ops.Add, src, _ when U.op src.(1) = Ops.Const ->
      Option.map (fun offset -> (Base_uop src.(0), offset))
        (U.const_int_value src.(1))
  | Ops.Add, src, _ when U.op src.(0) = Ops.Const ->
      Option.map (fun offset -> (Base_uop src.(1), offset))
        (U.const_int_value src.(0))
  | _, _, Some offset -> Some (Base_const, offset)
  | _ -> Some (Base_uop idx, 0)

let index_offset base offset =
  match base with
  | Base_uop base -> U.O.(base + U.const_int offset)
  | Base_const | Base_invalid -> U.const_int offset

let coalesce_divides expr width =
  width = 1 || U.divides expr width <> None

let coalesce_entry node =
  let of_index op ?value index =
    match U.as_index index with
    | Some { ptr = buf; idxs = [ idx ] } ->
        if U.addrspace buf = Some Dtype.Reg then None
        else
          let valid, idx =
            match invalid_where_index idx with
            | None -> (None, idx)
            | Some (valid, idx) -> (Some valid, idx)
          in
          (match index_base_offset idx with
           | Some (base, offset) ->
               Some { op; node; index; buf; base; valid; offset; value }
           | None -> None)
	    | Some _ | None ->
	        failwith
	          (Printf.sprintf "memory coalesing should be on INDEX, not %s"
	             (Ops.name (U.op index)))
  in
  match U.as_load node, U.as_store node with
  | Some { src; alt = None; gate = None }, _ -> of_index Mem_load src
  | _, Some { dst; value; gate = None } -> of_index Mem_store ~value dst
  | Some _, _ | _, Some _ ->
      failwith "memory coalesing does not support gated loads/stores"
  | _ -> None

let base_equal a b =
  match a, b with
  | Base_const, Base_const | Base_invalid, Base_invalid -> true
  | Base_uop a, Base_uop b -> U.compare_structure a b = 0
  | _ -> false

let valid_equal a b =
  match a, b with
  | None, None -> true
  | Some a, Some b -> U.compare_structure a b = 0
  | _ -> false

let same_memory_key a b =
  a.op = b.op && U.equal a.buf b.buf && base_equal a.base b.base
  && valid_equal a.valid b.valid

let offset_groups offsets =
  let rec finish current groups = match current with
    | [] -> List.rev groups
    | _ -> List.rev (List.rev current :: groups)
  in
  let rec loop current groups = function
  | [] -> finish current groups
  | x :: xs -> (
      match current with
      | last :: _ when x = last + 1 -> loop (x :: current) groups xs
      | _ -> loop [ x ] (List.rev current :: groups) xs)
  in
  match offsets with [] -> [] | x :: xs -> loop [ x ] [] xs

let entries_at entries offset =
  let entries = List.filter (fun entry -> entry.offset = offset) entries in
  let rec loop seen acc = function
  | [] -> List.rev acc
  | entry :: entries ->
      if List.exists (fun node -> U.compare_structure node entry.node = 0) seen
      then loop seen acc entries
      else loop (entry.node :: seen) (entry :: acc) entries
  in
  loop [] [] entries

let is_foldable_scalar = function
  | Dtype.Float32 | Dtype.Float16 | Dtype.Fp8e4m3 | Dtype.Fp8e5m2
  | Dtype.Fp8e4m3fnuz | Dtype.Fp8e5m2fnuz -> true
  | _ -> false

let fold_widths_for_value ren buf =
  if Renderer.device ren = "DSP" then [ 128; 64; 32; 16; 8; 4 ]
  else if U.addrspace buf = Some Dtype.Reg then []
  else if is_image_shape buf then [ 4 ]
  else
    let scalar = U.dtype buf in
    if not (is_foldable_scalar scalar) then []
    else if Renderer.supports_float4 ren then
      if Dtype.equal scalar Dtype.float16 && Helpers.allow_half8 then [ 8; 4; 2 ]
      else [ 4; 2 ]
    else []

let must_divide ren = Renderer.device ren <> "DSP"

let take n xs =
  let rec loop n acc xs =
    if n = 0 then List.rev acc, xs
    else match xs with
    | [] -> List.rev acc, []
    | x :: xs -> loop (n - 1) (x :: acc) xs
  in
  loop n [] xs

let gated_offset valid base =
  match valid with
  | None -> base
  | Some valid -> U.valid ~src:base ~cond:valid

let coalesce_load_group ren entries offsets =
  let first = List.hd entries in
  let widths = fold_widths_for_value ren first.buf @ [ 1 ] in
  let rec loop acc = function
  | [] -> acc
  | offsets ->
      let group_offset = List.hd offsets in
      let base = index_offset first.base group_offset in
      let len = List.length offsets in
      let width =
        List.find
          (fun width ->
            width <= len
            && ((not (must_divide ren)) || coalesce_divides base width))
          widths
      in
      let group, rest = take width offsets in
      let offset = gated_offset first.valid base in
      let idx =
        if width > 1 then
          U.shrink ~src:first.buf ~offset ~size:(U.const_int width)
        else U.index ~ptr:first.buf ~idxs:[ offset ] ()
      in
      let template = List.hd (entries_at entries group_offset) in
      let load =
        U.replace template.node ~src:[| idx |] ~dtype:(U.dtype template.node) ()
      in
      let replacements =
        List.mapi
          (fun lane offset ->
            entries_at entries offset
            |> List.map (fun entry ->
                   let value =
                     if width > 1 then
                       U.replace entry.index
                         ~src:[| load; U.const_int lane |]
                         ~dtype:(U.dtype entry.node) ()
                     else load
                   in
                   (entry.node, value)))
          group
        |> List.flatten
      in
      loop (List.rev_append replacements acc) rest
  in
  loop [] offsets

let coalesce_store_group ren entries offsets =
  let first = List.hd entries in
  let widths = fold_widths_for_value ren first.buf @ [ 1 ] in
  let rec loop acc = function
  | [] -> acc
  | offsets ->
      let group_offset = List.hd offsets in
      let base = index_offset first.base group_offset in
      let len = List.length offsets in
      let width =
        List.find
          (fun width ->
            width <= len
            && ((not (must_divide ren)) || coalesce_divides base width))
          widths
      in
      let group, rest = take width offsets in
      let grouped_entries = List.map (entries_at entries) group in
      let replacements =
        match List.find_opt (fun entries -> List.length entries <> 1) grouped_entries with
        | Some _ -> failwith "Coalese: multiple stores to the same offset"
        | None ->
            let offset = gated_offset first.valid base in
            let idx =
              if width > 1 then
                U.shrink ~src:first.buf ~offset ~size:(U.const_int width)
              else U.index ~ptr:first.buf ~idxs:[ offset ] ()
            in
            let stores = List.map List.hd grouped_entries in
            let values =
              List.map
                (fun entry ->
                  match entry.value with
                  | Some value -> value
                  | None -> assert false)
                stores
            in
            let value =
              match values with
              | [ value ] -> value
              | values -> coalesced_stack values
            in
            let store = U.replace (List.hd stores).node ~src:[| idx; value |] () in
            List.map (fun entry -> (entry.node, store)) stores
      in
      loop (List.rev_append replacements acc) rest
  in
  loop [] offsets

let coalesce_group ren entries =
  match entries with
  | [] -> []
  | first :: _ ->
      let offsets =
        entries
        |> List.map (fun entry -> entry.offset)
        |> List.sort_uniq Int.compare
      in
      let groups = offset_groups offsets in
      List.concat_map
        (fun offsets ->
          match first.op with
          | Mem_load -> coalesce_load_group ren entries offsets
          | Mem_store -> coalesce_store_group ren entries offsets)
        groups

let memory_coalesing ren root =
  if Helpers.getenv "DMC" 0 <> 0 then root
  else
    let rec group_entries groups = function
    | [] -> groups
    | entry :: rest ->
        let same, rest = List.partition (same_memory_key entry) rest in
        group_entries ((entry :: same) :: groups) rest
    in
    let replacements =
      U.toposort root
      |> List.filter_map coalesce_entry
      |> group_entries []
      |> List.concat_map (coalesce_group ren)
    in
    match replacements with
    | [] -> root
    | replacements -> U.substitute replacements root
