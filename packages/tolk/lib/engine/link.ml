(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop
module B = Device.Buffer

let cache = U.Weak_tbl.create 16

let rec constant u =
  match U.as_const u with
  | Some c -> Some c
  | None when Ops.Group.is_alu (U.op u) ->
      let values = List.map constant (U.children u) in
      if List.for_all Option.is_some values then
        U.exec_alu (U.op u) (U.dtype u) (List.map Option.get values)
      else None
  | None -> None

let words u = if U.op u = Ops.Stack then U.children u else [u]

let word_bytes c =
  let n = Dtype.itemsize (Const.dtype c) in
  let value = match Const.view c with
    | Const.Int value -> value
    | Const.Bool value -> Z.of_int (if value then 1 else 0)
    | _ -> invalid_arg "link: command words must be integers" in
  Bytes.init n (fun i -> Char.chr (Z.to_int (Z.extract value (8 * i) 8)))

let run ~resolve ?(allow_cache = true) linear =
  match if allow_cache then U.Weak_tbl.find_opt cache linear else None with
  | Some linked -> linked
  | None ->
      let refs = ref [] and can_cache = ref allow_cache in
      let retain buf =
        if not (List.exists (U.equal buf) !refs) then refs := buf :: !refs in
      let buffer u =
        let b = resolve u in
        retain (U.from_buffer b);
        b in
      let write buf offset bytes =
        let nbytes = Bytes.length bytes in
        if offset < 0 || nbytes > B.nbytes buf - offset then
          invalid_arg "link: patch outside its allocation";
        let view = B.view buf ~size:nbytes ~dtype:Dtype.uint8 ~offset in
        B.ensure_allocated view;
        B.copyin view bytes in
      let fold_store u =
        match U.as_store u with
        | Some {dst; value; gate = None} ->
            let value = match U.op value, U.children value with
              | Ops.Bitcast, [blob] when U.op blob = Ops.Binary -> blob
              | _ -> value in
            (match U.op value, U.arg value, U.as_index dst with
             | Ops.Binary, U.Arg.String blob, None ->
                 write (buffer dst) 0 (Bytes.of_string blob);
                 Some (U.noop ~dtype:Dtype.void ())
             | _, _, Some {ptr; idxs = [indices]} ->
                 let offsets = List.map constant (words indices)
                 and values = List.map constant (words value) in
                 if List.length offsets <> List.length values
                    || not (List.for_all Option.is_some (offsets @ values)) then None
                 else begin
                   let buf = buffer ptr in
                   List.iter2 (fun offset value ->
                       let bytes = word_bytes (Option.get value) in
                       let offset = match Const.view (Option.get offset) with
                         | Const.Int n -> Z.to_int (Z.mul n (Z.of_int (Bytes.length bytes)))
                         | _ -> invalid_arg "link: patch index must be an integer" in
                       write buf offset bytes) offsets values;
                   Some (U.noop ~dtype:Dtype.void ())
                 end
             | _ -> None)
        | _ -> None in
      let rec rewrite u =
        match U.op u with
        | Ops.Param when U.node_tag u <> None ->
            let buf =
              if U.node_tag u = Some "lt_input" then begin
                can_cache := false;
                resolve (U.replace u ~node_tag:None ())
              end else
                let device = match U.device_of u with
                  | Some (U.Single d) -> d
                  | _ -> invalid_arg "link: placeholder needs one device" in
                let volatile = match U.as_param u with
                  | Some {param; _} -> param.volatile | None -> assert false in
                let spec = {B.Buffer_spec.default with host = volatile;
                  cpu_access = true; uncached = volatile} in
                Device.create_buffer ~size:(max 1 (U.max_numel u))
                  ~dtype:(U.dtype u) ~spec (Device.get device) in
            Some (U.from_buffer buf)
        | Ops.Getaddr ->
            let source = (U.src u).(0) in
            (* Ordinary input parameters remain runtime-bound. *)
            if List.exists (fun b -> U.op b = Ops.Param)
                (U.toposort ~enter_calls:false source) then None
            else
              let buf = buffer source in
              let device = match U.arg u with
                | U.Arg.Device (U.Single d) -> Some d
                | U.Arg.Empty -> None
                | _ -> invalid_arg "link: address needs one device" in
              let address = B.addr ?device buf in
              Some (U.const (Const.int64 Dtype.uint64 (Int64.of_nativeint address)))
        | Ops.Store -> fold_store u
        | Ops.End ->
            (match U.as_end u with
             | Some {value; ranges = [range]} ->
                 (match U.as_range range with
                  | Some {size; _} ->
                      (match U.const_int_value size with
                       | Some count when count >= 0 ->
                           let folded = List.init count (fun i ->
                               let body = U.substitute [range, U.const_int i] value in
                               U.graph_rewrite rewrite body) in
                           if List.for_all (fun n -> U.op n = Ops.Noop) folded
                           then Some (U.noop ~dtype:Dtype.void ()) else None
                       | _ -> None)
                  | None -> None)
             | _ -> None)
        | Ops.After when not (U.is_bound_var u) ->
            let value = (U.src u).(0) in
            let deps = List.tl (U.children u)
              |> List.filter (fun dep -> U.op dep <> Ops.Noop) in
            if deps = [] then Some value
            else if U.op value = Ops.Call then
              if List.length deps = Array.length (U.src u) - 1 then None
              else Some (U.after ~src:value ~deps)
            else invalid_arg "link: unresolved initialization dependency"
        | op when Ops.Group.is_alu op -> Option.map U.const (constant u)
        | _ -> None in
      let linked = U.graph_rewrite ~name:"link" rewrite linear in
      let linked = match U.children linked, !refs with
        | first :: rest, (_ :: _ as deps) ->
            U.replace linked ~src:(Array.of_list (U.after ~src:first ~deps :: rest)) ()
        | _ -> linked in
      if !can_cache then U.Weak_tbl.replace cache linear linked;
      linked
