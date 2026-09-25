(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop
module B = Device.Buffer

type cached = {
  names : string array;
  graph : (Device.t, U.t) Ephemeron.Kn.t;
}

let cache = U.Weak_tbl.create 16
let cache_lock = Mutex.create ()
let with_cache_lock f = Storage.with_operation (fun () -> Mutex.protect cache_lock f)

let find_cached linear =
  match with_cache_lock (fun () -> U.Weak_tbl.find_opt cache linear) with
  | None -> None
  | Some entry -> Ephemeron.Kn.query entry.graph (Array.map Device.get entry.names)

let rec constant u =
  match U.as_const u with
  | Some c -> Some c
  | None when List.mem (U.op u) [Ops.Cast; Ops.Bitcast] ->
      (match U.children u with
       | [src] -> Option.bind (constant src) (fun value ->
           U.replace u ~src:[|U.const value|] () |> Symbolic.simplify |> U.as_const)
       | _ -> None)
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

let rec run ~resolve ?(allow_cache = true) linear =
  match if allow_cache then find_cached linear else None with
  | Some linked -> linked
  | None ->
      let refs = ref [] and can_cache = ref allow_cache in
      let owners = ref [] in
      let own device =
        if not (List.exists (( == ) device) !owners) then owners := device :: !owners in
      let own_name name =
        let device = Device.get name in
        own device;
        Option.iter (fun queue -> own (Device.get queue.Device.host)) (Device.queue device) in
      let rec collect_owners linear =
        List.iter (fun node ->
            (match U.device_of node with
             | Some (U.Single name) -> own_name name
             | Some (U.Multi names) -> List.iter own_name names
             | Some (U.Index _) | None -> ());
            (match U.arg node with
             | U.Arg.Call_info {aux = Some info; _} ->
                 List.iter own_name (info.host :: info.devices @
                   List.concat_map (fun (owner, source) -> [owner; source]) info.host_deps)
             | _ -> ());
            match U.as_call node with
            | Some {body; _} when U.op body = Ops.Custom_function
                && U.Arg.as_string (U.arg body) = Some "loop" ->
                collect_owners (U.src body).(0)
            | _ -> ()) (U.toposort ~enter_calls:false linear) in
      collect_owners linear;
      let retain buf =
        if not (List.exists (U.equal buf) !refs) then refs := buf :: !refs in
      let buffer u =
        let b = resolve u in
        own_name (B.device b);
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
                 Some (U.noop ())
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
                   Some (U.noop ())
                 end
             | _ -> None)
        | _ -> None in
      let rec rewrite u =
        match U.op u with
        | Ops.Param when U.node_tag u <> None
            || (match U.as_param u with Some {param; _} -> param.allocation <> None | None -> false) ->
            let buf =
              if U.node_tag u = Some "lt_input" then begin
                can_cache := false;
                resolve (U.replace u ~node_tag:None ())
              end else
                let device = match U.device_of u with
                  | Some (U.Single d) -> d
                  | _ -> invalid_arg "link: placeholder needs one device" in
                let owner = Device.get device in
                own owner;
                match Device.bufferize owner u with
                | Some buf -> buf
                | None ->
                let volatile = match U.as_param u with
                  | Some {param = {allocation = Some (kind, _); _}; _} ->
                      invalid_arg ("link: unsupported allocation " ^ kind ^ " on " ^ device)
                  | Some {param; _} -> param.volatile | None -> assert false in
                let spec = {B.Buffer_spec.default with host = volatile;
                  cpu_access = true; uncached = volatile ||
                    Option.fold ~none:false ~some:(String.starts_with ~prefix:"cmdbuf")
                      (U.node_tag u)} in
                Device.create_buffer ~size:(max 1 (U.max_numel u))
                  ~dtype:(U.dtype u) ~spec owner in
            Some (U.from_buffer buf)
        | Ops.Call ->
            (match U.as_call u with
             | Some {body; _} when U.op body = Ops.Custom_function
                 && U.Arg.as_string (U.arg body) = Some "loop" ->
                 let src = Array.copy (U.src body) in
                 let linked = run ~resolve ~allow_cache src.(0) in
                 if U.equal linked src.(0) then None else begin
                   src.(0) <- linked;
                   let args = Array.copy (U.src u) in
                   args.(0) <- U.replace body ~src ();
                   Some (U.replace u ~src:args ())
                 end
             | _ -> None)
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
                           then Some (U.noop ()) else None
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
        | op when Ops.Group.is_alu op || op = Ops.Cast || op = Ops.Bitcast -> Option.map U.const (constant u)
        | _ -> None in
      let linked = U.graph_rewrite ~name:"link" rewrite linear in
      let linked = match U.children linked, !refs with
        | first :: rest, (_ :: _ as deps) ->
            U.replace linked ~src:(Array.of_list (U.after ~src:first ~deps :: rest)) ()
        | _ -> linked in
      if not !can_cache then linked else
        let owners = List.sort (fun a b -> Int.compare (Device.id a) (Device.id b)) !owners
            |> Array.of_list in
        let names = Array.map Device.name owners in
        let entry = {names; graph = Ephemeron.Kn.make owners linked} in
        with_cache_lock (fun () ->
            let previous = match U.Weak_tbl.find_opt cache linear with
              | Some cached when cached.names = names -> Ephemeron.Kn.query cached.graph owners
              | Some _ | None -> None in
            match previous with
            | Some winner -> winner
            | None -> U.Weak_tbl.replace cache linear entry; linked)
