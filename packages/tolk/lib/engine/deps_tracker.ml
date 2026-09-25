(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type allocation = Graph of int | Parameter of int | Storage of int

type region = { base : allocation; lane : int option; start : int; stop : int }
type 'a t = {
  writes : ((allocation * int option), (int * int * 'a) list) Hashtbl.t;
  reads : ((allocation * int option), (int * int * 'a) list) Hashtbl.t;
}

let create () = {writes = Hashtbl.create 16; reads = Hashtbl.create 16}
let ranges table key = Option.value (Hashtbl.find_opt table key) ~default:[]

let access t regions ~writes token =
  let dependencies = ref [] in
  List.iteri (fun i {base; lane; start; stop} ->
      if start < 0 || stop < start then invalid_arg "Deps_tracker.access: invalid byte interval";
      let overlap table =
        List.iter (fun (a, b, previous) ->
            if a < stop && start < b && start < stop then
              dependencies := previous :: !dependencies)
          (ranges table (base, lane)) in
      overlap t.writes;
      if List.mem i writes then overlap t.reads) regions;
  List.iteri (fun i {base; lane; start; stop} ->
      if start < stop then begin
        let key = base, lane in
        if List.mem i writes then begin
          let trim table =
            let left = List.concat_map (fun (a, b, previous) ->
                (if a < min start b then [a, min start b, previous] else [])
                @ (if max stop a < b then [max stop a, b, previous] else []))
                (ranges table key) in
            Hashtbl.replace table key left in
          trim t.writes;
          trim t.reads;
          Hashtbl.replace t.writes key ((start, stop, token) :: ranges t.writes key)
        end else
          Hashtbl.replace t.reads key ((start, stop, token) :: ranges t.reads key)
      end) regions;
  List.rev !dependencies

let buffer b =
  let start = Device.Buffer.offset b in
  {base = Storage (Device.Buffer.base_id b); lane = None; start;
   stop = start + Device.Buffer.nbytes b}

let uop u =
  let module U = Tolk_uop.Uop in
  let add a b = Tolk_uop.Bound.(to_int (add (int a) (int b))) in
  let rec unwrap u lane offset =
    match U.contiguous_view u with
    | Some (base, delta) when not (U.equal base u) -> unwrap base lane (add offset delta)
    | _ ->
        match U.op u, U.arg u with
        | Tolk_uop.Ops.Mselect, U.Arg.Int i ->
            let src = (U.src u).(0) in
            if U.op src = Tolk_uop.Ops.Mstack then unwrap (U.src src).(i) None offset
            else unwrap src (Some i) offset
        | Tolk_uop.Ops.Mstack, _ when Option.is_some lane ->
            unwrap (U.src u).(Option.get lane) None offset
        | Tolk_uop.Ops.Buffer, U.Arg.Param_arg {buffer = Some buffers; _} ->
            let buffer = match lane, buffers with
              | None, [b] -> Some b
              | Some i, bs -> Some (List.nth bs i)
              | None, _ -> None in
            (match buffer with
             | Some b -> Storage (Device.Buffer.base_id b), None,
                 add offset (Device.Buffer.offset b)
             | None -> Graph (U.tag u), lane, offset)
        | Tolk_uop.Ops.Param, U.Arg.Param_arg {slot; allocation = None; _}
            when U.node_tag u = None -> Parameter slot, lane, offset
        | (Tolk_uop.Ops.Param | Tolk_uop.Ops.Buffer), _ -> Graph (U.tag u), lane, offset
        | _ -> invalid_arg "Deps_tracker.uop: expected contiguous storage" in
  let base, lane, start = unwrap u None 0 in
  let nbytes = Tolk_uop.Bound.(to_int
      (mul (int (U.max_numel u)) (int (Tolk_uop.Dtype.itemsize (U.dtype u))))) in
  {base; lane; start; stop = add start nbytes}
