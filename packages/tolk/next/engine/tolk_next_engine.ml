(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_next

(* Devices *)

(* The kind of device a vendor library claims, if any. *)
let vendor d =
  if Metal.claims d then Some "METAL"
  else if Option.is_some (Nx_cuda_device.of_device d) then Some "CUDA"
  else if Option.is_some (Nx_amd_device.of_device d) then Some "AMD"
  else if Option.is_some (Nx_nv_device.of_device d) then Some "NV"
  else None

let target d =
  match vendor d with
  | Some kind -> Helpers.target ~arch:(Nx_device.arch d) kind
  | None when d != Nx_device.disk && Nx_device.shares_host_memory d ->
      let host = Nx_device.host_of d in
      let cpu = if host == Nx_device.host then "native" else "generic" in
      let t = Helpers.target ~arch:(Nx_device.arch host ^ "," ^ cpu) "CPU" in
      { t with renderer = "CLANG" }
  | None ->
      invalid_arg
        (Printf.sprintf "Tolk_next_engine.target: %s runs no program"
           (Nx_device.name d))

let find fn devices name =
  match List.assoc_opt name devices with
  | Some d -> d
  | None ->
      invalid_arg
        (Printf.sprintf "Tolk_next_engine.%s: no device is named %s" fn name)

let device devices name =
  let d = find "device" devices name in
  { Hcq2.target = target d; queues = None }

(* Host programs *)

module Program = struct
  type t = {
    program : Nx_device.Program.t;
    buffers : Device.Tiny_elf.param list; (* in the order of the globals *)
    vars : Ops.t list;
  }

  let load d prg =
    if (target d).device <> "CPU" then
      invalid_arg
        (Printf.sprintf "Tolk_next_engine.Program.load: %s runs no host program"
           (Nx_device.name d));
    let elf = Device.Tiny_elf.of_program prg in
    let info =
      match Ops.arg prg with Ops.Program info -> info | _ -> assert false
    in
    let nbuffers = List.length info.globals in
    let buffers = List.filteri (fun i _ -> i < nbuffers) elf.signature in
    match
      Nx_device.Program.load (Nx_device.host_of d) ~binary:elf.lib
        ~name:elf.name
    with
    | Ok program -> { program; buffers; vars = info.vars }
    | Error why -> failwith why

  let value vars v =
    match Ops.arg v with
    | Ops.Param p -> (
        match
          (Option.bind p.name (fun n -> List.assoc_opt n vars), p.bound)
        with
        | Some x, _ -> x
        | None, Some b -> Dtype.Value.to_int b
        | None, None ->
            invalid_arg
              (Printf.sprintf "Tolk_next_engine: variable %s is unbound"
                 (Option.value p.name ~default:(string_of_int p.slot))))
    | _ -> assert false

  let run ?(vars = []) p buffers =
    let fn = "Tolk_next_engine.Program.run" in
    let n = List.length p.buffers in
    if List.length buffers <> n then
      invalid_arg
        (Printf.sprintf "%s: %d buffers for %d parameters" fn
           (List.length buffers) n);
    List.iter2
      (fun (param : Device.Tiny_elf.param) b ->
        let need =
          List.fold_left ( * ) (Dtype.itemsize param.dtype) param.shape
        in
        if Nx_device.Buffer.nbytes b < need then
          invalid_arg
            (Printf.sprintf "%s: slot %d holds %d bytes, not %d" fn param.slot
               (Nx_device.Buffer.nbytes b)
               need))
      p.buffers buffers;
    Nx_device.Program.call p.program (Array.of_list buffers)
      (Array.of_list (List.map (value vars) p.vars))
end

(* Linked schedules *)

let strf = Printf.sprintf

(* A call of a schedule, as linked: a host program with a program loaded for
   each device of its first argument, or a copy. *)
type call =
  | Kernel of { call : Ops.t; lanes : Program.t list }
  | Copy of { dst : Ops.t; src : Ops.t }

type t = {
  lock : Mutex.t; (* runs share the storage: one at a time *)
  calls : call list;
  params : (int * Ops.t) list; (* the parameters the runs bind, by slot *)
  storage : Nx_device.Buffer.t list Ops.Tbl.t; (* by node, a buffer per device *)
}

let names u =
  match Ops.device u with
  | Some (Single d) -> [ d ]
  | Some (Multi ds) -> ds
  | None -> invalid_arg (Format.asprintf "%a has no device" Ops.pp u)

let is_slot u =
  Ops.op u = Op.Param
  && Option.is_none (Ops.tag u)
  && Ops.addrspace u <> Some Dtype.Alu

let bytes u = Ops.max_numel u * Dtype.itemsize (Ops.dtype u)

(* The storage each device holds of [base], a parameter bound by [slots] or
   storage bound at link. *)
let rec holds t slots base =
  match Ops.op base with
  | Op.Param -> (
      match Ops.arg base with
      | Ops.Param p -> slots.(p.slot)
      | _ -> assert false)
  | Op.Buffer -> Ops.Tbl.find t.storage base
  | Op.Mstack -> List.concat_map (holds t slots) (Ops.src base)
  | _ -> invalid_arg (Format.asprintf "%a is not storage" Ops.pp base)

(* The buffers of the view [u], one per device, or one shared by them all. *)
let view t slots u =
  let base, shard, off = Hcq2.unwrap_lane u in
  let buffers = holds t slots base in
  let buffers =
    match shard with Some i -> [ List.nth buffers i ] | None -> buffers
  in
  let n = bytes u in
  List.map
    (fun b ->
      if off = 0 && Nx_device.Buffer.nbytes b = n then b
      else Nx_device.Buffer.view b ~offset:off Nx_dtype.Scalar.UInt8 n)
    buffers

let lane buffers i = match buffers with [ b ] -> b | bs -> List.nth bs i

let int_of_const u =
  match Ops.arg u with
  | Ops.Const (`Int z) -> Z.to_int z
  | Ops.Const (`Bool b) -> Bool.to_int b
  | _ -> invalid_arg (Format.asprintf "%a is no integer" Ops.pp u)

let value vars lane u =
  if Ops.op u = Op.Const then int_of_const u
  else
    match (Ops.expr u, Ops.arg u) with
    | "_device_num", _ -> lane
    | name, Ops.Param p -> (
        match (List.assoc_opt name vars, p.bound) with
        | Some x, _ -> x
        | None, Some b -> Dtype.Value.to_int b
        | None, None ->
            invalid_arg
              (strf "Tolk_next_engine.run: variable %s is unbound" name))
    | _ -> assert false

let link ~devices ?(bound = []) linear =
  let fn = "Tolk_next_engine.link" in
  if Ops.op linear <> Op.Linear then
    invalid_arg (strf "%s: not a compiled schedule" fn);
  let device name = find "link" devices name in
  let storage = Ops.Tbl.create 16 in
  List.iter
    (fun (u, bs) ->
      let ns = names u in
      if Ops.op u <> Op.Buffer || List.length bs <> List.length ns then
        invalid_arg
          (Format.asprintf "%s: %a is not bound to a buffer on each device" fn
             Ops.pp u);
      List.iter2
        (fun n b ->
          if
            Nx_device.Buffer.device b != device n
            || Nx_device.Buffer.nbytes b < bytes u
          then
            invalid_arg
              (Format.asprintf "%s: %a is not bound to %d bytes of %s" fn Ops.pp
                 u (bytes u) n))
        ns bs;
      Ops.Tbl.replace storage u bs)
    bound;
  let calls = List.map Ops.without_after (Ops.src linear) in
  let args = List.concat_map Realize.get_call_arg_uops calls in
  let nodes = List.concat_map (fun a -> Ops.toposort a) args in
  List.iter
    (fun u ->
      if Ops.op u = Op.Buffer && not (Ops.Tbl.mem storage u) then
        Ops.Tbl.replace storage u
          (List.map
             (fun n ->
               Nx_device.Buffer.create (device n) Nx_dtype.Scalar.UInt8
                 (max (bytes u) 1))
             (names u)))
    nodes;
  let params =
    List.sort_uniq compare
      (List.filter_map
         (fun u ->
           match Ops.arg u with
           | Ops.Param p when is_slot u -> Some (p.slot, u)
           | _ -> None)
         nodes)
  in
  let linked call =
    let body = Ops.body call in
    match Ops.op body with
    | Op.Store -> (
        match Realize.get_call_arg_uops call with
        | [ dst; src ] -> Copy { dst; src }
        | _ -> invalid_arg (strf "%s: a copy of other than two buffers" fn))
    | Op.Program ->
        let first = List.hd (Realize.get_call_arg_uops call) in
        Kernel
          {
            call;
            lanes =
              List.map (fun n -> Program.load (device n) body) (names first);
          }
    | _ ->
        invalid_arg
          (Format.asprintf "%s: %a is no call to run" fn Op.pp (Ops.op body))
  in
  { lock = Mutex.create (); calls = List.map linked calls; params; storage }

let check_slots t slots =
  List.iter
    (fun (slot, u) ->
      let fail fmt =
        Printf.ksprintf
          (fun m -> invalid_arg ("Tolk_next_engine.run: " ^ m))
          fmt
      in
      if slot >= Array.length slots then fail "no buffers for slot %d" slot;
      let ns = names u and bs = slots.(slot) in
      if List.length bs <> List.length ns then
        fail "slot %d takes %d buffers, not %d" slot (List.length ns)
          (List.length bs);
      List.iter
        (fun b ->
          if Nx_device.Buffer.nbytes b < bytes u then
            fail "slot %d takes %d bytes, not %d" slot (bytes u)
              (Nx_device.Buffer.nbytes b))
        bs)
    t.params

let run_call ~vars t slots = function
  | Copy { dst; src } ->
      let dsts = view t slots dst and srcs = view t slots src in
      List.iteri
        (fun i dst -> Nx_device.Buffer.copy ~src:(lane srcs i) ~dst)
        dsts
  | Kernel { call; lanes } ->
      let prg = Ops.body call in
      let args = List.map (view t slots) (Realize.get_call_arg_uops call) in
      let info =
        match Ops.arg prg with Ops.Program i -> i | _ -> assert false
      in
      let vals = Realize.get_call_var_uops call prg in
      List.iteri
        (fun i (p : Program.t) ->
          let buffers =
            List.map (fun g -> lane (List.nth args g) i) info.globals
          in
          Nx_device.Program.call p.program (Array.of_list buffers)
            (Array.of_list (List.map (value vars i) vals)))
        lanes

let run ?(vars = []) t slots =
  check_slots t slots;
  Mutex.protect t.lock (fun () -> List.iter (run_call ~vars t slots) t.calls)

(* Measuring *)

(* Only NV invalidates its caches on demand. *)
let invalidate_caches d =
  Option.iter Nx_nv_device.invalidate_caches (Nx_nv_device.of_device d)

let measure ?(cold = false) ?(vars = []) ~devices name prg =
  let d = find "measure" devices name in
  if cold then invalidate_caches d;
  let p = Program.load d prg in
  let buffers =
    List.map
      (fun (param : Device.Tiny_elf.param) ->
        Nx_device.Buffer.create d Nx_dtype.Scalar.UInt8
          (max 1
             (List.fold_left ( * ) (Dtype.itemsize param.dtype) param.shape)))
      p.buffers
  in
  let t0 = Nx_device.Profile.now () in
  Program.run ~vars p buffers;
  Float.of_int (Nx_device.Profile.now () - t0) *. 1e-9
