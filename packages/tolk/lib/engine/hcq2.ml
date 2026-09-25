(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop

type call = { call : U.t; device : string; queue : string }
type plan = {
  queues : (string * string * U.t list) list;
  timelines : U.t list;
  signals : U.t list;
}

let uint n = U.const (Const.int Dtype.uint64 n)
let ins name args = U.ins ~mnemonic:name ~operands:args ()
let view src start size = U.shrink ~src ~offset:(U.const_int start) ~size:(U.const_int size)
let timeline device = U.placeholder ~shape:[2] ~dtype:Dtype.uint64 ~slot:0
    ~device:(U.Single device) ~volatile:true () |> U.with_tag "timeline"
let timeline_value device = U.load
    ~src:(U.index ~ptr:(timeline device) ~idxs:[U.const_int 1] ()) ()
let unique xs = List.sort_uniq String.compare xs
let hardware device =
  List.mem (List.hd (String.split_on_char ':' device)) ["NV"; "AMD"; "CUDA"; "METAL"]

let arguments call =
  match U.as_call call with
  | Some {body; args} ->
      let args = List.filter (fun a -> not (U.is_bound_var a)) args in
      (match U.as_program_info body with
       | Some info ->
           let selected = List.map (List.nth args) info.globals in
           let writes = List.mapi (fun i slot -> i, slot) info.globals
             |> List.filter_map (fun (i, slot) -> if List.mem slot info.outs then Some i else None) in
           selected, writes
       | None when U.op body = Ops.Store -> args, [0]
       | None -> invalid_arg "Hcq2.plan: expected PROGRAM or STORE")
  | None -> invalid_arg "Hcq2.plan: expected CALL"

let plan ?(profile = false) calls =
  let calls = Array.of_list (List.map (fun c -> {c with device = Device.canonicalize c.device}) calls) in
  let devices = ref [] and queues = Hashtbl.create 8 and peers = Hashtbl.create 8 in
  let ensure device =
    if not (Hashtbl.mem queues device) then begin
      devices := !devices @ [device]; Hashtbl.add queues device []
    end in
  let previous = Array.make (Array.length calls) None and last = Hashtbl.create 8 in
  Array.iteri (fun tag c ->
      ensure c.device;
      let qs = Hashtbl.find queues c.device in
      if not (List.mem c.queue qs) then Hashtbl.replace queues c.device (qs @ [c.queue]);
      let key = c.device, c.queue in
      previous.(tag) <- Hashtbl.find_opt last key;
      Hashtbl.replace last key tag;
      let args, _ = arguments c.call in
      let foreign = List.concat_map (fun u -> match U.device_of u with
          | Some (U.Single d) -> [Device.canonicalize d]
          | Some (U.Multi ds) -> List.map Device.canonicalize ds
          | _ -> []) args
        |> List.filter (fun d -> d <> c.device && hardware d) |> unique in
      List.iter ensure foreign;
      let old = Option.value (Hashtbl.find_opt peers key) ~default:[] in
      Hashtbl.replace peers key (unique (old @ foreign))) calls;
  let epilogue device = match Hashtbl.find queues device with
    | [q] -> q | _ -> "COMPUTE:0" in
  let slots = List.map (fun device ->
      let n = 2 * (List.length (Hashtbl.find queues device) + 1
                   + if profile then 2 * Array.length calls else 0) in
      device, (U.placeholder ~shape:[n] ~dtype:Dtype.uint64 ~slot:0
        ~device:(U.Single device) ~volatile:true () |> U.with_tag "slots")) !devices in
  let slot device i = view (List.assoc device slots) (2 * i) 2 in
  let signal (device, queue) =
    let rec index i = function
      | q :: _ when q = queue -> i
      | _ :: rest -> index (i + 1) rest
      | [] -> invalid_arg "Hcq2.plan: unknown queue" in
    slot device (index 0 (Hashtbl.find queues device)) in
  let signal_tags = Hashtbl.create 16 in
  Hashtbl.iter (fun ((device, queue) as key) tag ->
      if queue <> epilogue device || Hashtbl.find peers key <> [] then
        Hashtbl.replace signal_tags tag ()) last;
  let deps = Deps_tracker.create () in
  let waits = Array.mapi (fun tag c ->
      let args, writes = arguments c.call in
      let latest = Hashtbl.create 8 in
      Deps_tracker.access deps (List.map Deps_tracker.uop args) ~writes (c.device, c.queue, tag)
      |> List.iter (fun (d, q, t) ->
          if t < tag && (d, q) <> (c.device, c.queue) then
            let old = Option.value (Hashtbl.find_opt latest (d, q)) ~default:(-1) in
            Hashtbl.replace latest (d, q) (max old t));
      if Hashtbl.length latest > 0 && String.starts_with ~prefix:"NV" c.device
         && String.starts_with ~prefix:"COMPUTE" c.queue then
        Option.iter (fun p -> Hashtbl.replace latest (c.device, c.queue) p) previous.(tag);
      Hashtbl.to_seq latest |> List.of_seq |> List.sort compare
      |> List.map (fun (key, t) ->
          Hashtbl.replace signal_tags t ();
          ins "wait" [signal key; uint (t + 1)])) calls in
  let commands = Hashtbl.create 8 and order = ref [] in
  let start ((device, _) as key) =
    let peers = Option.value (Hashtbl.find_opt peers key) ~default:[] in
    ins "barrier" [] :: List.map (fun d -> ins "wait" [timeline d; timeline_value d]) (device :: peers) in
  let append key nodes =
    if not (Hashtbl.mem commands key) then begin
      order := !order @ [key]; Hashtbl.add commands key (List.rev (start key))
    end;
    Hashtbl.replace commands key (List.rev_append nodes (Hashtbl.find commands key)) in
  Array.iteri (fun tag c ->
      let stamp i = ins "timestamp" [slot c.device
          (List.length (Hashtbl.find queues c.device) + 1 + 2 * tag + i)] in
      let nodes = waits.(tag) @ (if profile then [stamp 0] else []) @ [c.call]
        @ (if profile then [stamp 1] else [])
        @ (if Hashtbl.mem signal_tags tag then
             [ins "store" [signal (c.device, c.queue); uint (tag + 1)]] else []) in
      append (c.device, c.queue) nodes) calls;
  List.iter (fun device ->
      let queue = epilogue device in
      let others = List.filter (fun q -> q <> queue) (Hashtbl.find queues device)
        |> List.map (fun q -> device, q) in
      let foreign = Hashtbl.to_seq peers |> List.of_seq
        |> List.filter_map (fun (key, ds) -> if List.mem device ds then Some key else None)
        |> List.sort compare in
      let waits = List.map (fun key -> ins "wait" [signal key; uint (Hashtbl.find last key + 1)]) (others @ foreign) in
      let value = U.alu_binary ~op:Ops.Add ~lhs:(timeline_value device) ~rhs:(uint 1) in
      append (device, queue) (waits @ [ins "store" [timeline device; value]])) !devices;
  { queues = List.map (fun (d, q) -> d, q, List.rev (Hashtbl.find commands (d, q))) !order;
    timelines = List.map (fun d -> slot d (List.length (Hashtbl.find queues d))) !devices;
    signals = List.concat_map (fun d -> List.map (fun q -> signal (d, q)) (Hashtbl.find queues d)) !devices }

let ccall ?(host = "CPU") ?(libs = []) ?(after = []) ~name ~dtype args =
  let ptr = U.placeholder ~shape:[1] ~dtype:Dtype.uint64 ~slot:0
      ~device:(U.Single host) ~allocation:("cfunc", Marshal.to_string (libs, name) [Marshal.No_sharing]) () in
  let fxn = U.load ~src:(U.index ~ptr ~idxs:[U.const_int 0] ()) () in
  let fxn, args = match args with
    | first :: rest -> fxn, U.after ~src:first ~deps:after :: rest
    | [] -> U.after ~src:fxn ~deps:after, [] in
  U.call ~body:(U.custom_function ~name ~srcs:[fxn]) ~args
    ~info:{grad_fxn = None; name = None; precompile = false;
      precompile_backward = false; aux = None; dtype}

let tagged u = U.node_tag u <> None || match U.as_param u with
  | Some {param; _} -> param.allocation <> None | None -> false

let rec link_value u =
  match U.op u with
  | Ops.Getaddr -> List.for_all (fun p -> U.op p <> Ops.Param || tagged p)
      (U.toposort ~enter_calls:false (U.src u).(0))
  | Ops.Param -> tagged u
  | Ops.Buffer -> U.addrspace u = Some Dtype.Global
  | Ops.Load | Ops.After -> false
  | _ when U.is_variable u -> false
  | _ -> List.for_all link_value (U.children u)

let patch ?blob ?(after = []) buf rows =
  let initialized = match blob with
    | None -> buf
    | Some bytes -> U.set ~target:buf ~value:(U.binary bytes) () in
  let stores = List.map (fun (offset, value) ->
      (* Byte words already use the storage width; reinterpret the value so
         signed bytes do not turn the storage view into an elementwise cast. *)
      let value = if Dtype.itemsize (U.dtype value) = 1
        then U.bitcast ~src:value ~dtype:Dtype.uint8 else value in
      let dtype = U.dtype value in
      let target = if link_value value then initialized
        else U.after ~src:initialized ~deps:after in
      let ptr = U.bitcast ~src:(view target offset (Dtype.itemsize dtype)) ~dtype in
      U.store ~dst:(U.index ~ptr ~idxs:[U.const_int 0] ()) ~value ()) rows in
  U.after ~src:initialized ~deps:stores

let fence devices plan =
  let last = ref [] in
  List.iter2 (fun device slot ->
      let initialized = patch ~blob:(String.make (U.max_numel slot * 8) '\000') slot [] in
      let at b i = U.index ~ptr:b ~idxs:[U.const_int i] () in
      let timeline = timeline device in
      let old = timeline_value device in
      let target = U.load ~src:(at (U.after ~src:initialized ~deps:(!last @ [old])) 0) () in
      let loop = U.loop ~axis:(U.fresh_buffer_slot ()) in
      let done_ = U.load ~src:(at (U.after ~src:timeline ~deps:[target; loop]) 0) () in
      let wait = U.backedge ~body:done_ ~loop
          ~cond:(U.alu_binary ~op:Ops.Cmplt ~lhs:done_ ~rhs:target) in
      let next = U.alu_binary ~op:Ops.Add ~lhs:old ~rhs:(uint 1) in
      let bump = U.store ~dst:(at (U.after ~src:timeline ~deps:[wait]) 1) ~value:next () in
      last := [U.store ~dst:(at (U.after ~src:initialized ~deps:[bump]) 0) ~value:next ()])
    devices plan.timelines;
  List.iter (fun signal ->
      last := [U.store ~dst:(U.index ~ptr:(U.after ~src:signal ~deps:!last)
          ~idxs:[U.const_int 0] ()) ~value:(uint 0) ()]) plan.signals;
  U.group !last

let storage_views u =
  match U.op u, U.children u with
  | Ops.Bitcast, [view] when U.op view = Ops.Shrink ->
      let src = U.src view in
      let base = src.(0) in
      let old_width = Dtype.itemsize (U.dtype base) and width = Dtype.itemsize (U.dtype u) in
      (match U.const_int_value src.(1), U.const_int_value src.(2) with
       | Some offset, Some size ->
           let bytes n = Bound.(to_int (mul (int n) (int old_width))) in
           let offset = bytes offset and size = bytes size in
           if offset mod width <> 0 || size mod width <> 0
              || bytes (U.max_numel base) mod width <> 0 then None
           else Some (U.shrink ~src:(U.bitcast ~src:base ~dtype:(U.dtype u))
             ~offset:(U.const_int (offset / width))
             ~size:(U.const_int (size / width)))
       | _ -> None)
  | _ -> None

let lower_call queue devices calls sink =
  let patches = ref [] in
  let hoist u =
    if U.op u <> Ops.After then None else
      let links, rest = List.partition (fun s ->
          List.mem (U.op s) [Ops.Store; Ops.End] && link_value s) (List.tl (U.children u)) in
      if links = [] then None else begin
        patches := !patches @ links;
        Some (U.after ~src:(U.src u).(0) ~deps:rest)
      end in
  let sink = U.graph_rewrite ~name:"encode queues" queue.Device.encode sink in
  let sink = U.graph_rewrite ~name:"lower queue accesses" queue.lower sink in
  let sink = U.graph_rewrite ~name:"hoist link patches" ~enter_calls:true hoist sink in
  let addresses = U.toposort ~enter_calls:true sink |> List.filter (fun u -> U.op u = Ops.Getaddr) in
  let runtime, linked = List.partition (fun g -> not (link_value g)) addresses in
  let addresses = runtime @ linked in
  let table = U.placeholder ~shape:[max 1 (List.length addresses)] ~dtype:Dtype.uint64
      ~slot:0 ~device:(U.Single queue.host) () |> U.with_tag "inputs" in
  let mappings = List.mapi (fun i addr ->
      addr, U.load ~src:(U.index ~ptr:table ~idxs:[U.const_int i] ()) ()) addresses in
  let sink = U.substitute ~walk:true ~enter_calls:true mappings sink in
  List.iteri (fun i addr ->
      patches := !patches @ [U.store ~dst:(U.index ~ptr:table
          ~idxs:[U.const_int (List.length runtime + i)] ()) ~value:addr ()]) linked;
  let parameters = U.toposort ~enter_calls:true sink |> List.filter (fun u -> U.op u = Ops.Param) in
  let bufs, vals = List.partition (fun p -> U.addrspace p <> Some Dtype.Alu) parameters in
  let buffer_params = List.mapi (fun slot p ->
      let volatile = match U.as_param p with Some {param; _} -> param.volatile | None -> false in
      p, U.param ~slot ~dtype:(U.dtype p) ~shape:(U.const_int (U.max_numel p))
        ~device:(U.Single queue.host) ~volatile ()) bufs in
  let names = List.fold_left (fun acc p ->
      let name = Option.get (U.program_var_name p) in
      if List.mem name acc then acc else acc @ [name]) [] vals in
  let val_params = List.map (fun p -> match U.arg p with
      | U.Arg.Param_arg param -> p, U.replace p ~arg:(U.Arg.Param_arg
          {param with slot = List.length bufs +
            Option.get (List.find_index (( = ) (Option.get (U.program_var_name p))) names)}) ()
      | _ -> assert false) vals in
  let sink = U.substitute ~walk:true ~enter_calls:true (buffer_params @ val_params) sink in
  let sink = U.graph_rewrite ~name:"queue storage views" ~enter_calls:true storage_views sink in
  let program = queue.compile sink in
  let rec position i u = function
    | n :: _ when U.equal n u -> i
    | _ :: rest -> position (i + 1) u rest
    | [] -> -1 in
  let dedup xs = List.fold_left (fun acc u ->
      if List.exists (U.equal u) acc then acc else acc @ [u]) [] xs in
  let originals = List.concat_map (fun c -> fst (arguments c.call)) calls in
  let sources = List.map (fun g -> (U.src g).(0)) runtime in
  let args = dedup (bufs @ originals @ sources) in
  let written = List.concat_map (fun c ->
      let buffers, writes = arguments c.call in List.map (List.nth buffers) writes) calls |> dedup in
  let inputs = List.map (fun g ->
      let device = match U.arg g with
        | U.Arg.Device (U.Single d) -> d
        | _ -> (match U.device_of (U.src g).(0) with Some (U.Single d) -> d
                | _ -> invalid_arg "Hcq2.lower_call: address needs one device") in
      position 0 (U.src g).(0) args, device) runtime in
  let aux = U.{devices; host = queue.host; table = position 0 table bufs;
    inputs; outputs = List.map (fun u -> position 0 u args) written; kernels = List.length calls} in
  let call = U.call ~body:program ~args
      ~info:{grad_fxn = None; name = Some "hcq_submit"; precompile = false;
        precompile_backward = false; dtype = Dtype.void; aux = Some aux} in
  U.after ~src:call ~deps:!patches

let compile_batch calls =
  let plan = plan calls in
  let devices = List.fold_left (fun ds (d, _, _) ->
      if List.mem d ds then ds else ds @ [d]) [] plan.queues in
  let first = Device.get (List.hd devices) in
  let queue = Option.get (Device.queue first) in
  let fence = fence devices plan in
  let previous = ref [fence] in
  let submits = List.map (fun (device, kind, commands) ->
      let prefix = String.lowercase_ascii (List.hd (String.split_on_char ':' device)) in
      let kind = String.lowercase_ascii (List.hd (String.split_on_char ':' kind)) in
      let submit = U.custom_function ~name:("submit_" ^ prefix ^ "_" ^ kind)
          ~srcs:[U.linear commands; U.group !previous] in
      previous := [submit]; submit) plan.queues in
  let module E = Program_spec.Estimates in
  let estimates = List.fold_left (fun total c ->
      let cost = match U.as_call c.call with
        | Some {body; args} when U.op body = Ops.Store ->
            let dst = List.hd args in
            let nbytes = Bound.(to_int (mul (int (U.max_numel dst)) (int (Dtype.itemsize (U.dtype dst))))) in
            E.{ops = Int 0; lds = Int nbytes; mem = Int nbytes}
        | Some {body; _} ->
            (match U.children body with
             | sink :: _ -> (match U.as_kernel_info sink with
                 | Some {estimates = Some e; _} -> E.of_uop e | _ -> E.zero)
             | _ -> E.zero)
        | None -> E.zero in
      E.(total + cost)) E.zero calls in
  let kernel_info = U.{name = "hcq_submit"; applied_opts = []; opts_to_apply = None;
    estimates = Some (E.to_uop estimates); beam = 0} in
  (* Owned arguments can still be rebound by a consumer. Encode against
     parameters, then restore the original argument nodes outside the host
     program so execution resolves their current bindings on every call. The
     dependency plan above keeps the original allocation alias information. *)
  let nodes = U.toposort ~enter_calls:false (U.linear (List.map (fun c -> c.call) calls)) in
  let slot = List.fold_left (fun slot u -> match U.as_param u with
      | Some {param; _} -> max slot (param.slot + 1) | None -> slot) 0 nodes in
  let mappings = nodes |> List.filter (fun u ->
      U.op u = Ops.Buffer && U.addrspace u = Some Dtype.Global)
    |> List.mapi (fun i u -> u, U.param ~slot:(slot + i) ~dtype:(U.dtype u)
        ~shape:(U.const_int (U.max_numel u)) ?device:(U.device_of u) ()) in
  let substitute = U.substitute ~walk:true mappings in
  let calls = List.map (fun c -> {c with call = substitute c.call}) calls in
  let sink = substitute (U.sink ~kernel_info submits) in
  let lowered = lower_call queue devices calls sink in
  U.substitute ~walk:true (List.map (fun (a, b) -> b, a) mappings) lowered

let compile linear =
  let result = ref [] and batch = ref [] and group = ref None in
  let flush () =
    if !batch <> [] then result := compile_batch (List.rev !batch) :: !result;
    batch := []; group := None in
  let enqueue call = match U.as_call call with
    | Some {body; args} when (U.op body = Ops.Program || U.op body = Ops.Store)
        && (match U.arg call with U.Arg.Call_info {aux = None; _} -> true | _ -> false)
        && not (List.exists (fun u -> match U.device_of u with Some (U.Multi _) -> true | _ -> false) args) ->
        let args = List.filter (fun u -> not (U.is_bound_var u)) args in
        let args = if U.op body = Ops.Store then List.rev args else args in
        List.find_map (fun arg -> match U.device_of arg with
            | Some (U.Single device) ->
                let dev = Device.get device in
                (match Device.queue dev with
                 | Some q when U.op body = Ops.Program || q.copy ->
                     Some {call; device = Device.name dev;
                       queue = if U.op body = Ops.Program then "COMPUTE:0" else "COPY:0"}
                 | _ -> None)
            | _ -> None) args
    | _ -> None in
  List.iter (fun call -> match enqueue call with
      | None -> flush (); result := call :: !result
      | Some c ->
          (* Peer groups will extend this boundary when backend peer mapping is ported. *)
          if !group <> Some c.device then flush ();
          group := Some c.device; batch := c :: !batch) (U.children linear);
  flush ();
  U.linear (List.rev !result)
