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
  timestamps : (string * U.t * U.t) list;
  timelines : U.t list;
  independent_accesses : (U.t * U.t) list;
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

(* Vector clocks retain the actual FIFO/wait ordering. Only accesses in
   unordered calls need a runtime non-aliasing check; donation within a queue
   and reuse behind a cross-queue wait remain legal. *)
let independent_accesses calls predecessors =
  let keys = Array.to_list calls |> List.map (fun c -> c.device, c.queue)
      |> List.sort_uniq compare |> Array.of_list in
  let queue c = Option.get (Array.find_index (( = ) (c.device, c.queue)) keys) in
  let clocks = Array.init (Array.length calls) (fun _ -> Array.make (Array.length keys) (-1)) in
  let history = Array.make (Array.length keys) [] in
  let accesses = Array.map (fun c ->
      let args, writes = arguments c.call in
      List.mapi (fun i arg -> arg, List.mem i writes) args) calls in
  let pairs = Hashtbl.create 16 in
  Array.iteri (fun tag c ->
      let own = queue c and clock = clocks.(tag) in
      List.iter (fun prior ->
          Array.iteri (fun q t -> clock.(q) <- max clock.(q) t) clocks.(prior))
        predecessors.(tag);
      Array.iteri (fun q previous ->
          let rec visit = function
            | prior :: rest when prior > clock.(q) ->
                List.iter (fun (a, wa) -> List.iter (fun (b, wb) ->
                    if wa || wb then begin
                      let a, b = if U.tag a < U.tag b then a, b else b, a in
                      Hashtbl.replace pairs (U.tag a, U.tag b) (a, b)
                    end) accesses.(prior)) accesses.(tag);
                visit rest
            | _ -> () in
          visit previous) history;
      clock.(own) <- tag;
      history.(own) <- tag :: history.(own)) calls;
  Hashtbl.to_seq pairs |> List.of_seq |> List.sort (fun (a, _) (b, _) -> compare a b)
    |> List.map snd

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
  let predecessors = Array.make (Array.length calls) [] in
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
      predecessors.(tag) <- Option.to_list previous.(tag)
          @ (Hashtbl.to_seq_values latest |> List.of_seq);
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
  let timestamps = if not profile then [] else Array.to_list calls |> List.mapi (fun tag c ->
      let first = List.length (Hashtbl.find queues c.device) + 1 + 2 * tag in
      c.device, slot c.device first, slot c.device (first + 1)) in
  { timestamps; independent_accesses = independent_accesses calls predecessors; queues = List.map (fun (d, q) -> d, q, List.rev (Hashtbl.find commands (d, q))) !order;
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
  | Ops.After, load :: deps when U.op load = Ops.Load ->
      (* Address-table substitution can turn an ordered address into a load.
         Keep the ordering on storage so the load executes after its deps. *)
      let src = Array.copy (U.src load) in
      src.(0) <- U.after ~src:src.(0) ~deps;
      Some (U.replace load ~src ())
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

let lower_batch queue devices calls original_calls independent_accesses timestamps sink =
  let patches = ref [] in
  let hoist u =
    if U.op u <> Ops.After then None else
      let links, rest = List.partition (fun s ->
          List.mem (U.op s) [Ops.Store; Ops.End] && link_value s) (List.tl (U.children u)) in
      if links = [] then None else begin
        patches := !patches @ links;
        Some (U.after ~src:(U.src u).(0) ~deps:rest)
      end in
  let hooks name = Option.value (Device.queue (Device.get name)) ~default:queue in
  let encode u = match U.op u, U.children u with
    | Ops.Custom_function, linear :: _ ->
        (match U.arg linear with
         | U.Arg.Device (U.Single name) -> (hooks name).Device.encode u
         | _ -> queue.Device.encode u)
    | _ -> queue.Device.encode u in
  let lower u = match U.device_of u with
    | Some (U.Single name) -> (hooks name).Device.lower u
    | _ -> queue.Device.lower u in
  let sink = U.graph_rewrite ~name:"encode queues" encode sink in
  let sink = U.graph_rewrite ~name:"lower queue accesses" lower sink in
  (* Address-table substitution must retain the writes that prepare pointed-to
     storage. The address itself is static even when its contents are patched
     on every submission. *)
  let address_dependencies u = match U.op u, U.children u with
    | Ops.Getaddr, [source] when U.op source = Ops.After ->
        Some (U.after ~src:(U.replace u ~src:[| (U.src source).(0) |] ())
          ~deps:(List.tl (U.children source)))
    | _ -> None in
  let sink = U.graph_rewrite ~name:"retain address dependencies" ~enter_calls:true
      address_dependencies sink in
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
  let originals = List.concat_map (fun c -> fst (arguments c.call)) calls
      @ List.concat_map (fun call -> fst (arguments call)) original_calls
      @ List.concat_map (fun (a, b) -> [a; b]) independent_accesses in
  let sources = List.map (fun g -> (U.src g).(0)) runtime in
  let timestamp_buffers = List.map (fun (_, start, _) -> U.buf_uop start) timestamps in
  let args = dedup (bufs @ originals @ sources @ timestamp_buffers) in
  let written = List.concat_map (fun c ->
      let buffers, writes = arguments c.call in List.map (List.nth buffers) writes) calls |> dedup in
  let inputs = List.map (fun g ->
      let device = match U.arg g with
        | U.Arg.Device (U.Single d) -> d
        | _ -> (match U.device_of (U.src g).(0) with Some (U.Single d) -> d
                | _ -> invalid_arg "Hcq2.lower_call: address needs one device") in
      position 0 (U.src g).(0) args, device) runtime in
  let fallback = List.map (fun call ->
      let original = Option.get (U.as_call call) in
      let actuals = List.map (fun arg ->
          let slot = position 0 arg args in
          if slot < 0 then arg else U.param_like arg ~slot) original.args in
      U.replace call ~src:(Array.of_list (original.body :: actuals)) ()) original_calls in
  let aux = U.{fallback; devices; host = queue.host; table = position 0 table bufs;
    timings = (if timestamps = [] then [] else List.map2 (fun (device, start, finish) (c : call) ->
      device, c.queue, position 0 (U.buf_uop start) args,
      (Deps_tracker.uop start).start / 8 + 1, (Deps_tracker.uop finish).start / 8 + 1) timestamps calls);
    independent_accesses = List.map (fun (a, b) -> position 0 a args, position 0 b args) independent_accesses;
    host_deps = List.concat_map (fun (c : call) ->
        fst (arguments c.call) |> List.filter_map (fun arg ->
          match U.device_of arg with
          | Some (U.Single owner) when not (List.mem (Device.canonicalize owner) devices) ->
              Some (Device.canonicalize owner, c.device)
          | _ -> None)) calls |> List.sort_uniq Stdlib.compare;
    inputs; outputs = List.map (fun u -> position 0 u args) written;
    accesses = List.map (fun c -> List.map (fun u -> position 0 u args)
        (fst (arguments c.call))) calls} in
  let call = U.call ~body:program ~args
      ~info:{grad_fxn = None; name = Some (Label "hcq_submit"); precompile = false;
        precompile_backward = false; dtype = Dtype.void; aux = Some aux} in
  U.after ~src:call ~deps:!patches

let lower_call ~devices sink =
  let queue = match devices with
    | [] -> invalid_arg "Hcq2.lower_call: no submission device"
    | device :: _ ->
        (match Device.queue (Device.get device) with
         | Some queue -> queue
         | None -> invalid_arg "Hcq2.lower_call: device has no compiled queue") in
  lower_batch queue devices [] [] [] [] sink

let compile_batch ~profile ~original_calls ~reordered_accesses calls =
  let plan = plan ~profile calls in
  let devices = List.fold_left (fun ds (d, _, _) ->
      if List.mem d ds then ds else ds @ [d]) [] plan.queues in
  let first = Device.get (List.hd devices) in
  let queue = Option.get (Device.queue first) in
  let fence = fence devices plan in
  let previous = ref [fence] in
  let submits = List.map (fun (device, kind, commands) ->
      let prefix = String.lowercase_ascii (List.hd (String.split_on_char ':' device)) in
      let kind = String.lowercase_ascii kind |> String.map (fun c -> if c = ':' then '_' else c) in
      let submit = U.custom_function ~name:("submit_" ^ prefix ^ "_" ^ kind)
          ~srcs:[U.replace (U.linear commands) ~arg:(U.Arg.Device (U.Single device)) (); U.group !previous] in
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
  lower_batch queue devices calls original_calls
    (plan.independent_accesses @ reordered_accesses) plan.timestamps
    (U.sink ~kernel_info submits)

let enqueue call = match U.as_call call with
  | Some {body; args} when (U.op body = Ops.Program || U.op body = Ops.Store)
      && (U.op body <> Ops.Store || match args with dst :: _ -> U.max_numel dst <> 0 | [] -> false)
      && (match U.arg call with U.Arg.Call_info {aux = None; _} -> true | _ -> false)
      && not (List.exists (fun u -> match U.device_of u with Some (U.Multi _) -> true | _ -> false) args) ->
      let args = List.filter (fun u -> not (U.is_bound_var u)) args in
      let args = if U.op body = Ops.Store then List.rev args else args in
      List.find_map (fun arg -> match U.device_of arg with
          | Some (U.Single device) ->
              let dev = Device.get device in
              (match Device.queue dev with
               | Some q ->
                   let fits = match q.max_kernel_bindings, U.as_program_info body with
                     | Some most, Some info -> List.length info.globals + List.length info.vars <= most
                     | _ -> true in
                   let queue = if U.op body <> Ops.Program then q.copy call
                     else if fits then Some "COMPUTE:0" else None in
                   Option.map (fun queue -> {call; device = Device.name dev; queue}) queue
               | None -> None)
          | _ -> None) args
  | _ -> None

(* Queue backends encode one device per CALL. Expand multi-device arguments
   before batching, sharing single-device arguments across lanes as upstream. *)
let unwrap_call call = match U.as_call call with
  | Some {body; args} when List.exists (fun arg ->
      match U.device_of arg with Some (U.Multi _) -> true | _ -> false) args ->
      let count = List.fold_left (fun n arg -> match U.device_of arg with
          | Some (U.Multi devices) -> max n (List.length devices) | _ -> n) 1 args in
      let select lane arg =
        if U.is_bound_var arg then arg else match U.op arg, U.device_of arg with
        | Ops.Mstack, _ -> (U.src arg).(lane)
        | _, Some (U.Multi _) -> U.mselect ~src:arg ~index:lane
        | _ -> arg in
      let dnum = U.variable ~name:"_device_num" ~min_val:0 ~max_val:(count - 1)
          ~dtype:Dtype.int32 () in
      let lane i = U.replace call ~src:(Array.of_list
          (body :: List.map (select i) args @ [U.bind ~var:dnum ~value:(U.const_int i)])) () in
      let first = lane 0 in
      if Option.is_none (enqueue first) then [call]
      else first :: List.init (count - 1) (fun i -> lane (i + 1))
  | _ -> [call]

(* A failed peer import becomes two ordinary queue legs through host memory.
   Allocate per prepared schedule: independent linked batches must not race
   over shared staging slots before either batch's retirement fence. *)
let stage_copies ~resolve linear =
  let module B = Device.Buffer in
  let buffers = Hashtbl.create 2 and changed = ref false in
  let staging host = match Hashtbl.find_opt buffers host with
    | Some buf -> buf
    | None ->
        let spec = {Device.Buffer_spec.default with host = true; cpu_access = true; nolru = true} in
        let buf = Device.create_buffer ~size:(128 lsl 20) ~dtype:Dtype.uint8 ~spec (Device.get host) in
        Hashtbl.add buffers host buf;
        buf in
  let expand call = match U.as_call call, enqueue call with
    | Some {body; args = [dst; src]}, Some selected when U.op body = Ops.Store ->
        let target = resolve dst and source = resolve src in
        let mapped = try
          ignore (B.addr ~device:selected.device target : nativeint);
          ignore (B.addr ~device:selected.device source : nativeint);
          true
        with Storage.Mapping_unavailable _ -> false in
        if mapped || B.base_id target = B.base_id source then [call] else begin
          let host = (Option.get (Device.queue (Device.get selected.device))).host in
          let buffer = staging host in
          let base = U.from_buffer buffer in
          let dst = U.bitcast ~src:dst ~dtype:Dtype.uint8
          and src = U.bitcast ~src ~dtype:Dtype.uint8 in
          let size = B.nbytes source and chunk = B.nbytes buffer / 2 in
          if B.nbytes target <> size || U.max_numel src <> size || U.max_numel dst <> size then
            invalid_arg "stage copy: storage size differs from its compiled extent";
          let calls = List.init (if size = 0 then 0 else 1 + (size - 1) / chunk) (fun i ->
              let offset = i * chunk and length = min chunk (size - i * chunk) in
              let slot = view base ((i mod 2) * chunk) length in
              [U.store_call ~dst:slot ~src:(view src offset length);
               U.store_call ~dst:(view dst offset length) ~src:slot]) |> List.concat in
          (* All queued legs must be able to import the staging allocation.
             Otherwise retain the ordinary bounded host-copy fallback. *)
          List.iter (fun call -> Option.iter (fun selected ->
              ignore (B.addr ~device:selected.device buffer : nativeint)) (enqueue call)) calls;
          changed := true;
          calls
        end
    | _ -> [call] in
  try
    let calls = List.concat_map expand (U.children linear) in
    if !changed then Some (U.linear calls) else None
  with Storage.Mapping_unavailable _ -> None

let compile_copy ~to_program c = match U.as_call c.call with
  | Some {body; args = [dst; src]} when U.op body = Ops.Store
      && String.starts_with ~prefix:"COMPUTE:" c.queue ->
      let device = Device.get c.device in
      let bytes arg = U.bitcast ~src:arg ~dtype:Dtype.uint8 in
      let dst = bytes dst and src = bytes src in
      let size = U.max_numel src in
      if U.max_numel dst <> size then invalid_arg "queue copy: buffer sizes differ";
      let ptr slot = U.param ~slot ~dtype:Dtype.uint8 ~shape:(U.const_int size)
          ~device:(U.Single c.device) () in
      let range = U.range ~size:(U.const_int size) ~axis:0 ~kind:Axis_type.Weak () in
      let index ptr = U.index ~ptr ~idxs:[range] () in
      let store = U.store ~dst:(index (ptr 0)) ~value:(U.load ~src:(index (ptr 1)) ()) () in
      let kernel_info = U.{name = "copy"; applied_opts = []; opts_to_apply = None;
        estimates = None; beam = 0} in
      let body = to_program device
          (U.sink ~kernel_info [U.end_ ~value:store ~ranges:[range]]) in
      {c with call = U.replace c.call ~src:[|body; dst; src|] ()}
  | _ -> c

let compile ~to_program ?(profile = false) linear =
  let linear = U.linear (List.concat_map unwrap_call (U.children linear)) in
  let peers = U.children linear |> List.concat_map (fun call -> match U.as_call call with
      | Some {body; args} when U.op body = Ops.Store ->
          List.filter_map (fun arg -> match U.device_of arg with
              | Some (U.Single name) ->
                  let name = Device.canonicalize name in
                  if List.hd (String.split_on_char ':' name) = "AMD" then Some name else None
              | _ -> None) args
      | _ -> []) |> List.sort_uniq String.compare in
  let count = max 1 (Helpers.getenv "HCQ_NUM_SDMA"
      (if Helpers.Context_var.get Helpers.all2all >= 1 then min (List.length peers) 8 else 1)) in
  let assign_copy c = match U.as_call c.call with
    | Some {body; args = [dst; src]} when U.op body = Ops.Store && c.queue = "COPY:0" ->
        let position arg = match U.device_of arg with
          | Some (U.Single name) -> List.find_index (String.equal (Device.canonicalize name)) peers
          | _ -> None in
        (match position dst, position src with
         | Some dst, Some src ->
             let n = List.length peers in
             let index = ((dst - src - 1 + n) mod n) mod count in
             {c with queue = "COPY:" ^ string_of_int index}
         | _ -> c)
    | _ -> c in
  let result = ref [] and batches = ref [] in
  let dependencies = ref (Deps_tracker.create ()) in
  let placements = Hashtbl.create 16 and next = ref 0 in
  let flush () =
    List.iter (fun (_, calls, checks) ->
        let calls, original_calls = List.split (List.rev !calls) in
        result := compile_batch ~profile ~original_calls ~reordered_accesses:!checks calls :: !result)
      !batches;
    batches := []; dependencies := Deps_tracker.create ();
    Hashtbl.clear placements; next := 0 in
  List.iter (fun call -> match enqueue call with
      | None -> flush (); result := call :: !result
      | Some c ->
          let c = compile_copy ~to_program (assign_copy c) in
          let peer_group = Device.peer_group (Device.get c.device) in
          let args, writes = arguments c.call in
          let deps = Deps_tracker.access !dependencies (List.map Deps_tracker.uop args)
              ~writes !next |> List.map (Hashtbl.find placements) in
          let candidate = List.mapi (fun i (group, _, _) -> i, group) !batches
              |> List.rev |> List.find_opt (fun (_, group) -> group = peer_group) in
          let index, calls = match candidate with
            | Some (i, _) when List.for_all (fun dependency -> dependency <= i) deps ->
                let _, calls, checks = List.nth !batches i in
                (* Moving a call across another group creates new alias
                   assumptions. Keep them in the earlier batch's runtime
                   checks, including arguments used only by the later group. *)
                List.iteri (fun j (_, other_calls, _) -> if j > i then
                    List.iter (fun (other, _) ->
                        let other_args, other_writes = arguments other.call in
                        List.iteri (fun a arg -> List.iteri (fun b other_arg ->
                            if List.mem a writes || List.mem b other_writes then
                              checks := (arg, other_arg) :: !checks) other_args) args)
                      !other_calls) !batches;
                i, calls
            | _ ->
                let calls = ref [] and checks = ref [] in
                let i = List.length !batches in
                batches := !batches @ [peer_group, calls, checks];
                i, calls in
          calls := (c, call) :: !calls;
          Hashtbl.add placements !next index;
          incr next) (U.children linear);
  flush ();
  U.linear (List.rev !result)
