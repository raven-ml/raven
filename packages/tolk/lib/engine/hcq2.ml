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
