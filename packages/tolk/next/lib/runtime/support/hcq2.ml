(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let rule = Pattern_matcher.rule
let rule_ctx = Pattern_matcher.rule_ctx
let ops = Op.Set.of_list
let strf = Printf.sprintf
let dedup l = Helpers.dedup (module Ops) l
let u64 n = int ~dtype:Dtype.Uint64 n
let ins name src = v Op.Ins ~src ~arg:(Code { code = name; dtype = Dtype.Void })
let binary s = v Op.Binary ~arg:(Bytes s)
let part u a b = shrink u [ Some (Int a, Int b) ]
let is_int o = op o = Op.Const

let const_value o =
  match Ops.value o with
  | #Dtype.value as v -> v
  | `Invalid -> invalid_arg "a word is not invalid"

let int_of o = Dtype.Value.to_int (const_value o)
let tag_is s u = match tag u with Some (Tag.String t) -> t = s | _ -> false

let devices_of u =
  match device u with
  | Some (Single d) -> [ d ]
  | Some (Multi ds) -> ds
  | None -> []

(* Command queues *)

type commands = {
  exec : Ops.t -> Ops.t -> unit;
  copy : Ops.t -> Ops.t -> int -> unit;
  wait : Ops.t -> Ops.t -> unit;
  signal : Ops.t -> Ops.t -> unit;
  timestamp : Ops.t -> unit;
  memory_barrier : unit -> unit;
  submit : Ops.t -> Ops.t;
}

module Queue = struct
  type t = {
    lin : Ops.t;
    devices : string list;
    name : string;
    mutable blob : Bytes.t;
    mutable size : int;
    mutable patches : (Ops.t * Ops.t) list;  (** Last first. *)
  }

  let make lin =
    match arg lin with
    | Queue { devices; queue } ->
        {
          lin;
          devices;
          name = queue;
          blob = Bytes.create 256;
          size = 0;
          patches = [];
        }
    | _ -> invalid_arg "a submission's commands name their queue"

  let devices q = q.devices
  let name q = q.name
  let size q = q.size

  let append q s =
    let n = String.length s in
    if q.size + n > Bytes.length q.blob then begin
      let b = Bytes.create (max (q.size + n) (2 * Bytes.length q.blob)) in
      Bytes.blit q.blob 0 b 0 q.size;
      q.blob <- b
    end;
    Bytes.blit_string s 0 q.blob q.size n;
    q.size <- q.size + n

  let contents q = Bytes.sub_string q.blob 0 q.size

  let le n z =
    String.init n (fun i -> Char.chr (Z.to_int (Z.extract z (8 * i) 8)))

  let q q words =
    List.iter
      (fun w ->
        let rec uncast c = if op c = Op.Cast then uncast (nth c 0) else c in
        let c = uncast w and n = Dtype.itemsize (dtype w) in
        match (op c, arg c) with
        | Op.Binary, Bytes s -> append q s
        | Op.Const, _ -> append q (le n (Dtype.Value.to_z (const_value c)))
        | _ ->
            q.patches <- (int q.size, w) :: q.patches;
            append q (String.make n '\000'))
      words;
    q.size

  let dword n = int ~dtype:Dtype.Uint32 (n land 0xffff_ffff)

  let get_dword q off =
    Int32.to_int (Bytes.get_int32_le q.blob off) land 0xffff_ffff

  let set_dword q off n = Bytes.set_int32_le q.blob off (Int32.of_int n)

  let reset q =
    q.size <- 0;
    q.patches <- []
end

(* A queue runs its commands in order: each call and instruction, and a range
   around commands, which repeats their bytes and moves each repeated word by
   its trip. *)
let rec encode_command q cmds u =
  let bad () =
    invalid_arg (Format.asprintf "a queue has no command for %a" Op.pp (op u))
  in
  match (op u, src u) with
  | Op.Call, body :: args -> (
      match (op body, args) with
      | Op.Program, _ -> cmds.exec u body
      | Op.Store, dst :: src :: _ ->
          cmds.copy dst src (max_numel src * Dtype.itemsize (dtype src))
      | _ -> bad ())
  | Op.Ins, s -> (
      match (arg u, s) with
      | Code { code = "barrier"; _ }, [] -> cmds.memory_barrier ()
      | Code { code = "wait"; _ }, [ dst; value ] -> cmds.wait dst value
      | Code { code = "timestamp"; _ }, [ dst ] -> cmds.timestamp dst
      | Code { code = "store"; _ }, [ dst; value ] -> cmds.signal dst value
      | _ -> bad ())
  | Op.End, [ body; r ] when op body = Op.Linear && op r = Op.Range ->
      loop q cmds body r
  | _ -> bad ()

and loop q cmds body r =
  let start = q.Queue.size and first = List.length q.patches in
  List.iter (encode_command q cmds) (src body);
  let trip = q.size - start in
  let fresh, older =
    List.partition_map
      (fun (i, p) ->
        if i < List.length q.patches - first then Left p else Right p)
      (List.mapi (fun i p -> (i, p)) q.patches)
  in
  (* Each trip gets its words as dwords: a trip need not be 8-byte aligned. *)
  let dwords =
    List.concat_map
      (fun (o, w) ->
        List.init
          (Dtype.itemsize (dtype w) / 4)
          (fun k ->
            ( add (add o (int (4 * k))) (mul r (int trip)),
              cast (shr w (int (32 * k))) Dtype.Uint32 )))
      (List.rev fresh)
  in
  q.patches <- List.rev_append dwords older;
  let body = Bytes.sub_string q.blob start trip in
  for _ = 1 to Dtype.Value.to_int (vmax r) do
    Queue.append q body
  done

(* Helpers *)

(* A view's storage and byte offset; the offset of a view inside a range around
   a call reads the range's variable. *)
let rec view_offset v =
  match op v with
  | Op.Bitcast | Op.After -> view_offset (nth v 0)
  | Op.Shrink -> (
      let base, off = view_offset (nth v 0) in
      match marg v with
      | Shrink [ (o, _) ] ->
          (base, Sint.(off + (o * Int (Dtype.itemsize (dtype v)))))
      | _ -> invalid_arg "a view of storage is one-dimensional")
  | _ -> (v, Int 0)

let lane_offset v =
  let sel, off = view_offset v in
  if op sel <> Op.Mselect then (sel, None, off)
  else
    let base, inner = view_offset (nth sel 0) in
    let lane = match arg sel with Shard i -> Some i | _ -> None in
    (base, lane, Sint.(off + inner))

let const_offset = function
  | Int o -> o
  | Sym _ -> invalid_arg "a view of storage is a shrink by constants"

let unwrap_view v =
  let base, off = view_offset v in
  (base, const_offset off)

let unwrap_lane v =
  let base, lane, off = lane_offset v in
  (base, lane, const_offset off)

let select_lane u lane =
  if op u = Op.Mstack then List.nth (src u) lane
  else if List.length (devices_of u) > 1 then mselect u lane
  else u

let to_name parts =
  String.map
    (function ':' -> '_' | c -> c)
    (String.lowercase_ascii (String.concat "_" parts))

let signal_word d =
  placeholder ~slot:0 ~device:(Multi [ d ]) ~volatile:true
    ~tag:(Tag.String "timeline") [ 1 ] Dtype.Uint64

(* Timeline values are variables of the host program, which the engine binds on
   each run: only the runtime writes the submitted value (D1). *)
let timeline_var what d =
  variable ~dtype:Dtype.Uint64
    (to_name [ what; d ])
    (`Int Z.zero)
    (`Int Z.(shift_left one 62 - one))

let submitted d = timeline_var "submitted" d
let value d = timeline_var "value" d

let range_value r =
  variable
    ~dtype:(Dtype.strong (dtype r))
    ("range_" ^ range_str r)
    (`Int Z.zero) (vmax r)

(* Host functions and C structures *)

let layout_args ?(offset = 0) args =
  let signature =
    List.mapi
      (fun slot w ->
        { Device.Tiny_elf.name = None; slot; dtype = dtype w; shape = [] })
      args
  in
  List.map2
    (fun (o, _) w -> (o, w))
    (Device.Tiny_elf.iter_sig ~offset signature)
    args

let pack_args rows size =
  let rows = List.stable_sort (fun (a, _) (b, _) -> Int.compare a b) rows in
  let words, end_ =
    List.fold_left
      (fun (words, end_) (o, w) ->
        let words =
          if o <> end_ then w :: binary (String.make (o - end_) '\000') :: words
          else w :: words
        in
        (words, o + Dtype.itemsize (dtype w)))
      ([], 0) rows
  in
  List.rev (binary (String.make (size - end_) '\000') :: words)

let ccall ~host ~lib ?(ret = Dtype.Void) f args =
  let ptr =
    placeholder ~slot:0 ~device:(Single host)
      ~tag:(Tag.Tuple [ String "cfunc"; String lib; String f ])
      [ 1 ] Dtype.Uint64
  in
  call ~ret_dtype:ret (custom_function f [ load (index ptr [ int 0 ]) [] ]) args

type c_struct = {
  struct_name : string;
  struct_size : int;
  fields : (string * int * int) list;
}

(* A C field as the unsigned integer of its size. *)
let cdtype = function
  | 1 -> Dtype.Uint8
  | 2 -> Dtype.Uint16
  | 4 -> Dtype.Uint32
  | 8 -> Dtype.Uint64
  | n -> invalid_arg (strf "no C integer of %d bytes" n)

let field s f =
  match List.find_opt (fun (n, _, _) -> n = f) s.fields with
  | Some (_, off, size) -> (off, size)
  | None -> invalid_arg (strf "struct %s has no field %s" s.struct_name f)

let cfield buf s f =
  let off, size = field s f in
  index (bitcast (part buf off (off + size)) (cdtype size)) [ int 0 ]

(* Patches *)

let is_input_addr g =
  let base, _, _ = lane_offset (nth g 0) in
  op base = Op.Param && tag base = None

let rec is_link_patch w =
  match op w with
  | Op.Getaddr -> (
      (* An address that moves with a range is known only when the program
         runs. *)
      (not (is_input_addr w))
      && match lane_offset (nth w 0) with _, _, Int _ -> true | _ -> false)
  | Op.Param -> tag w <> None
  | Op.Buffer -> addrspace w = Some Dtype.Global
  | Op.Load | Op.After -> false
  | _ when is_variable w -> false
  | _ -> List.for_all is_link_patch (src w)

let patch ?blob buf rows =
  let key (o, w) =
    let dt = dtype w in
    let vmin_o = if is_int o then int_of o else Dtype.Value.to_int (vmin o) in
    ( dt,
      ((vmin_o mod Dtype.itemsize dt) + Dtype.itemsize dt) mod Dtype.itemsize dt,
      is_int o && is_link_patch w,
      if is_int o then [] else Nodes.to_list (ranges o) )
  in
  let same (d0, p0, l0, r0) (d1, p1, l1, r1) =
    Dtype.equal d0 d1 && p0 = p1 && l0 = l1 && List.equal ( == ) r0 r1
  in
  let keyed = List.map (fun r -> (key r, r)) rows in
  let groups =
    List.fold_left
      (fun ks (k, _) -> if List.exists (same k) ks then ks else k :: ks)
      [] keyed
    |> List.rev
  in
  let dep =
    Option.to_list
      (Option.map (fun b -> store buf (bitcast (binary b) (dtype buf))) blob)
  in
  (* The patches follow the blob, as link writes them. *)
  let base = after buf dep in
  let stores =
    List.map
      (fun ((dt, phase, _, rngs) as k) ->
        let grp =
          List.filter_map
            (fun (k', r) -> if same k' k then Some r else None)
            keyed
        in
        let n = Dtype.itemsize dt in
        let view =
          bitcast
            (part base phase (phase + ((max_numel buf - phase) / n * n)))
            dt
        in
        let offs =
          List.map
            (fun (o, _) ->
              if is_int o then int ((int_of o - phase) / n)
              else div ~rounding:`Floor (sub o (int phase)) (int n))
            grp
        in
        end_ (store (index view [ stack offs ]) (stack (List.map snd grp))) rngs)
      groups
  in
  after buf (dep @ stores)

let cstruct ~host s fields =
  let rows =
    List.map
      (fun (f, v) ->
        let off, size = field s f in
        (int off, ccast v (cdtype size)))
      fields
  in
  let buf =
    placeholder ~device:(Single host) ~volatile:true
      ~tag:(Tag.String s.struct_name) [ s.struct_size ] Dtype.Uint8
  in
  patch buf rows ~blob:(String.make s.struct_size '\000')

(* Devices *)

type queues = {
  commands : Queue.t -> commands;
  copy_queue : bool;
  host : string;
  reaches : string -> bool;
}

type device = { target : Helpers.Target.t; queues : queues option }

let kind devices d = (devices d).target.device
let enqueues devices d = Option.is_some (devices d).queues

let queues devices d =
  match (devices d).queues with
  | Some q -> q
  | None -> invalid_arg (strf "%s has no command queues" d)

let all_devices_in devices ds = List.for_all (enqueues devices) ds

let get_enqueue_devs devices call =
  if op call <> Op.Call then None
  else
    let b = body call in
    match (op b, Realize.get_call_arg_uops call) with
    | (Op.Program | Op.Store), (_ :: _ as bufs) ->
        (* Copies push from the source: writes to a peer are faster than
           reads. *)
        let bufs = if op b = Op.Store then List.rev bufs else bufs in
        let devs =
          match
            List.find_opt (fun b -> all_devices_in devices (devices_of b)) bufs
          with
          | Some b -> devices_of b
          | None -> devices_of (List.hd bufs)
        in
        if not (all_devices_in devices devs) then
          None (* Unified memory copies on the host. *)
        else if op b = Op.Store && kind devices (List.hd devs) = "METAL" then
          None
        else Some devs
    | _ -> None

let make_submit ~devices ~queue kind cmds =
  let fn =
    to_name [ "submit"; kind; List.hd (String.split_on_char ':' queue) ]
  in
  custom_function fn [ v Op.Linear ~src:cmds ~arg:(Queue { devices; queue }) ]

(* Unwrap multi *)

let unwrap_call devices call =
  match get_enqueue_devs devices call with
  | None -> None
  | Some _ ->
      let n =
        List.fold_left
          (fun n a -> max n (List.length (devices_of a)))
          1
          (Realize.get_call_arg_uops call)
      in
      if n = 1 then None
      else
        let dnum =
          variable ~dtype:Dtype.Int32 "_device_num" (`Int Z.zero)
            (`Int (Z.of_int (n - 1)))
        in
        let lane i =
          replace call
            ~src:
              (body call
               :: List.map
                    (fun a -> if is_bound_var a then a else select_lane a i)
                    (src_without_body call)
              @ [ bind dnum (`Int (Z.of_int i)) ])
        in
        Some (v Op.Linear ~src:(List.init n lane))

(* Staging copies *)

let staging_size = 128 lsl 20
let staging_slots = 2

let stage_copy ~devices ~lower_and_compile call dst src =
  match get_enqueue_devs devices call with
  | None -> None
  | Some devs ->
      let device = List.hd devs in
      let q = queues devices device in
      (* A device's queues address its own memory. *)
      let reached b =
        List.for_all (fun d -> d = device || q.reaches d) (devices_of b)
      in
      if not (reached dst && reached src) then begin
        let staging =
          placeholder ~slot:0 ~device:(Single q.host)
            ~tag:(Tag.String "staging") [ staging_size ] Dtype.Uint8
        in
        let it = Dtype.itemsize (dtype src) and numel = max_numel src in
        let chunk = staging_size / staging_slots / it in
        let rec copies i off =
          if off >= numel then []
          else
            let n = min chunk (numel - off) in
            let so = i mod staging_slots * chunk * it in
            let stage = part staging so (so + (n * it)) in
            store_call stage (part src off (off + n))
            :: store_call (part dst off (off + n)) stage
            :: copies (i + 1) (off + chunk)
        in
        Some (v Op.Linear ~src:(copies 0 0))
      end
      else if q.copy_queue then None
      else
        let byte i b =
          param i Dtype.Uint8 ~shape:[ Int (nbytes b) ] ~device:(Single device)
        in
        let out = byte 0 dst and inp = byte 1 src in
        let r = range (Int (nbytes src)) [ 0 ] in
        let ast =
          sink ~kernel:(kernel_info ())
            [ end_ (store (index out [ r ]) (load (index inp [ r ]) [])) [ r ] ]
        in
        Some
          (lower_and_compile
             (replace call ~src:(ast :: List.tl (Ops.src call))))

let pm_prep ~devices ~lower_and_compile =
  Pattern_matcher.concat
    [
      Pattern_matcher.v
        [
          rule (Upat.op ~name:"call" Op.Call) (fun m ->
              unwrap_call devices (m "call"));
          rule
            (Upat.op ~name:"call" ~allow_any_len:true
               ~src:[ Upat.op Op.Store; Upat.var "dst"; Upat.var "src" ]
               Op.Call)
            (fun m ->
              stage_copy ~devices ~lower_and_compile (m "call") (m "dst")
                (m "src"));
        ];
      Schedule.pm_flatten_linear;
    ]

(* Dependencies *)

module Deps = struct
  module Key = struct
    type t = Ops.t * int option

    let equal (b0, l0) (b1, l1) = b0 == b1 && l0 = l1
    let hash (b, l) = Hashtbl.hash (Ops.hash b, l)
  end

  module H = Hashtbl.Make (Key)

  (* The byte ranges of a storage and shard, each with its access. *)
  type 'a t = {
    writes : (int * int * 'a) list H.t;
    reads : (int * int * 'a) list H.t;
  }

  let make () = { writes = H.create 16; reads = H.create 16 }
  let get m k = Option.value (H.find_opt m k) ~default:[]

  let access t bufs ~writes x =
    let ranges =
      List.map
        (fun b ->
          (* A view that moves with a range covers every place it moves to. *)
          let base, lane, off = lane_offset b in
          let lo, hi =
            match off with
            | Int o -> (o, o)
            | Sym o -> Dtype.Value.(to_int (vmin o), to_int (vmax o))
          in
          ((base, lane), lo, hi + (max_numel b * Dtype.itemsize (dtype b))))
        bufs
    in
    let overlapping m (k, s, e) =
      List.filter_map
        (fun (st, en, d) -> if st < e && s < en then Some d else None)
        (List.rev (get m k))
    in
    let waits =
      List.concat
        (List.mapi
           (fun i r ->
             overlapping t.writes r
             @ if List.mem i writes then overlapping t.reads r else [])
           ranges)
    in
    List.iteri
      (fun i (k, s, e) ->
        if List.mem i writes then begin
          let trim m =
            let kept =
              List.concat_map
                (fun ((st, en, d) as entry) ->
                  if st = en then []
                  else if en <= s || e <= st then [ entry ]
                  else
                    (if e < en then [ (e, en, d) ] else [])
                    @ if st < s then [ (st, s, d) ] else [])
                (get m k)
            in
            H.replace m k kept
          in
          trim t.writes;
          trim t.reads;
          H.replace t.writes k ((s, e, x) :: get t.writes k)
        end
        else H.replace t.reads k ((s, e, x) :: get t.reads k))
      ranges;
    List.fold_left
      (fun seen d -> if List.memq d seen then seen else d :: seen)
      [] waits
    |> List.rev
end

(* Batches *)

type entry = { call : Ops.t; devs : string list; queue : string }

(* Ordered tables: the keys in the order they were first added. *)
module Ordered = struct
  type ('k, 'v) t = { mutable items : ('k * 'v) list }

  let create () = { items = [] }
  let find t k = List.assoc_opt k t.items

  let set t k x =
    if List.mem_assoc k t.items then
      t.items <-
        List.map (fun (k', v) -> if k' = k then (k', x) else (k', v)) t.items
    else t.items <- t.items @ [ (k, x) ]

  let keys t = List.map fst t.items
end

type ctx = {
  devices : string -> device;
  batch : entry array;
  profile : bool;
  tracker : (string * string * int) Deps.t;
  queues : (string, string list) Ordered.t;
      (** Each device's queues, in first use. *)
  last : (string * string, int) Ordered.t;
  prev : int option array;
  mutable signal_tags : int list;
  slots : (string * Ops.t) list;
  peers : (string * string, string list) Ordered.t;
      (** The other devices whose memory a queue touches. *)
}

let dev_queues ctx d = Option.value (Ordered.find ctx.queues d) ~default:[]

let epilogue_queue ctx dev =
  match dev_queues ctx dev with [ q ] -> q | _ -> "COMPUTE:0"

let make_ctx devices batch profile =
  let queues = Ordered.create ()
  and last = Ordered.create ()
  and peers = Ordered.create () in
  let prev =
    Array.mapi
      (fun tag { call; devs; queue } ->
        let d = List.hd devs in
        let qs = Option.value (Ordered.find queues d) ~default:[] in
        if not (List.mem queue qs) then Ordered.set queues d (qs @ [ queue ]);
        let p = Ordered.find last (d, queue) in
        Ordered.set last (d, queue) tag;
        let touched =
          List.concat_map devices_of (Realize.get_call_arg_uops call)
          |> List.filter (fun x -> enqueues devices x && x <> d)
          |> List.sort_uniq String.compare
        in
        List.iter
          (fun x ->
            let ps = Option.value (Ordered.find peers (d, queue)) ~default:[] in
            if not (List.mem x ps) then Ordered.set peers (d, queue) (ps @ [ x ]);
            if Ordered.find queues x = None then Ordered.set queues x [])
          touched;
        p)
      batch
  in
  let ctx =
    {
      devices;
      batch;
      profile;
      tracker = Deps.make ();
      queues;
      last;
      prev;
      signal_tags = [];
      slots = [];
      peers;
    }
  in
  let signal_tags =
    List.filter_map
      (fun ((dev, q), tag) ->
        if q <> epilogue_queue ctx dev || Ordered.find peers (dev, q) <> None
        then Some tag
        else None)
      last.items
  in
  (* A slot is [signal][timestamp], 16 bytes: the queue signals, then two per
     call when profiling. *)
  let slots =
    List.map
      (fun (dev, qs) ->
        let n =
          List.length qs + if profile then 2 * Array.length batch else 0
        in
        ( dev,
          placeholder ~device:(Multi [ dev ]) ~volatile:true
            ~tag:(Tag.String "slots")
            [ 2 * n ]
            Dtype.Uint64 ))
      queues.items
  in
  { ctx with signal_tags; slots }

let slot ctx dev i = part (List.assoc dev ctx.slots) (2 * i) ((2 * i) + 2)

let queue_signal ctx dev queue =
  let rec pos i = function
    | [] -> invalid_arg (strf "%s has no queue %s" dev queue)
    | q :: _ when q = queue -> i
    | _ :: qs -> pos (i + 1) qs
  in
  slot ctx dev (pos 0 (dev_queues ctx dev))

let stamps ctx dev tag =
  if ctx.profile then
    let st = List.length (dev_queues ctx dev) + (2 * tag) in
    [ st; st + 1 ]
  else []

let wait_ins ctx { call; devs; queue } tag =
  let device = List.hd devs in
  let bufs = Realize.get_call_arg_uops call
  and writes = fst (Realize.get_call_outs_ins call) in
  (* The latest call to wait on of each producer's queue: calls of one queue run
     in order. *)
  let latest = Ordered.create () in
  List.iter
    (fun (d, q, t) ->
      if t < tag && (d, q) <> (device, queue) then
        Ordered.set latest (d, q)
          (max t (Option.value (Ordered.find latest (d, q)) ~default:0)))
    (Deps.access ctx.tracker bufs ~writes (device, queue, tag));
  (* On NV, a wait breaks the chaining of launches, so the queue also waits for
     its previous launch. *)
  (match ctx.prev.(tag) with
  | Some p
    when latest.items <> []
         && kind ctx.devices device = "NV"
         && String.starts_with ~prefix:"COMPUTE" queue ->
      Ordered.set latest (device, queue) p
  | _ -> ());
  ctx.signal_tags <- List.map snd latest.items @ ctx.signal_tags;
  List.map
    (fun ((d, q), t) -> ins "wait" [ queue_signal ctx d q; u64 (t + 1) ])
    latest.items

(* A queue first waits for the earlier work of its device and of the peers it
   touches. *)
let start_ins ctx dev queue =
  ins "barrier" []
  :: List.map
       (fun d -> ins "wait" [ signal_word d; submitted d ])
       (dev
       :: List.sort String.compare
            (Option.value (Ordered.find ctx.peers (dev, queue)) ~default:[]))

let build_queues ctx =
  (* All the waits first, to find the calls that must signal. *)
  let call_waits = Array.mapi (fun tag e -> wait_ins ctx e tag) ctx.batch in
  let queues = Ordered.create () in
  let extend k cmds =
    Ordered.set queues k
      (Option.value (Ordered.find queues k) ~default:[] @ cmds)
  in
  Array.iteri
    (fun tag { call; devs; queue } ->
      let k = (devs, queue) in
      if Ordered.find queues k = None then
        extend k (start_ins ctx (List.hd devs) queue);
      let ts =
        List.map
          (fun i -> ins "timestamp" [ slot ctx (List.hd devs) i ])
          (stamps ctx (List.hd devs) tag)
      in
      let before, after_ =
        match ts with [ a; b ] -> ([ a ], [ b ]) | _ -> ([], [])
      in
      extend k (call_waits.(tag) @ before @ [ call ] @ after_);
      if List.mem tag ctx.signal_tags then
        extend k
          [
            ins "store" [ queue_signal ctx (List.hd devs) queue; u64 (tag + 1) ];
          ])
    ctx.batch;
  (* One queue advances the device's timeline once its other queues are done,
     and those of the peers that touched the device. *)
  List.iter
    (fun dev ->
      let queue = epilogue_queue ctx dev in
      let others =
        List.filter_map
          (fun q -> if q <> queue then Some (dev, q) else None)
          (dev_queues ctx dev)
        @ List.sort Stdlib.compare
            (List.filter_map
               (fun (k, ds) -> if List.mem dev ds then Some k else None)
               ctx.peers.items)
      in
      let waits =
        List.map
          (fun (d, q) ->
            ins "wait"
              [
                queue_signal ctx d q;
                u64 (Option.get (Ordered.find ctx.last (d, q)) + 1);
              ])
          others
      in
      let bump = ins "store" [ signal_word dev; value dev ] in
      let k = ([ dev ], queue) in
      (* Several copy queues may need a compute stream of their own; a peer
         without calls starts like any queue. *)
      if
        Ordered.find queues k = None
        && not (List.exists (fun (d, _) -> d = dev) (Ordered.keys ctx.last))
      then extend k (start_ins ctx dev queue);
      extend k (waits @ [ bump ]))
    (Ordered.keys ctx.queues);
  queues.items

let finalize_batch ctx =
  let queues = build_queues ctx in
  (* Re-arm the batch's signals before submitting the queues in first use. *)
  let signals =
    List.concat_map
      (fun (dev, qs) -> List.map (queue_signal ctx dev) qs)
      ctx.queues.items
  in
  let fence = custom_function "hcq_fence" signals in
  let submits =
    List.fold_left
      (fun submits ((devs, queue), cmds) ->
        let prev = match submits with [] -> [] | s :: _ -> [ s ] in
        after
          (make_submit ~devices:devs ~queue
             (kind ctx.devices (List.hd devs))
             cmds)
          (fence :: prev)
        :: submits)
      [] queues
    |> List.rev
  in
  let sink =
    sink ~tag:(Tag.Int 1)
      ~kernel:
        (kernel_info ~name:"hcq_submit" ~estimates:Renderer.Estimates.zero ())
      submits
  in
  let kernel tag { call; devs; _ } =
    let args = Realize.get_call_arg_uops call in
    let globals =
      match arg (body call) with
      | Program p -> p.globals
      | _ -> List.init (List.length args) Fun.id
    in
    let lanes = List.map (fun g -> lane_offset (List.nth args g)) globals in
    let outs, ins = Realize.get_call_outs_ins call in
    {
      devices = devs;
      name = Realize.get_call_name call args;
      estimates = Realize.estimate_uop call;
      stamps = List.map (fun s -> (2 * s) + 1) (stamps ctx (List.hd devs) tag);
      profile_key =
        (if op (body call) = Op.Program then Some (key (body call)) else None);
      input_slots =
        (if
           List.for_all
             (fun (b, lane, _) -> op b = Op.Param && lane = None)
             lanes
         then
           List.map
             (fun (b, _, _) ->
               match arg b with Param p -> p.slot | _ -> assert false)
             lanes
         else []);
      outs;
      ins;
    }
  in
  let kernels = Array.to_list (Array.mapi kernel ctx.batch) in
  let info =
    {
      device = Ordered.keys ctx.queues;
      kernels;
      estimates =
        Renderer.Estimates.simplify
          (List.fold_left
             (fun acc (k : hcq_kernel) ->
               Renderer.Estimates.add acc k.estimates)
             Renderer.Estimates.zero kernels);
      nargs = 0;
      table = -1;
      inputs = [];
      slots = [];
      written_bufs =
        dedup
          (List.concat_map
             (fun e -> Realize.get_call_written_bufs e.call)
             (Array.to_list ctx.batch));
    }
  in
  call ~aux:info sink (if ctx.profile then List.map snd ctx.slots else [])

let rec sched_batches ~devices ~profile l =
  (* The calls in a range are batched on their own and run once per trip: their
     batch reads the range as a variable. *)
  let trip e =
    if op e <> Op.End then e
    else
      let body = nth e 0 and rs = List.tl (src e) in
      let vars = List.map (fun r -> (r, range_value r)) rs in
      let calls = if op body = Op.Linear then src body else [ body ] in
      let inner =
        sched_batches ~devices ~profile
          (v Op.Linear ~src:(List.map (fun c -> substitute c vars) calls))
      in
      end_ (match src inner with [ c ] -> c | _ -> inner) rs
  in
  let entries = List.map trip (src l) in
  let devs = List.map (get_enqueue_devs devices) entries in
  let is_copy c = op c = Op.Call && op (body c) = Op.Store in
  let peers =
    List.concat_map
      (fun c ->
        if is_copy c then
          List.concat_map devices_of (Realize.get_call_arg_uops c)
          |> List.filter (fun d -> kind devices d = "AMD")
        else [])
      entries
    |> List.sort_uniq String.compare
  in
  let npeers = List.length peers in
  let num_queues =
    max 1
      (Helpers.getenv "HCQ_NUM_SDMA"
         (if Helpers.Context_var.value Helpers.all2all >= 1 then min npeers 8
          else 1))
  in
  let index d =
    let rec go i = function
      | [] -> raise Not_found
      | x :: xs -> if x = d then i else go (i + 1) xs
    in
    go 0 peers
  in
  let queue c =
    if op c = Op.Call && op (body c) = Op.Program then "COMPUTE:0"
    else if
      is_copy c
      && List.for_all
           (fun b ->
             match device b with
             | Some (Single d) -> List.mem d peers
             | _ -> false)
           (Realize.get_call_arg_uops c)
    then
      let dev i = List.hd (devices_of (nth c i)) in
      let m a n = ((a mod n) + n) mod n in
      strf "COPY:%d"
        (m (m (index (dev 1) - index (dev 2) - 1) npeers) num_queues)
    else "COPY:0"
  in
  (* Runs of calls enqueued or not; the enqueued ones batch by kind of device,
     in the order the kinds first appear. *)
  let rec runs acc = function
    | [] -> List.rev acc
    | (c, d) :: rest -> (
        match acc with
        | (hcq, grp) :: acc' when hcq = Option.is_some d ->
            runs ((hcq, (c, d) :: grp) :: acc') rest
        | _ -> runs ((Option.is_some d, [ (c, d) ]) :: acc) rest)
  in
  let batched =
    List.concat_map
      (fun (hcq, grp) ->
        let grp = List.rev grp in
        if not hcq then List.map fst grp
        else
          let groups = Ordered.create () in
          List.iter
            (fun (c, d) ->
              let d = Option.get d in
              let k = kind devices (List.hd d) in
              let e = { call = c; devs = d; queue = queue c } in
              Ordered.set groups k
                (Option.value (Ordered.find groups k) ~default:[] @ [ e ]))
            grp;
          List.map
            (fun (_, batch) ->
              finalize_batch (make_ctx devices (Array.of_list batch) profile))
            groups.items)
      (runs [] (List.combine entries devs))
  in
  replace l ~src:batched

(* Encoding *)

let rec bufferize_cmdbuf q name (device : Ops.device) =
  let stream = Queue.contents q and patches = List.rev q.Queue.patches in
  (* One loop writes the words used at several offsets. *)
  let rt =
    List.filter
      (fun (o, w) ->
        is_int o
        && int_of o mod 4 = 0
        && List.mem (Dtype.itemsize (dtype w)) [ 4; 8 ]
        && not (is_link_patch w))
      patches
    |> List.stable_sort (fun (_, a) (_, b) -> String.compare (key a) (key b))
  in
  let uses =
    List.fold_left
      (fun uses (o, w) ->
        match uses with
        | (w', at) :: rest when w' == w -> (w', o :: at) :: rest
        | _ -> (w, [ o ]) :: uses)
      [] rt
    |> List.rev_map (fun (w, at) -> (w, List.rev_map int_of at))
  in
  let looped =
    List.filter_map
      (fun (w, at) ->
        let rngs = Nodes.to_list (ranges w) in
        if List.length at > 1 || rngs <> [] then
          let r =
            match rngs with
            | r :: _ -> r
            | [] -> range (Int (List.length at)) [ unique_num () ]
          in
          Some (w, (at, r))
        else None)
      uses
  in
  let dwords =
    List.concat_map
      (fun (w, (at, r)) ->
        let table =
          String.concat "" (List.map (Queue.le 4) (List.map Z.of_int at))
        in
        List.init
          (Dtype.itemsize (dtype w) / 4)
          (fun k ->
            ( add
                (load (index (bitcast (binary table) Dtype.Uint32) [ r ]) [])
                (int (4 * k)),
              cast (shr w (int (32 * k))) Dtype.Uint32 )))
      looped
  in
  let patches =
    List.filter (fun (_, w) -> not (List.mem_assq w looped)) patches @ dwords
  in
  (* Nested linears, such as kernel arguments, merge into a buffer per name,
     written before the stream. *)
  let nested =
    dedup
      (List.concat_map
         (fun (_, w) ->
           List.filter_map
             (fun g ->
               if op g = Op.Getaddr && op (nth g 0) = Op.Linear then
                 Some (nth g 0)
               else None)
             (toposort w))
         patches)
  in
  let lname l = match arg l with String s -> s | _ -> "" in
  let names = List.sort_uniq String.compare (List.map lname nested) in
  let bufs =
    List.map
      (fun n ->
        Queue.reset q;
        let offs =
          List.map
            (fun l ->
              let o =
                Queue.q q
                  [
                    binary
                      (String.make ((128 - (q.size mod 128)) mod 128) '\000');
                  ]
              in
              (l, (o, Queue.q q (src l))))
            (List.filter (fun l -> lname l = n) nested)
        in
        (offs, bufferize_cmdbuf q n (Multi q.devices)))
      names
  in
  let views =
    List.concat_map
      (fun (offs, buf) ->
        List.map (fun (l, (o, e)) -> (l, part (without_after buf) o e)) offs)
      bufs
  in
  let buf =
    placeholder ~device
      ~tag:(Tag.String (to_name [ name; q.name ]))
      [ String.length stream ]
      Dtype.Uint8
  in
  let words = src (substitute (Ops.sink (List.map snd patches)) views) in
  after
    (patch buf (List.combine (List.map fst patches) words) ~blob:stream)
    (List.map snd bufs)

let bufferize_cmdbuf_on = bufferize_cmdbuf
let bufferize_cmdbuf q name device = bufferize_cmdbuf_on q name (Single device)

let encode_submit devices submit =
  let q = Queue.make (nth submit 0) in
  let cmds = (queues devices (List.hd q.devices)).commands q in
  List.iter (encode_command q cmds) (src q.lin);
  cmds.submit (bufferize_cmdbuf_on q "cmdbuf" (Multi q.devices))

(* The fence re-arms the batch's queue signals. Waiting for the batch's previous
   run is the engine's, before it runs the host program (D1). *)
let hcq_fence f =
  (* The re-arming follows the run's timeline values, as it follows the
     timeline's loads without D1: it happens on every run, never at link. *)
  let timelines =
    List.map submitted
      (List.sort_uniq String.compare
         (List.map (fun s -> List.hd (devices_of s)) (src f)))
  in
  let last =
    List.fold_left
      (fun last sig_ ->
        let base, off = unwrap_view sig_ in
        let i = off / Dtype.itemsize (dtype sig_) in
        [
          store
            (index (after base last) [ int i ])
            (const ~dtype:(dtype base) (`Int Z.zero));
        ])
      timelines (src f)
  in
  match last with
  | [ s ] -> barrier s []
  | _ -> invalid_arg "a fence re-arms signals"

let pm_hcq_encode devices =
  Pattern_matcher.v
    [
      rule (Upat.op ~name:"submit" ~allow_any_len:true Op.Custom_function)
        (fun m ->
          let s = m "submit" in
          match (arg s, src s) with
          | String "hcq_fence", _ -> Some (hcq_fence s)
          | String name, [ lin ]
            when String.starts_with ~prefix:"submit_" name && op lin = Op.Linear
            ->
              Some (encode_submit devices s)
          | _ -> None);
      (* Once blocks are lowered, the stores they make are chained in their
         order. *)
      rule
        (Upat.after ~name:"a" ~allow_any_len:true
           (Upat.var ~dtype:[ Dtype.Void ] "root")
           [])
        (fun m ->
          let root = m "root" and deps = List.tl (src (m "a")) in
          Some
            (substitute ~walk:true root
               (List.filter_map
                  (fun s ->
                    if op s = Op.Store then
                      Some (buf_uop s, after (buf_uop s) deps)
                    else None)
                  (toposort root))));
    ]

(* The words known at link leave the host program: link writes them. *)
let pm_patches =
  Pattern_matcher.v
    [
      rule_ctx (Upat.op ~name:"a" Op.After) (fun ctx m ->
          let a = m "a" in
          let links, rest =
            List.partition
              (fun s -> op s = Op.Store && is_link_patch s)
              (List.tl (src a))
          in
          if links = [] then None
          else begin
            ctx := !ctx @ links;
            Some (after (nth a 0) rest)
          end);
    ]

(* Lowering a batch *)

let bitcast_view x v b =
  match marg v with
  | Shrink [ (Int o, Int n) ] ->
      let k = Dtype.itemsize (dtype x) and m = Dtype.itemsize (dtype b) in
      if o * k mod m = 0 && n * k mod m = 0 && max_numel x * k mod m = 0 then
        Some (part (bitcast x (dtype b)) (o * k / m) ((o + n) * k / m))
      else None
  | _ -> None

let pm_views =
  Pattern_matcher.v
    [
      (* A shrink of a shrink is one shrink. *)
      rule
        (Upat.f ~name:"s" ~allow_any_len:true
           (Upat.op ~name:"x" Op.Shrink)
           Op.Shrink)
        (fun m ->
          match (marg (m "x"), marg (m "s")) with
          | Shrink xs, Shrink ss ->
              Some
                (mop
                   (nth (m "x") 0)
                   (Shrink
                      (List.map2 (fun (o, _) (p, n) -> (Sint.(o + p), n)) xs ss)))
          | _ -> None);
      (* A bitcast of a one-dimensional view of storage is a view of the
         bitcast, so movements fold the view into the index. *)
      rule
        (Upat.named "b"
           (Upat.bitcast
              (Upat.f ~name:"v" ~allow_any_len:true
                 (Upat.or_after ~name:"x"
                    (Upat.v ~op:(ops [ Op.Param; Op.Buffer ]) ()))
                 Op.Shrink)))
        (fun m ->
          if List.length (shape (m "v")) = 1 then
            bitcast_view (m "x") (m "v") (m "b")
          else None);
    ]

let pm_renumber =
  Pattern_matcher.v
    [
      rule_ctx (Upat.op ~name:"u" Op.Range) (fun next m ->
          let u = m "u" in
          match arg u with
          | Range { axis_id = _ :: rest; axis_type } ->
              incr next;
              Some
                (replace u
                   ~arg:(Range { axis_id = (!next - 1) :: rest; axis_type }))
          | _ -> None);
      rule_ctx (Upat.op ~name:"u" Op.Buffer) (fun next m ->
          let u = m "u" in
          match arg u with
          | Param p when p.addrspace = Some Dtype.Reg ->
              incr next;
              Some (replace u ~arg:(Param { p with slot = !next - 1 }))
          | _ -> None);
    ]

let getaddr_device g =
  match arg g with
  | Device (Single d) | Device (Multi (d :: _)) -> d
  | _ -> invalid_arg "an address is taken on a device"

let batch_info call =
  match arg call with
  | Call { aux = Some info; _ } -> info
  | _ -> invalid_arg "a batch is a call that submits command queues"

let lower_call ~devices call =
  let info = batch_info call in
  if info.nargs <> 0 then invalid_arg "the batch is lowered already";
  let host = (queues devices (List.hd info.device)).host in
  let lt_patches = ref [] in
  let body =
    graph_rewrite ~ctx:lt_patches ~bpm:pm_patches (body call)
      (Pattern_matcher.with_ctx (pm_hcq_encode devices))
  in
  let body =
    graph_rewrite ~ctx:lt_patches ~bpm:pm_patches body (Pattern_matcher.v [])
  in
  (* An address is its storage's and a byte offset: afters drop, since an
     address depends on nothing, and views share their storage's slot. *)
  (* An offset that moves with a range is added when the program runs; the
     table holds the address it moves from. *)
  let normalize g =
    let base, off = view_offset (nth g 0) in
    let off, moving =
      match off with Int o -> (o, None) | Sym o -> (0, Some o)
    in
    ( getaddr ~device:(getaddr_device g)
        (part (bitcast base Dtype.Uint8) off (nbytes base)),
      moving )
  in
  let normalized =
    List.filter_map
      (fun g -> if op g = Op.Getaddr then Some (g, normalize g) else None)
      (toposort body)
  in
  (* Addresses load from a table: the inputs' written on each run, the others at
     link. *)
  let input_addrs, link_addrs =
    List.partition is_input_addr
      (dedup (List.map (fun (_, (n, _)) -> n) normalized))
  in
  let addrs = input_addrs @ link_addrs in
  let table =
    placeholder ~device:(Single host) ~tag:(Tag.String "inputs")
      [ List.length addrs ]
      Dtype.Uint64
  in
  let slot_of g =
    let rec go i = function
      | [] -> raise Not_found
      | x :: xs -> if x == g then i else go (i + 1) xs
    in
    go 0 addrs
  in
  let body =
    substitute body
      (List.map
         (fun (g, (n, moving)) ->
           let addr = load (index table [ int (slot_of n) ]) [] in
           match moving with
           | None -> (g, addr)
           | Some o -> (g, add addr (cast o Dtype.Uint64)))
         normalized)
  in
  (lt_patches :=
     !lt_patches
     @
     match
       src
         (patch table (List.map (fun g -> (int (8 * slot_of g), g)) link_addrs))
     with
     | _ :: stores -> stores
     | [] -> []);
  (* The placeholders of one tag, device, type and volatility become views of
     one, each 128-byte aligned. *)
  let words =
    List.filter
      (fun u ->
        op u = Op.Param
        && (match tag u with
          | None | Some (Tag.String "program") -> false
          | _ -> true)
        && match arg u with Param p -> p.slot <> 0 | _ -> false)
      (toposort body)
  in
  let pkey u =
    match arg u with
    | Param p -> (tag u, device u, dtype u, p.volatile)
    | _ -> assert false
  in
  let same_key (t0, d0, dt0, v0) (t1, d1, dt1, v1) =
    Option.equal Tag.equal t0 t1
    && Option.equal equal_device d0 d1
    && Dtype.equal dt0 dt1 && v0 = v1
  in
  let groups =
    List.fold_left
      (fun ks u ->
        if List.exists (same_key (pkey u)) ks then ks else pkey u :: ks)
      [] words
    |> List.rev_map (fun k -> List.filter (fun u -> same_key (pkey u) k) words)
  in
  let views =
    List.concat_map
      (fun g ->
        match g with
        | [] | [ _ ] -> []
        | g0 :: _ ->
            let sizes =
              List.map
                (fun u ->
                  Helpers.round_up (nbytes u) 128 / Dtype.itemsize (dtype u))
                g
            in
            let total = List.fold_left ( + ) 0 sizes in
            let merged =
              match arg g0 with
              | Param p -> replace g0 ~arg:(Param { p with size = Some total })
              | _ -> assert false
            in
            snd
              (List.fold_left2
                 (fun (o, vs) u n ->
                   (o + n, (u, part merged o (o + max_numel u)) :: vs))
                 (0, []) g sizes))
      groups
  in
  let body =
    substitute
      ~extra_pm:
        (Pattern_matcher.with_ctx
           (Pattern_matcher.concat [ Prepare.pm_mops; pm_views ]))
      ~enter_calls:true body views
  in
  let patches = src (substitute (Ops.sink (dedup !lt_patches)) views) in
  (* The placeholders become the body's parameters in visit order, and the
     variables follow them, by name. *)
  let bufs, alus =
    List.partition
      (fun u -> tag u <> None)
      (List.filter (fun u -> op u = Op.Param) (toposort body))
  in
  let bufs = dedup (src_without_body call @ bufs) in
  let names =
    List.fold_left
      (fun ns a -> if List.mem (expr a) ns then ns else ns @ [ expr a ])
      [] alus
  in
  let params =
    List.mapi
      (fun i b ->
        let volatile, name =
          match arg b with Param p -> (p.volatile, p.name) | _ -> (false, None)
        in
        ( b,
          param i (dtype b) ~shape:(shape b) ~device:(Single host) ~volatile
            ~name:(strf "%s_%d" (Option.value name ~default:"None") i) ))
      bufs
  in
  let vals =
    List.map
      (fun a ->
        let rec pos i = function
          | [] -> raise Not_found
          | n :: ns -> if n = expr a then i else pos (i + 1) ns
        in
        match arg a with
        | Param p ->
            ( a,
              replace a
                ~arg:(Param { p with slot = List.length bufs + pos 0 names }) )
        | _ -> assert false)
      alus
  in
  let sink =
    graph_rewrite ~ctx:(ref 0) ~walk:true ~enter_calls:true
      (substitute ~enter_calls:true body (params @ vals))
      pm_renumber
  in
  let index_of b =
    let rec go i = function
      | [] -> -1
      | x :: xs -> if x == b then i else go (i + 1) xs
    in
    go 0 bufs
  in
  let info =
    {
      info with
      nargs = List.length bufs;
      table = index_of table;
      inputs =
        List.map
          (fun g ->
            let base, off = unwrap_view (nth g 0) in
            (base, off, getaddr_device g))
          input_addrs;
      slots =
        List.filter_map
          (fun (i, b) ->
            if tag_is "slots" b then Some (List.hd (devices_of b), i) else None)
          (List.mapi (fun i b -> (i, b)) bufs);
    }
  in
  let call_arg =
    match arg call with Call c -> Call { c with aux = Some info } | a -> a
  in
  after (replace call ~src:(sink :: bufs) ~arg:call_arg) patches

let pm_encode devices =
  Pattern_matcher.v
    [
      rule
        (Upat.op ~name:"call" ~allow_any_len:true
           ~src:[ Upat.op Op.Sink ]
           Op.Call)
        (fun m ->
          let c = m "call" in
          match arg c with
          | Call { aux = Some { nargs = 0; _ }; _ } ->
              Some (lower_call ~devices c)
          | _ -> None);
    ]

(* Compiling *)

let rec is_batch c =
  let c = without_after c in
  match (op c, arg c) with
  | Op.End, _ -> is_batch (nth c 0)
  | Op.Linear, _ -> List.exists is_batch (src c)
  | _, Call { aux = Some _; _ } -> true
  | _ -> false

let hcq_compile ~devices ~lower_and_compile ~profile linear =
  if List.exists is_batch (src linear) then linear
  else
    let linear =
      graph_rewrite ~ctx:() linear (pm_prep ~devices ~lower_and_compile)
    in
    let lin =
      graph_rewrite ~ctx:() ~walk:true
        (sched_batches ~devices ~profile linear)
        (pm_encode devices)
    in
    Helpers.context
      [ B (Helpers.emulated_dtypes, []) ]
      (fun () -> lower_and_compile lin)

(* A kernel that asks for no beam search asks for one of the setting's width. *)
let pm_beam width =
  Pattern_matcher.v
    [
      rule
        (Upat.op ~name:"call" ~allow_any_len:true
           ~src:[ Upat.op ~name:"sink" Op.Sink ]
           Op.Call)
        (fun m ->
          match arg (m "sink") with
          | Kernel k when k.beam = 0 ->
              let sink =
                replace (m "sink") ~arg:(Kernel { k with beam = width })
              in
              Some (replace (m "call") ~src:(sink :: List.tl (src (m "call"))))
          | _ -> None);
    ]

let compile_linear ?search ?profile ~devices linear =
  let profile =
    Option.value profile ~default:(Helpers.Context_var.value Helpers.debug >= 2)
  in
  let targets d = (devices d).target in
  let lower_and_compile = Realize.lower_and_compile ?search ~targets in
  let width = Helpers.Context_var.value Helpers.beam in
  let linear =
    if width >= 1 then graph_rewrite ~ctx:() ~walk:true linear (pm_beam width)
    else linear
  in
  hcq_compile ~devices ~lower_and_compile ~profile (lower_and_compile linear)
