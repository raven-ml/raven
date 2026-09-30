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
  | None
    when d != Nx_device.disk
         && (Nx_device.host_of d == d || Nx_device.shares_host_memory d) ->
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

(* Engine devices *)

type device = {
  device : Nx_device.t;
  compiler : Hcq2.device;
  placeholder : Ops.t -> Nx_device.Buffer.t option;
  submitting : unit -> unit;
}

(* The compiler has no queue encoder of CUDA, AMD or NV yet: those devices run
   their calls one by one. *)
let device devices name =
  let d = find "device" devices name in
  (* The disk runs no program: its copies are the runtime's. *)
  let target =
    if d == Nx_device.disk then Helpers.target "DISK" else target d
  in
  match Metal.queues devices name d with
  | Some (queues, placeholder, submitting) ->
      {
        device = d;
        compiler = { Hcq2.target; queues = Some queues };
        placeholder;
        submitting;
      }
  | None ->
      {
        device = d;
        compiler = { Hcq2.target; queues = None };
        placeholder = (fun _ -> None);
        submitting = ignore;
      }

(* Host programs *)

module Program = struct
  type t = {
    program : Nx_device.Program.t;
    buffers : Device.Tiny_elf.param list; (* in the order of the globals *)
    globals : int list; (* the argument slot of each *)
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
    | Ok program ->
        { program; buffers; globals = info.globals; vars = info.vars }
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
let fail fn fmt = Printf.ksprintf (fun m -> invalid_arg (fn ^ ": " ^ m)) fmt

module B = Nx_device.Buffer

(* A batch, as linked: its host program, the devices whose queues it submits,
   its arguments, and the values its previous run signals on each device. *)
type batch = {
  info : Ops.hcq_info;
  named : string -> Nx_device.t; (* the devices of the schedule, by name *)
  host_program : Program.t;
  queues : Nx_device.t list;
  submitting : (unit -> unit) list; (* each device's, before the host program *)
  arguments : B.t list; (* by argument slot *)
  table : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t;
  reached : B.t list; (* the linked storage its words address *)
  last : int array;
  mutable kept : B.t list; (* the borrows its run's addresses map *)
}

(* A call of a schedule, as linked: a host program with a program loaded for
   each device of its first argument, a copy, a batch, or calls run once for
   each value of ranges. *)
type call =
  | Kernel of { call : Ops.t; lanes : Program.t list }
  | Copy of { call : Ops.t; dst : Ops.t; src : Ops.t }
  | Batch of batch
  | Range of { ranges : Ops.t list; body : call list }

type t = {
  lock : Mutex.t; (* runs share the storage: one at a time *)
  calls : call list;
  params : (int * Ops.t) list; (* the parameters the runs bind, by slot *)
  storage : B.t list Ops.Tbl.t; (* by node, a buffer per device *)
  borrows : B.t list; (* the borrows link's addresses map *)
  named : string -> Nx_device.t; (* the devices of the schedule, by name *)
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

let at b off n =
  if
    (off = 0
    && B.nbytes b = n)
    [@mutate off "a view of a whole buffer is the buffer"]
  then b
  else B.view b ~offset:off Nx_dtype.Scalar.UInt8 n

(* The storage each device holds of [base]: a parameter [slots] binds, storage
   or a placeholder linked, or the stack of one view per device. *)
let rec holds storage slots vars base =
  match (Ops.op base, Ops.arg base) with
  | Op.Param, Ops.Param p when Option.is_none (Ops.tag base) -> slots.(p.slot)
  | (Op.Param | Op.Buffer), _ -> Ops.Tbl.find storage base
  | Op.Mstack, _ -> List.concat_map (view storage slots vars) (Ops.src base)
  | _ -> invalid_arg (Format.asprintf "%a is not storage" Ops.pp base)

(* The buffers of the view [u], one per device, or one shared by them all, with
   its offset's variables bound by [vars]. *)
and view storage slots vars u =
  let base, shard, off = Hcq2.lane_offset u in
  let buffers = holds storage slots vars base in
  let buffers =
    match shard with Some i -> [ List.nth buffers i ] | None -> buffers
  in
  let off = Ops.sym_infer off vars in
  List.map (fun b -> at b off (bytes u)) buffers

let sint u =
  match Ops.arg u with Ops.Const (`Int z) -> Ops.Int (Z.to_int z) | _ -> Sym u

let lane buffers i = match buffers with [ b ] -> b | bs -> List.nth bs i

(* [b]'s address on [d], through [d]'s borrow of it, which [keep] keeps. *)
let address keep d b =
  if (B.device b == d) [@mutate off "a device's borrow of its own buffer is it"]
  then B.address b
  else
    match B.borrow d b with
    | Ok m ->
        keep m;
        B.address m
    | Error why ->
        invalid_arg
          (strf "Tolk_next_engine: %s cannot address memory of %s: %s"
             (Nx_device.name d)
             (Nx_device.name (B.device b))
             why)

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

(* Link patches *)

(* The value of a word known at link, with each address resolved. *)
let rec word addr u : Dtype.value =
  let v x : Dtype.value = match word addr x with #Dtype.value as x -> x in
  match (Ops.op u, Ops.arg u) with
  | Op.Const, Ops.Const (#Dtype.value as c) -> c
  | Op.Getaddr, Ops.Device d ->
      let dn = match d with Single n | Multi (n :: _) -> n | Multi [] -> "" in
      `Int (Z.of_nativeint (addr dn (Ops.nth u 0)))
  | Op.Cast, _ -> Dtype.truncate (Ops.dtype u) (v (Ops.nth u 0))
  | Op.Bitcast, _ ->
      let x = Ops.nth u 0 in
      Dtype.bitcast (Ops.dtype x) (Ops.dtype u) (v x)
  | o, _ -> (
      match
        Ops.exec_alu o (Ops.dtype u)
          (List.map (fun x -> (v x :> Dtype.const)) (Ops.src u))
      with
      | #Dtype.value as x -> x
      | `Invalid ->
          invalid_arg (Format.asprintf "Tolk_next_engine.link: %a" Ops.pp u))

(* The little-endian bytes of [v], a value of [dt]. *)
let le dt (v : Dtype.value) =
  let n = Dtype.itemsize dt in
  let unsigned =
    match n with
    | 1 -> Dtype.Uint8
    | 2 -> Dtype.Uint16
    | 4 -> Dtype.Uint32
    | _ -> Dtype.Uint64
  in
  let z =
    match (dt, v) with
    | Dtype.Bool, `Bool b -> Z.of_int (Bool.to_int b)
    | _ -> (
        match Dtype.bitcast dt unsigned v with `Int z -> z | _ -> assert false)
  in
  String.init n (fun k -> Char.chr (Z.to_int (Z.extract z (8 * k) 8)))

let write b off s =
  let n = String.length s in
  if (n > 0) [@mutate off "a copy of no bytes writes nothing"] then
    let src = Bigarray.(Array1.init char c_layout n (String.get s)) in
    B.copy ~src:(B.of_bigarray src) ~dst:(at b off n)

(* The linked buffer the view [u] is of, the shard it selects, and the view's
   byte offset in it. *)
let linked_at storage u =
  let base, shard, off = Hcq2.unwrap_lane u in
  (List.nth (holds storage [||] [] base) (Option.value shard ~default:0), off)

(* Writes the link patch [p], a store of bytes or of words at constant element
   indices, into the linked storage its view is of. *)
let apply storage addr p =
  let dst = Ops.nth p 0 and v = Ops.nth p 1 in
  let base_at = linked_at storage in
  let bytes_of u =
    match (Ops.op u, Ops.arg u) with
    | Op.Binary, Ops.Bytes s -> Some s
    | Op.Bitcast, _ -> (
        match Ops.arg (Ops.nth u 0) with Ops.Bytes s -> Some s | _ -> None)
    | _ -> None
  in
  match (bytes_of v, Ops.op dst) with
  | Some s, _ ->
      let b, off = base_at dst in
      write b off s
  | None, Op.Index -> (
      let viewed = Ops.nth dst 0 in
      let b, off = base_at viewed in
      let dt = Ops.dtype v in
      let n = Dtype.itemsize dt in
      (* A group of one row is its index and its word, unstacked. *)
      let lanes u = if Ops.op u = Op.Stack then Ops.src u else [ u ] in
      match (lanes (Ops.nth dst 1), lanes v) with
      | offs, words when List.length offs = List.length words ->
          List.iter2
            (fun o w ->
              write b (off + (int_of_const o * n)) (le dt (word addr w)))
            offs words
      | _ -> invalid_arg "Tolk_next_engine.link: a patch of mismatched stacks")
  | None, _ ->
      invalid_arg
        (Format.asprintf "Tolk_next_engine.link: %a is no link patch" Ops.pp p)

(* Batches *)

let signal_word_tag = Ops.Tag.String "timeline"

(* The storage of a batch's placeholder [u]: the signal word of its device for
   ["timeline"], the address of a C function for a [("cfunc", lib, f)] tuple,
   and pinned memory, which the host program writes, for any other. *)
(* The vendor whose commands name a placeholder gives its storage: its own
   device's, or a device of the batch's, such as the vendor that owns a C
   function its host program calls. *)
let placeholder device queues u =
  let dev = device (List.hd (names u)) in
  let d = dev.device in
  let named =
    List.find_map
      (fun n -> (device n).placeholder u)
      (List.hd (names u) :: queues)
  in
  match named with
  | Some b -> b
  | None -> (
      match Ops.tag u with
      | Some t when Ops.Tag.equal t signal_word_tag -> Nx_device.signal_word d
      | Some (Ops.Tag.Tuple [ String "cfunc"; String lib; String _ ]) ->
          invalid_arg (strf "Tolk_next_engine.link: no C library %s" lib)
      | _ -> B.create ~pinned:true d Nx_dtype.Scalar.UInt8 (max 1 (bytes u)))

let hcq_info call =
  match Ops.arg call with Ops.Call { aux; _ } -> aux | _ -> None

let link_batch ~device ~storage ~keep call patches =
  let info = Option.get (hcq_info call) in
  let args = Realize.get_call_arg_uops call in
  let is_placeholder u = Ops.op u = Op.Param && Option.is_some (Ops.tag u) in
  (* The placeholders the patches address are the batch's too, though its host
     program may not take them, such as a signal word only its queues read. *)
  let patched = List.concat_map Ops.toposort patches in
  List.iter
    (fun u ->
      if is_placeholder u && not (Ops.Tbl.mem storage u) then
        Ops.Tbl.replace storage u [ placeholder device info.device u ])
    (args @ patched);
  let reached =
    List.filter_map
      (fun u ->
        if Ops.op u = Op.Buffer || is_placeholder u then
          Some (Ops.Tbl.find storage u)
        else None)
      patched
    |> List.concat
  in
  let addr dn u =
    let b, off = linked_at storage u in
    Nativeint.add (address keep (device dn).device b) (Nativeint.of_int off)
  in
  List.iter (apply storage addr) patches;
  let arguments = List.map (fun u -> List.hd (Ops.Tbl.find storage u)) args in
  let queues = List.map (fun n -> (device n).device) info.device in
  (* The host programs of a device's queues run on the host they name. *)
  let host =
    match (device (List.hd info.device)).compiler.queues with
    | Some q -> (device q.host).device
    | None ->
        invalid_arg "Tolk_next_engine.link: a batch of a device without queues"
  in
  let table =
    if info.table < 0 then Bigarray.(Array1.create int64 c_layout 0)
    else
      let b = List.nth arguments info.table in
      let hb =
        match B.borrow host b with Ok m -> m | Error why -> invalid_arg why
      in
      keep hb;
      B.bigarray Bigarray.int64 hb
  in
  {
    info;
    named = (fun n -> (device n).device);
    host_program = Program.load host (Ops.body call);
    submitting = List.map (fun n -> (device n).submitting) info.device;
    queues;
    arguments;
    table;
    reached;
    last = Array.make (List.length queues) 0;
    kept = [];
  }

let run_batch ~vars storage slots b =
  let info = b.info in
  let inputs =
    List.map
      (fun (base, off, dev) ->
        let base, shard, inner = Hcq2.unwrap_lane base in
        let bs = holds storage slots vars base in
        ( (match shard with Some i -> List.nth bs i | None -> List.hd bs),
          off + inner,
          dev ))
      info.inputs
  in
  let touches =
    b.arguments @ b.reached @ List.map (fun (x, _, _) -> x) inputs
  in
  Nx_device.submit b.queues ~touches (fun s ->
      List.iteri
        (fun i d ->
          if (b.last.(i) > 0) [@mutate off "a wait for 0 returns at once"] then
            Nx_device.Submission.wait s d b.last.(i))
        b.queues;
      List.iter
        (fun (d', v) ->
          if not (List.memq d' b.queues) then Nx_device.Submission.wait s d' v)
        (Nx_device.Submission.waits s);
      let kept = ref [] in
      List.iteri
        (fun k (x, off, dev) ->
          let d = b.named dev in
          b.table.{k} <-
            Int64.of_nativeint
              (Nativeint.add
                 (address (fun m -> kept := m :: !kept) d x)
                 (Nativeint.of_int off)))
        inputs;
      b.kept <- !kept;
      let prg = b.host_program in
      let timeline name dn =
        let d = b.named dn in
        if name = Ops.expr (Hcq2.submitted dn) then Some (Nx_device.submitted d)
        else if name = Ops.expr (Hcq2.value dn) then
          Some (Nx_device.Submission.value s d)
        else None
      in
      let bind u =
        match List.find_map (timeline (Ops.expr u)) info.device with
        | Some v -> v
        | None -> value vars 0 u
      in
      List.iter (fun f -> f ()) b.submitting;
      Nx_device.Program.call prg.program
        (Array.of_list (List.map (List.nth b.arguments) prg.globals))
        (Array.of_list (List.map bind prg.vars));
      if Nx_device.Profile.enabled () then
        List.iter
          (fun (k : Ops.hcq_kernel) ->
            match k.stamps with
            | first :: _ ->
                List.iter
                  (fun dn ->
                    let d = b.named dn in
                    let slots =
                      List.nth b.arguments (List.assoc dn info.slots)
                    in
                    Nx_device.Submission.record s d
                      ~lane:
                        (if Option.is_some k.profile_key then "compute"
                         else "copy")
                      ~name:k.name
                      (B.view slots
                         ~offset:(8 * (first - 1))
                         Nx_dtype.Scalar.UInt64 4))
                  k.devices
            | [] -> ())
          info.kernels;
      List.iteri
        (fun i d -> b.last.(i) <- Nx_device.Submission.value s d)
        b.queues)

(* The calls of a schedule's entry: itself, or those a range is around. *)
let rec calls_of entry =
  match Ops.op entry with
  | Op.End -> calls_of (Ops.nth entry 0)
  | Op.Linear -> List.concat_map calls_of (Ops.src entry)
  | _ -> [ entry ]

let link ~devices ?(bound = []) linear =
  let fn = "Tolk_next_engine.link" in
  if Ops.op linear <> Op.Linear then fail fn "not a compiled schedule";
  let device = devices in
  let nx name = (devices name).device in
  let storage = Ops.Tbl.create 16 in
  List.iter
    (fun (u, bs) ->
      let ns = names u in
      if Ops.op u <> Op.Buffer || List.length bs <> List.length ns then
        fail fn "%s is not bound to a buffer on each device"
          (Format.asprintf "%a" Ops.pp u);
      List.iter2
        (fun n b ->
          if B.device b != nx n || B.nbytes b < bytes u then
            fail fn "a storage node is not bound to %d bytes of %s" (bytes u) n)
        ns bs;
      Ops.Tbl.replace storage u bs)
    bound;
  let entries = Ops.src linear in
  let nodes =
    List.concat_map
      (fun e ->
        let c = Ops.without_after e in
        let patches = if Ops.op e = Op.After then List.tl (Ops.src e) else [] in
        (* A batch's inputs are in its table, not among its arguments. *)
        let inputs =
          match hcq_info c with
          | Some i -> List.map (fun (base, _, _) -> base) i.inputs
          | None -> []
        in
        List.concat_map Ops.toposort
          (Realize.get_call_arg_uops c @ patches @ inputs))
      (List.concat_map calls_of entries)
  in
  List.iter
    (fun u ->
      if Ops.op u = Op.Buffer && not (Ops.Tbl.mem storage u) then
        Ops.Tbl.replace storage u
          (List.map
             (fun n -> B.create (nx n) Nx_dtype.Scalar.UInt8 (max (bytes u) 1))
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
  let borrows = ref [] in
  let keep m = borrows := m :: !borrows in
  let rec linked entry =
    match Ops.op entry with
    | Op.End ->
        let body = Ops.nth entry 0 in
        Range
          {
            ranges = List.tl (Ops.src entry);
            body =
              List.map linked
                (if Ops.op body = Op.Linear then Ops.src body else [ body ]);
          }
    | _ -> linked_call entry
  and linked_call entry =
    let call = Ops.without_after entry in
    let body = Ops.body call in
    match (Ops.op body, hcq_info call) with
    | Op.Store, _ -> (
        match Realize.get_call_arg_uops call with
        | [ dst; src ] -> Copy { call; dst; src }
        | _ -> fail fn "a copy of other than two buffers")
    | Op.Program, Some _ ->
        let patches =
          if Ops.op entry = Op.After then List.tl (Ops.src entry) else []
        in
        Batch (link_batch ~device ~storage ~keep call patches)
    | Op.Program, None ->
        let first = List.hd (Realize.get_call_arg_uops call) in
        Kernel
          {
            call;
            lanes = List.map (fun n -> Program.load (nx n) body) (names first);
          }
    | o, _ -> fail fn "%s is no call to run" (Format.asprintf "%a" Op.pp o)
  in
  let calls = List.map linked entries in
  {
    lock = Mutex.create ();
    calls;
    params;
    storage;
    borrows = !borrows;
    named = nx;
  }

let check_slots t slots =
  let fn = "Tolk_next_engine.run" in
  List.iter
    (fun (slot, u) ->
      if slot >= Array.length slots then fail fn "no buffers for slot %d" slot;
      let ns = names u and bs = slots.(slot) in
      if List.length bs <> List.length ns then
        fail fn "slot %d takes %d buffers, not %d" slot (List.length ns)
          (List.length bs);
      List.iter2
        (fun n b ->
          if B.device b != t.named n then
            fail fn "slot %d takes a buffer of %s, not of %s" slot n
              (Nx_device.name (B.device b));
          if B.nbytes b < bytes u then
            fail fn "slot %d takes %d bytes, not %d" slot (bytes u) (B.nbytes b))
        ns bs)
    t.params

(* Reports *)

let reporting () = Helpers.Context_var.value Helpers.debug >= 2
let kernels_run = Atomic.make 0

(* One line per kernel, as [DEBUG=2] prints it: its device, how many kernels ran
   before it, its name, its number of arguments and, when known, its time and
   throughput. *)
let report ~device ~name ~args ~vars (e : Ops.estimates) time =
  let count = Atomic.fetch_and_add kernels_run 1 + 1 in
  let timing =
    match time with
    | None -> ""
    | Some s ->
        let per x = Float.of_int (Ops.sym_infer x vars) /. Float.max s 1e-20 in
        Printf.sprintf " tm %s (%7.0f GFLOPS %4.0f GB/s)"
          (Helpers.time_to_str ~w:9 s)
          (per e.ops *. 1e-9)
          (per e.mem *. 1e-9)
  in
  let device = String.sub device 0 (Int.min 7 (String.length device)) in
  Printf.printf "*** %-7s %4d %s arg %2d%s\n%!" device count
    (Helpers.ansipad name 46) args timing

(* The seconds [f] takes on the host clock. *)
let seconds f =
  let t0 = Nx_device.Profile.now () in
  f ();
  Float.of_int (Nx_device.Profile.now () - t0) *. 1e-9

(* A batch's kernels run on its devices, which stamp them: under a profile of
   its own, the batch reports each kernel's span once its devices synchronized.
   While a profile is taken elsewhere, the spans are that profile's, and the
   batch reports no time. *)
let run_reported ~vars storage slots b =
  let own =
    if Nx_device.Profile.enabled () then None
    else try Some (Nx_device.Profile.start ()) with Invalid_argument _ -> None
  in
  let events =
    match run_batch ~vars storage slots b with
    | () -> (
        match own with
        | Some p -> Nx_device.Profile.stop p
        | None ->
            List.iter Nx_device.synchronize b.queues;
            [])
    | exception e ->
        Option.iter (fun p -> ignore (Nx_device.Profile.stop p)) own;
        raise e
  in
  let spans = ref events in
  let span d name =
    let rec take = function
      | [] -> (None, [])
      | Nx_device.Profile.Span sp :: rest when sp.device == d && sp.name = name
        ->
          (Some (Float.of_int (sp.stop - sp.start) *. 1e-9), rest)
      | e :: rest ->
          let found, rest = take rest in
          (found, e :: rest)
    in
    let found, rest = take !spans in
    spans := rest;
    found
  in
  List.iter
    (fun (k : Ops.hcq_kernel) ->
      List.iter
        (fun dn ->
          report ~device:dn ~name:k.name
            ~args:(List.length k.input_slots)
            ~vars k.estimates
            (span (b.named dn) k.name))
        k.devices)
    b.info.kernels

let rec run_call ~vars t slots = function
  | Copy { call; dst; src } ->
      let dsts = view t.storage slots vars dst
      and srcs = view t.storage slots vars src in
      List.iteri
        (fun i dst ->
          let copy () = B.copy ~src:(lane srcs i) ~dst in
          if reporting () then
            report
              ~device:(Nx_device.name (B.device dst))
              ~name:
                (Realize.get_call_name ~var_vals:vars call
                   (Realize.get_call_arg_uops call))
              ~args:2 ~vars
              (Realize.estimate_uop call)
              (Some (seconds copy))
          else copy ())
        dsts
  | Kernel { call; lanes } ->
      let prg = Ops.body call in
      let args =
        List.map (view t.storage slots vars) (Realize.get_call_arg_uops call)
      in
      let info =
        match Ops.arg prg with Ops.Program i -> i | _ -> assert false
      in
      let vals = Realize.get_call_var_uops call prg in
      List.iteri
        (fun i (p : Program.t) ->
          let buffers =
            List.map (fun g -> lane (List.nth args g) i) info.globals
          in
          (* A host program is outside the devices' ordering: the work that
             touched its buffers, such as a batch's before it, completes
             first. *)
          List.iter Nx_device.synchronize
            (List.fold_left
               (fun ds b ->
                 let d = B.device b in
                 if
                   List.memq d ds
                   [@mutate
                     off "a second synchronization finds nothing to wait for"]
                 then ds
                 else d :: ds)
               [] buffers);
          let call_program () =
            Nx_device.Program.call p.program (Array.of_list buffers)
              (Array.of_list (List.map (value vars i) vals))
          in
          if reporting () then
            report
              ~device:(Nx_device.name (B.device (List.hd buffers)))
              ~name:
                (Realize.get_call_name ~var_vals:vars call
                   (Realize.get_call_arg_uops call))
              ~args:(List.length buffers) ~vars
              (Realize.estimate_uop call)
              (Some (seconds call_program))
          else call_program ())
        lanes
  | Batch b when reporting () -> run_reported ~vars t.storage slots b
  | Batch b -> run_batch ~vars t.storage slots b
  | Range { ranges; body } ->
      let rec trips vars = function
        | [] -> List.iter (run_call ~vars t slots) body
        | r :: rest ->
            let name = Ops.expr (Hcq2.range_value r) in
            for i = 0 to Ops.sym_infer (sint (Ops.nth r 0)) vars - 1 do
              trips ((name, i) :: vars) rest
            done
      in
      trips vars ranges

let run ?(vars = []) t slots =
  check_slots t slots;
  let n = List.length t.calls in
  if Helpers.Context_var.value Helpers.debug >= 1 && n >= 10 then
    Printf.printf "jit execs %d calls\n%!" n;
  Mutex.protect t.lock (fun () -> List.iter (run_call ~vars t slots) t.calls)

(* Measuring *)

(* Only NV invalidates its caches on demand. *)
let invalidate_caches d =
  Option.iter Nx_nv_device.invalidate_caches (Nx_nv_device.of_device d)

(* The span of [f]'s work named [name] on [d], as [d] stamps it, or, while a
   profile is taken elsewhere, [f] and [d]'s synchronization on the host
   clock. *)
let timed d name f =
  let now = Nx_device.Profile.now in
  if Nx_device.Profile.enabled () then (
    let t0 = now () in
    f ();
    Nx_device.synchronize d;
    now () - t0)
  else
    let p = Nx_device.Profile.start () in
    let events =
      match f () with
      | () -> Nx_device.Profile.stop p
      | exception e ->
          ignore (Nx_device.Profile.stop p);
          raise e
    in
    let span = function
      | Nx_device.Profile.Span sp when sp.device == d && sp.name = name ->
          Some (sp.stop - sp.start)
      | _ -> None
    in
    match List.find_map span events with
    | Some ns -> ns
    | None -> invalid_arg ("Tolk_next_engine.measure: no span of " ^ name)

(* A run shorter than a tick of its clock measures 0: the mean of runs that take
   [enough_ns] in all is off by at most a tick in [enough_ns]. *)
let enough_ns = 10_000
let most_runs = 1000

let mean_ns run =
  let rec go runs total =
    if total >= enough_ns || runs >= most_runs then
      Float.of_int total /. Float.of_int runs
    else go (runs + 1) (total + run ())
  in
  go 1 (run ())

let measure ?(cold = false) ?(vars = []) ~devices name prg =
  let dev = devices name in
  let d = dev.device in
  let elf = Device.Tiny_elf.of_program prg in
  let info = match Ops.arg prg with Ops.Program i -> i | _ -> assert false in
  let buffers =
    List.filteri (fun i _ -> i < List.length info.globals) elf.signature
  in
  let scratch (param : Device.Tiny_elf.param) =
    B.create d Nx_dtype.Scalar.UInt8
      (max 1 (List.fold_left ( * ) (Dtype.itemsize param.dtype) param.shape))
  in
  let run =
    match dev.compiler.queues with
    | None ->
        let p = Program.load d prg in
        let bs = List.map scratch buffers in
        fun () ->
          timed (Nx_device.host_of d) elf.name (fun () ->
              Program.run ~vars p bs)
    | Some _ ->
        let nslots = 1 + List.fold_left max 0 info.globals in
        let buffer slot =
          List.nth buffers
            (Option.get (List.find_index (Int.equal slot) info.globals))
        in
        let param slot =
          let b = buffer slot in
          Ops.param
            ~shape:(List.map (fun n -> Ops.Int n) b.shape)
            ~device:(Single name) slot b.dtype
        in
        let linear =
          Hcq2.compile_linear ~profile:true
            ~devices:(fun n -> (devices n).compiler)
            (Ops.v Op.Linear ~src:[ Ops.call prg (List.init nslots param) ])
        in
        let s = link ~devices linear in
        let slots = Array.init nslots (fun slot -> [ scratch (buffer slot) ]) in
        fun () -> timed d elf.name (fun () -> run ~vars s slots)
  in
  mean_ns (fun () ->
      if cold then invalidate_caches d;
      run ())
  *. 1e-9
