(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

let strf = Printf.sprintf
let fail fn fmt = Printf.ksprintf (fun m -> invalid_arg (fn ^ ": " ^ m)) fmt

(* Devices *)

(* The kind of device a vendor's driver declares, if any. *)
let vendor d =
  let declares k = Option.is_some (Rig.capability d k) in
  if declares Rig_metal_abi.key then Some "METAL"
  else if declares Rig_cuda_abi.key then Some "CUDA"
  else if declares Rig_amd_abi.key then Some "AMD"
  else if declares Rig_nv_abi.key then Some "NV"
  else None

(* The target of a device kind, which names no interface and no indices. *)
let of_kind ?(renderer = "") ?(arch = "") device =
  { Helpers.Target.device; renderer; arch; interface = ""; indices = "" }

let target d =
  match vendor d with
  | Some kind -> of_kind ~arch:(Rig.arch d) kind
  | None
    when Rig.computes d && (Rig.equal (Rig.host_of d) d || Rig.runs_on_host d)
    ->
      let host = Rig.host_of d in
      let cpu = if Rig.equal host Rig.host then "native" else "generic" in
      of_kind ~renderer:"CLANG" ~arch:(Rig.arch host ^ "," ^ cpu) "CPU"
  | None ->
      invalid_arg (strf "Tolk_engine.target: %s runs no program" (Rig.name d))

let renderer d =
  match Device.renderer (target d) with Ok r -> r | Error why -> failwith why

(* Whether [d] runs copy parts: a GPU with a copy engine, or a memory device. *)
let copies d =
  match vendor d with
  | Some ("CUDA" | "AMD" | "NV") -> true
  | Some _ -> false
  | None -> Rig.runs_on_host d && not (Rig.equal d Rig.host)

(* Whether [c] copies between memory of [d] and of [d'] in a submission of its
   own: it copies and reaches both, and both are memory the host computes on, or
   neither is. A GPU maps host memory by pages, which a host buffer under 64 KiB
   may not start on, so a copy between the two is the host's
   ([Rig.Buffer.copy]), which stages where it must. *)
let copies_between c d d' =
  copies c && Rig.reaches c d && Rig.reaches c d'
  && Rig.runs_on_host d = Rig.runs_on_host d'

(* Views *)

(* A view's storage and byte offset; the offset of a view inside a range around
   a call reads the range. *)
let rec view_offset v =
  match Ops.op v with
  | Op.Bitcast | Op.After -> view_offset (Ops.nth v 0)
  | Op.Shrink -> (
      let base, off = view_offset (Ops.nth v 0) in
      match Shape.marg v with
      | Shrink [ (o, _) ] ->
          (base, Shape.Sint.(off + (o * Int (Dtype.itemsize (Ops.dtype v)))))
      | _ -> invalid_arg "Tolk_engine: a view of storage is one-dimensional")
  | _ -> (v, Ops.Int 0)

(* [view_offset], through a shard selection too: the storage, the shard
   selected, if any, and the byte offset. *)
let lane_offset v =
  let sel, off = view_offset v in
  if Ops.op sel <> Op.Mselect then (sel, None, off)
  else
    let base, inner = view_offset (Ops.nth sel 0) in
    let shard = match Ops.arg sel with Ops.Shard i -> Some i | _ -> None in
    (base, shard, Shape.Sint.(off + inner))

let names u =
  match Ops.device u with
  | Some (Single d) -> [ d ]
  | Some (Multi ds) -> ds
  | None -> invalid_arg (Format.asprintf "%a has no device" Ops.pp u)

let is_slot u =
  Ops.op u = Op.Param
  && Option.is_none (Ops.tag u)
  && Ops.addrspace u <> Some Dtype.Alu

let bytes u = Shape.max_numel u * Dtype.itemsize (Ops.dtype u)

let sint u =
  match Ops.arg u with
  | Ops.Const (`Int z) -> Ops.Int (Bigint.to_int z)
  | _ -> Sym u

let is_variable u =
  Ops.op u = Op.Param
  && (Ops.addrspace u = Some Dtype.Alu || Shape.is_variable u)

(* Variables *)

(* A run's variables, a cell each, set where the run binds them. Linking gives
   each name a cell, so that a run reads a value without looking up its name. *)
type env = { values : int array; set : bool array }

let cell cells name =
  match Hashtbl.find_opt cells name with
  | Some i -> i
  | None ->
      let i = Hashtbl.length cells in
      Hashtbl.add cells name i;
      i

(* How a run reads the variable [u]: its binding, or else the value it is bound
   to. *)
let reader cells u =
  let name = Ops.expr u in
  let i = cell cells name in
  let unset =
    match Ops.arg u with
    | Ops.Param { bound = Some b; _ } ->
        let b = Dtype.Value.to_int b in
        fun () -> b
    | _ ->
        fun () ->
          invalid_arg (strf "Tolk_engine.run: variable %s is unbound" name)
  in
  fun env -> if env.set.(i) then env.values.(i) else unset ()

(* Values as forms of words *)

(* The value [Σ factor · word + add] of words of the run's ints: the words that
   hold the run's variables and the values computed from them, and those where
   loops store their trips. *)
type affine = { terms : (int * int) list; add : int }

let constant add = { terms = []; add }
let word w = { terms = [ (w, 1) ]; add = 0 }

let scale k a =
  { terms = List.map (fun (w, f) -> (w, k * f)) a.terms; add = k * a.add }

let plus a b =
  let add terms (w, f) =
    match List.assoc_opt w terms with
    | Some g -> (w, f + g) :: List.remove_assoc w terms
    | None -> (w, f) :: terms
  in
  { terms = List.fold_left add a.terms b.terms; add = a.add + b.add }

(* What a word of the ints holds: a value each run computes before its first
   step, or a loop's trip. *)
type word_kind = Computed of (env -> int) | Trip

(* Inputs *)

(* A buffer a run is given: a lane of a parameter's buffers, or a buffer bound
   at link. *)
type source = Lane of { slot : int; lane : int } | Bound of Rig.Buffer.t

(* An input of the program: a buffer a run is given, or the bytes of one at an
   offset the run's words compute. *)
type input =
  | Whole of source
  | Part of { source : source; first : int array -> int; length : int }

(* Where a view lies: in memory made at link, of its bytes, or in an input, at
   an offset. *)
type place = Memory of { memory : int; bytes : int } | Input of int
type located = { place : place; offset : affine; device : Rig.t }

(* Linking *)

(* The submission being built: its device's index, its parts and slots, in
   reverse. *)
type pending = {
  on : int;
  mutable parts : Rig_program.part list;
  slots : Rig_program.slot Dynarray.t;
  mutable nparts : int;
}

(* A kernel of the schedule, for reports and timing: the device it runs on, its
   function's name and its call. *)
type kernel = { device : Rig.t; name : string; call : Ops.t }

(* What linking builds: the program's devices, memory, images, inputs and words,
   each by index, and the steps of the loop being built. *)
type builder = {
  fn : string;
  named : string -> Rig.t;
  separate : bool;
  bound : Rig.Buffer.t list Ops.Tbl.t;
  cells : (string, int) Hashtbl.t;
  rigs : Rig.t Dynarray.t;
  memory : Rig_program.memory Dynarray.t;
  memories : (int, int) Hashtbl.t Ops.Tbl.t;
  images : Rig_program.image Dynarray.t;
  image_of : (int * string, int) Hashtbl.t;
  inputs : (input * Rig_program.input) Dynarray.t;
  lane_inputs : (int * int, int) Hashtbl.t;
  mutable bound_inputs : (Rig.Buffer.t * int) list;
  mutable params : (int * Rig.t list * int) list;
  words : word_kind Dynarray.t;
  var_words : (string, int) Hashtbl.t;
  computed : (int * int) list Ops.Tbl.t;
  trips : int Ops.Tbl.t;
  mutable steps : Rig_program.step list;
  mutable pending : pending option;
  mutable kernels : kernel list;
}

(* Appends [x] to [a], and is its index. *)
let push a x =
  Dynarray.add_last a x;
  Dynarray.length a - 1

let device_index b d =
  match Dynarray.find_index (Rig.equal d) b.rigs with
  | Some i -> i
  | None -> push b.rigs d

let new_word b kind = push b.words kind

let var_word b u =
  let name = Ops.expr u in
  match Hashtbl.find_opt b.var_words name with
  | Some w -> w
  | None ->
      let w = new_word b (Computed (reader b.cells u)) in
      Hashtbl.add b.var_words name w;
      w

(* Whether [u] reads a range: a value that moves with its loop's trips. *)
let moves u =
  List.exists (fun x -> Ops.op x = Op.Range) (Ops.toposort ~calls:Skip u)

let reads_variable u = List.exists is_variable (Ops.toposort ~calls:Skip u)

(* The word that holds [u], an expression of the run's variables on lane [lane],
   which each run computes. *)
let computed_word b lane u =
  let ws = Option.value (Ops.Tbl.find_opt b.computed u) ~default:[] in
  match List.assoc_opt lane ws with
  | Some w -> w
  | None ->
      let read v =
        match Ops.arg v with
        | Ops.Param { name = Some "_device_num"; _ } -> fun _ -> lane
        | _ -> reader b.cells v
      in
      let w = new_word b (Computed (Shape.sym_compile (Sym u) read)) in
      Ops.Tbl.replace b.computed u ((lane, w) :: ws);
      w

let int_of_const u =
  match Ops.arg u with
  | Ops.Const (`Int z) -> Bigint.to_int z
  | Ops.Const (`Bool x) -> Bool.to_int x
  | _ -> invalid_arg (Format.asprintf "%a is no integer" Ops.pp u)

(* [u] on lane [lane] as a form of words: a range's trip moves it by a constant
   factor, and the rest of it is computed for the run. *)
let rec affine b lane u =
  match (Ops.op u, Ops.arg u) with
  | Op.Const, _ -> constant (int_of_const u)
  | Op.Range, _ -> (
      match Ops.Tbl.find_opt b.trips u with
      | Some w -> word w
      | None -> fail b.fn "%s reads a range outside its loop" (Ops.expr u))
  | Op.Param, Ops.Param { name = Some "_device_num"; _ } -> constant lane
  | Op.Param, _ when is_variable u -> word (var_word b u)
  | Op.Add, _ ->
      plus (affine b lane (Ops.nth u 0)) (affine b lane (Ops.nth u 1))
  | Op.Mul, _ -> (
      let x = affine b lane (Ops.nth u 0) and y = affine b lane (Ops.nth u 1) in
      match (x.terms, y.terms) with
      | [], _ -> scale x.add y
      | _, [] -> scale y.add x
      | _ -> computed b lane u)
  | Op.Cast, _ when Dtype.is_int (Ops.dtype u) -> affine b lane (Ops.nth u 0)
  | _ -> computed b lane u

and computed b lane u =
  if moves u then
    fail b.fn "%s moves with its loop's trips by no constant factor"
      (Ops.expr u)
  else if not (reads_variable u) then constant (Shape.sym_infer (Sym u) [])
  else word (computed_word b lane u)

let affine_sint b lane : Ops.sint -> affine = function
  | Int n -> constant n
  | Sym u -> affine b lane u

(* A value of one word, such as a grid's size or a loop's trips. *)
let value b lane (s : Ops.sint) : Rig_program.value =
  let a = affine_sint b lane s in
  match (a.terms, s) with
  | [], _ -> Fixed a.add
  | [ (w, 1) ], _ when a.add = 0 -> Int w
  | _, Sym u when not (moves u) -> Int (computed_word b lane u)
  | _ -> fail b.fn "a value of one word moves with its loop's trips"

(* Storage *)

let memory b base lane d =
  let lanes =
    match Ops.Tbl.find_opt b.memories base with
    | Some l -> l
    | None ->
        let l = Hashtbl.create 1 in
        Ops.Tbl.replace b.memories base l;
        l
  in
  match Hashtbl.find_opt lanes lane with
  | Some m -> m
  | None ->
      let m =
        Rig_program.Alloc
          {
            device = device_index b d;
            kind = Device;
            bytes = max 1 (bytes base);
            init = { bytes = ""; holes = [||] };
            copies = One;
          }
      in
      let i = push b.memory m in
      Hashtbl.replace lanes lane i;
      i

let add_input b input desc = push b.inputs (input, desc)

let lane_input b ~slot ~lane d n =
  match Hashtbl.find_opt b.lane_inputs (slot, lane) with
  | Some i -> i
  | None ->
      let i =
        add_input b
          (Whole (Lane { slot; lane }))
          { device = device_index b d; bytes = n }
      in
      Hashtbl.replace b.lane_inputs (slot, lane) i;
      i

let bound_input b buf =
  match List.assq_opt buf b.bound_inputs with
  | Some i -> i
  | None ->
      let i =
        add_input b (Whole (Bound buf))
          {
            device = device_index b (Rig.Buffer.device buf);
            bytes = Rig.Buffer.length buf;
          }
      in
      b.bound_inputs <- (buf, i) :: b.bound_inputs;
      i

let note_param b slot ds n =
  if not (List.exists (fun (s, _, _) -> s = slot) b.params) then
    b.params <- (slot, ds, n) :: b.params

(* Where the view [u] of lane [lane] lies: a lane of a parameter, which a run
   gives; of bound storage; of storage made at link; or of a stack of one view
   per device. A storage of one device serves every lane. *)
let rec locate b u lane =
  let base, shard, off = lane_offset u in
  let offset = affine_sint b lane off in
  let pick ns =
    match shard with
    | Some j -> j
    | None -> if List.compare_length_with ns 1 = 0 then 0 else lane
  in
  match (Ops.op base, Ops.arg base) with
  | Op.Param, Ops.Param p when is_slot base ->
      let ns = names base in
      let ds = List.map b.named ns in
      let n = bytes base in
      note_param b p.slot ds n;
      let j = pick ns in
      let d = List.nth ds j in
      {
        place = Input (lane_input b ~slot:p.slot ~lane:j d n);
        offset;
        device = d;
      }
  | Op.Buffer, _ -> (
      let ns = names base in
      let j = pick ns in
      match Ops.Tbl.find_opt b.bound base with
      | Some bs ->
          let buf = List.nth bs j in
          {
            place = Input (bound_input b buf);
            offset;
            device = Rig.Buffer.device buf;
          }
      | None ->
          let d = b.named (List.nth ns j) in
          let memory = memory b base j d in
          {
            place = Memory { memory; bytes = max 1 (bytes base) };
            offset;
            device = d;
          })
  | Op.Mstack, _ ->
      let k = match shard with Some k -> k | None -> lane in
      let l = locate b (List.nth (Ops.src base) k) 0 in
      { l with offset = plus l.offset offset }
  | _ -> fail b.fn "%s is not storage" (Format.asprintf "%a" Ops.pp base)

(* Steps *)

let emit b step = b.steps <- step :: b.steps

let flush b =
  match b.pending with
  | None -> ()
  | Some p ->
      b.pending <- None;
      emit b
        (Submit
           {
             device = p.on;
             parts = Iarray.of_list (List.rev p.parts);
             slots = Iarray.of_list (Dynarray.to_list p.slots);
             fixed = Iarray.of_list [];
           })

(* The submission of [on] that takes the next part: one of at most the parts
   its device always runs ([Rig.most_parts]). *)
let pending b on =
  match b.pending with
  | Some p
    when p.on = on && p.nparts < Rig.most_parts (Dynarray.get b.rigs on) ->
      p
  | _ ->
      flush b;
      let p = { on; parts = []; slots = Dynarray.create (); nparts = 0 } in
      b.pending <- Some p;
      p

let add_part b p part =
  p.parts <- part :: p.parts;
  p.nparts <- p.nparts + 1;
  if b.separate then flush b

let slot_index p (s : Rig_program.slot) =
  match Dynarray.find_index (( = ) s) p.slots with
  | Some i -> i
  | None -> push p.slots s

(* The steps [f] emits, as the body of a loop. *)
let nested b f =
  flush b;
  let outer = b.steps in
  b.steps <- [];
  f ();
  flush b;
  let body = Iarray.of_list (List.rev b.steps) in
  b.steps <- outer;
  body

(* Launches *)

(* A block repays waking the host's threads and joining them once it does 2^18
   operations. *)
let block_ops = 1 lsl 18

(* The blocks a split launch of [extent] iterations and [ops] operations runs
   in, and the iterations of each but the last. *)
let blocks ~extent ~ops =
  let n = max 1 (min extent (ops / block_ops)) in
  (n, max 1 ((extent + n - 1) / n))

let le bytes at width v =
  match width with
  | 1 -> Bytes.set_int8 bytes at v
  | 2 -> Bytes.set_int16_le bytes at v
  | 4 -> Bytes.set_int32_le bytes at (Int32.of_int v)
  | _ -> Bytes.set_int64_le bytes at (Int64.of_int v)

(* Writes [a] into [width] bytes at [at]: its constant, or a hole. *)
let place b bytes holes at width a =
  match a.terms with
  | [] -> le bytes at width a.add
  | terms ->
      let width : Rig_program.width =
        match width with
        | 4 -> W32
        | 8 -> W64
        | _ -> fail b.fn "a value of %d bytes moves with the run" width
      in
      let terms =
        Iarray.of_list (List.map (fun (w, f) -> (Rig_program.Int w, f)) terms)
      in
      holes :=
        { Rig_program.at; width; terms; add = a.add; shift = 0 } :: !holes

let whole (l : located) : Rig_program.slot =
  match l.place with
  | Memory { memory; bytes } -> Memory { memory; offset = 0; length = bytes }
  | Input i -> Input i

let program_info prg =
  match Ops.arg prg with
  | Ops.Program info -> info
  | _ -> invalid_arg "Tolk_engine: a call of no compiled program"

let kernel_of prg =
  match Ops.arg (Ops.nth prg 0) with
  | Ops.Kernel k -> k
  | _ -> invalid_arg "Tolk_engine: a program of no kernel"

let image b dev binary =
  match Hashtbl.find_opt b.image_of (dev, binary) with
  | Some i -> i
  | None ->
      let i =
        push b.images
          {
            Rig_program.device = dev;
            binary = { bytes = binary; holes = [||] };
          }
      in
      Hashtbl.replace b.image_of (dev, binary) i;
      i

let rec split_at n l =
  if n = 0 then ([], l)
  else
    match l with
    | x :: rest ->
        let a, rest = split_at (n - 1) rest in
        (x :: a, rest)
    | [] -> ([], [])

type grid = Rig_program.value * Rig_program.value * Rig_program.value

let one : grid = (Fixed 1, Fixed 1, Fixed 1)

let triple : Rig_program.value list -> grid = function
  | [ x; y; z ] -> (x, y, z)
  | [ x; y ] -> (x, y, Fixed 1)
  | [ x ] -> (x, Fixed 1, Fixed 1)
  | _ -> (Fixed 1, Fixed 1, Fixed 1)

(* How a run computes [s], which reads no range, on lane [lane]. *)
let counter b lane (s : Ops.sint) =
  match s with
  | Int n -> fun _ -> n
  | Sym u when moves u ->
      fail b.fn "a split loop's iterations move with its loop's trips"
  | Sym _ ->
      let read v =
        match Ops.arg v with
        | Ops.Param { name = Some "_device_num"; _ } -> fun _ -> lane
        | _ -> reader b.cells v
      in
      Shape.sym_compile s read

(* The grid of a host program that splits its loop of [extent] iterations into
   blocks: a group of threads per block, a thread per iteration. *)
let split_grid b lane extent (k : Ops.kernel_info) : grid * grid =
  let ops = match k.estimates with Some e -> e.ops | None -> extent in
  match (extent, ops) with
  | Int extent, Int ops ->
      let n, t = blocks ~extent ~ops in
      ((Fixed n, Fixed 1, Fixed 1), (Fixed t, Fixed 1, Fixed 1))
  | extent, ops ->
      let extent = counter b lane extent and ops = counter b lane ops in
      let n env = fst (blocks ~extent:(extent env) ~ops:(ops env)) in
      let t env = snd (blocks ~extent:(extent env) ~ops:(ops env)) in
      ( (Int (new_word b (Computed n)), Fixed 1, Fixed 1),
        (Int (new_word b (Computed t)), Fixed 1, Fixed 1) )

(* The launch of the compiled program [prg] that [call] makes on its lane
   [lane], whose device is [d]. A host program runs on its device's host. *)
let launch b call prg lane d =
  let info = program_info prg and k = kernel_of prg in
  let elf = Device.Tiny_elf.of_program prg in
  let kind = (target d).device in
  if kind <> info.target.device then
    fail b.fn "%s runs no %s program" (Rig.name d) info.target.device;
  let host = kind = "CPU" in
  let on = if host then Rig.host_of d else d in
  let dev = device_index b on in
  let image = image b dev elf.lib in
  let args = Realize.get_call_arg_uops call in
  let bound slot =
    match k.split with Some s -> slot = s.lo || slot = s.hi | None -> false
  in
  let values = List.combine info.vars (Realize.get_call_var_uops call prg) in
  let vars =
    List.filter
      (fun (v, _) ->
        match Ops.arg v with Ops.Param p -> not (bound p.slot) | _ -> true)
      values
  in
  (* Its parameters: a pointer to each buffer, each variable but a split's
     bounds, and a split loop's iterations, each aligned to its size, as the
     program reads them ({!Tolk.Codegen}). *)
  let fields =
    List.map (fun _ -> Dtype.Uint64) info.globals
    @ List.map (fun (v, _) -> Ops.dtype v) vars
    @ if Option.is_some k.split then [ Dtype.Int64 ] else []
  in
  let offsets =
    Device.Tiny_elf.iter_sig
      (List.mapi
         (fun slot dtype ->
           { Device.Tiny_elf.name = None; slot; dtype; shape = [] })
         fields)
  in
  (* A structure's size is a multiple of its widest field's. *)
  let size, align =
    List.fold_left
      (fun (m, a) (o, dt) ->
        let n = Dtype.itemsize dt in
        (max m (o + n), max a n))
      (0, 4) offsets
  in
  let params = Bytes.make (Helpers.round_up size align) '\000' in
  let holes = ref [] in
  let buffers_at, rest = split_at (List.length info.globals) offsets in
  let vars_at, extent_at = split_at (List.length vars) rest in
  let p = pending b dev in
  let refs =
    List.map2
      (fun g (at, _) ->
        let l = locate b (List.nth args g) lane in
        place b params holes at 8 l.offset;
        let use : Rig_program.use =
          if List.mem g info.outs then Write else Read
        in
        { Rig_program.slot = slot_index p (whole l); at; use })
      info.globals buffers_at
  in
  List.iter2
    (fun (_, v) (at, dt) ->
      place b params holes at (Dtype.itemsize dt) (affine b lane v))
    vars vars_at;
  let groups, threads =
    match k.split with
    | Some s ->
        (* The bound [hi] of the whole loop is its iterations. *)
        let extent =
          List.find_map
            (fun (v, x) ->
              match Ops.arg v with
              | Ops.Param p when p.slot = s.hi -> Some (sint x)
              | _ -> None)
            values
          |> Option.value ~default:s.iterations
        in
        List.iter
          (fun (at, _) -> place b params holes at 8 (affine_sint b lane extent))
          extent_at;
        split_grid b lane extent k
    | None when host -> (one, one)
    | None ->
        ( triple (List.map (value b lane) info.global_size),
          triple (List.map (value b lane) info.local_size) )
  in
  b.kernels <- { device = on; name = elf.name; call } :: b.kernels;
  add_part b p
    (Launch
       {
         image;
         name = elf.name;
         params =
           { bytes = Bytes.to_string params; holes = Iarray.of_list !holes };
         refs = Iarray.of_list refs;
         groups;
         threads;
         shared = Fixed 0;
       })

(* Copies *)

(* [l] as a copy's end: its memory or input whole, at the offset each step
   computes from the run's ints. *)
let copy_end (l : located) :
    Rig_program.slot * (Rig_program.value * int) iarray =
  let slot : Rig_program.slot =
    match l.place with
    | Memory { memory; bytes } -> Memory { memory; offset = 0; length = bytes }
    | Input i -> Input i
  in
  let words = List.map (fun (w, f) -> (Rig_program.Int w, f)) l.offset.terms in
  let add =
    if l.offset.add = 0 then [] else [ (Rig_program.Fixed l.offset.add, 1) ]
  in
  (slot, Iarray.of_list (add @ words))

(* The [n] bytes of [l] as a move's end: memory made at link at a constant
   offset, a bound buffer whole, or the bytes of an input at an offset each run
   computes. A move copies its ends whole ([Rig.Buffer.copy]). *)
let move_end b (l : located) n : Rig_program.slot =
  match l.place with
  | Memory { memory; _ } -> (
      match l.offset.terms with
      | [] -> Memory { memory; offset = l.offset.add; length = n }
      | _ -> fail b.fn "a move's offset into storage made at link moves")
  | Input i ->
      let input, desc = Dynarray.get b.inputs i in
      let source = match input with Whole s -> s | Part _ -> assert false in
      let a = l.offset in
      let is_trip (w, _) =
        match Dynarray.get b.words w with Trip -> true | Computed _ -> false
      in
      if List.exists is_trip a.terms then
        fail b.fn "a move's offset into an input moves with its loop's trips";
      let whole =
        a.terms = [] && a.add = 0
        &&
        match source with
        | Bound buf -> Rig.Buffer.length buf = n
        | Lane _ -> false
      in
      if whole then Input i
      else
        let first ints =
          List.fold_left (fun s (w, f) -> s + (f * ints.(w))) a.add a.terms
        in
        Input
          (add_input b
             (Part { source; first; length = n })
             { device = desc.device; bytes = n })

(* The copy [call] makes on its lane [lane]: a copy part of the source's device
   or the destination's, the source's first, where it copies between the two
   ([copies_between]), or else the host's copy. *)
let copy b call lane =
  match Realize.get_call_arg_uops call with
  | [ dst; src ] ->
      let length = bytes dst in
      let d = locate b dst lane and s = locate b src lane in
      let dst_slot, dst_offset = copy_end d
      and src_slot, src_offset = copy_end s in
      let runs c = copies_between c s.device d.device in
      begin match List.find_opt runs [ s.device; d.device ] with
      | Some c ->
          let p = pending b (device_index b c) in
          let src = slot_index p src_slot in
          let dst = slot_index p dst_slot in
          add_part b p (Copy { src; src_offset; dst; dst_offset; length })
      | None ->
          flush b;
          emit b
            (Move { src = move_end b s length; dst = move_end b d length })
      end
  | _ -> fail b.fn "a copy of other than two buffers"

(* Calls *)

let rec entry b e =
  match Ops.op e with
  | Op.End ->
      let rec ranges = function
        | [] -> body b (Ops.nth e 0)
        | r :: rest -> loop b r None (fun () -> ranges rest)
      in
      ranges (List.tl (Ops.src e))
  | Op.Backedge ->
      let flag = locate b (Ops.nth e 2) 0 in
      if flag.offset.terms <> [] then
        fail b.fn "a back edge's flag moves with the run's words";
      let flag = (fst (copy_end flag), flag.offset.add) in
      loop b (Ops.nth e 1) (Some flag) (fun () -> body b (Ops.nth e 0))
  | Op.Linear -> body b e
  | _ -> call b e

and body b e =
  if Ops.op e = Op.Linear then List.iter (entry b) (Ops.src e) else entry b e

(* A loop over the range [r], or one trip where [r] is no range, around the
   steps [f] emits. *)
and loop b r flag f =
  let trips, trip =
    if Ops.op r <> Op.Range then (Rig_program.Fixed 1, None)
    else
      let trips = value b 0 (sint (Ops.nth r 0)) in
      let w = new_word b Trip in
      Ops.Tbl.replace b.trips r w;
      (trips, Some w)
  in
  let body = nested b f in
  Ops.Tbl.remove b.trips r;
  emit b (Loop { trips; trip; flag; body })

and call b e =
  let c = Ops.without_after e in
  let prg = Ops.body c in
  (* Each parameter a call takes is a slot of the runs, whether its program
     reads it or not. *)
  List.iter
    (fun u ->
      let base, _, _ = lane_offset u in
      match Ops.arg base with
      | Ops.Param p when is_slot base ->
          note_param b p.slot (List.map b.named (names base)) (bytes base)
      | _ -> ())
    (Realize.get_call_arg_uops c);
  match Ops.op prg with
  | Op.Store ->
      let dst = List.hd (Realize.get_call_arg_uops c) in
      List.iteri (fun lane _ -> copy b c lane) (names dst)
  | Op.Program ->
      let first = List.hd (Realize.get_call_arg_uops c) in
      List.iteri (fun lane n -> launch b c prg lane (b.named n)) (names first)
  | o -> fail b.fn "%s is no call to run" (Format.asprintf "%a" Op.pp o)

(* Linked schedules *)

type t = {
  lock : Mutex.t; (* the frame is shared: one run at a time *)
  loaded : Rig_program.loaded;
  params : (int * Rig.t list * int) list;
      (* the parameters the runs bind: slot, devices and bytes *)
  cells : (string, int) Hashtbl.t; (* the variables' cells, by name *)
  env : env; (* the variables of the run being made *)
  words : word_kind array;
  inputs : input array;
  frame : Rig_program.frame;
  calls : int;
  kernels : kernel list;
}

let link_with ~separate ~devices ?(bound = []) linear =
  let fn = "Tolk_engine.link" in
  if Ops.op linear <> Op.Linear then fail fn "not a compiled schedule";
  let tbl = Ops.Tbl.create 16 in
  List.iter
    (fun (u, bs) ->
      let ns = names u in
      if Ops.op u <> Op.Buffer || List.length bs <> List.length ns then
        fail fn "%s is not bound to a buffer on each device"
          (Format.asprintf "%a" Ops.pp u);
      List.iter2
        (fun n buf ->
          if
            (not (Rig.equal (Rig.Buffer.device buf) (devices n)))
            || Rig.Buffer.length buf < bytes u
          then
            fail fn "a storage node is not bound to %d bytes of %s" (bytes u) n)
        ns bs;
      Ops.Tbl.replace tbl u bs)
    bound;
  let b =
    {
      fn;
      named = devices;
      separate;
      bound = tbl;
      cells = Hashtbl.create 16;
      rigs = Dynarray.create ();
      memory = Dynarray.create ();
      memories = Ops.Tbl.create 16;
      images = Dynarray.create ();
      image_of = Hashtbl.create 16;
      inputs = Dynarray.create ();
      lane_inputs = Hashtbl.create 16;
      bound_inputs = [];
      params = [];
      words = Dynarray.create ();
      var_words = Hashtbl.create 16;
      computed = Ops.Tbl.create 16;
      trips = Ops.Tbl.create 16;
      steps = [];
      pending = None;
      kernels = [];
    }
  in
  List.iter (entry b) (Ops.src linear);
  flush b;
  let rigs = Dynarray.to_list b.rigs in
  (match rigs with
  | d :: rest ->
      let h = Rig.host_of d in
      List.iter
        (fun d' ->
          if not (Rig.equal (Rig.host_of d') h) then
            fail fn "%s and %s are of two machines" (Rig.name d) (Rig.name d'))
        rest
  | [] -> ());
  let inputs = Dynarray.to_list b.inputs in
  let words = Dynarray.to_array b.words in
  let desc =
    {
      Rig_program.devices = Iarray.of_list (List.map Rig.arch rigs);
      memory = Iarray.of_list (Dynarray.to_list b.memory);
      images = Iarray.of_list (Dynarray.to_list b.images);
      inputs = Iarray.of_list (List.map snd inputs);
      ints = Array.length words;
      steps = Iarray.of_list (List.rev b.steps);
    }
  in
  let loaded =
    match Rig_program.load desc (Iarray.of_list rigs) with
    | Ok l -> l
    | Error why -> failwith (fn ^ ": " ^ why)
  in
  let n = Hashtbl.length b.cells in
  let inputs = Array.of_list (List.map fst inputs) in
  {
    lock = Mutex.create ();
    loaded;
    params = List.rev b.params;
    cells = b.cells;
    env = { values = Array.make n 0; set = Array.make n false };
    words;
    inputs;
    frame =
      {
        inputs = Array.make (Array.length inputs) (Rig.Buffer.of_string "");
        ints = Array.make (Array.length words) 0;
      };
    calls = List.length (Ops.src linear);
    kernels = List.rev b.kernels;
  }

let link ~devices ?bound linear =
  link_with ~separate:(Setting.value Setting.debug >= 2) ~devices ?bound linear

let check_slots t slots =
  let fn = "Tolk_engine.run" in
  List.iter
    (fun (slot, ds, n) ->
      if slot >= Array.length slots then fail fn "no buffers for slot %d" slot;
      let bs = slots.(slot) in
      if List.length bs <> List.length ds then
        fail fn "slot %d takes %d buffers, not %d" slot (List.length ds)
          (List.length bs);
      List.iter2
        (fun d buf ->
          if not (Rig.equal (Rig.Buffer.device buf) d) then
            fail fn "slot %d takes a buffer of %s, not of %s" slot (Rig.name d)
              (Rig.name (Rig.Buffer.device buf));
          if Rig.Buffer.length buf < n then
            fail fn "slot %d takes %d bytes, not %d" slot n
              (Rig.Buffer.length buf))
        ds bs)
    t.params

let buffer slots = function
  | Bound buf -> buf
  | Lane { slot; lane } -> (
      match slots.(slot) with [ buf ] -> buf | bs -> List.nth bs lane)

(* Writes the run's frame: its words, then its inputs. *)
let frame t ~vars slots =
  let env = t.env in
  Array.fill env.set 0 (Array.length env.set) false;
  (* The first binding of a name binds it. *)
  List.iter
    (fun (name, v) ->
      match Hashtbl.find_opt t.cells name with
      | Some i when not env.set.(i) ->
          env.values.(i) <- v;
          env.set.(i) <- true
      | Some _ | None -> ())
    vars;
  let ints = t.frame.ints in
  for w = 0 to Array.length t.words - 1 do
    match t.words.(w) with Computed f -> ints.(w) <- f env | Trip -> ()
  done;
  for k = 0 to Array.length t.inputs - 1 do
    t.frame.inputs.(k) <-
      (match t.inputs.(k) with
      | Whole s -> buffer slots s
      | Part { source; first; length } ->
          Rig.Buffer.view (buffer slots source) ~first:(first ints) ~length)
  done

(* Reports *)

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
        let per x =
          Float.of_int (Shape.sym_infer x vars) /. Float.max s 1e-20
        in
        Printf.sprintf " tm %s (%7.0f GFLOPS %4.0f GB/s)"
          (Helpers.time_to_str ~w:9 s)
          (per e.ops *. 1e-9)
          (per e.mem *. 1e-9)
  in
  let device = String.sub device 0 (Int.min 7 (String.length device)) in
  Printf.printf "*** %-7s %4d %s arg %2d%s\n%!" device count
    (Helpers.ansipad name 46) args timing

(* The nanoseconds of each span of the kernel [k] in [events]. *)
let spans events k =
  List.filter_map
    (function
      | Rig.Profile.Span sp
        when Rig.equal sp.device k.device && sp.name = k.name ->
          Some (sp.stop - sp.start)
      | _ -> None)
    events

(* The variables the run binds, by name, for the reports. *)
let bindings t =
  Hashtbl.fold
    (fun name i vars ->
      if t.env.set.(i) then (name, t.env.values.(i)) :: vars else vars)
    t.cells []

(* A line for each kernel of each span of [events], timed where the span is one
   kernel's. *)
let print_reports t events =
  let vars = bindings t in
  let line k time =
    let args = Realize.get_call_arg_uops k.call in
    report ~device:(Rig.name k.device)
      ~name:(Realize.get_call_name ~var_vals:vars k.call args)
      ~args:(List.length args) ~vars
      (Realize.estimate_uop k.call)
      time
  in
  let kernel d name =
    List.find_opt (fun k -> Rig.equal k.device d && k.name = name) t.kernels
  in
  List.iter
    (function
      | Rig.Profile.Span sp -> (
          match String.split_on_char ',' sp.name |> List.map String.trim with
          | [ name ] ->
              Option.iter
                (fun k ->
                  line k (Some (Float.of_int (sp.stop - sp.start) *. 1e-9)))
                (kernel sp.device name)
          | names ->
              List.iter
                (fun name ->
                  Option.iter (fun k -> line k None) (kernel sp.device name))
                names)
      | _ -> ())
    events

(* Running *)

let submit t ~vars slots =
  check_slots t slots;
  Mutex.protect t.lock (fun () ->
      frame t ~vars slots;
      Rig_program.run t.loaded t.frame)

let run ?(vars = []) t slots =
  let debug = Setting.value Setting.debug in
  if debug >= 1 && t.calls >= 10 then
    Printf.printf "jit execs %d calls\n%!" t.calls;
  if debug < 2 then ignore (submit t ~vars slots)
  else
    let _, events = Rig.Profile.take (fun () -> submit t ~vars slots) in
    print_reports t events

let slots t =
  let n = List.fold_left (fun n (slot, _, _) -> max n (slot + 1)) 0 t.params in
  let slots = Array.make n [] in
  List.iter
    (fun (slot, ds, bytes) ->
      slots.(slot) <- List.map (fun d -> Rig.Buffer.create d (max 1 bytes)) ds)
    t.params;
  slots

(* Timing *)

let link_call ~devices call =
  if Ops.op call <> Op.Call || Ops.op (Ops.body call) <> Op.Program then
    fail "Tolk_engine.link_call" "not a call of a compiled program";
  let targets n = target (devices n) in
  link_with ~separate:true ~devices
    (Realize.compile_linear ~targets (Ops.v Op.Linear ~src:[ call ]))

let clock t =
  if List.for_all (fun k -> Option.is_some (vendor k.device)) t.kernels then
    Search.Device
  else Search.Host

let time ?(vars = []) t slots =
  let wait points = Array.iter Rig.Point.wait points in
  let (), events = Rig.Profile.take (fun () -> wait (submit t ~vars slots)) in
  let ns =
    List.fold_left
      (fun ns k ->
        match spans events k with
        | [] -> (
            match Rig.lost k.device with
            | Some why -> raise (Rig.Lost (k.device, why))
            | None -> ns)
        | s -> List.fold_left ( + ) ns s)
      0 t.kernels
  in
  Float.of_int ns *. 1e-9
