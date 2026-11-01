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
      let here = Rig.equal host Rig.host in
      let cpu = if here then "native" else "generic" in
      (* The functions of a Windows process's libraries follow a convention of
         their own, which the object is compiled for (rig's host code). *)
      let windows = here && Sys.win32 in
      let system = if windows then "-windows" else "" in
      of_kind ~renderer:"CLANG" ~arch:(Rig.arch host ^ system ^ "," ^ cpu) "CPU"
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

let name u =
  match Ops.device u with
  | Some d -> d
  | None -> invalid_arg (Format.asprintf "%a has no device" Ops.pp u)

let is_slot u = Ops.op u = Op.Param && Ops.addrspace u <> Some Dtype.Alu

let bytes u = Shape.max_numel u * Dtype.itemsize (Ops.dtype u)

(* The alignment and phase of storage ({!Ops.type-param_arg}). *)
let start u =
  match Ops.arg u with Ops.Param p -> (p.align, p.phase) | _ -> (1, 0)

(* The bytes by which [buf]'s first byte lies past a multiple of [align]. *)
let phase buf align =
  let rec find j =
    if Rig.Buffer.aligned buf ~at:(-j) align then j else find (j + 1)
  in
  find 0

(* Whether [buf] starts where storage of alignment [align] and phase [p]
   does. *)
let starts_at buf (align, p) =
  align = 1 || Rig.Buffer.aligned buf ~at:(-p) align

(* [n] bytes of [d] that start where storage of alignment [align] and phase [p]
   does: a buffer of its own where [d] starts one there ([Rig.alignment]), and
   else a view into [align - 1] more bytes, with the buffer it views. *)
let made_at d n (align, p) =
  if p = 0 && Rig.alignment d >= align then
    let buf = Rig.Buffer.create d n in
    (buf, buf)
  else
    let buf = Rig.Buffer.create d (n + align - 1) in
    let first = (p - phase buf align + align) mod align in
    (buf, Rig.Buffer.view buf ~first ~length:n)

let create_at d n start = snd (made_at d n start)

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

(* An input of the program, a buffer a run is given: a parameter's buffer, by
   slot, or a buffer bound at link, with its hold, which keeps every call from
   consuming what each run accesses while the schedule lives. *)
type input = Slot of int | Bound of Rig.Buffer.t * Rig.Claim.hold

type storage = Linked | Per_run

(* Where a view lies: in memory made at link, of its bytes from its byte
   [first], or in an input, at an offset; and the alignment and phase its
   storage starts at. *)
type place =
  | Memory of { memory : int; first : int; bytes : int }
  | Input of int

type located = {
  place : place;
  offset : affine;
  device : Rig.t;
  start : int * int;
}

(* Linking *)

(* The submission being built: its device's index, its parts and slots, in
   reverse. *)
type pending = {
  on : int;
  mutable parts : Rig_program.part list;
  slots : Rig_program.slot Dynarray.t;
}

(* A kernel of the schedule, for reports and timing: the device it runs on, its
   function's name and its call. *)
type kernel = {
  device : Rig.t;
  name : string;
  call : Ops.t;
  estimates : Ops.estimates;
}

(* What linking builds: the program's devices, memory, images, inputs and words,
   each by index, and the steps of the loop being built. *)
type builder = {
  fn : string;
  named : string -> Rig.t;
  separate : bool;
  bound : Rig.Buffer.t Ops.Tbl.t;
  cells : (string, int) Hashtbl.t;
  rigs : Rig.t Dynarray.t;
  memory : Rig_program.memory Dynarray.t;
  memories : int Ops.Tbl.t;
  images : Rig_program.image Dynarray.t;
  image_of : (int * string, int) Hashtbl.t;
  inputs : (input * Rig_program.input) Dynarray.t;
  slot_inputs : (int, int) Hashtbl.t;
  mutable bound_inputs : (Rig.Buffer.t * int) list;
  mutable params : (int * Rig.t * int) list;
  starts : (int, int * int) Hashtbl.t;
  words : word_kind Dynarray.t;
  var_words : (string, int) Hashtbl.t;
  computed : int Ops.Tbl.t;
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

(* A free variable is one word per name; a bound one, a word of its own,
   since bound variables of one name may hold several values, one per call. *)
let var_word b u =
  if Shape.is_bound_var u then (
    match Ops.Tbl.find_opt b.computed u with
    | Some w -> w
    | None ->
        let w = new_word b (Computed (reader b.cells u)) in
        Ops.Tbl.replace b.computed u w;
        w)
  else
    let name = Ops.expr u in
    match Hashtbl.find_opt b.var_words name with
    | Some w -> w
    | None ->
        let w = new_word b (Computed (reader b.cells u)) in
        Hashtbl.add b.var_words name w;
        w

(* Whether [u] reads a range: a value that moves with its loop's trips. An
   after's value is its first source; the others only order it, as a loop's
   bound is ordered after a sibling loop. *)
let moves u =
  let cache = Ops.Tbl.create 16 in
  Ops.topovisit u
    (fun n ->
      match (Ops.op n, Ops.src n) with
      | Range, _ -> true
      | After, x :: _ -> Ops.Tbl.find cache x
      | _, s -> List.exists (Ops.Tbl.find cache) s)
    cache

let reads_variable u = List.exists is_variable (Ops.toposort ~calls:Skip u)

(* The word that holds [u], an expression of the run's variables, which each
   run computes. *)
let computed_word b u =
  match Ops.Tbl.find_opt b.computed u with
  | Some w -> w
  | None ->
      let w = new_word b (Computed (Shape.sym_compile (Sym u) (reader b.cells))) in
      Ops.Tbl.replace b.computed u w;
      w

let int_of_const u =
  match Ops.arg u with
  | Ops.Const (`Int z) -> Bigint.to_int z
  | Ops.Const (`Bool x) -> Bool.to_int x
  | _ -> invalid_arg (Format.asprintf "%a is no integer" Ops.pp u)

(* [u] as a form of words: a range's trip moves it by a constant factor, and
   the rest of it is computed for the run. *)
let rec affine b u =
  match (Ops.op u, Ops.arg u) with
  | Op.Const, _ -> constant (int_of_const u)
  | Op.Range, _ -> (
      match Ops.Tbl.find_opt b.trips u with
      | Some w -> word w
      | None -> fail b.fn "%s reads a range outside its loop" (Ops.expr u))
  | Op.Param, _ when is_variable u -> word (var_word b u)
  | Op.Add, _ ->
      plus (affine b (Ops.nth u 0)) (affine b (Ops.nth u 1))
  | Op.Mul, _ -> (
      let x = affine b (Ops.nth u 0) and y = affine b (Ops.nth u 1) in
      match (x.terms, y.terms) with
      | [], _ -> scale x.add y
      | _, [] -> scale y.add x
      | _ -> computed b u)
  | Op.Cast, _ when Dtype.is_int (Ops.dtype u) -> affine b (Ops.nth u 0)
  | Op.After, _ -> affine b (Ops.nth u 0)
  | _ -> computed b u

and computed b u =
  if moves u then
    fail b.fn "%s moves with its loop's trips by no constant factor"
      (Ops.expr u)
  else if not (reads_variable u) then constant (Shape.sym_infer (Sym u) [])
  else word (computed_word b u)

let affine_sint b : Ops.sint -> affine = function
  | Int n -> constant n
  | Sym u -> affine b u

(* A value of one word, such as a grid's size or a loop's trips. *)
let value b (s : Ops.sint) : Rig_program.value =
  let a = affine_sint b s in
  match (a.terms, s) with
  | [], _ -> Fixed a.add
  | [ (w, 1) ], _ when a.add = 0 -> Int w
  | _, Sym u when not (moves u) -> Int (computed_word b u)
  | _ -> fail b.fn "a value of one word moves with its loop's trips"

(* Storage *)

(* The memory of storage [base] made at link on [d], whose first [first] bytes
   lie before the storage's. *)
let memory b base d ~first =
  match Ops.Tbl.find_opt b.memories base with
  | Some m -> m
  | None ->
      let m =
        Rig_program.Alloc
          {
            device = device_index b d;
            kind = Device;
            bytes = first + max 1 (bytes base);
            init = { bytes = ""; holes = [||] };
            copies = One;
          }
      in
      let i = push b.memory m in
      Ops.Tbl.replace b.memories base i;
      i

let add_input b input desc = push b.inputs (input, desc)

let slot_input b ~slot d n =
  match Hashtbl.find_opt b.slot_inputs slot with
  | Some i -> i
  | None ->
      let i = add_input b (Slot slot) { device = device_index b d; bytes = n } in
      Hashtbl.replace b.slot_inputs slot i;
      i

(* A hold of the bound [buf], refused in the link's name. *)
let hold b buf =
  match Rig.Claim.hold [ buf ] with
  | h -> h
  | exception Invalid_argument why -> fail b.fn "a bound buffer: %s" why

let bound_input b buf =
  match List.assq_opt buf b.bound_inputs with
  | Some i -> i
  | None ->
      let i =
        add_input b (Bound (buf, hold b buf))
          {
            device = device_index b (Rig.Buffer.device buf);
            bytes = Rig.Buffer.length buf;
          }
      in
      b.bound_inputs <- (buf, i) :: b.bound_inputs;
      i

let note_param b slot d n =
  if not (List.exists (fun (s, _, _) -> s = slot) b.params) then
    b.params <- (slot, d, n) :: b.params

(* The bytes before storage made at link on [d] in its memory, which starts on
   [d]'s alignment ([Rig.alignment]): the storage's phase. An io device's
   storage has none. *)
let first_byte b base d =
  let align, p = start base in
  if not (Rig.computes d) then 0
  else if Rig.alignment d < align then
    fail b.fn "%s starts memory on %d bytes, not the %d of %s" (Rig.name d)
      (Rig.alignment d) align
      (Format.asprintf "%a" Ops.pp base)
  else p

(* Storage made for [base] at link. *)
let memory_place b base offset =
  let d = b.named (name base) in
  let first = first_byte b base d in
  let memory = memory b base d ~first in
  {
    place = Memory { memory; first; bytes = max 1 (bytes base) };
    offset;
    device = d;
    start = start base;
  }

(* Where the view [u] lies: in a parameter's buffer, which a run gives; in
   bound storage; or in storage made at link. *)
let locate b u =
  let base, off = view_offset u in
  let offset = affine_sint b off in
  match (Ops.op base, Ops.arg base) with
  | Op.Param, Ops.Param p when is_slot base ->
      let d = b.named (name base) and n = bytes base in
      note_param b p.slot d n;
      {
        place = Input (slot_input b ~slot:p.slot d n);
        offset;
        device = d;
        start = start base;
      }
  | Op.Alloc, _ -> memory_place b base offset
  | Op.Buffer, _ -> (
      match Ops.Tbl.find_opt b.bound base with
      | Some buf ->
          {
            place = Input (bound_input b buf);
            offset;
            device = Rig.Buffer.device buf;
            start = start base;
          }
      | None -> memory_place b base offset)
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

(* The submission of [on] that takes the next part. *)
let pending b on =
  match b.pending with
  | Some p when p.on = on -> p
  | _ ->
      flush b;
      let p = { on; parts = []; slots = Dynarray.create () } in
      b.pending <- Some p;
      p

let add_part b p part =
  p.parts <- part :: p.parts;
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
  | Memory { memory; first; bytes } ->
      Memory { memory; offset = first; length = bytes }
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

type grid = Rig_program.value * Rig_program.value * Rig_program.value

let one : grid = (Fixed 1, Fixed 1, Fixed 1)

let triple : Rig_program.value list -> grid = function
  | [ x; y; z ] -> (x, y, z)
  | [ x; y ] -> (x, y, Fixed 1)
  | [ x ] -> (x, Fixed 1, Fixed 1)
  | _ -> (Fixed 1, Fixed 1, Fixed 1)

(* How a run computes [s], which reads no range. *)
let counter b (s : Ops.sint) =
  match s with
  | Int n -> fun _ -> n
  | Sym u when moves u ->
      fail b.fn "a split loop's iterations move with its loop's trips"
  | Sym _ -> Shape.sym_compile s (reader b.cells)

(* The grid of a host program that splits its loop of [extent] iterations into
   blocks: a group of threads per block, a thread per iteration. *)
let split_grid b (extent : Ops.sint) (ops : Ops.sint) : grid * grid =
  match (extent, ops) with
  | Int extent, Int ops ->
      let n, t = blocks ~extent ~ops in
      ((Fixed n, Fixed 1, Fixed 1), (Fixed t, Fixed 1, Fixed 1))
  | extent, ops ->
      let extent = counter b extent and ops = counter b ops in
      let n env = fst (blocks ~extent:(extent env) ~ops:(ops env)) in
      let t env = snd (blocks ~extent:(extent env) ~ops:(ops env)) in
      ( (Int (new_word b (Computed n)), Fixed 1, Fixed 1),
        (Int (new_word b (Computed t)), Fixed 1, Fixed 1) )

(* A kernel's vector accesses lie on their width only where each buffer it
   accesses starts at its storage's alignment and phase ({!Tolk.Codegen}).
   Linking checks a bound buffer here, and each run the buffers of parameters. A
   copy moves bytes wherever they start. *)
let note_start b (l : located) =
  match l.place with
  | Memory _ -> ()
  | Input i -> (
      match fst (Dynarray.get b.inputs i) with
      | Slot slot -> (
          match Hashtbl.find_opt b.starts slot with
          | None -> Hashtbl.replace b.starts slot l.start
          | Some s when s = l.start -> ()
          | Some (align, p) ->
              let align', p' = l.start in
              fail b.fn
                "slot %d is accessed %d bytes past a multiple of %d and %d \
                 past %d"
                slot p align p' align')
      | Bound (buf, _) when not (starts_at buf l.start) ->
          let align, p = l.start in
          fail b.fn
            "a bound buffer of %s starts %d bytes past a multiple of %d, not %d"
            (Rig.name (Rig.Buffer.device buf))
            (phase buf align) align p
      | Bound _ -> ())

(* The launch of the compiled program [prg] that [call] makes on [d]. A host
   program runs on its device's host. *)
let launch b call prg d =
  let info = program_info prg and k = kernel_of prg in
  let elf = Device.Tiny_elf.of_program prg in
  let kind = (target d).device in
  if kind <> info.target.device then
    fail b.fn "%s runs no %s program" (Rig.name d) info.target.device;
  let host = kind = "CPU" in
  let on = if host then Rig.host_of d else d in
  let dev = device_index b on in
  let image = image b dev elf.lib in
  let slot_of p = match Ops.arg p with Ops.Param p -> p.slot | _ -> -1 in
  (* Its parameters, in slot order, the order the program takes them, as
     parameters or packed into its bytes: a pointer to each buffer and each
     scalar, each aligned to its size, as the program reads them
     ({!Tolk.Codegen}). A split program's bounds are those of its whole loop,
     which its entry narrows to each block's. *)
  let taken =
    List.filter (fun u -> Ops.op u = Op.Param) (Ops.src (Ops.nth prg 1))
  in
  let position (p, _) =
    Option.value (List.find_index (Ops.equal p) taken) ~default:max_int
  in
  let fields =
    List.stable_sort
      (fun a b -> Int.compare (position a) (position b))
      (List.combine (Shape.kernel_params prg)
         (Realize.get_call_kernel_args call prg))
  in
  let signature =
    List.map
      (fun (p, _) ->
        let addrspace =
          match Ops.arg p with Ops.Param a -> a.addrspace | _ -> Dtype.Global
        in
        {
          Device.Tiny_elf.name = None;
          addrspace;
          dtype = Ops.dtype p;
          shape = [];
        })
      fields
  in
  let offsets = Device.Tiny_elf.iter_sig signature in
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
  let p = pending b dev in
  let refs =
    List.concat
      (List.map2
         (fun (prm, a) (at, dt) ->
           if Ops.addrspace prm = Some Dtype.Alu then begin
             place b params holes at (Dtype.itemsize dt) (affine b a);
             []
           end
           else
             let l = locate b a in
             note_start b l;
             place b params holes at 8 l.offset;
             let use : Rig_program.use =
               if List.mem (slot_of prm) info.outs then Write else Read
             in
             [ { Rig_program.slot = slot_index p (whole l); at; use } ])
         fields offsets)
  in
  (* A grid, as a split loop's size, reads a nameless scalar as its call
     passes it, in its slot. *)
  let args = Ops.src_without_body call in
  let passed : Ops.sint -> Ops.sint = function
    | Int n -> Int n
    | Sym u ->
        let slotted v =
          match Ops.arg v with
          | Ops.Param { name = None; slot; addrspace = Dtype.Alu; _ }
            when slot >= 0 && slot < List.length args ->
              Some (v, List.nth args slot)
          | _ -> None
        in
        let subs = List.filter_map slotted (Ops.toposort ~calls:Skip u) in
        Sym (Ops.substitute ~calls:Skip ~pass:Once u subs)
  in
  let groups, threads =
    match k.split with
    | Some s ->
        (* The bound [hi] of the whole loop is its iterations. *)
        let extent =
          List.find_map
            (fun (prm, x) -> if slot_of prm = s.hi then Some (sint x) else None)
            fields
          |> Option.value ~default:(passed s.iterations)
        in
        let ops =
          match k.estimates with Some e -> passed e.ops | None -> extent
        in
        split_grid b extent ops
    | None when host -> (one, one)
    | None ->
        let dims d = triple (List.map (fun s -> value b (passed s)) d) in
        (dims info.global_size, dims info.local_size)
  in
  let estimates =
    let e = Realize.estimate_uop call in
    { Ops.ops = passed e.ops; lds = passed e.lds; mem = passed e.mem }
  in
  b.kernels <- { device = on; name = elf.name; call; estimates } :: b.kernels;
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

(* [l] as the end of a copy part or a move: its memory or input whole, at the
   offset each step computes from the run's ints. *)
let copy_end (l : located) :
    Rig_program.slot * (Rig_program.value * int) iarray =
  let slot : Rig_program.slot =
    match l.place with
    | Memory { memory; first; bytes } ->
        Memory { memory; offset = first; length = bytes }
    | Input i -> Input i
  in
  let words = List.map (fun (w, f) -> (Rig_program.Int w, f)) l.offset.terms in
  let add =
    if l.offset.add = 0 then [] else [ (Rig_program.Fixed l.offset.add, 1) ]
  in
  (slot, Iarray.of_list (add @ words))

(* The copy [call] makes: a copy part of the source's device or the
   destination's, the source's first, where it copies between the two
   ([copies_between]), or else a move of the program. *)
let copy b call =
  match Realize.get_call_arg_uops call with
  | [ dst; src ] ->
      let length = bytes dst in
      let d = locate b dst and s = locate b src in
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
            (Move
               {
                 src = src_slot;
                 src_offset;
                 dst = dst_slot;
                 dst_offset;
                 length;
               })
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
      let flag = locate b (Ops.nth e 2) in
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
      let trips = value b (sint (Ops.nth r 0)) in
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
      let base, _ = view_offset u in
      match Ops.arg base with
      | Ops.Param p when is_slot base ->
          note_param b p.slot (b.named (name base)) (bytes base)
      | _ -> ())
    (Realize.get_call_arg_uops c);
  match Ops.op prg with
  | Op.Store -> copy b c
  | Op.Program ->
      let first = List.hd (Realize.get_call_arg_uops c) in
      launch b c prg (b.named (name first))
  | o -> fail b.fn "%s is no call to run" (Format.asprintf "%a" Op.pp o)

(* Linked schedules *)

(* Storage each run of a schedule linked per run makes: [bytes] of [device]
   starting at the alignment and phase [at], in the slot that the calls from
   its first to its last read it from. *)
type made = { slot : int; device : Rig.t; bytes : int; at : int * int }

(* A loaded program of rig's that runs consecutive calls of a schedule. *)
type part = {
  loaded : Rig_program.loaded;
  params : (int * Rig.t * int) list;
      (* the parameters the runs bind: slot, device and bytes *)
  starts : (int * (int * int)) list;
      (* the parameters kernels read, by slot, and the alignment and phase
         their buffers start at *)
  cells : (string, int) Hashtbl.t; (* the variables' cells, by name *)
  env : env; (* the variables of the run being made *)
  words : word_kind array;
  inputs : input array;
  frame : Rig_program.frame;
  idle : Rig.Buffer.t;
      (* what the frame's inputs hold between runs, so that the schedule keeps
         none of a run's buffers *)
  makes : made list; (* the storage a run makes before the part *)
  frees : made list; (* and returns after it *)
}

type t = {
  lock : Mutex.t; (* the frames are shared: one run at a time *)
  parts : part array;
  given : (int * Rig.t * int) list;
      (* the parameters the caller binds: slot, device and bytes *)
  given_starts : (int * (int * int)) list;
      (* the given parameters kernels read, and where their buffers start *)
  width : int; (* the slots a run binds, the made storage's included *)
  made : made array;
  calls : int;
  kernels : kernel list;
}

(* The calls [entries] that [b] links as a part. *)
let assemble b entries ~makes ~frees =
  List.iter (entry b) entries;
  flush b;
  let rigs = Dynarray.to_list b.rigs in
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
    | Error why -> failwith (b.fn ^ ": " ^ why)
  in
  let n = Hashtbl.length b.cells in
  let inputs = Array.of_list (List.map fst inputs) in
  let idle = Rig.Buffer.of_string "" in
  {
    loaded;
    params = List.rev b.params;
    starts = Hashtbl.fold (fun k v l -> (k, v) :: l) b.starts [];
    cells = b.cells;
    env = { values = Array.make n 0; set = Array.make n false };
    words;
    inputs;
    frame =
      {
        inputs = Array.make (Array.length inputs) idle;
        ints = Array.make (Array.length words) 0;
        step_points = Rig_program.step_points loaded;
      };
    idle;
    makes;
    frees;
  }

(* Ends the holds of the buffers bound in [b], whose link raised, so that a
   donation right after the raise consumes their memory. *)
let release_bound (b : builder) =
  Dynarray.iter
    (function Bound (_, h), _ -> Rig.Claim.release h | Slot _, _ -> ())
    b.inputs

let builder ~fn ~separate ~devices bound =
  {
    fn;
    named = devices;
    separate;
    bound;
    cells = Hashtbl.create 16;
    rigs = Dynarray.create ();
    memory = Dynarray.create ();
    memories = Ops.Tbl.create 16;
    images = Dynarray.create ();
    image_of = Hashtbl.create 16;
    inputs = Dynarray.create ();
    slot_inputs = Hashtbl.create 16;
    bound_inputs = [];
    params = [];
    starts = Hashtbl.create 16;
    words = Dynarray.create ();
    var_words = Hashtbl.create 16;
    computed = Ops.Tbl.create 16;
    trips = Ops.Tbl.create 16;
    steps = [];
    pending = None;
    kernels = [];
  }

(* Splitting a schedule linked per run

   Each storage node that no buffer is bound to becomes a parameter in a slot
   past the caller's, which each run binds to storage it makes before the first
   call that reads the node and returns after the last. The calls are cut into
   parts wherever storage starts or ends between two of them: a part is one
   program of rig's, whose inputs it binds once, before its first step. A range
   around calls is one call. Where calls 0 and 1 read [a] and call 2 reads [b],
   calls 0 and 1 are a part, which makes [a] and returns it, and call 2 another,
   which makes [b] and returns it. *)

(* The storage nodes that [e] reads and no buffer is bound to. *)
let made_storage bound e =
  List.filter
    (fun u ->
      match Ops.op u with
      | Op.Alloc -> true
      | Op.Buffer -> not (Ops.Tbl.mem bound u)
      | _ -> false)
    (Ops.toposort ~calls:Skip e)

let first_free_slot linear =
  List.fold_left
    (fun n u ->
      match Ops.arg u with
      | Ops.Param p when is_slot u -> max n (p.slot + 1)
      | _ -> n)
    0
    (Ops.toposort ~calls:Skip linear)

(* The parts of [linear] linked per run: each part's calls, and the storage made
   before it and returned after it. *)
let per_run_parts ~named bound linear =
  let entries = Array.of_list (Ops.src linear) in
  let reads = Array.map (made_storage bound) entries in
  let first = Ops.Tbl.create 16 and last = Ops.Tbl.create 16 in
  let order = ref [] in
  Array.iteri
    (fun i us ->
      List.iter
        (fun u ->
          if not (Ops.Tbl.mem first u) then begin
            Ops.Tbl.replace first u i;
            order := u :: !order
          end;
          Ops.Tbl.replace last u i)
        us)
    reads;
  let next = first_free_slot linear in
  let made =
    List.mapi
      (fun k u ->
        let d = named (name u) in
        let at = if Rig.computes d then start u else (1, 0) in
        (u, { slot = next + k; device = d; bytes = max 1 (bytes u); at }))
      (List.rev !order)
  in
  let param (u, m) =
    let align, phase = start u in
    ( u,
      Shape.param
        ~shape:[ Ops.Int (Shape.max_numel u) ]
        ?device:(Ops.device u) ~phase ~align m.slot (Ops.dtype u) )
  in
  let entries =
    Array.of_list
      (Ops.src
         (Ops.substitute ~calls:Skip ~pass:Once linear (List.map param made)))
  in
  let starting i = List.filter (fun (u, _) -> Ops.Tbl.find first u = i) made
  and ending i = List.filter (fun (u, _) -> Ops.Tbl.find last u = i) made in
  let n = Array.length entries in
  let parts = ref [] and calls = ref [] and makes = ref [] and frees = ref [] in
  for i = 0 to n - 1 do
    calls := entries.(i) :: !calls;
    makes := List.map snd (starting i) @ !makes;
    frees := List.map snd (ending i) @ !frees;
    if i = n - 1 || !frees <> [] || starting (i + 1) <> [] then begin
      parts := (List.rev !calls, List.rev !makes, List.rev !frees) :: !parts;
      calls := [];
      makes := [];
      frees := []
    end
  done;
  (List.rev !parts, Array.of_list (List.map snd made))

let link_with ~separate ~devices ?(bound = []) ?(storage = Linked) linear =
  let fn = "Tolk_engine.link" in
  if Ops.op linear <> Op.Linear then fail fn "not a compiled schedule";
  let tbl = Ops.Tbl.create 16 in
  List.iter
    (fun (u, buf) ->
      if Ops.op u <> Op.Buffer then
        fail fn "%s is not storage" (Format.asprintf "%a" Ops.pp u);
      let n = name u in
      if
        (not (Rig.equal (Rig.Buffer.device buf) (devices n)))
        || Rig.Buffer.length buf < bytes u
      then fail fn "a storage node is not bound to %d bytes of %s" (bytes u) n;
      Ops.Tbl.replace tbl u buf)
    bound;
  let parts, made =
    match storage with
    | Linked -> ([ (Ops.src linear, [], []) ], [||])
    | Per_run -> per_run_parts ~named:devices tbl linear
  in
  let built = ref [] in
  match
    List.map
      (fun (entries, makes, frees) ->
        let b = builder ~fn ~separate ~devices tbl in
        built := b :: !built;
        (b, assemble b entries ~makes ~frees))
      parts
  with
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      List.iter release_bound !built;
      Printexc.raise_with_backtrace e bt
  | linked ->
      let is_made slot = Array.exists (fun m -> m.slot = slot) made in
      let given = ref [] and given_starts = ref [] in
      List.iter
        (fun (_, (p : part)) ->
          List.iter
            (fun ((slot, _, _) as g) ->
              let seen = List.exists (fun (s, _, _) -> s = slot) !given in
              if not (is_made slot || seen) then given := g :: !given)
            p.params;
          List.iter
            (fun (slot, ((align, ph) as s)) ->
              if not (is_made slot) then
                match List.assoc_opt slot !given_starts with
                | None -> given_starts := (slot, s) :: !given_starts
                | Some s' when s' = s -> ()
                | Some (align', ph') ->
                    List.iter release_bound !built;
                    fail fn
                      "slot %d is accessed %d bytes past a multiple of %d and \
                       %d past %d"
                      slot ph align ph' align')
            p.starts)
        linked;
      let width =
        List.fold_left
          (fun n (s, _, _) -> max n (s + 1))
          (Array.fold_left (fun n m -> max n (m.slot + 1)) 0 made)
          !given
      in
      {
        lock = Mutex.create ();
        parts = Array.of_list (List.map snd linked);
        given = List.rev !given;
        given_starts = !given_starts;
        width;
        made;
        calls = List.length (Ops.src linear);
        kernels =
          List.concat_map (fun ((b : builder), _) -> List.rev b.kernels) linked;
      }

let link ~devices ?bound ?storage linear =
  link_with
    ~separate:(Setting.value Setting.debug >= 2)
    ~devices ?bound ?storage linear

(* The buffer [slots] binds to [slot], which [check_slots] found. *)
let slot_buffer slots slot = Option.get slots.(slot)

(* Checks, without allocating, that each buffer a kernel accesses starts at its
   parameter's phase. *)
let rec check_starts fn slots = function
  | [] -> ()
  | (slot, start) :: rest ->
      let buf = slot_buffer slots slot in
      if not (starts_at buf start) then begin
        let align, p = start in
        fail fn
          "slot %d's buffer of %s starts %d bytes past a multiple of %d, not %d"
          slot
          (Rig.name (Rig.Buffer.device buf))
          (phase buf align) align p
      end;
      check_starts fn slots rest

let check_slots t slots =
  let fn = "Tolk_engine.run" in
  List.iter
    (fun (slot, d, n) ->
      match if slot < Array.length slots then slots.(slot) else None with
      | None -> fail fn "no buffer for slot %d" slot
      | Some buf ->
          if not (Rig.equal (Rig.Buffer.device buf) d) then
            fail fn "slot %d takes a buffer of %s, not of %s" slot (Rig.name d)
              (Rig.name (Rig.Buffer.device buf));
          if Rig.Buffer.length buf < n then
            fail fn "slot %d takes %d bytes, not %d" slot n
              (Rig.Buffer.length buf))
    t.given;
  check_starts fn slots t.given_starts

(* Writes the words of [p]'s frame: the variables [vars] binds, and the values
   computed from them. *)
let words (p : part) ~vars =
  let env = p.env in
  Array.fill env.set 0 (Array.length env.set) false;
  (* The first binding of a name binds it. *)
  List.iter
    (fun (name, v) ->
      match Hashtbl.find_opt p.cells name with
      | Some i when not env.set.(i) ->
          env.values.(i) <- v;
          env.set.(i) <- true
      | Some _ | None -> ())
    vars;
  let ints = p.frame.ints in
  for w = 0 to Array.length p.words - 1 do
    match p.words.(w) with Computed f -> ints.(w) <- f env | Trip -> ()
  done

(* Writes the inputs of [p]'s frame from [slots]. *)
let inputs (p : part) slots =
  for k = 0 to Array.length p.inputs - 1 do
    p.frame.inputs.(k) <-
      (match p.inputs.(k) with
      | Bound (buf, _) -> buf
      | Slot slot -> slot_buffer slots slot)
  done

(* Lets [p]'s frame drop the run's buffers. *)
let idle (p : part) =
  Array.fill p.frame.inputs 0 (Array.length p.frame.inputs) p.idle

(* Runs [t]'s parts in order on [slots], calling [each] with each part and its
   ends. A part's storage is made before it runs, in its slot, and returned
   after the last part that reads it, its memory coming back once the work on it
   is done; on a raise, all of it is returned. *)
let run_parts t ~vars slots each =
  let all = Array.make t.width None in
  Array.blit slots 0 all 0 (Int.min (Array.length slots) t.width);
  Array.iter (fun p -> words p ~vars) t.parts;
  let held = Array.make t.width None in
  let return m =
    match held.(m.slot) with
    | None -> ()
    | Some whole ->
        held.(m.slot) <- None;
        all.(m.slot) <- None;
        ignore (Rig.Buffer.free ~why:"a run's storage" whole : bool)
  in
  let make m =
    let whole, buf = made_at m.device m.bytes m.at in
    held.(m.slot) <- Some whole;
    all.(m.slot) <- Some buf
  in
  let run_part (p : part) =
    List.iter make p.makes;
    inputs p all;
    Fun.protect
      ~finally:(fun () -> idle p)
      (fun () -> each p (Rig_program.run p.loaded p.frame));
    List.iter return p.frees
  in
  Fun.protect
    ~finally:(fun () -> Array.iter return t.made)
    (fun () -> Array.iter run_part t.parts)

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
        (* A variable the run does not bind takes its bound value. *)
        let var n vars =
          match (List.assoc_opt (Ops.expr n) vars, Ops.arg n) with
          | Some x, _ -> x
          | None, Ops.Param { bound = Some x; _ } -> Dtype.Value.to_int x
          | None, _ -> invalid_arg ("the variable " ^ Ops.expr n ^ " has no value")
        in
        let per x =
          Float.of_int (Shape.sym_compile x var vars) /. Float.max s 1e-20
        in
        Printf.sprintf " tm %s (%7.0f GFLOPS %4.0f GB/s)"
          (Helpers.time_to_str ~w:9 s)
          (per e.ops *. 1e-9)
          (per e.mem *. 1e-9)
  in
  let device = String.sub device 0 (Int.min 7 (String.length device)) in
  Printf.printf "*** %-7s %4d %s arg %2d%s\n%!" device count
    (Helpers.ansipad name 46) args timing

(* The variables the run binds, by name, for the reports. *)
let bindings t =
  Array.fold_left
    (fun vars (p : part) ->
      Hashtbl.fold
        (fun name i vars ->
          if p.env.set.(i) && not (List.mem_assoc name vars) then
            (name, p.env.values.(i)) :: vars
          else vars)
        p.cells vars)
    [] t.parts

(* A line for each kernel of each submission's work in [events], timed where the
   work is one kernel's. *)
let print_reports t events =
  let vars = bindings t in
  let line k time =
    let args = Realize.get_call_arg_uops k.call in
    report ~device:(Rig.name k.device)
      ~name:(Realize.get_call_name ~var_vals:vars k.call args)
      ~args:(List.length args) ~vars k.estimates time
  in
  let kernel d name =
    List.find_opt
      (fun (k : kernel) -> Rig.equal k.device d && k.name = name)
      t.kernels
  in
  List.iter
    (function
      | Rig.Profile.Work w -> (
          match String.split_on_char ',' w.name |> List.map String.trim with
          | [ name ] ->
              Option.iter
                (fun k ->
                  line k (Some (Float.of_int (w.stop - w.start) *. 1e-9)))
                (kernel w.device name)
          | names ->
              List.iter
                (fun name ->
                  Option.iter (fun k -> line k None) (kernel w.device name))
                names)
      | _ -> ())
    events

(* Running *)

let submit t ~vars slots =
  check_slots t slots;
  Mutex.protect t.lock (fun () -> run_parts t ~vars slots (fun _ _ -> ()))

let run ?(vars = []) t slots =
  let debug = Setting.value Setting.debug in
  if debug >= 1 && t.calls >= 10 then
    Printf.printf "jit execs %d calls\n%!" t.calls;
  if debug < 2 then ignore (submit t ~vars slots)
  else
    let _, events = Rig.Profile.take (fun () -> submit t ~vars slots) in
    print_reports t events

let slots t =
  let n = List.fold_left (fun n (slot, _, _) -> max n (slot + 1)) 0 t.given in
  let slots = Array.make n None in
  List.iter
    (fun (slot, d, bytes) ->
      let buf =
        match List.assoc_opt slot t.given_starts with
        | Some start -> create_at d (max 1 bytes) start
        | None -> Rig.Buffer.create d (max 1 bytes)
      in
      slots.(slot) <- Some buf)
    t.given;
  slots

(* Timing *)

let link_call ~devices call =
  if Ops.op call <> Op.Call || Ops.op (Ops.body call) <> Op.Program then
    fail "Tolk_engine.link_call" "not a call of a compiled program";
  let targets n = target (devices n) in
  link_with ~separate:true ~devices
    (Realize.compile_linear ~targets (Ops.v Op.Linear ~src:[ call ]))

let clock t =
  if
    List.for_all
      (fun (k : kernel) -> Option.is_some (vendor k.device))
      t.kernels
  then Search.Device
  else Search.Host

(* Whether the work [w] runs a kernel of [t]: its name lists the kernels of its
   submission, joined by [", "]. A submission of copies alone names none. *)
let runs_kernel t (w : Rig.Profile.event) =
  match w with
  | Work { device; name; _ } ->
      let names = List.map String.trim (String.split_on_char ',' name) in
      List.exists
        (fun (k : kernel) -> Rig.equal k.device device && List.mem k.name names)
        t.kernels
  | _ -> false

(* The run's work in a profile is the [Work] whose point a step of the run
   reported in the frame, which the lock keeps the run's until it is copied:
   another domain's work of the same kernels has other points. *)
let time ?(vars = []) t slots =
  check_slots t slots;
  let steps = ref [] and ends = ref [] in
  let run () =
    Mutex.protect t.lock (fun () ->
        run_parts t ~vars slots (fun p e ->
            steps := Array.copy p.frame.step_points :: !steps;
            ends := e :: !ends));
    List.iter (Array.iter Rig.Point.wait) !ends
  in
  let (), events = Rig.Profile.take run in
  let ns = ref 0 in
  List.iter
    (function
      | Rig.Profile.Work w as e
        when List.exists (Array.mem w.point) !steps && runs_kernel t e ->
          ns := !ns + (w.stop - w.start)
      | _ -> ())
    events;
  Float.of_int !ns *. 1e-9
