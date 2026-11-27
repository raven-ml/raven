(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call refuses what nx.kernel's rules refuse, then its family plans it as one
   call ({!Plan}), one submission of its launches or several in order, its
   parts, which the door submits:

   map, reduce, scan --> device (Rig.Local) --> Operands.v --> a map's plan of
   its program and facts, kept, else Moves.map where a map moves bits, else Loop
   --> Plan.t --> door --> the submission of its key, its kernels' pipelines
   made at their first use and their families' prepared, its slots bound to the
   call's arrays, the device's workspace, tickets and constants and fresh
   buffers, its launches' values --> Rig.submit, for each part --> fresh buffers
   freed

   A map that finds every cache warm allocates its facts, its slots' buffers and
   the door's work, and nothing that outlives it. *)

module A = Nx_array
module D = Nx_array.Dtype
module Sub = Rig.Submission

let name = "nx.metal"

(* The GPU family the metallib's kernels are checked on. *)
let gpu_family = "Apple7"

(* Devices *)

(* What this library keeps for a device [rig]: the metallib's [image] and its
   kernels' functions by {!Kernels.index}, each resolved at its first use; the
   workspace, which holds a call's temporaries, grown to {!Plan.kept_workspace}
   bytes; the tickets, zero words that every call leaves zero, grown to
   {!Plan.kept_tickets} bytes, so no launch zeroes them; in [subs] the
   submissions of the calls made, by their key ({!Plan.key}), a cache for each
   first kernel; in [plans] the maps' plans, by their program and facts
   ({!call}), in [buckets] caches by the facts' hash; in [constants] the
   constants' buffers by key; in [keys] the functions of the specialised steps'
   keys made, each of an image of its own; and in [key_subs] the submissions of
   the calls that launch a key, with the keys they hold. Every call that uses
   the workspace or the tickets names them as written, so rig orders those
   calls. *)
type device = {
  rig : Rig.t;
  image : Rig.Image.t;
  funcs : Rig.Image.func option Atomic.t array;
  workspace : Rig.Buffer.t Atomic.t;
  tickets : Rig.Buffer.t Atomic.t;
  subs : (int array, Sub.t) Cache.t array;
  plans : (call, entry) Cache.t array;
  constants : (string, Rig.Buffer.t) Cache.t;
  keys : (Kernels.t * int array, Rig.Image.func) Cache.t;
  key_subs : (int array, Sub.t * int) Cache.t;
}

(* A map: its program, physically, its facts' key ({!Operands.t}) and the epoch
   ({!Plan.epoch}) its plan was made at. *)
and call = { prog : Nx_kernel.Prog.t; epoch : int; ints : int array }

(* A map's plan, and the bytes an entry of it holds. *)
and entry = { answer : (Plan.t option, A.answer) result; bytes : int }

(* The caches of map plans a device keeps, chosen by the facts' hash. *)
let buckets = 256

(* A device of the Apple7 or Apple8 family, which loads the metallib. *)
let load d =
  let arch = Rig.arch d in
  if not (String.equal arch gpu_family || String.equal arch "Apple8") then None
  else
    match Rig.Image.load d Metallib.metallib with
    | Error _ -> None
    | Ok image ->
        let n = Array.length Kernels.all in
        Some
          {
            rig = d;
            image;
            funcs = Array.init n (fun _ -> Atomic.make None);
            workspace = Atomic.make (Rig.Buffer.create d 0);
            tickets = Atomic.make (Rig.Buffer.create d 0);
            subs = Array.init n (fun _ -> Cache.make ());
            plans = Array.init buckets (fun _ -> Cache.make ());
            constants = Cache.make ();
            keys = Cache.make ();
            key_subs = Cache.make ();
          }

(* Each device's [load], [None] where it does not compute. *)
let devices = Rig.Local.new_key load
let device d = Rig.Local.get devices d
let computes_on d = Option.is_some (device d)

(* Pipelines

   Metal makes a kernel's pipeline at its first resolve, in milliseconds for a
   small kernel and a fifth of a second for a large one. A call prepares its
   launches' families, so that Metal makes their pipelines concurrently, then
   resolves its own launches' kernels alone, so that it waits for no other's.
   Two domains may resolve one kernel at once; rig gives both one function. *)

let resolve dv k =
  let what = Kernels.name k in
  match Rig.Image.resolve dv.image what with
  | None -> failwith ("nx.metal: the metallib has no kernel " ^ what)
  | Some f -> (
      match Rig.Image.threads f with
      | Some t when t < Kernels.threads k ->
          failwith
            (Printf.sprintf
               "nx.metal: %s's groups hold at most %d threads, not %d" what t
               (Kernels.threads k))
      | Some _ | None -> f)

(* Starts making the pipelines of [k]'s family where [k]'s is not made, [k]'s
   first, so that a family larger than Metal compiles at once does not queue it
   behind the others. *)
let prepare_family dv k =
  if Option.is_none (Atomic.get dv.funcs.(Kernels.index k)) then begin
    let fam = Kernels.family k in
    Rig.Image.prepare dv.image (Kernels.name k);
    Array.iter
      (fun j ->
        if j <> k && Kernels.family j = fam then
          Rig.Image.prepare dv.image (Kernels.name j))
      Kernels.all
  end

let func dv k =
  let cell = dv.funcs.(Kernels.index k) in
  match Atomic.get cell with
  | Some f -> f
  | None ->
      prepare_family dv k;
      let f = resolve dv k in
      Atomic.set cell (Some f);
      f

(* Keys

   A specialised step's pipeline is made for its key, its function and its
   function constants' values: an image of the steps' metallib behind a header
   of those values (rig_metal.mli, Specialised metallibs), the bytes "RMSC",
   their count and each value, 32-bit little-endian words. A device keeps the
   functions of the [kept_keys] keys made last; one it dropped is made again,
   from Metal's cache where it holds it. A call starts its new keys together, so
   that Metal makes their pipelines concurrently, then waits for each. *)
let kept_keys = 256

(* The keys a device keeps, [kept_keys] but in tests. *)
let keys_bound = Atomic.make kept_keys

let bound_keys n =
  if n < 1 || n > kept_keys then invalid_arg "Nx_metal.bound_keys";
  Atomic.set keys_bound n

let header constants =
  let n = Array.length constants in
  let b = Bytes.create (8 + (4 * n)) in
  Bytes.blit_string "RMSC" 0 b 0 4;
  Bytes.set_int32_le b 4 (Int32.of_int n);
  Array.iteri
    (fun i v -> Bytes.set_int32_le b (8 + (4 * i)) (Int32.of_int v))
    constants;
  Bytes.unsafe_to_string b

(* Loads [k]'s image specialised by [cs] and starts making its pipeline. *)
let start dv (k, cs) =
  match Rig.Image.load dv.rig (header cs ^ Fft_metallib.metallib) with
  | Error why -> failwith ("nx.metal: " ^ why)
  | Ok image ->
      Rig.Image.prepare image (Kernels.name k);
      image

(* The function of key [(k, cs)] from its started [image], whose groups take
   [threads] threads, kept. *)
let finish dv ((k, cs) as key) image ~threads =
  let what = Plan.label k cs in
  match Rig.Image.resolve image (Kernels.name k) with
  | None -> failwith ("nx.metal: the steps' metallib has no kernel " ^ what)
  | Some f -> (
      match Rig.Image.threads f with
      | Some t when t < threads ->
          failwith
            (Printf.sprintf
               "nx.metal: %s's groups hold at most %d threads, not %d" what t
               threads)
      | Some _ | None -> Cache.keep ~count:(Atomic.get keys_bound) dv.keys key f
      )

(* The function of each key of [launches], each with its groups' threads, in a
   table of the call's own: those [dv] keeps, found, then the others, all
   started before any is waited for. The table holds them while the call makes
   its parts, so that a key [dv] drops meanwhile ([keys_bound]) is not made
   again. *)
let call_keys dv (launches : (Kernels.t * int array * int) list) =
  let table = Hashtbl.create 8 and todo = ref [] in
  let add (k, cs, threads) =
    let key = (k, cs) in
    if not (Hashtbl.mem table key || List.mem_assoc key !todo) then
      match Cache.find dv.keys key with
      | Some f -> Hashtbl.replace table key f
      | None -> todo := (key, threads) :: !todo
  in
  List.iter add launches;
  let started =
    List.map (fun (key, threads) -> (key, start dv key, threads)) !todo
  in
  List.iter
    (fun (key, image, threads) ->
      Hashtbl.replace table key (finish dv key image ~threads))
    started;
  table

(* The threads Metal lets a group of [k]'s pipeline on [dv] have, which its
   registers bound. *)
let threads_of dv k =
  match Rig.Image.threads (func dv k) with
  | Some t -> t
  | None -> Kernels.threads k

(* [cell]'s buffer of at least [need] bytes, grown to twice its length or [need]
   by [make], at most [cap]. A race stores only a larger one, so it never
   shrinks. A call whose buffer is large enough reads it without reaching here,
   which would allocate [make]'s closure. *)
let grow cell need ~cap make =
  let w = Atomic.get cell in
  let fresh = make (Int.min cap (Int.max need (2 * Rig.Buffer.length w))) in
  let rec store () =
    let w = Atomic.get cell in
    if Rig.Buffer.length w >= need then w
    else if Atomic.compare_and_set cell w fresh then fresh
    else store ()
  in
  store ()

let workspace dv need =
  let w = Atomic.get dv.workspace in
  if Rig.Buffer.length w >= need then w
  else
    grow dv.workspace need ~cap:Plan.kept_workspace (Rig.Buffer.create dv.rig)

let zeros d bytes =
  let b = Rig.Buffer.create d bytes in
  Rig.Buffer.fill b '\000';
  b

let tickets dv need =
  let t = Atomic.get dv.tickets in
  if Rig.Buffer.length t >= need then t
  else grow dv.tickets need ~cap:Plan.kept_tickets (zeros dv.rig)

(* Submissions

   A device keeps the submission of each part's key ({!Plan.key}), made at its
   first use, in a cache for each first kernel ({!Cache}) that holds the
   [kept_subs] most recently made, so that none grows with the calls a program
   makes and a lookup reads few keys. Calls on several domains submit one at
   once, each with its own values ({!Plan.args}). A part that is not kept
   ({!Plan.finish}) takes a submission of its own. *)
let kept_subs = 64
let keyed (l : Plan.launch) = Array.length l.constants > 0

(* The function of each launch of [ps], every one made before any part is
   submitted: their kernels' families prepared and their keys started, so that
   Metal makes the pipelines concurrently, then each waited for, so that no
   kernel first used in a later part compiles while the GPU waits behind the
   earlier ones. *)
let funcs dv (ps : Plan.part list) =
  let used = Array.make (Array.length Kernels.all) false in
  let seen = Hashtbl.create 8 and keys = ref [] in
  let note (l : Plan.launch) =
    if keyed l then begin
      let key = (l.kernel, l.constants) in
      if not (Hashtbl.mem seen key) then begin
        Hashtbl.add seen key ();
        let threads, _, _ = l.threads in
        keys := (l.kernel, l.constants, threads) :: !keys
      end
    end
    else
      let i = Kernels.index l.kernel in
      if not used.(i) then begin
        used.(i) <- true;
        prepare_family dv l.kernel
      end
  in
  List.iter (fun (p : Plan.part) -> Array.iter note p.run) ps;
  let table = call_keys dv !keys in
  Array.iteri (fun i u -> if u then ignore (func dv Kernels.all.(i))) used;
  fun (l : Plan.launch) ->
    if keyed l then Hashtbl.find table (l.kernel, l.constants)
    else func dv l.kernel

(* The launches of [p] that launch a key. *)
let keys (p : Plan.part) =
  Array.fold_left (fun n l -> if keyed l then n + 1 else n) 0 p.run

(* A submission holds its launches' functions, so those of parts that launch
   keys are kept in one cache, bounded by the keys they hold as well as by their
   count. *)
let kept_sub dv (p : Plan.part) =
  if not p.kept then None
  else if keys p > 0 then
    Option.map fst (Cache.find_by Plan.matches dv.key_subs p)
  else Cache.find_by Plan.matches dv.subs.(Kernels.index p.run.(0).kernel) p

(* [p]'s submission made with the functions [f], kept where [p] is. *)
let made dv f (p : Plan.part) =
  let s = Sub.make dv.rig (Plan.parts f p) in
  if not p.kept then s
  else
    let n = keys p in
    if n > 0 then
      fst
        (Cache.keep ~count:kept_subs
           ~bytes:(Atomic.get keys_bound, snd)
           dv.key_subs (Plan.key p) (s, n))
    else
      let cell = dv.subs.(Kernels.index p.run.(0).kernel) in
      Cache.keep ~count:kept_subs cell (Plan.key p) s

let submission dv p =
  match kept_sub dv p with Some s -> s | None -> made dv (funcs dv [ p ]) p

(* Constants

   The most bytes of constants a device keeps, {!Plan.kept_constants} but in
   tests. *)
let constants_bound = Atomic.make Plan.kept_constants

let bound_constants n =
  if n < 0 || n > Plan.kept_constants then
    invalid_arg "Nx_metal.bound_constants";
  Atomic.set constants_bound n

let wrong_length key ~length n =
  invalid_arg
    (Printf.sprintf "Nx_metal: constant %S of %d bytes, not %d" key n length)

(* The buffer of the constant [key] of [length] bytes [bytes ()] on [dv], and
   whether [dv] keeps it: the one it keeps, else a new one that [bytes ()] are
   uploaded into, kept with the newest others while they number at most
   {!Plan.kept_constant_count} and fit [bound]. A buffer the cache drops is left
   to rig's reclaim, never freed: a call on another domain may have just found
   it. *)
let constant dv ~bound key ~length bytes =
  match Cache.find dv.constants key with
  | Some b ->
      let n = Rig.Buffer.length b in
      if n <> length then wrong_length key ~length n;
      (b, true)
  | None ->
      let bytes = bytes () in
      let n = Bigarray.Array1.dim bytes in
      if n <> length then wrong_length key ~length n;
      let b = Rig.Buffer.create dv.rig n in
      (match Rig.Buffer.copy ~src:(Rig.Buffer.of_bigarray bytes) ~dst:b with
      | () -> ()
      | exception e ->
          ignore (Rig.Buffer.free b);
          raise e);
      let size = Rig.Buffer.length and count = Plan.kept_constant_count in
      (Cache.keep ~count ~bytes:(bound, size) dv.constants key b, n <= bound)

(* Binding

   [owned.(i)] is whether the buffer of slot [i] is the call's own, which the
   door frees once it has submitted the call's last part. The call's operands
   are [written], then [read] ({!Operands}). *)

let operand ~written ~read k =
  let n = Iarray.length written in
  let (A.Any x) =
    if k < n then Iarray.get written k else Iarray.get read (k - n)
  in
  A.buffer x

let bind dv ~written ~read (c : Plan.t) owned i (s : Plan.slot) =
  match s with
  | Operand k -> operand ~written ~read k
  | Workspace -> workspace dv c.workspace
  | Tickets -> tickets dv c.tickets
  | Fresh n ->
      owned.(i) <- true;
      Rig.Buffer.create dv.rig n
  | Constant { key; length; bytes } ->
      let bound = Atomic.get constants_bound in
      let b, kept = constant dv ~bound key ~length bytes in
      owned.(i) <- not kept;
      b
  | Uploaded s ->
      let n = String.length s in
      let b = Rig.Buffer.create dv.rig n in
      (match Rig.Buffer.blit_from_string s 0 b 0 n with
      | () -> ()
      | exception e ->
          ignore (Rig.Buffer.free b);
          raise e);
      owned.(i) <- true;
      b

let free_owned owned buffers =
  Array.iteri
    (fun i own -> if own then ignore (Rig.Buffer.free buffers.(i)))
    owned

(* [c]'s slots' buffers, bound in order. A bind that raises frees the call's own
   buffers bound before it. *)
let bind_all dv ~written ~read (c : Plan.t) owned =
  let made = ref [] in
  let bind_one i s =
    let b = bind dv ~written ~read c owned i s in
    if owned.(i) then made := b :: !made;
    b
  in
  match Array.mapi bind_one c.slots with
  | buffers -> buffers
  | exception e ->
      List.iter (fun b -> ignore (Rig.Buffer.free b)) !made;
      raise e

(* After [p]'s submit at [point], commits the device's work where [p] passes
   {!Plan.commit_after}. The commit is why [reached] is called, and its answer
   is ignored: it ends the open command buffer, so that one holds at most twice
   {!Plan.part_budget} of a domain's work. *)
let commit (p : Plan.part) point =
  if p.commits then ignore (Rig.Point.reached point)

(* Submits [c]'s part with its slots' buffers and its launches' values, then
   frees its own buffers, which return once that work is done, also where the
   submit raises. *)
let issue_one dv ~written ~read (c : Plan.t) =
  let p = c.parts.(0) in
  let s = submission dv p in
  let owned = Array.make (Array.length c.slots) false in
  let buffers = bind_all dv ~written ~read c owned in
  Fun.protect ~finally:(fun () -> free_owned owned buffers) @@ fun () ->
  commit p (Rig.submit s ~buffers ~args:(Plan.args p))

(* Submits [c]'s parts in order, without waiting for any. Before the first: the
   kept submissions found, every pipeline of the others made, and the slots'
   buffers bound, so that what can raise for the call's resources, a slot's
   memory or a kernel's pipeline, raises with nothing submitted. Part k + 1's
   submission and values are made once part k is submitted, so that the GPU
   starts on part 1 while the host makes the rest. The host stays ahead: a make
   takes less time a launch than the launch's dispatch.

   A later part's make is valid, so it raises only for host memory: its device
   is part 1's, its functions were made before part 1 and the device holds their
   metallib, it fixes no memory, its refs are ordered and name its slots from 0
   in the order its launches first use them ({!Plan.part}), and its blocks take
   no more bytes than part 1's, whose make was valid, or one launch's, far under
   rig's bound (plan.ml's [cuts], {!Plan.finish}). *)
let issue_parts dv ~written ~read (c : Plan.t) =
  let found = Array.map (kept_sub dv) c.parts in
  let unmade =
    List.filteri (fun i _ -> Option.is_none found.(i)) (Array.to_list c.parts)
  in
  let f = funcs dv unmade in
  let owned = Array.make (Array.length c.slots) false in
  let buffers = bind_all dv ~written ~read c owned in
  Fun.protect ~finally:(fun () -> free_owned owned buffers) @@ fun () ->
  Array.iteri
    (fun i (p : Plan.part) ->
      let s =
        match found.(i) with
        | Some s -> s
        | None -> (
            (* An earlier part of the same key may have kept one since. *)
            match kept_sub dv p with
            | Some s -> s
            | None -> made dv f p)
      in
      let buffers = Array.map (fun k -> buffers.(k)) p.binds in
      commit p (Rig.submit s ~buffers ~args:(Plan.args p)))
    c.parts

let issue (dv, written, read, (c : Plan.t)) =
  if Array.length c.parts = 1 then issue_one dv ~written ~read c
  else issue_parts dv ~written ~read c

let nothing () = ()
let dead (A.Any x) = Option.is_some (Rig.Buffer.dead (A.buffer x))

let rec any_dead xs i =
  i < Iarray.length xs && (dead (Iarray.get xs i) || any_dead xs (i + 1))

(* Whether a buffer of [written] or [read] is dead, which the door refuses
   before a plan reads its facts. *)
let refused ~written ~read = any_dead written 0 || any_dead read 0

(* [f ()], unless {!refused}. *)
let live ~written ~read f =
  if refused ~written ~read then A.door ~written ~read nothing () else f ()

(* A call that refuses nothing: on [d], [plan] of its device through the door,
   claiming [written] and [read], its operands in that order. A plan of no
   launch, a call of no element, has no work. *)
let run d ~written ~read plan =
  match device d with
  | None -> A.Declined
  | Some _ when refused ~written ~read -> A.door ~written ~read nothing ()
  | Some dv -> (
      match plan dv with
      | Error answer -> answer
      | Ok None -> A.door ~written ~read nothing ()
      | Ok (Some c) -> A.door ~written ~read issue (dv, written, read, c))

let shape_of a = Iarray.to_array (A.Layout.shape (A.layout a))

(* Loops *)

(* The arrays a loop reads: its loads' arrays, and its carry. *)
let load_arrays ?carry (loads : Nx_kernel.load iarray) =
  let xs =
    Iarray.map
      (function Nx_kernel.Plain x -> A.Any x | Padded { x; _ } -> A.Any x)
      loads
  in
  match carry with
  | None -> xs
  | Some c -> Iarray.append xs (Iarray.of_list [ c ])

(* Refusals: Nx_kernel's rules, which read dtypes and counts, then the shapes
   they leave to a library: each load's once loaded and the destinations', as
   {!Nx_kernel.Shape} gives them. *)

let any_dtype (A.Any a) = D.Any (A.dtype a)

let arrays =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype = (fun a i -> any_dtype (Iarray.get a i));
  }

let loads_of =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype =
      (fun (a : Nx_kernel.load iarray) i ->
        match Iarray.get a i with
        | Plain x -> D.Any (A.dtype x)
        | Padded { x; _ } -> D.Any (A.dtype x));
  }

(* The rank of a loop over the shape of [dsts]' first, [0] for none. *)
let rank_of dsts =
  if Iarray.length dsts = 0 then 0
  else
    let (A.Any d) = Iarray.get dsts 0 in
    A.Layout.rank (A.layout d)

(* Whether each load has the loop's [shape] once loaded and each of [dsts] the
   shape [want]. *)
let fits ~shape loads ~want dsts =
  let loaded : Nx_kernel.load -> bool = function
    | Plain x -> shape_of x = shape
    | Padded { x; pad; _ } -> Nx_kernel.Shape.padded pad (shape_of x) = Ok shape
  in
  Iarray.for_all loaded loads
  && Iarray.for_all (fun (A.Any y) -> shape_of y = want) dsts

(* A map that moves bits takes the copy kernels, which move a run of consecutive
   elements whose ends lie on 16 bytes 16 bytes a thread, and other operands an
   element a thread; every other, the loop interpreter. *)
let planned dv p xs =
  match Moves.map p xs with
  | Error A.Declined -> Loop.map ~holds:(threads_of dv) p xs
  | r -> r

(* Plans

   A device keeps each map's answer from the planner by its program, physically,
   and its facts ({!Operands}), which are all the planner reads but the device's
   pipelines and the bounds ({!Plan.epoch}). An answer is a plan, [Ok None] or
   [Error Declined], each of the key alone: the door refuses a dead buffer
   before the planner, and no refusal of the door's or the device's is kept. A
   program made for each call never finds its plan, nor a padded load whose fill
   changes.

   Each bucket keeps its newest entries within [plans_bound / buckets] bytes, an
   entry's its plan's words and constants, its key's and its program's, which it
   keeps reachable: 2 MiB a device, and no plan over 8 KiB, more than about 7
   launches, whose call takes at least about 35 µs of GPU against 2 to 3 µs of
   planning. *)
let kept_plans_bytes = 2 * 1024 * 1024

(* The bytes of a program's node, input or output on the OCaml heap, at most:
   its slot, a block of at most 4 fields and a header, and its operation's block
   or its constant's bits. *)
let node_bytes = 64
let plans_bound = Atomic.make kept_plans_bytes

let bound_plans n =
  if n < 0 || n > kept_plans_bytes then invalid_arg "Nx_metal.bound_plans";
  Atomic.set plans_bound n;
  Plan.renew ()

let plan_bytes (c : Plan.t) =
  let words n (l : Plan.launch) = n + Bytes.length l.words in
  let constant n : Plan.slot -> int = function
    | Constant { length; _ } -> n + length
    | Operand _ | Uploaded _ | Workspace | Tickets | Fresh _ -> n
  in
  Array.fold_left constant (Array.fold_left words 0 c.launches) c.slots

let entry_bytes (k : call) answer =
  let plan =
    match answer with Ok (Some c) -> plan_bytes c | Ok None | Error _ -> 0
  in
  let p = k.prog in
  let nodes =
    Iarray.length p.nodes + Iarray.length p.ins + Iarray.length p.outs
  in
  (8 * Array.length k.ints) + (node_bytes * nodes) + plan

let rec same_ints a b i =
  i = Array.length a || (a.(i) = b.(i) && same_ints a b (i + 1))

let same_call (k : call) (q : call) =
  k.prog == q.prog && k.epoch = q.epoch
  && Array.length k.ints = Array.length q.ints
  && same_ints k.ints q.ints 0

let entry_size e = e.bytes

let map_plan dv p (xs : Operands.t) =
  let ints = Operands.key xs in
  let q = { prog = p; epoch = Plan.epoch (); ints } in
  let cell = dv.plans.(Operands.hash ints land (buckets - 1)) in
  match Cache.find_by same_call cell q with
  | Some e ->
      (match e.answer with
      | Ok (Some c) -> Plan.replay c
      | Ok None | Error _ -> ());
      e.answer
  | None ->
      let answer = planned dv p xs in
      let most = Atomic.get plans_bound / buckets in
      let e = { answer; bytes = entry_bytes q answer } in
      (Cache.keep ~bytes:(most, entry_size) cell q e).answer

let map (p : Nx_kernel.Prog.t) ~dsts loads =
  match
    Nx_kernel.map_refusal p ~rank:(rank_of dsts) ~dsts:arrays dsts
      ~loads:loads_of loads
  with
  | Some (refused, _) -> refused
  | None when Iarray.length dsts = 0 ->
      A.door ~written:dsts ~read:(load_arrays loads) nothing ()
  | None ->
      let (A.Any y) = Iarray.get dsts 0 in
      let shape = shape_of y in
      if not (fits ~shape loads ~want:shape dsts) then A.Shape_mismatch
      else
        run (A.device y) ~written:dsts ~read:(load_arrays loads) (fun dv ->
            map_plan dv p (Operands.v ~dsts loads))

(* Reductions *)

let reductions =
  {
    Nx_kernel.reductions = Iarray.length;
    kind = (fun rs i -> fst (Iarray.get rs i));
    output = (fun rs i -> snd (Iarray.get rs i));
  }

(* The program of a reduction's value of no term into [dsts], [rs]'s
   destinations in order: [+0] for a sum, [1] for a product, [-∞] for a
   Logsumexp and NaN for Moments, each in its output's carrier, cast. *)
let identities (p : Nx_kernel.Prog.t) rs (dsts : A.any iarray) =
  let nodes = ref [] and outs = ref [] and at = ref 0 in
  let push n =
    nodes := n :: !nodes;
    List.length !nodes - 1
  in
  Iarray.iter
    (fun ((r : Nx_kernel.reduction), k) ->
      let (D.Any c) =
        Nx_kernel.carrier (Nx_kernel.Prog.dtype p (Iarray.get p.outs k))
      in
      let value =
        match r with
        | Monoid Prod -> D.one c
        | Logsumexp -> D.of_float c Float.neg_infinity
        | Moments -> D.of_float c Float.nan
        | Monoid _ | Arg _ -> D.zero c
      in
      let const =
        push (Nx_kernel.Prog.Const (D.Any c, Nx_kernel.Prog.bits c value))
      in
      for _ = 1 to Loop.destinations r do
        let (A.Any y) = Iarray.get dsts !at in
        let n = push (Nx_kernel.Prog.Op1 (Cast, D.Any (A.dtype y), const)) in
        outs := n :: !outs;
        incr at
      done)
    rs;
  Nx_kernel.Prog.v ~ins:[||]
    (Iarray.of_list (List.rev !nodes))
    ~outs:(Iarray.of_list (List.rev !outs))

let reduce p ~shape ~axes rs ~dsts loads =
  match
    Nx_kernel.reduce_refusal p ~rank:(Iarray.length shape) ~reductions rs
      ~dsts:arrays dsts ~loads:loads_of loads
  with
  | Some (refused, _) -> refused
  | None -> (
      let s = Iarray.to_array shape in
      let kinds =
        Array.init (Iarray.length rs) (fun i -> fst (Iarray.get rs i))
      in
      match Nx_kernel.Shape.reduce ~axes kinds s with
      | Ok want when fits ~shape:s loads ~want dsts ->
          (* No term: each destination element takes the identity, where the
             call of one term computes. *)
          let (A.Any y) = Iarray.get dsts 0 in
          let read = load_arrays loads in
          if Array.mem 0 s && not (Array.mem 0 want) then
            match device (A.device y) with
            | None -> A.Declined
            | Some dv -> (
                live ~written:dsts ~read @@ fun () ->
                let holds = threads_of dv in
                let xs = Operands.v ~dsts loads in
                match Loop.reduce ~holds p ~shape ~axes rs xs with
                | Error answer -> answer
                | Ok _ -> map (identities p rs dsts) ~dsts [||])
          else
            run (A.device y) ~written:dsts ~read (fun dv ->
                let holds = threads_of dv in
                Loop.reduce ~holds p ~shape ~axes rs (Operands.v ~dsts loads))
      | Ok _ | Error _ -> A.Shape_mismatch)

(* Scans *)

let scan ?carry p ~axis ((r, k) as rk) ~dsts loads =
  match
    Nx_kernel.scan_refusal p ~rank:(rank_of dsts) r ~output:k ~dsts:arrays dsts
      ?carry:(Option.map any_dtype carry)
      ~loads:loads_of loads
  with
  | Some (refused, _) -> refused
  | None -> (
      let (A.Any y) = Iarray.get dsts 0 in
      let shape = shape_of y in
      let carried =
        match carry with
        | None -> true
        | Some (A.Any c) -> Nx_kernel.Shape.carry ~axis shape = Ok (shape_of c)
      in
      match Nx_kernel.Shape.scan ~axis shape with
      | Ok want when carried && fits ~shape loads ~want dsts ->
          run (A.device y) ~written:dsts ~read:(load_arrays ?carry loads)
            (fun dv ->
              Loop.scan ~holds:(threads_of dv) p ~axis rk
                (Operands.v ?carry ~dsts loads))
      | Ok _ | Error _ -> A.Shape_mismatch)

(* Contractions *)

(* The arrays a contraction reads: [a], [b], then [init], the segments' ids and
   the scales where given. *)
let contract_reads ~init ~segments (oa : Nx_kernel.operand)
    (ob : Nx_kernel.operand) =
  let scale (o : Nx_kernel.operand) =
    match o.scale with
    | None -> []
    | Some (Exponent e) -> [ A.Any e ]
    | Some (Value v) -> [ A.Any v ]
  in
  let ids =
    match (segments : _ Nx_kernel.segments option) with
    | None -> []
    | Some (Rows r) -> [ A.Any r.ids ]
    | Some (Contracting g) -> [ A.Any g.ids ]
  in
  Iarray.of_list
    (oa.array :: ob.array
     :: Option.to_list (Option.map (fun i -> A.Any i) init)
    @ ids @ scale oa @ scale ob)

(* [o]'s shape and its scale's, as {!Nx_kernel.Shape.contract} takes them. *)
let operand_shapes (o : Nx_kernel.operand) =
  let (A.Any x) = o.array in
  let scale =
    match o.scale with
    | None -> None
    | Some (Exponent e) -> Some (shape_of e)
    | Some (Value v) -> Some (shape_of v)
  in
  (shape_of x, scale)

(* Whether the call's shapes fit: {!Nx_kernel.Shape.contract}'s, [dst]'s and
   [init]'s. *)
let contract_fits ~batch ~contracting ~segments ~dst ~init a b =
  let segments =
    match (segments : _ Nx_kernel.segments option) with
    | None -> None
    | Some (Rows r) -> Some (Nx_kernel.Rows { r with ids = shape_of r.ids })
    | Some (Contracting c) ->
        Some (Nx_kernel.Contracting { c with ids = shape_of c.ids })
  in
  match
    Nx_kernel.Shape.contract ~batch ~contracting ~segments (operand_shapes a)
      (operand_shapes b)
  with
  | Ok y -> (
      y = shape_of dst
      && match init with None -> true | Some i -> y = shape_of i)
  | Error _ -> false

(* A mixed call's casts take one buffer of at most [pieces_bound] bytes, the
   call cut into pieces past it ({!Pieces}). *)
let pieces_bound = Atomic.make Plan.kept_workspace

let bound_pieces n =
  if n < 1 || n > Plan.kept_workspace then invalid_arg "Nx_metal.bound_pieces";
  Atomic.set pieces_bound n;
  Plan.renew ()

let bound_outputs n =
  if n < 1 || n > Contract.largest_outputs then
    invalid_arg "Nx_metal.bound_outputs";
  Atomic.set Contract.most_outputs n;
  Plan.renew ()

(* [x] cut along each axis [i] of [cuts] to its indices from [lo] to below [lo +
   len]. *)
let slice (x : ('v, 's) A.t) cuts =
  let l = A.layout x in
  let r = A.Layout.rank l in
  let dims = Array.init r (A.Layout.dim l) in
  let st = Array.init r (A.Layout.stride l) in
  let offset = ref (A.Layout.offset l) in
  List.iter
    (fun (i, lo, len) ->
      offset := !offset + (lo * st.(i));
      dims.(i) <- len)
    cuts;
  let l =
    A.Layout.v ~offset:!offset ~strides:(Iarray.of_array st)
      (Iarray.of_array dims)
  in
  A.v (A.dtype x) l (A.buffer x)

let slice_any (A.Any x) cuts = A.Any (slice x cuts)

(* The cuts of a scale of [x] for [x]'s [cuts]: an axis of [x]'s extent cut as
   [x]; one of extent 1 kept; else its blocks that the cut's indices lie in,
   which {!Pieces.axis}'s [step] aligns. *)
let scale_cuts (A.Any x) (A.Any s) cuts =
  let lx = A.layout x and ls = A.layout s in
  List.map
    (fun (i, lo, len) ->
      let ex = A.Layout.dim lx i and es = A.Layout.dim ls i in
      if es = ex then (i, lo, len)
      else if es = 1 then (i, 0, 1)
      else
        let block = ex / es in
        (i, lo / block, ((lo + len + block - 1) / block) - (lo / block)))
    cuts

let slice_operand (o : Nx_kernel.operand) cuts : Nx_kernel.operand =
  let scale : Nx_kernel.scale option =
    match o.scale with
    | None -> None
    | Some (Exponent e) ->
        Some (Exponent (slice e (scale_cuts o.array (A.Any e) cuts)))
    | Some (Value v) ->
        Some (Value (slice v (scale_cuts o.array (A.Any v) cuts)))
  in
  { array = slice_any o.array cuts; scale }

(* The blocks of [o]'s scale along its axis [i], 1 for none or a block of every
   index. *)
let scale_block (o : Nx_kernel.operand) i =
  let block (A.Any s) =
    let (A.Any x) = o.array in
    let ex = A.Layout.dim (A.layout x) i and es = A.Layout.dim (A.layout s) i in
    if es = 1 || es = ex then 1 else ex / es
  in
  match o.scale with
  | None -> 1
  | Some (Exponent e) -> block (A.Any e)
  | Some (Value v) -> block (A.Any v)

(* The programs that cast an operand of each dtype into each stage's. *)
let stage_dtypes = D.[ Any Float32; Any Float16; Any Bfloat16 ]

let cast_progs =
  List.map
    (fun src ->
      List.map
        (fun dst ->
          Nx_kernel.Prog.v ~ins:(Iarray.of_list [ src ])
            (Iarray.of_list Nx_kernel.Prog.[ In 0; Op1 (Cast, dst, 0) ])
            ~outs:(Iarray.of_list [ 1 ]))
        stage_dtypes)
    D.all

let cast_prog (D.Any src) (D.Any dst) =
  let row = List.nth cast_progs (D.code src) in
  let rec find ps ds =
    match (ps, ds) with
    | p :: ps, D.Any d :: ds -> if D.code d = D.code dst then p else find ps ds
    | _ -> invalid_arg "Nx_metal.cast_prog: a stage's dtype"
  in
  find row stage_dtypes

(* The bytes of a cast of [x] into [dt]: of its distinct elements, each axis of
   stride 0 one element, along the axes [i] where [axes i]. *)
let cast_bytes (D.Any dt) (A.Any x) axes =
  let l = A.layout x in
  let n = ref 1 in
  for i = 0 to A.Layout.rank l - 1 do
    if axes i && A.Layout.stride l i <> 0 then n := !n * A.Layout.dim l i
  done;
  D.bytes dt !n

(* A cast of an operand [x] into [dt] at byte [at] of [buffer]: [x]'s distinct
   elements, each axis of stride 0 at extent 1, C-contiguous in [x]'s memory
   order (its axes by decreasing stride), so that a weight broadcast along a
   batch is cast once; the view of [x] that fills it; and the view of it the
   contraction reads, stride 0 restored. *)
type cast = { into : A.any; from : A.any; view : A.any }

let cast_into buffer ~at (D.Any dt) (A.Any x) =
  let l = A.layout x in
  let r = A.Layout.rank l in
  let dims = Array.init r (A.Layout.dim l) in
  let st = Array.init r (A.Layout.stride l) in
  let kept = Array.mapi (fun i e -> if st.(i) = 0 then 1 else e) dims in
  let outer i j =
    let c = compare (abs st.(j)) (abs st.(i)) in
    if c <> 0 then c else compare i j
  in
  let order = List.sort outer (List.init r Fun.id) in
  let dense = Array.make r 0 in
  ignore
    (List.fold_right
       (fun i step ->
         dense.(i) <- step;
         step * kept.(i))
       order 1);
  let offset = at / D.bytes dt 1 in
  let at strides shape =
    A.Layout.v ~offset ~strides:(Iarray.of_array strides)
      (Iarray.of_array shape)
  in
  let seen = Array.mapi (fun i s -> if st.(i) = 0 then 0 else s) dense in
  let from =
    A.Layout.v ~offset:(A.Layout.offset l) ~strides:(Iarray.of_array st)
      (Iarray.of_array kept)
  in
  {
    into = A.Any (A.v dt (at dense kept) buffer);
    from = A.Any (A.v (A.dtype x) from (A.buffer x));
    view = A.Any (A.v dt (at seen dims) buffer);
  }

let load_of (A.Any x) = Nx_kernel.Plain x
let region = 256
let round_up n m = (n + m - 1) / m * m

(* The programs that copy an element of each dtype. *)
let identity_progs =
  List.map
    (fun d ->
      Nx_kernel.Prog.v ~ins:(Iarray.of_list [ d ])
        (Iarray.of_list Nx_kernel.Prog.[ In 0 ])
        ~outs:(Iarray.of_list [ 0 ]))
    D.all

let identity_prog (D.Any d) = List.nth identity_progs (D.code d)

(* A C-contiguous array of [dt] and [shape] at byte [at] of [buffer], a multiple
   of {!region}. *)
let temp_at buffer ~at dt shape =
  let offset = at * 8 / D.bits dt in
  let strides = Array.make (Array.length shape) 1 in
  for i = Array.length shape - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * shape.(i + 1)
  done;
  let l =
    A.Layout.v ~offset ~strides:(Iarray.of_array strides)
      (Iarray.of_array shape)
  in
  A.v dt l buffer

(* The geometry of a contraction's outputs: for each of [dst]'s axes, the
   batch's then [a]'s free axes then [b]'s, its axis of [a] and of [b]; a Rows
   call's ids along the batch and [a_axis]. *)
type geometry = {
  a_axes : int option array;
  b_axes : int option array;
  ids_axes : int option array;
  batch_axes : int;
  row_axes : int;
}

let geometry ~batch ~contracting ~segments (A.Any a) (A.Any b) =
  let paired pairs pick i =
    Iarray.exists (fun p -> pick p = i) batch
    || Iarray.exists (fun p -> pick p = i) pairs
  in
  let drop =
    match (segments : _ Nx_kernel.segments option) with
    | Some (Rows r) -> r.b_axis
    | Some (Contracting _) | None -> -1
  in
  let free x pick skip =
    List.filter
      (fun i -> (not (paired contracting pick i)) && i <> skip)
      (List.init (A.Layout.rank (A.layout x)) Fun.id)
  in
  let af = free a fst (-1) and bf = free b snd drop in
  let nb = Iarray.length batch in
  let r = nb + List.length af + List.length bf in
  let a_axes = Array.make r None and b_axes = Array.make r None in
  let ids_axes = Array.make r None in
  Iarray.iteri
    (fun d (i, j) ->
      a_axes.(d) <- Some i;
      b_axes.(d) <- Some j;
      ids_axes.(d) <- Some d)
    batch;
  List.iteri
    (fun q i ->
      a_axes.(nb + q) <- Some i;
      match segments with
      | Some (Rows r) when r.a_axis = i -> ids_axes.(nb + q) <- Some nb
      | _ -> ())
    af;
  List.iteri (fun q j -> b_axes.(nb + List.length af + q) <- Some j) bf;
  { a_axes; b_axes; ids_axes; batch_axes = nb; row_axes = List.length af }

(* A box's cuts of an array along [axes]: [dst]'s own, or an operand's. *)
let cuts_of axes (x : Pieces.box) =
  List.concat
    (List.init (Array.length axes) (fun d ->
         match axes.(d) with
         | Some i -> [ (i, x.lo.(d), x.len.(d)) ]
         | None -> []))

(* The call that {!Contract.call} names [r] in pieces: of [oa]'s and [ob]'s
   casts into the dtypes [r.a] and [r.b], in pieces whose casts take one buffer
   of at most {!pieces_bound} bytes ({!Pieces.cut}), a cast line that passes it
   read through a band of the piece's call; but where the vector kernels take
   the call, whose buffer holds one line of each cast operand at least: each
   piece is a call of slices of the operands, the casts of the cast ones, and of
   [init], [dst] and the ids, which keeps the whole call's choices ([~whole])
   and so its bits. A piece is of at most {!Contract.most_outputs} outputs
   ({!Pieces.refine}). Where [r.temp], the casts take half the buffer where
   there are any, and each piece computes into a C-contiguous temporary of
   [dst]'s dtype in the rest, then copies it into its slice of [dst]: the copy
   is exact, so [dst] gets the bits a C-contiguous one gets. A piece's casts go
   into its regions of the buffer, but where they would cast the slice the
   previous piece cast. Every cast, piece and copy is planned before any is
   submitted, so a plan that declines declines the call before any work; then
   one door claims [written] and the buffer as written and [read] as read, and
   submits each piece's casts, then its call, then its copy, in order: on one
   device rig runs each submission after the ones before it, so a cast or a call
   never overwrites what an earlier piece reads. A piece's call reads its slice
   of [init] before its copy writes [dst] there, and the pieces partition the
   outputs, so an [init] identical to [dst] is read before it is written. The
   buffer is freed once submitted; rig keeps it until the work ends. [plan
   ~whole ~dst ~init ~segments oa ob] plans one call, with its written and read
   arrays. Each cast, piece and copy is a plan of its own, which runs as several
   parts where its launches pass a part's seconds ({!Plan.finish}), so the
   pieces together take no bound. *)
let recast dv ~written ~read ~batch ~contracting ~segments ~dst ~init ~plan
    (r : Contract.recast) (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let ca = r.a and cb = r.b in
  let g = geometry ~batch ~contracting ~segments oa.array ob.array in
  let shape = shape_of dst in
  let line axes cast (o : Nx_kernel.operand) =
    match cast with
    | None -> 0
    | Some dt ->
        let mapped i = Array.exists (fun a -> a = Some i) axes in
        cast_bytes dt o.array (fun i -> not (mapped i))
  in
  let varies axes cast (o : Nx_kernel.operand) d =
    let (A.Any x) = o.array in
    match (cast, axes.(d)) with
    | Some _, Some i -> A.Layout.stride (A.layout x) i <> 0
    | _ -> false
  in
  let step d =
    let block axes o =
      match axes.(d) with Some i -> scale_block o i | None -> 1
    in
    let p = block g.a_axes oa and q = block g.b_axes ob in
    let rec gcd a b = if b = 0 then a else gcd b (a mod b) in
    p / gcd p q * q
  in
  let axes =
    Array.mapi
      (fun d extent ->
        {
          Pieces.extent;
          a = varies g.a_axes ca oa d;
          b = varies g.b_axes cb ob d;
          step = step d;
        })
      shape
  in
  let la = line g.a_axes ca oa and lb = line g.b_axes cb ob in
  (* The casts' share of the buffer, [w], and the temporary's. *)
  let w, wt =
    let w = Atomic.get pieces_bound in
    if not r.temp then (w, 0)
    else if la + lb > 0 then (w / 2, w - (w / 2))
    else (w, w)
  in
  let bits = D.bits (A.dtype dst) in
  let pieces =
    let p =
      Pieces.cut ~w ~batch:g.batch_axes ~rows:g.row_axes axes ~a:la ~b:lb
    in
    (* At most {!Contract.most_outputs} outputs, and a temporary of at most [wt]
       bytes. *)
    let most = Atomic.get Contract.most_outputs in
    let most = if r.temp then Int.min most (8 * wt / bits) else most in
    Array.of_list
      (List.concat_map (Pieces.refine ~most axes) (Array.to_list p.boxes))
  in
  let product lo hi = Array.fold_left ( * ) 1 (Array.sub shape lo (hi - lo)) in
  let nb = g.batch_axes and nr = g.batch_axes + g.row_axes in
  let whole = (product 0 nb, product nb nr, product nr (Array.length shape)) in
  let vector =
    Contract.vectored ~whole ~batch ~contracting ~segments (ca, cb) oa ob
  in
  let can_a, can_b = Contract.bandable ~contracting oa ob in
  (* A box past [w] reads each cast line that passes its share through a band
     ({!Contract.band}), which takes the share: what the other's cast leaves
     where that takes at most [w / 2], else [w / 2]. Only the box's one line of
     an operand ({!Pieces.cut}) is banded, and never in a call the vector
     kernels take or of terms in several runs of strides: such a line is cast
     whole, and the other line's share is what it leaves, a byte at least (a
     band then takes its least, a span of terms a range). Where neither can band
     (each line's terms make several runs, or one's do beside several lines of
     the other operand), both are cast whole, past [w]. *)
  let bands (x : Pieces.box) =
    let sa, sb = Pieces.bytes axes ~a:la ~b:lb x in
    let one s l = l > 0 && s <= l in
    let band_a = can_a && one sa la and band_b = can_b && one sb lb in
    let share s = Int.max 1 (w - s) in
    if vector || sa + sb <= w then (0, 0)
    else if not band_a then if band_b then (0, share sa) else (0, 0)
    else if not band_b then (share sb, 0)
    else if 2 * sa <= w then (0, w - sa)
    else if 2 * sb <= w then (w - sb, 0)
    else (w / 2, w / 2)
  in
  let a_bytes, b_bytes =
    Array.fold_left
      (fun (ma, mb) x ->
        let sa, sb = Pieces.bytes axes ~a:la ~b:lb x in
        let ba, bb = bands x in
        ( Int.max ma (if ba > 0 then 0 else sa),
          Int.max mb (if bb > 0 then 0 else sb) ))
      (0, 0) pieces
  in
  let at_b = round_up a_bytes region in
  let at_t = round_up (at_b + b_bytes) region in
  let size =
    if not r.temp then at_b + b_bytes
    else
      Array.fold_left
        (fun m (x : Pieces.box) ->
          Int.max m (at_t + (((Array.fold_left ( * ) 1 x.len * bits) + 7) / 8)))
        at_t pieces
  in
  let buffer = Rig.Buffer.create dv.rig size in
  Fun.protect ~finally:(fun () -> ignore (Rig.Buffer.free buffer)) @@ fun () ->
  let last = [| None; None |] in
  let piece (x : Pieces.box) =
    let ba, bb = bands x in
    let operand side axes cast banded at (o : Nx_kernel.operand) =
      let cuts = cuts_of axes x in
      (* The slice's distinct elements: cuts along an axis of [o] of stride 0
         read the same ones. A slice's axis of extent 1 may take stride 0, so
         [o]'s own strides decide. *)
      let (A.Any full) = o.array in
      let seen =
        List.filter
          (fun (i, _, _) -> A.Layout.stride (A.layout full) i <> 0)
          cuts
      in
      let o = slice_operand o cuts in
      match cast with
      | None -> (o, None)
      | Some _ when banded ->
          last.(side) <- None;
          (o, None)
      | Some dt ->
          let c = cast_into buffer ~at dt o.array in
          let again = last.(side) = Some seen in
          last.(side) <- Some seen;
          ({ o with array = c.view }, if again then None else Some c)
    in
    let oa, xa = operand 0 g.a_axes ca (ba > 0) 0 oa in
    let ob, xb = operand 1 g.b_axes cb (bb > 0) at_b ob in
    let all = Array.map Option.some (Array.init (Array.length shape) Fun.id) in
    let part = slice dst (cuts_of all x) in
    let dst =
      if r.temp then temp_at buffer ~at:at_t (A.dtype dst) x.len else part
    in
    let copy = if r.temp then Some (A.Any dst, A.Any part) else None in
    let init = Option.map (fun i -> slice i (cuts_of all x)) init in
    let segments =
      match (segments : _ Nx_kernel.segments option) with
      | Some (Rows r) ->
          Some
            (Nx_kernel.Rows { r with ids = slice r.ids (cuts_of g.ids_axes x) })
      | s -> s
    in
    let casts = List.filter_map Fun.id [ xa; xb ] in
    let band =
      if ba + bb = 0 then None else Some { Contract.a_band = ba; b_band = bb }
    in
    (casts, plan ~whole ?band ~dst ~init ~segments oa ob, copy)
  in
  let cast_plan c =
    let (A.Any x) = c.from in
    let (A.Any y) = c.into in
    let p = cast_prog (D.Any (A.dtype x)) (D.Any (A.dtype y)) in
    let xs =
      Operands.v
        ~dsts:(Iarray.of_list [ c.into ])
        (Iarray.of_list [ load_of c.from ])
    in
    match map_plan dv p xs with
    | Error answer -> Error answer
    | Ok p -> Ok (c, p)
  in
  (* The copy of a piece's temporary into its slice of [dst], where its call
     writes it. *)
  let copy_plan call = function
    | Some (A.Any t, A.Any d) when Option.is_some call -> (
        let xs =
          Operands.v
            ~dsts:(Iarray.of_list [ A.Any d ])
            (Iarray.of_list [ load_of (A.Any t) ])
        in
        match map_plan dv (identity_prog (D.Any (A.dtype t))) xs with
        | Error answer -> Error answer
        | Ok p -> Ok (Option.map (fun p -> (A.Any d, A.Any t, p)) p))
    | Some _ | None -> Ok None
  in
  let rec planned acc i =
    if i = Array.length pieces then Ok (List.rev acc)
    else
      let casts, (w, rd, answer), copy = piece pieces.(i) in
      match answer with
      | Error answer -> Error answer
      | Ok (Contract.Recast _) -> invalid_arg "Nx_metal.contract: a cast recast"
      | Ok (Contract.Planned call) -> (
          let rec cast_plans ps = function
            | [] -> Ok (List.rev ps)
            | c :: cs -> (
                match cast_plan c with
                | Error a -> Error a
                | Ok p -> cast_plans (p :: ps) cs)
          in
          match (cast_plans [] casts, copy_plan call copy) with
          | Error answer, _ | _, Error answer -> Error answer
          | Ok cs, Ok copy -> planned ((cs, (w, rd, call), copy) :: acc) (i + 1)
          )
  in
  match planned [] 0 with
  | Error answer -> answer
  | Ok steps ->
      let one x = Iarray.of_list [ x ] in
      let submit () =
        List.iter
          (fun (cs, (w, rd, call), copy) ->
            List.iter
              (fun (c, p) ->
                Option.iter (fun p -> issue (dv, one c.into, one c.from, p)) p)
              cs;
            Option.iter (fun p -> issue (dv, w, rd, p)) call;
            Option.iter (fun (d, t, p) -> issue (dv, one d, one t, p)) copy)
          steps
      in
      let bytes =
        A.Any
          (A.v D.Uint8
             (A.Layout.contiguous (Iarray.of_list [ Rig.Buffer.length buffer ]))
             buffer)
      in
      A.door
        ~written:(Iarray.append written (Iarray.of_list [ bytes ]))
        ~read submit ()

(* The plan of a contraction into [dst], the door's operands [written] then
   [read]. *)
let contract_call ?whole ?band ~batch ~contracting ~acc ~written ~read ~dst
    ~init ~segments oa ob =
  let ops = Iarray.to_array (Iarray.append written read) in
  Contract.call ?whole ?band ~ops ~batch ~contracting ~segments ~acc
    ~dst:(A.Any dst)
    ~init:(Option.map (fun i -> A.Any i) init)
    oa ob

let contract ~batch ~contracting ~segments ~acc ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let (D.Any acc_dt) = acc in
  if not (Nx_kernel.accumulates acc_dt) then A.Wrong_dtype
  else if not (contract_fits ~batch ~contracting ~segments ~dst ~init oa ob)
  then A.Shape_mismatch
  else
    match device (A.device dst) with
    | None -> A.Declined
    | Some dv -> (
        let written = Iarray.of_list [ A.Any dst ] in
        let read = contract_reads ~init ~segments oa ob in
        live ~written ~read @@ fun () ->
        match
          contract_call ~batch ~contracting ~acc ~written ~read ~dst ~init
            ~segments oa ob
        with
        | Error answer -> answer
        | Ok (Contract.Planned None) -> A.door ~written ~read nothing ()
        | Ok (Contract.Planned (Some c)) ->
            A.door ~written ~read issue (dv, written, read, c)
        | Ok (Contract.Recast r) ->
            let plan ~whole ?band ~dst ~init ~segments oa ob =
              let written = Iarray.of_list [ A.Any dst ] in
              let read = contract_reads ~init ~segments oa ob in
              ( written,
                read,
                contract_call ~whole ?band ~batch ~contracting ~acc ~written
                  ~read ~dst ~init ~segments oa ob )
            in
            recast dv ~written ~read ~batch ~contracting ~segments ~dst ~init
              ~plan r oa ob)

(* Fourier transforms *)

(* A complex64 transform of a plan whose every launch has a kernel: each emitted
   for [direction], the operand and [dst] at their buffers' bytes of a line, the
   workspace a piece of the call's. Any other declines. The door numbers [dst] 0
   and [x] 1, and a buffer takes the slot of its first operand
   ({!Plan.slot}). *)
let fft_plan plan ~inverse ~axes ~dst x =
  let bases = (Fft.line_offset (A.buffer x), Fft.line_offset (A.buffer dst)) in
  match
    plan ~bases ~axes:(Iarray.to_array axes) (A.layout x) ~dst:(A.layout dst)
  with
  | None -> Error A.Declined
  | Some call when not (Fft.runs call) -> Error A.Declined
  | Some call when Array.length call.launches = 0 -> Ok None
  | Some call ->
      let b = Plan.make () in
      let ws = if call.workspace > 0 then Plan.take b call.workspace else 0 in
      let place : Fft.buffer -> int * int = function
        | Operand when A.buffer x == A.buffer dst -> (Plan.slot b 0, 0)
        | Operand -> (Plan.slot b 1, 0)
        | Dst -> (Plan.slot b 0, 0)
        | Workspace at -> (Plan.workspace b, ws + at)
      in
      Array.iter (Fft.emit b ~inverse ~place) call.launches;
      Ok (Some (Plan.finish b))

let c2c (type s) direction ~axes ~(dst : (Complex.t, s) A.t)
    (x : (Complex.t, s) A.t) =
  Nx_kernel.check_dst "nx.metal c2c" "dst" dst;
  match Nx_kernel.Shape.c2c ~axes (shape_of x) with
  | Error _ -> A.Shape_mismatch
  | Ok want when want <> shape_of dst -> A.Shape_mismatch
  | Ok _ -> (
      match A.dtype x with
      | D.Complex128 -> A.Declined
      | D.Complex64 ->
          run (A.device dst) ~written:(Iarray.of_list [ A.Any dst ])
            ~read:(Iarray.of_list [ A.Any x ]) (fun _ ->
              let inverse = direction = Nx_kernel.Inverse in
              fft_plan Fft.c2c ~inverse ~axes ~dst x))

(* An r2c runs its plan's launches forward; a c2r, the inverse transform's real
   parts, runs them as an inverse. *)
let r2c (type c r) ~axes ~(dst : (Complex.t, c) A.t) (x : (float, r) A.t) =
  Nx_kernel.check_dst "nx.metal r2c" "dst" dst;
  match Nx_kernel.Shape.r2c ~axes (shape_of x) with
  | Error _ -> A.Shape_mismatch
  | Ok want when want <> shape_of dst -> A.Shape_mismatch
  | Ok _ -> (
      match (A.dtype x, A.dtype dst) with
      | D.Float32, D.Complex64 ->
          run (A.device dst) ~written:(Iarray.of_list [ A.Any dst ])
            ~read:(Iarray.of_list [ A.Any x ]) (fun _ ->
              fft_plan Fft.r2c ~inverse:false ~axes ~dst x)
      | _ -> A.Declined)

let c2r (type c r) ~axes ~(dst : (float, r) A.t) (x : (Complex.t, c) A.t) =
  Nx_kernel.check_dst "nx.metal c2r" "dst" dst;
  let s = shape_of dst and k = Iarray.length axes - 1 in
  let a = if k >= 0 then Iarray.get axes k else -1 in
  if a < 0 || a >= Array.length s then A.Shape_mismatch
  else
    match Nx_kernel.Shape.c2r ~axes ~n:s.(a) (shape_of x) with
    | Error _ -> A.Shape_mismatch
    | Ok want when want <> s -> A.Shape_mismatch
    | Ok _ -> (
        match (A.dtype x, A.dtype dst) with
        | D.Complex64, D.Float32 ->
            run (A.device dst) ~written:(Iarray.of_list [ A.Any dst ])
              ~read:(Iarray.of_list [ A.Any x ]) (fun _ ->
                fft_plan Fft.c2r ~inverse:true ~axes ~dst x)
        | _ -> A.Declined)

(* Sorts and numberings *)

let sort ~axis ~descending ~values ~positions x =
  Nx_kernel.check_dst "nx.metal sort" "values" values;
  Nx_kernel.check_dst "nx.metal sort" "positions" positions;
  let vs = shape_of values in
  let k =
    if axis >= 0 && axis < Array.length vs then Some vs.(axis) else None
  in
  if shape_of positions <> vs then A.Shape_mismatch
  else
    match Nx_kernel.Shape.sort ~axis ~k (shape_of x) with
    | Ok s when s = vs ->
        run (A.device values)
          ~written:(Iarray.of_array [| A.Any values; A.Any positions |])
          ~read:(Iarray.of_array [| A.Any x |])
          (fun dv ->
            Sort.sort ~holds:(threads_of dv) ~axis ~descending ~values
              ~positions x)
    | Ok _ | Error _ -> A.Shape_mismatch

(* A numbering's [size] is [first]'s length, and [counts] has [first]'s
   shape. *)
let unique ~ids ~first ~counts ~count keys =
  Nx_kernel.check_dst "nx.metal unique" "ids" ids;
  Nx_kernel.check_dst "nx.metal unique" "first" first;
  Nx_kernel.check_dst "nx.metal unique" "counts" counts;
  Nx_kernel.check_dst "nx.metal unique" "count" count;
  let want =
    match shape_of first with
    | [| size |] -> Nx_kernel.Shape.unique ~size (shape_of keys)
    | _ -> Error "first is not a vector"
  in
  match want with
  | Ok got
    when shape_of counts = shape_of first
         && got = (shape_of ids, shape_of first, shape_of count) ->
      run (A.device ids)
        ~written:
          (Iarray.of_array
             [| A.Any ids; A.Any first; A.Any counts; A.Any count |])
        ~read:(Iarray.of_array [| A.Any keys |])
        (fun dv ->
          Sort.unique ~holds:(threads_of dv) ~ids ~first ~counts ~count keys)
  | Ok _ | Error _ -> A.Shape_mismatch

(* Gathers and assemblies

   Each refuses what {!Nx_kernel.S} states it refuses, by {!Nx_kernel.Shape}'s
   rules, then plans one call of {!Index}. *)

(* Whether [want], the shapes {!Nx_kernel.Shape} gives, are [got]. *)
let fit want got = match want with Ok s -> s = got | Error _ -> false

let gather ~axis ~dst idx x =
  Nx_kernel.check_dst "nx.metal gather" "dst" dst;
  let want = Nx_kernel.Shape.gather ~axis (shape_of idx) (shape_of x) in
  if not (fit want (shape_of dst)) then A.Shape_mismatch
  else
    let written = [| A.Any dst |] and read = [| A.Any idx; A.Any x |] in
    run (A.device dst) ~written:(Iarray.of_array written)
      ~read:(Iarray.of_array read) (fun _ ->
        Index.gather ~ops:(Array.append written read) ~axis ~dst idx x)

let bound_index = Index.bound

(* A range gather's [size] is [dst]'s extent along the runs' axis. *)
let gather_ranges ~dst ~starts ~ends x =
  Nx_kernel.check_dst "nx.metal gather_ranges" "dst" dst;
  let ys = shape_of dst and a = A.Layout.rank (A.layout starts) - 1 in
  let want =
    if a < 0 || a >= Array.length ys then Error "no runs' axis"
    else
      Nx_kernel.Shape.gather_ranges ~size:ys.(a) (shape_of starts)
        (shape_of ends) (shape_of x)
  in
  if not (fit want ys) then A.Shape_mismatch
  else
    let written = [| A.Any dst |] in
    let read = [| A.Any starts; A.Any ends; A.Any x |] in
    run (A.device dst) ~written:(Iarray.of_array written)
      ~read:(Iarray.of_array read) (fun _ ->
        Index.gather_ranges
          ~ops:(Array.append written read)
          ~dst ~starts ~ends x)

let assemble ~fill ~dst pieces =
  Nx_kernel.check_dst "nx.metal assemble" "dst" dst;
  let n = Iarray.length pieces in
  let regions = Array.init n (fun i -> fst (Iarray.get pieces i)) in
  let want = Nx_kernel.Shape.assemble (shape_of dst) regions in
  let got = Array.init n (fun i -> shape_of (snd (Iarray.get pieces i))) in
  if not (fit want got) then A.Shape_mismatch
  else
    let read = Iarray.map (fun (_, x) -> A.Any x) pieces in
    run (A.device dst) ~written:(Iarray.of_list [ A.Any dst ]) ~read (fun _ ->
        let ops = Array.append [| A.Any dst |] (Iarray.to_array read) in
        Index.assemble ~ops ~fill ~dst pieces)

(* A fold adds: its dtype is one Prog.Add takes, and Metal computes on no
   float64 or complex128 value. *)
let fold p ~dst x =
  Nx_kernel.check_dst "nx.metal fold" "dst" dst;
  if not (Nx_kernel.Prog.accepts2 (Binary Add) (A.dtype x)) then A.Wrong_dtype
  else if not (fit (Nx_kernel.Shape.padded p (shape_of dst)) (shape_of x)) then
    A.Shape_mismatch
  else
    match D.Any (A.dtype x) with
    | D.Any (D.Float64 | D.Complex128) -> A.Declined
    | _ ->
        let written = [| A.Any dst |] and read = [| A.Any x |] in
        run (A.device dst) ~written:(Iarray.of_array written)
          ~read:(Iarray.of_array read) (fun _ ->
            Index.fold ~ops:(Array.append written read) p ~dst x)

(* Families not yet computed here

   Each declines every call, which the contract admits of any call, before it
   reads an array, once its destinations are C-contiguous, as
   Nx_kernel.check_dst checks. *)

let scatter _ ~unique:_ ~axis:_ ~dst ~into:_ _ _ =
  Nx_kernel.check_dst "nx.metal scatter" "dst" dst;
  A.Declined

(* Matrix routines *)

(* [plan] of the threads Metal lets each kernel's groups have, where the dtypes
   are taken and [want], the shapes {!Nx_kernel.Shape} gives, are [got]: on
   [written]'s first device through the door. *)
let routine ~takes want got ~(written : A.any iarray) ~(read : A.any iarray)
    plan =
  if not takes then A.Wrong_dtype
  else
    match want with
    | Ok s when s = got ->
        let (A.Any y) = Iarray.get written 0 in
        run (A.device y) ~written ~read (fun dv -> plan (threads_of dv))
    | Ok _ | Error _ -> A.Shape_mismatch

let full a = Nx_kernel.full_precision (A.dtype a)

let cholesky t ~dst a =
  Nx_kernel.check_dst "nx.metal cholesky" "dst" dst;
  routine ~takes:(full a)
    (Nx_kernel.Shape.cholesky (shape_of a))
    (shape_of dst) ~written:[| A.Any dst |] ~read:[| A.Any a |]
    (fun holds ->
      Factor.cholesky ~holds t (Operands.v ~dsts:[| A.Any dst |] [| Plain a |]))

let lu ~lu ~pivots ~perm a =
  Nx_kernel.check_dst "nx.metal lu" "lu" lu;
  Nx_kernel.check_dst "nx.metal lu" "pivots" pivots;
  Nx_kernel.check_dst "nx.metal lu" "perm" perm;
  routine ~takes:(full a)
    (Nx_kernel.Shape.lu (shape_of a))
    (shape_of lu, shape_of pivots, shape_of perm)
    ~written:[| A.Any lu; A.Any pivots; A.Any perm |]
    ~read:[| A.Any a |]
    (fun holds ->
      Factor.lu ~holds
        (Operands.v
           ~dsts:[| A.Any lu; A.Any pivots; A.Any perm |]
           [| Plain a |]))

(* [q] and [r] state the factors: reduced or complete, whichever fits. *)
let qr ~q ~r a =
  Nx_kernel.check_dst "nx.metal qr" "q" q;
  Nx_kernel.check_dst "nx.metal qr" "r" r;
  let got = (shape_of q, shape_of r) in
  let fits f = Nx_kernel.Shape.qr f (shape_of a) = Ok got in
  routine ~takes:(full a)
    (if fits Reduced then Ok got else Nx_kernel.Shape.qr Complete (shape_of a))
    got ~written:[| A.Any q; A.Any r |] ~read:[| A.Any a |]
    (fun holds ->
      Factor.qr ~holds (Operands.v ~dsts:[| A.Any q; A.Any r |] [| Plain a |]))

(* Whether [r], a result of floats or complex numbers, is of [a]'s precision. *)
let paired a r = Nx_kernel.same_precision (A.dtype a) (A.dtype r)

let svd ~s ?vectors a =
  Nx_kernel.check_dst "nx.metal svd" "s" s;
  Option.iter
    (fun (~u, ~vh) ->
      Nx_kernel.check_dst "nx.metal svd" "u" u;
      Nx_kernel.check_dst "nx.metal svd" "vh" vh)
    vectors;
  let shape f = Nx_kernel.Shape.svd f (shape_of a) in
  let takes = full a && paired a s in
  let plan dsts holds = Spectral.svd ~holds (Operands.v ~dsts [| Plain a |]) in
  match vectors with
  | None ->
      routine ~takes
        (Result.map (fun (s, _, _) -> s) (shape Reduced))
        (shape_of s) ~written:[| A.Any s |] ~read:[| A.Any a |]
        (plan [| A.Any s |])
  | Some (~u, ~vh) ->
      let got = (shape_of s, shape_of u, shape_of vh) in
      let dsts : A.any iarray = [| A.Any s; A.Any u; A.Any vh |] in
      routine ~takes
        (if shape Reduced = Ok got then Ok got else shape Complete)
        got ~written:dsts ~read:[| A.Any a |] (plan dsts)

let eigh ~w ?v a =
  Nx_kernel.check_dst "nx.metal eigh" "w" w;
  Option.iter (Nx_kernel.check_dst "nx.metal eigh" "v") v;
  let want = Nx_kernel.Shape.eigh (shape_of a) in
  let takes = full a && paired a w in
  let plan dsts holds = Spectral.eigh ~holds (Operands.v ~dsts [| Plain a |]) in
  match v with
  | None ->
      routine ~takes (Result.map fst want) (shape_of w) ~written:[| A.Any w |]
        ~read:[| A.Any a |] (plan [| A.Any w |])
  | Some v ->
      let dsts : A.any iarray = [| A.Any w; A.Any v |] in
      routine ~takes want
        (shape_of w, shape_of v)
        ~written:dsts ~read:[| A.Any a |] (plan dsts)

let eig ~w ?v a =
  Nx_kernel.check_dst "nx.metal eig" "w" w;
  Option.iter (Nx_kernel.check_dst "nx.metal eig" "v") v;
  let want = Nx_kernel.Shape.eigh (shape_of a) in
  let takes =
    full a && paired a w && Option.fold ~none:true ~some:(paired a) v
  in
  let plan dsts holds = Spectral.eig ~holds (Operands.v ~dsts [| Plain a |]) in
  match v with
  | None ->
      routine ~takes (Result.map fst want) (shape_of w) ~written:[| A.Any w |]
        ~read:[| A.Any a |] (plan [| A.Any w |])
  | Some v ->
      let dsts : A.any iarray = [| A.Any w; A.Any v |] in
      routine ~takes want
        (shape_of w, shape_of v)
        ~written:dsts ~read:[| A.Any a |] (plan dsts)

let solve_triangular t ~adjoint ~unit_diagonal ~dst a b =
  Nx_kernel.check_dst "nx.metal solve_triangular" "dst" dst;
  routine ~takes:(full a)
    (Nx_kernel.Shape.solve_triangular (shape_of a) (shape_of b))
    (shape_of dst) ~written:[| A.Any dst |] ~read:[| A.Any a; A.Any b |]
    (fun holds ->
      Factor.solve_triangular ~holds t ~adjoint ~unit_diagonal
        (Operands.v ~dsts:[| A.Any dst |] [| Plain a; Plain b |]))

(* Attention *)

let float64 (type a b) (dt : (a, b) D.t) =
  match dt with D.Float64 -> true | _ -> false

let attention (type w r c v s) (p : Nx_kernel.Prog.t) ~scale ~(o : (w, r) A.t)
    ~(lse : (float, c) A.t) ?lengths (q : (v, s) A.t)
    (kv : ((v, s) A.t, (int64, D.int64_elt) A.t) Nx_kernel.kv)
    (loads : A.any iarray) =
  Nx_kernel.check_dst "nx.metal attention" "o" o;
  Nx_kernel.check_dst "nx.metal attention" "lse" lse;
  let rank = A.Layout.rank (A.layout q) in
  match
    Nx_kernel.attention_refusal p ~q:(A.dtype q) ~rank ~o:(A.dtype o)
      ~acc:(A.dtype lse) ~loads:arrays loads
  with
  | Some (answer, _) -> answer
  | None -> (
      let k, v, table =
        match kv with
        | Nx_kernel.Dense { k; v } -> (k, v, None)
        | Paged { k; v; table } -> (k, v, Some (A.Any table))
      in
      let shapes =
        match kv with
        | Nx_kernel.Dense _ ->
            Nx_kernel.Dense { k = shape_of k; v = shape_of v }
        | Paged { table; _ } ->
            Paged { k = shape_of k; v = shape_of v; table = shape_of table }
      in
      match
        Nx_kernel.Shape.attention (shape_of q) shapes
          ~lengths:(Option.map shape_of lengths)
          (Iarray.to_array (Iarray.map (fun (A.Any x) -> shape_of x) loads))
      with
      | Error _ -> A.Shape_mismatch
      | Ok (os, ls) when os <> shape_of o || ls <> shape_of lse ->
          A.Shape_mismatch
      | Ok _ ->
          if
            float64 (A.dtype lse)
            || float64 (A.dtype q)
            || not (Attention.keeps_all p)
          then A.Declined
          else
            let written = Iarray.of_list [ A.Any o; A.Any lse ] in
            let read =
              Iarray.append
                (Iarray.of_list
                   ([ A.Any q; A.Any k; A.Any v ]
                   @ Option.to_list table
                   @ Option.to_list (Option.map (fun l -> A.Any l) lengths)))
                loads
            in
            run (A.device o) ~written ~read (fun dv ->
                let xs =
                  Operands.v ~dsts:written
                    (Iarray.map (fun (A.Any x) -> Nx_kernel.Plain x) read)
                in
                Attention.plan ~holds:(threads_of dv) p ~scale
                  ~paged:(Option.is_some table)
                  ~lengths:(Option.is_some lengths) xs))

module Kernels = Kernels
module Plan = Plan
module Operands = Operands
module Moves = Moves
module Loop = Loop
module Contract = Contract
module Pieces = Pieces
module Factor = Factor
module Fft_tables = Fft_tables
module Fft = Fft
module Sort = Sort
module Index = Index
module Spectral = Spectral

let bound_sorts = Sort.bound_sorts
let bound_fresh = Plan.bound_fresh
let bound_kept = Plan.bound_kept
let bound_iterations = Spectral.bound_iterations
let bound_direct = Loop.bound_direct
let bound_copy = Moves.bound_copy
let bound_fft_wide = Fft.bound_wide
let bound_offsets = Attention.bound_offsets

let kept_calls d =
  match device d with
  | None -> 0
  | Some dv ->
      Array.fold_left
        (fun n c -> n + Cache.count c)
        (Cache.count dv.key_subs) dv.subs

let kept_keys_on d =
  match device d with None -> 0 | Some dv -> Cache.count dv.keys

let kept_call_keys_on d =
  match device d with
  | None -> 0
  | Some dv ->
      List.fold_left (fun n (_, k) -> n + k) 0 (Cache.values dv.key_subs)

let forget_keys d =
  match device d with
  | None ->
      invalid_arg "Nx_metal.forget_keys: no Metal device nx.metal computes on"
  | Some dv ->
      Cache.clear dv.keys;
      Cache.clear dv.key_subs

let prepare d calls =
  match device d with
  | None -> invalid_arg "Nx_metal.prepare: no Metal device nx.metal computes on"
  | Some dv ->
      let keys =
        List.concat_map
          (fun (c : Fft.call) ->
            List.filter_map
              (fun (l : Fft.launch) ->
                Option.map (fun (k, cs) -> (k, cs, l.threads)) (Fft.key l))
              (Array.to_list c.launches))
          calls
      in
      ignore (call_keys dv keys)

let resolved d k =
  match device d with
  | None ->
      invalid_arg "Nx_metal.resolved: no Metal device nx.metal computes on"
  | Some dv -> Option.is_some (Atomic.get dv.funcs.(Kernels.index k))

let holds d k =
  match device d with
  | None -> invalid_arg "Nx_metal.holds: no Metal device nx.metal computes on"
  | Some dv -> threads_of dv k

let kept_constants d =
  match device d with None -> 0 | Some dv -> Cache.count dv.constants

let kept_plans d =
  match device d with
  | None -> 0
  | Some dv -> Array.fold_left (fun n c -> n + Cache.count c) 0 dv.plans

let call d ~written ~read c = run d ~written ~read (fun _ -> Ok (Some c))

(* A routine's product, for tests: its matrices where they lie, or with
   [workspace] each copied into a piece of the workspace laid out as it is, the
   product computed there and y copied back. *)
let product ?(init_first = false) ?(neg = false) ?(conj_a = false)
    ?(conj_b = false) ?tri ?ranges ?(workspace = false) ~acc ~init ~y a m =
  let o = { Contract.init_first; neg; conj_a; conj_b; tri; ranges } in
  let (A.Any yy) = y in
  let ops = [| y; a; m |] in
  (* Operand [k]'s buffer's first operand, which names its slot. *)
  let first k =
    let (A.Any x) = ops.(k) in
    let rec go i =
      let (A.Any z) = ops.(i) in
      if A.buffer z == A.buffer x then i else go (i + 1)
    in
    go 0
  in
  run (A.device yy) ~written:[| y |] ~read:[| a; m |] (fun _ ->
      let b = Plan.make () in
      let arr k = Contract.array (first k) ops.(k) in
      match
        if not workspace then
          Contract.product b o ~acc ~init ~y:(arr 0) (arr 1) (arr 2)
        else begin
          (* A piece of the workspace laid out as [x] is in its buffer. *)
          let piece (A.Any x) =
            let l = A.layout x and dt = A.dtype x in
            let at = Plan.take b (snd (A.Layout.span l) * (D.bits dt / 8)) in
            Contract.work ~at (D.Any dt) l
          in
          let into k =
            let w = piece ops.(k) in
            Contract.copy b ~src:(arr k) ~dst:w;
            w
          in
          let wa = into 1 and wm = into 2 in
          (* [tri] keeps the outputs of the tiles it skips. *)
          let wy = if init || tri <> None then into 0 else piece y in
          Contract.product b o ~acc ~init ~y:wy wa wm;
          Contract.copy b ~src:wy ~dst:(arr 0)
        end
      with
      | () -> Ok (Plan.finish_some b)
      | exception Contract.Decline -> Error A.Declined)
