(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Sub = Rig.Submission

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type slot =
  | Operand of int
  | Workspace
  | Tickets
  | Fresh of int
  | Constant of { key : string; length : int; bytes : unit -> bigbytes }
  | Uploaded of string

let kept_workspace = 64 * 1024 * 1024
let kept_tickets = 4 * 1024 * 1024
let kept_constants = 64 * 1024 * 1024
let kept_constant_count = 16

(* The count of changes to a bound a plan reads. *)
let epoch_count = Atomic.make 0
let epoch () = Atomic.get epoch_count
let renew () = Atomic.incr epoch_count

(* The bounds {!finish} applies, below the kept sizes in tests. *)
let workspace_bound = Atomic.make kept_workspace
let tickets_bound = Atomic.make kept_tickets

let bound_fresh ~workspace ~tickets =
  if workspace < 0 || workspace > kept_workspace then
    invalid_arg "Nx_metal.bound_fresh: workspace";
  if tickets < 0 || tickets > kept_tickets then
    invalid_arg "Nx_metal.bound_fresh: tickets";
  Atomic.set workspace_bound workspace;
  Atomic.set tickets_bound tickets;
  renew ()

(* The most launches of a call {!finish} keeps, below every count in tests. *)
let kept_bound = Atomic.make max_int

let bound_kept ~launches =
  if launches < 0 then invalid_arg "Nx_metal.bound_kept: launches";
  Atomic.set kept_bound launches;
  renew ()

type ref = { word : int; slot : int; write : bool }

(* A launch {!traced} saw. *)
type traced = {
  kernel : Kernels.t;
  constants : int array;
  groups : int * int * int;
  resident : int;
  tickets : bool;
  words : Bytes.t;
}

type cost = { bytes : int; ops : int; chain : int }

type launch = {
  kernel : Kernels.t;
  constants : int array;
  groups : int * int * int;
  threads : int * int * int;
  shared : int;
  resident : int;
  cost : cost;
  words : Bytes.t;
  refs : ref array;
}

type part = {
  run : launch array;
  binds : int array;
  local : int array;
  kept : bool;
  commits : bool;
}

type t = {
  launches : launch array;
  slots : slot array;
  workspace : int;
  tickets : int;
  parts : part array;
  zeroing : int;
  tickets_slot : int;
}

(* Time budget

   The M1 of 7 GPU cores is the Apple7 family's slowest. Its rates: 0.85 of its
   LPDDR4X's 68.25 GB/s, since the M1 Max reaches 0.93 of its own 400 GB/s in a
   copy, and 1.8 counted in whole lines from a source strided by a line; 0.91 of
   a float32 operation a lane and a cycle, 128 lanes a core, at a nominal 1.278
   GHz, the best case: 128 independent fmas a step, written out, reach 0.91 of
   one a lane and a cycle in a full grid of the M1 Max, and the M1 is taken to
   reach the same fraction; a dependent chain's 35.5 ns a link and a dependent
   device read's 472 ns, measured on the M1 Max at 1.296 GHz in a lone thread,
   rounded to 25e6 links a second and 16 links a read.

   The rates hold at the GPU's sustained clock. A launch issued after idle
   starts lower, at down to about 0.3 of it, so its work can take up to about 3
   times its estimate. The budget leaves that margin: 50 ms is a tenth of the
   0.5 s after which this family's watchdog has ended a command buffer. A larger
   budget must keep it.

   A launch has a fixed cost, the command processor's, which no count of the
   launch's states. It bounds microseconds: a choice between plans of different
   launch counts. Measured on the M1 Max, scaled to the M1 by the clock alone
   (1.296/1.278), as no M1 is at hand to show its command processor's cost
   differs, and taken 1.48 times: a dispatch of an empty kernel in a chain of
   dependent dispatches takes 1.24 to 1.37 us. *)

let line = 128
let bandwidth = 0.85 *. 68.25e9
let cores = 7
let lanes = 128
let nominal_clock = 1.278e9
let op_rate = 0.91 *. nominal_clock
let lane_rate = 25.0e6
let read_links = 16
let divide_ops = 16
let wide_divide_ops = 96
let native_divide_ops = 4096
let core_threads = 3072
let core_memory = 64 * 1024
let budget = 0.05
let most_refs = Kernels.loop_buffers + 1
let part_budget = 10.0
let part_launches = 100_000

(* rig_metal commits a command buffer once 256 values have joined it. *)
let commit_after = part_budget /. 256.
let margin = 1.48 *. (1.296 /. 1.278)
let dispatch = margin *. 1.37e-6

(* {!estimate} and {!seconds} without their optional argument, inlined, so that
   a caller here that compares or sums the result boxes no float and no
   option. *)
let[@inline] rated ~bandwidth ~cores ~op_rate ~threads ~groups:(x, y, z)
    ~resident c =
  let groups = x * y * z in
  let filled = Int.min (Int.min groups cores * lanes) (groups * threads) in
  let wave = resident * cores in
  let waves = (groups + wave - 1) / wave in
  (float c.bytes /. bandwidth)
  +. (float c.ops /. (float filled *. op_rate))
  +. (float waves *. float c.chain /. lane_rate)

let[@inline] time ~threads ~groups ~resident c =
  dispatch +. rated ~bandwidth ~cores ~op_rate ~threads ~groups ~resident c

let[@inline] estimate ~bandwidth ~cores ~op_rate ~threads ~groups
    ?(resident = 1) c =
  rated ~bandwidth ~cores ~op_rate ~threads ~groups ~resident c

let seconds ~threads ~groups ?(resident = 1) c =
  time ~threads ~groups ~resident c

(* rig rounds threadgroup memory to 16 bytes. *)
let holds ~threads shared =
  let by_threads = core_threads / threads in
  if shared = 0 then by_threads
  else Int.min by_threads (core_memory / ((shared + 15) land lnot 15))

let resident ~granted k ~shared =
  let threads = Kernels.threads k in
  Int.min (holds ~threads shared) (granted k / threads)

let most ~fits groups =
  if fits groups then Some groups
  else if not (fits 1) then None
  else
    (* [lo] fits, [hi] does not. *)
    let rec go lo hi =
      if hi - lo <= 1 then lo
      else
        let m = (lo + hi) / 2 in
        if fits m then go m hi else go lo m
    in
    Some (go 1 groups)

(* The most seconds of a launch, below {!budget} in tests. *)
let budget_bound = Atomic.make budget
let allowed () = Atomic.get budget_bound

let bound_budget s =
  if not (s > 0. && s <= budget) then invalid_arg "Nx_metal.bound_budget";
  Atomic.set budget_bound s;
  renew ()

(* The most launches and seconds of a part, below {!part_launches} and
   {!part_budget} in tests. *)
let part_launches_bound = Atomic.make part_launches
let part_bound = Atomic.make part_budget
let part_allowed () = Atomic.get part_bound

let bound_parts ~launches ~seconds =
  if launches < 1 || launches > part_launches then
    invalid_arg "Nx_metal.bound_parts: launches";
  if not (seconds > 0. && seconds <= part_budget) then
    invalid_arg "Nx_metal.bound_parts: seconds";
  Atomic.set part_launches_bound launches;
  Atomic.set part_bound seconds;
  renew ()

(* Building

   Slots get indices in the order the plan first names them. The workspace is
   carved as a stack: a call's launches run in order, each after the last, so a
   piece is free once the launches that use it are planned. A ticket word is
   zero at each launch's start, since the launch before left it zero, so every
   launch takes its tickets from the slot's start. *)

type builder = {
  mutable launched : launch list;
  mutable named : slot list;
  mutable count : int;
  mutable workspace_slot : int;
  mutable tickets_slot : int;
  mutable top : int;
  mutable peak : int;
  mutable ticket_bytes : int;
}

let make () =
  {
    launched = [];
    named = [];
    count = 0;
    workspace_slot = -1;
    tickets_slot = -1;
    top = 0;
    peak = 0;
    ticket_bytes = 0;
  }

let add b s =
  b.named <- s :: b.named;
  b.count <- b.count + 1;
  b.count - 1

let rec find k i = function
  | [] -> -1
  | Operand j :: _ when j = k -> i
  | _ :: rest -> find k (i - 1) rest

let slot b k =
  if k < 0 then invalid_arg "Plan.slot: operand";
  match find k (b.count - 1) b.named with -1 -> add b (Operand k) | i -> i

let rec find_key key i = function
  | [] -> None
  | Constant c :: _ when String.equal c.key key -> Some (i, c.length)
  | _ :: rest -> find_key key (i - 1) rest

let constant b ~key ~length bytes =
  if length < 0 then invalid_arg "Plan.constant: length";
  match find_key key (b.count - 1) b.named with
  | None -> add b (Constant { key; length; bytes })
  | Some (i, l) when l = length -> i
  | Some _ -> invalid_arg (Printf.sprintf "Plan.constant: %S's length" key)

let rec find_bytes bytes i = function
  | [] -> -1
  | Uploaded s :: _ when String.equal s bytes -> i
  | _ :: rest -> find_bytes bytes (i - 1) rest

let uploaded b bytes =
  match find_bytes bytes (b.count - 1) b.named with
  | -1 -> add b (Uploaded bytes)
  | i -> i

let workspace b =
  if b.workspace_slot < 0 then b.workspace_slot <- add b Workspace;
  b.workspace_slot

let take b n =
  let at = (b.top + 255) land lnot 255 in
  b.top <- at + n;
  b.peak <- Int.max b.peak b.top;
  at

let mark b = b.top
let release b m = b.top <- m

let tickets_slot b =
  if b.tickets_slot < 0 then b.tickets_slot <- add b Tickets;
  b.tickets_slot

let tickets b n =
  b.ticket_bytes <- Int.max b.ticket_bytes (4 * n);
  0

(* Division by an invariant integer (Granlund and Montgomery, 1994): for [s =
   ⌈log₂ d⌉], [m = ⌊2³² (2^s - d) / d⌋ + 1] fits 32 bits, and [⌊n m / 2³²⌋ + n],
   shifted right by [s], is [n / d] for every [n] below 2³². The divisor of a
   number past 32 bits is 0, which no kernel reads. The word is m with s above
   it, an int, so that no plan boxes it; m's quotient is taken 16 bits at a
   time, since (2^s - d) 2³² passes 63 bits. *)
let divisor d =
  if d < 1 || d >= 1 lsl 32 then 0
  else
    let rec log2_up s = if 1 lsl s >= d then s else log2_up (s + 1) in
    let s = log2_up 0 in
    let a = ((1 lsl s) - d) lsl 16 in
    let r = (a mod d) lsl 16 in
    let m = ((a / d) lsl 16) + (r / d) + 1 in
    m lor (s lsl 32)

let words k = Bytes.make (8 * Kernels.words k) '\000'
let set w i v = Bytes.set_int64_le w (8 * i) (Int64.of_int v)
let group_memory = 32 * 1024
let most_groups = (1 lsl 32) - 1

(* The launches the domain planned since {!traced} began, the latest first. *)
let trace : traced list option Domain.DLS.key =
  Domain.DLS.new_key (fun () -> None)

let traced f =
  Domain.DLS.set trace (Some []);
  let taken () =
    let l = Option.value (Domain.DLS.get trace) ~default:[] in
    Domain.DLS.set trace None;
    List.rev l
  in
  match f () with
  | x -> (x, taken ())
  | exception e ->
      ignore (taken ());
      raise e

let unseen f =
  let seen = Domain.DLS.get trace in
  Domain.DLS.set trace None;
  Fun.protect ~finally:(fun () -> Domain.DLS.set trace seen) f

let record traced =
  match Domain.DLS.get trace with
  | None -> ()
  | Some l -> Domain.DLS.set trace (Some (traced :: l))

let axis_fits n = n > 0 && n <= most_groups

(* Each kernel's threads a group, by {!Kernels.index}, made once so that a
   launch's values take none of their own. *)
let group_threads = Array.map (fun k -> (Kernels.threads k, 1, 1)) Kernels.all

(* A specialised step's function takes its constants, every other kernel
   none. *)
let specialised k = Kernels.family k = Kernels.family_fft_steps

let launch b kernel ~groups ?(shared = 0) ?(resident = 1) ?threads
    ?(constants = [||]) ~cost words refs =
  let x, y, z = groups in
  if not (axis_fits x && axis_fits y && axis_fits z) then
    invalid_arg "Plan.launch: groups";
  if shared < 0 || shared > group_memory then invalid_arg "Plan.launch: shared";
  if specialised kernel <> (Array.length constants > 0) then
    invalid_arg "Plan.launch: constants";
  let most = Kernels.threads kernel in
  let threads = Option.value threads ~default:most in
  if threads < 1 || threads > most then invalid_arg "Plan.launch: threads";
  if resident < 1 || resident > holds ~threads shared then
    invalid_arg "Plan.launch: resident";
  if cost.bytes < 0 || cost.ops < 0 || cost.chain < 0 then
    invalid_arg "Plan.launch: cost";
  if time ~threads ~groups ~resident cost > allowed () then
    invalid_arg "Plan.launch: past the time budget";
  let n = Kernels.words kernel in
  if Bytes.length words <> 8 * n then invalid_arg "Plan.launch: words";
  if Array.exists (fun r -> r.word < 0 || r.word >= n) refs then
    invalid_arg "Plan.launch: a ref past the words";
  if Array.length refs > most_refs then invalid_arg "Plan.launch: refs";
  if Option.is_some (Domain.DLS.get trace) then begin
    let ticket r = r.slot = b.tickets_slot in
    let tickets = b.tickets_slot >= 0 && Array.exists ticket refs in
    record { kernel; constants; groups; resident; tickets; words }
  end;
  let threads =
    if threads = most then group_threads.(Kernels.index kernel)
    else (threads, 1, 1)
  in
  b.launched <-
    { kernel; constants; groups; threads; shared; resident; cost; words; refs }
    :: b.launched

(* A zeroing thread's work: its index and a store, at most what a copy's thread
   takes before its axes, measured at 101 operations. *)
let zero_ops = 128

(* The cost of zeroing [n] words in [groups] groups of [per] threads: each
   word's 4 bytes and a line at each end, and each thread's work. *)
let zero_cost ~per ~groups n =
  { bytes = (4 * n) + (2 * line); ops = zero_ops * per * groups; chain = 0 }

(* The launches that zero the [bytes] of the slot [slot], one word a thread,
   each within {!allowed}: as many words as {!seconds}' bounds leave, less one
   for rounding. The bounds are {!dispatch}; on the work, [(n + per - 1) w /
   (cores lanes op_rate) + ceil (per / lanes) w / op_rate] at [w] = {!zero_ops};
   and on the traffic, [(4 n + 2 line) / bandwidth]. *)
let zeroing ?(at = 0) slot bytes =
  let per = Kernels.threads Kernels.Zero in
  let w = float zero_ops in
  let word = (4. /. bandwidth) +. (w /. (float (cores * lanes) *. op_rate)) in
  let fixed =
    dispatch
    +. (float (2 * line) /. bandwidth)
    +. (float (per - 1) *. w /. (float (cores * lanes) *. op_rate))
    +. (float ((per + lanes - 1) / lanes) *. w /. op_rate)
  in
  let most = Int.max 1 (Float.to_int ((allowed () -. fixed) /. word) - 1) in
  let count = bytes / 4 in
  let piece first =
    let n = Int.min most (count - first) in
    let words = words Kernels.Zero in
    let groups = (n + per - 1) / per in
    set words Kernels.zero_dst (at + (4 * first));
    set words Kernels.zero_count n;
    {
      kernel = Kernels.Zero;
      constants = [||];
      threads = group_threads.(Kernels.index Kernels.Zero);
      groups = (groups, 1, 1);
      shared = 0;
      resident = 1;
      cost = zero_cost ~per ~groups n;
      words;
      refs = [| { word = Kernels.zero_dst; slot; write = true } |];
    }
  in
  List.init ((count + most - 1) / most) (fun i -> piece (i * most))

let zero b ~slot ~at bytes =
  if at < 0 || at mod 4 <> 0 || bytes < 0 || bytes mod 4 <> 0 then
    invalid_arg "Plan.zero";
  List.iter
    (fun (l : launch) ->
      launch b l.kernel ~groups:l.groups ~cost:l.cost l.words l.refs)
    (zeroing ~at slot bytes)

(* Refuses a slot of [slots] no launch's ref names, a ref that names no slot,
   and a ref that writes a constant. A key ({!key}) holds refs alone, so the
   submission of a call that named one more slot than its refs would take fewer
   buffers than the call binds. *)
let check_named slots launched =
  let n = Array.length slots in
  let named = Array.make n false in
  let name r =
    if r.slot < 0 || r.slot >= n then
      invalid_arg (Printf.sprintf "Plan.finish: no slot %d" r.slot);
    (match slots.(r.slot) with
    | (Constant _ | Uploaded _) when r.write ->
        invalid_arg (Printf.sprintf "Plan.finish: slot %d is written" r.slot)
    | Constant _ | Uploaded _ | Operand _ | Workspace | Tickets | Fresh _ -> ());
    named.(r.slot) <- true
  in
  List.iter (fun l -> Array.iter name l.refs) launched;
  Array.iteri
    (fun i named ->
      if not named then
        invalid_arg
          (Printf.sprintf "Plan.finish: slot %d is in no launch's refs" i))
    named

(* Parts

   A launch's parameters take its words rounded up to 256 bytes of the device's
   parameter arena, and a submit's blocks 32 bytes and twice its words for its
   values (Rig.submit). A part after the first stays within the first's of both,
   so that its submit grows neither: the arena grows only for a submission past
   the empty arena, and then to hold twice it. *)

let arena_bytes l = (Bytes.length l.words + 255) land lnot 255
let block_bytes l = 32 + (2 * Bytes.length l.words)

let[@inline] launch_seconds l =
  time ~threads:(Kernels.threads l.kernel) ~groups:l.groups ~resident:l.resident
    l.cost

(* The end of the part of [launches] from [first]: as many launches as keep it
   within its bounds and [arena] and [blocks] bytes, at least one. Cutting a
   call allocates nothing until it is known to be of several parts. *)
let part_end launches first ~arena ~blocks =
  let most = Atomic.get part_launches_bound and allowed = part_allowed () in
  let n = Array.length launches and l = launches.(first) in
  let last = ref (first + 1) and fits = ref true in
  let secs = ref (launch_seconds l) in
  let a = ref (arena_bytes l) and g = ref (block_bytes l) in
  while !fits && !last < n do
    let l = launches.(!last) in
    let s = launch_seconds l and la = arena_bytes l and lg = block_bytes l in
    fits :=
      !last - first < most
      && !secs +. s <= allowed
      && !a + la <= arena
      && !g + lg <= blocks;
    if !fits then begin
      secs := !secs +. s;
      a := !a + la;
      g := !g + lg;
      incr last
    end
  done;
  !last

(* Whether the launches of [launches] from [first] to before [last] pass
   {!commit_after} together. *)
let commits launches first last =
  let secs = ref 0. in
  for i = first to last - 1 do
    secs := !secs +. launch_seconds launches.(i)
  done;
  !secs > commit_after

(* The first launch of each part of [launches], the first of which ends at
   [first_end]. *)
let starts launches first_end =
  let total = Array.length launches in
  let a = ref 0 and g = ref 0 in
  for i = 0 to first_end - 1 do
    a := !a + arena_bytes launches.(i);
    g := !g + block_bytes launches.(i)
  done;
  let arena = !a and blocks = !g in
  let next i = part_end launches i ~arena ~blocks in
  let count = ref 1 and at = ref first_end in
  while !at < total do
    at := next !at;
    incr count
  done;
  let starts = Array.make !count 0 in
  at := first_end;
  for k = 1 to !count - 1 do
    starts.(k) <- !at;
    at := next !at
  done;
  starts

(* The part of [launches] from [first] to before [last], whose slots number the
   call's [n] slots in the order its launches first name them. *)
let part ~n ~kept launches first last =
  let run = Array.sub launches first (last - first) in
  let local = Array.make n (-1) and binds = ref [] and count = ref 0 in
  let name r =
    if local.(r.slot) < 0 then begin
      local.(r.slot) <- !count;
      binds := r.slot :: !binds;
      incr count
    end
  in
  Array.iter (fun l -> Array.iter name l.refs) run;
  {
    run;
    binds = Array.of_list (List.rev !binds);
    local;
    kept;
    commits = commits launches first last;
  }

let finish ?(most = max_int) b =
  if b.launched = [] then invalid_arg "Plan.finish: no launch";
  let slots = Array.of_list (List.rev b.named) in
  check_named slots b.launched;
  let tickets, zeroes =
    if b.tickets_slot < 0 || b.ticket_bytes <= Atomic.get tickets_bound then
      (b.ticket_bytes, [])
    else begin
      slots.(b.tickets_slot) <- Fresh b.ticket_bytes;
      (0, zeroing b.tickets_slot b.ticket_bytes)
    end
  in
  let zeroing = List.length zeroes in
  let launches = Array.of_list (zeroes @ List.rev b.launched) in
  let first_end = part_end launches 0 ~arena:max_int ~blocks:max_int in
  let several = first_end < Array.length launches in
  let workspace =
    if
      b.workspace_slot < 0
      || ((not several) && b.peak <= Atomic.get workspace_bound)
    then b.peak
    else begin
      slots.(b.workspace_slot) <- Fresh b.peak;
      0
    end
  in
  let kept_most = Int.min most (Atomic.get kept_bound) in
  let parts =
    if not several then
      let n = Array.length slots in
      let ids = Array.init n Fun.id in
      [|
        {
          run = launches;
          binds = ids;
          local = ids;
          kept = Array.length launches - zeroing <= kept_most;
          commits = commits launches 0 (Array.length launches);
        };
      |]
    else
      let n = Array.length slots and total = Array.length launches in
      let starts = starts launches first_end in
      Array.mapi
        (fun i first ->
          let last =
            if i + 1 < Array.length starts then starts.(i + 1) else total
          in
          let own = last - Int.max first zeroing in
          part ~n ~kept:(own <= kept_most) launches first last)
        starts
  in
  {
    launches;
    slots;
    workspace;
    tickets;
    parts;
    zeroing;
    tickets_slot = b.tickets_slot;
  }

let finish_some ?most b =
  if b.launched = [] then None else Some (finish ?most b)

let replay c =
  if Option.is_some (Domain.DLS.get trace) then
    for i = c.zeroing to Array.length c.launches - 1 do
      let l = c.launches.(i) in
      let ticket r = r.slot = c.tickets_slot in
      let tickets = c.tickets_slot >= 0 && Array.exists ticket l.refs in
      record
        {
          kernel = l.kernel;
          constants = l.constants;
          groups = l.groups;
          resident = l.resident;
          tickets;
          words = l.words;
        }
    done

(* Submissions

   A key is the count of launches, then each one's kernel, count of constants
   and its constants, count of refs and its refs, each a word, the part's slot
   and 1 if written. *)

let key p =
  let ls = p.run in
  let n =
    Array.fold_left
      (fun n l -> n + 3 + Array.length l.constants + (3 * Array.length l.refs))
      1 ls
  in
  let k = Array.make n (Array.length ls) in
  let at = ref 1 in
  Array.iter
    (fun l ->
      let m = Array.length l.constants in
      k.(!at) <- Kernels.index l.kernel;
      k.(!at + 1) <- m;
      Array.blit l.constants 0 k (!at + 2) m;
      at := !at + 2 + m;
      k.(!at) <- Array.length l.refs;
      at := !at + 1;
      Array.iter
        (fun r ->
          k.(!at) <- r.word;
          k.(!at + 1) <- p.local.(r.slot);
          k.(!at + 2) <- Bool.to_int r.write;
          at := !at + 3)
        l.refs)
    ls;
  k

let rec same_refs local k at refs j =
  j = Array.length refs
  ||
  let r = refs.(j) in
  k.(at) = r.word
  && k.(at + 1) = local.(r.slot)
  && k.(at + 2) = Bool.to_int r.write
  && same_refs local k (at + 3) refs (j + 1)

let rec same_constants k at cs j =
  j = Array.length cs
  || (k.(at) = cs.(j) && same_constants k (at + 1) cs (j + 1))

(* Whether [p]'s launches from [i] match [k] from [at]. *)
let rec same k at p i =
  if i = Array.length p.run then at = Array.length k
  else
    let l = p.run.(i) in
    let c = Array.length l.constants and m = Array.length l.refs in
    let refs = at + 2 + c in
    refs + 1 + (3 * m) <= Array.length k
    && k.(at) = Kernels.index l.kernel
    && k.(at + 1) = c
    && same_constants k (at + 2) l.constants 0
    && k.(refs) = m
    && same_refs p.local k (refs + 1) l.refs 0
    && same k (refs + 1 + (3 * m)) p (i + 1)

let matches k p =
  Array.length k > 0 && k.(0) = Array.length p.run && same k 1 p 0

let label k constants =
  String.concat " "
    (Kernels.name k :: Array.to_list (Array.map string_of_int constants))

let parts fs p =
  Array.map
    (fun l ->
      let refs =
        Array.map
          (fun r ->
            let slot = p.local.(r.slot) in
            if r.write then Sub.write slot ~at:(8 * r.word)
            else Sub.read slot ~at:(8 * r.word))
          l.refs
      in
      let label =
        if Array.length l.constants = 0 then None
        else Some (label l.kernel l.constants)
      in
      Sub.launch ~refs ?label (fs l) ~params:(8 * Kernels.words l.kernel))
    p.run

let launch_args l =
  {
    Sub.groups = l.groups;
    threads = l.threads;
    shared = l.shared;
    params = l.words;
  }

let args p = Array.map launch_args p.run
