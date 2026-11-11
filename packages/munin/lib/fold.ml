(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Smap = Map.Make (String)

(* Columns

   A key's samples live in three unboxed arrays shared by every history that
   holds them. A history sees the first [len] samples. A fold that extends a
   history appends past [len] in place when it claims the store, by moving
   [used] from [len] to [len + 1]; a second fold of the same history finds
   [used] past its [len] and copies. Readers of an older history read below
   their own [len] alone, so an append never changes what they see. [arrays] is
   replaced when the store grows; an Atomic publishes the copied prefix to other
   domains. *)

type arrays = {
  steps : int array;
  times : Float.Array.t;
  values : Float.Array.t;
}

type store = { arrays : arrays Atomic.t; used : int Atomic.t }
(* [ordered] is whether no step of the first [len] is below the one before,
   [last] the step of sample [len - 1]: a push keeps it in one comparison. *)
type col = { store : store; len : int; last : int; ordered : bool }

let alloc cap =
  {
    steps = Array.make cap 0;
    times = Float.Array.create cap;
    values = Float.Array.create cap;
  }

let copy_prefix a len cap =
  let b = alloc cap in
  Array.blit a.steps 0 b.steps 0 len;
  Float.Array.blit a.times 0 b.times 0 len;
  Float.Array.blit a.values 0 b.values 0 len;
  b

let empty_col () =
  {
    store = { arrays = Atomic.make (alloc 0); used = Atomic.make 0 };
    len = 0;
    last = 0;
    ordered = true;
  }

let write a i (s : Metric.sample) =
  Array.unsafe_set a.steps i s.step;
  Float.Array.unsafe_set a.times i s.time;
  Float.Array.unsafe_set a.values i s.value

let push c (s : Metric.sample) =
  let st = c.store in
  let ordered = c.ordered && (c.len = 0 || s.step >= c.last) in
  if Atomic.compare_and_set st.used c.len (c.len + 1) then begin
    let a = Atomic.get st.arrays in
    let a =
      if c.len < Array.length a.steps then a
      else
        let b = copy_prefix a c.len (max 16 (2 * c.len)) in
        Atomic.set st.arrays b;
        b
    in
    write a c.len s;
    { c with len = c.len + 1; last = s.step; ordered }
  end
  else begin
    let a =
      copy_prefix (Atomic.get st.arrays) c.len (max 16 (2 * (c.len + 1)))
    in
    write a c.len s;
    {
      store = { arrays = Atomic.make a; used = Atomic.make (c.len + 1) };
      len = c.len + 1;
      last = s.step;
      ordered;
    }
  end

let col_series c =
  let a = Atomic.get c.store.arrays in
  {
    Series.steps = a.steps;
    times = a.times;
    values = a.values;
    length = c.len;
    ordered = c.ordered;
  }

let sample_at c i : Metric.sample =
  let a = Atomic.get c.store.arrays in
  {
    step = a.steps.(i);
    time = Float.Array.get a.times i;
    value = Float.Array.get a.values i;
  }

(* State *)

type kind = [ `Metric | `Reading ]

type key = {
  kind : kind;
  acc : Acc.t; (* the statistics of the samples that are not orphans *)
  col : col option; (* those samples, when the history keeps the key's *)
  orph : col option;
  max_step : int; (* the largest step of a [metrics] sample not an orphan *)
}

type file = { key : string; step : int; time : float; path : string }
type resume = { step : int; time : float; provenance : Provenance.t }

(* [plan] classifies samples by the resumes of a log read before: a sample at
   offset [o] with step [s] is an orphan iff a resume after [o] names a step
   below [s]. [offsets] are the resumes' offsets, [mins.(i)] the least step of
   resumes [i] and later. [upto] is the last offset the plan covers; a resume at
   or before it was applied already. *)
type plan = { offsets : int array; mins : int array; upto : int }

type t = {
  run : Listing.t;
  path : string;
  ino : int;
  offset : int;
  keep : string list option;
  keys : key Smap.t;
  defs : (Metric.goal option * string option) Smap.t;
  summary : Value.t Smap.t;
  files : file list; (* reversed *)
  named : string list; (* every file record's path, orphans included *)
  scan : bool; (* a scan for resumes alone, which adds no sample *)
  uses : (string * int) list; (* reversed *)
  resumes : resume list; (* reversed *)
  known : (int * int) list; (* reversed: (offset, step) of each resume *)
  damaged : int;
  restarted : bool;
  last_end : (float * Record.status) option;
  plan : plan option;
}

exception Reread

let keeps st k = match st.keep with None -> true | Some l -> List.mem k l

let new_key st k kind =
  let col = if keeps st k then Some (empty_col ()) else None in
  {
    kind;
    acc = Acc.empty;
    col;
    orph = Option.map (fun _ -> empty_col ()) col;
    max_step = -1;
  }

let orphan_by_plan st ~offset step =
  match st.plan with
  | None -> false
  | Some p ->
      (* The first resume after [offset]. *)
      let n = Array.length p.offsets in
      let rec first lo hi =
        if lo >= hi then lo
        else
          let mid = (lo + hi) / 2 in
          if p.offsets.(mid) > offset then first lo mid else first (mid + 1) hi
      in
      let i = first 0 n in
      i < n && p.mins.(i) < step

(* [add_sample st ~offset ~kind ~step ~time keys (k, value)] is [keys] with the
   sample added: a record's samples update the keys map, and the fold's state
   once per record. *)
let add_sample st ~offset ~kind ~step ~time keys (k, value) =
  let key =
    match Smap.find_opt k keys with
    | Some key -> key
    | None -> new_key st k kind
  in
  let s = { Metric.step; time; value } in
  let key =
    if kind = `Metric && orphan_by_plan st ~offset step then
      { key with orph = Option.map (fun c -> push c s) key.orph }
    else
      {
        key with
        acc = Acc.add key.acc s;
        col = Option.map (fun c -> push c s) key.col;
        max_step =
          (if kind = `Metric then max key.max_step step else key.max_step);
      }
  in
  Smap.add k key keys

let kind_of st k =
  Option.map (fun (key : key) -> key.kind) (Smap.find_opt k st.keys)

(* A key the run named before was checked then. *)
let known st k = Smap.mem k st.keys || Smap.mem k st.defs || File.is_key k

let declared_kind st k =
  match kind_of st k with
  | Some _ as k -> k
  | None -> if Smap.mem k st.defs then Some `Metric else None

(* A resume at [step] makes orphans of the [metrics] samples above it. *)
let cut st step =
  let needs_reread =
    Smap.exists
      (fun _ (k : key) -> k.kind = `Metric && k.max_step > step && k.col = None)
      st.keys
  in
  if needs_reread then raise Reread;
  let keys =
    Smap.map
      (fun (k : key) ->
        if k.kind <> `Metric || k.max_step <= step then k
        else
          match (k.col, k.orph) with
          | Some c, Some o ->
              let kept = ref (empty_col ())
              and orph = ref o
              and acc = ref Acc.empty in
              let max_step = ref (-1) in
              for i = 0 to c.len - 1 do
                let s = sample_at c i in
                if s.step > step then orph := push !orph s
                else (
                  kept := push !kept s;
                  acc := Acc.add !acc s;
                  max_step := max !max_step s.step)
              done;
              {
                k with
                col = Some !kept;
                orph = Some !orph;
                acc = !acc;
                max_step = !max_step;
              }
          | _ -> k)
      st.keys
  in
  let files = List.filter (fun (f : file) -> f.step <= step) st.files in
  { st with keys; files }

let fold_record st ~offset (time, r) =
  let damaged st = { st with damaged = st.damaged + 1 } in
  match (r : Record.t) with
  | Start _ -> damaged st
  | Metric { key; goal; step_metric } ->
      if declared_kind st key = Some `Reading then damaged st
      else { st with defs = Smap.add key (goal, step_metric) st.defs }
  | Metrics { step; values } ->
      if
        List.exists
          (fun (k, _) -> declared_kind st k = Some `Reading || not (known st k))
          values
      then damaged st
      else
        let keys =
          if st.scan then st.keys
          else
            List.fold_left
              (add_sample st ~offset ~kind:`Metric ~step ~time)
              st.keys values
        in
        { st with keys }
  | Reading { step; values } ->
      if
        List.exists
          (fun (k, _) -> declared_kind st k = Some `Metric || not (known st k))
          values
      then damaged st
      else
        let keys =
          if st.scan then st.keys
          else
            List.fold_left
              (add_sample st ~offset ~kind:`Reading ~step ~time)
              st.keys values
        in
        { st with keys }
  | File { key; step; path = rel } ->
      let st = { st with named = rel :: st.named } in
      if orphan_by_plan st ~offset step then st
      else
        (* The record's path joins its segments with [/] on every system. *)
        let path =
          List.fold_left Filename.concat st.run.dir
            (String.split_on_char '/' rel)
        in
        { st with files = { key; step; time; path } :: st.files }
  | Summary values ->
      {
        st with
        summary =
          List.fold_left (fun m (k, v) -> Smap.add k v m) st.summary values;
      }
  | Use { name; version } -> { st with uses = (name, version) :: st.uses }
  | Resume { step; provenance } ->
      let covered =
        match st.plan with Some p -> offset <= p.upto | None -> false
      in
      let st = if covered then st else cut st step in
      {
        st with
        resumes = { step; time; provenance } :: st.resumes;
        known = (offset, step) :: st.known;
      }
  | End s -> { st with last_end = Some (time, s) }

let fold_line st ~offset line =
  if offset = 0 then st (* the start, read by the listing *)
  else
    match Record.decode line with
    | None -> { st with damaged = st.damaged + 1 }
    | Some r -> fold_record st ~offset r

(* Reading lines *)

let chunk = 65536

(* [fold_bytes st fd ~from ~target] folds the whole lines from [from], up to the
   first that ends past [target], and is the state with the offset after the
   last line folded. A line holding a NUL byte is not whole. *)
let fold_bytes st fd ~from ~target ~size =
  (* The start of a line that runs past a chunk, kept in one buffer so that a
     line of any length costs its bytes once. *)
  let pending = Buffer.create 256 in
  let rec go st pos =
    if pos >= size then st
    else
      let s = File.pread fd ~pos ~len:(min chunk (size - pos)) in
      if s = "" then st
      else
        let rec lines st i =
          match String.index_from_opt s i '\n' with
          | None ->
              Buffer.add_substring pending s i (String.length s - i);
              `More st
          | Some j ->
              let start = pos + i - Buffer.length pending in
              let line =
                if Buffer.length pending = 0 then String.sub s i (j - i)
                else begin
                  Buffer.add_substring pending s i (j - i);
                  let l = Buffer.contents pending in
                  Buffer.clear pending;
                  l
                end
              in
              if String.contains line '\000' then `Stop st
              else
                let st = fold_line st ~offset:start line in
                let st = { st with offset = pos + j + 1 } in
                if st.offset > target then `Stop st else lines st (j + 1)
        in
        match lines st 0 with
        | `Stop st -> st
        | `More st -> go st (pos + String.length s)
  in
  (* A read that already ends past [target], as a limit within the start does,
     reads no line. *)
  if from > target then st else go st from

let plan_of known ~upto =
  let k = Array.of_list (List.rev known) in
  let n = Array.length k in
  let offsets = Array.map fst k in
  let mins = Array.make n max_int in
  for i = n - 1 downto 0 do
    mins.(i) <- min (snd k.(i)) (if i + 1 < n then mins.(i + 1) else max_int)
  done;
  { offsets; mins; upto }

let fresh ~keep (run : Listing.t) =
  {
    run;
    path = Listing.log run.dir;
    ino = 0;
    offset = 0;
    keep;
    keys = Smap.empty;
    defs = Smap.empty;
    summary = Smap.empty;
    files = [];
    named = [];
    scan = false;
    uses = [];
    resumes = [];
    known = [];
    damaged = 0;
    restarted = false;
    last_end = None;
    plan = None;
  }

let finish_status st ~held ~mtime =
  let tags, notes = (Listing.tags st.run.dir, Listing.notes st.run.dir) in
  let status, ended_at =
    Listing.status ~held
      ~writer:(Listing.writer st.run.dir)
      ~log_mtime:mtime st.last_end
  in
  {
    st with
    run =
      {
        st.run with
        status;
        ended_at;
        updated_at = mtime;
        tags;
        notes;
        read_at = Unix.gettimeofday ();
      };
  }

(* [refold st fd ~target ~size ~known] folds the log again from its start with a
   plan of the resumes [known], (offset, step) in reverse order. *)
let refold st fd ~target ~size ~known =
  let upto = match known with (o, _) :: _ -> o | [] -> 0 in
  let base =
    {
      (fresh ~keep:st.keep st.run) with
      ino = st.ino;
      plan = Some (plan_of known ~upto);
    }
  in
  { (fold_bytes base fd ~from:0 ~target ~size) with plan = None }

(* The resumes of the log from [from] to [target], after [known]. A scan keeps
   no samples and never cuts, since its plan covers every offset. *)
let collect_resumes st fd ~from ~target ~size =
  let scan =
    {
      (fresh ~keep:(Some []) st.run) with
      known = st.known;
      scan = true;
      plan = Some { offsets = [||]; mins = [||]; upto = max_int };
    }
  in
  (fold_bytes scan fd ~from ~target ~size).known

(* [run_fold] is [fold_bytes], and when a resume orphans samples of a key whose
   samples are not kept, a fold again from the start that knows every resume. It
   also says whether it folded again. *)
let run_fold st fd ~from ~target ~size =
  match fold_bytes st fd ~from ~target ~size with
  | st -> (st, false)
  | exception Reread ->
      let known = collect_resumes st fd ~from ~target ~size in
      (refold st fd ~target ~size ~known, true)

let open_log st =
  let fd = File.open_read st.path in
  let held = File.probe fd in
  (fd, held, File.fstat fd)

let start_of st fd =
  let line =
    match Listing.first_line fd with
    | None -> failwith (st.path ^ ": no whole start record")
    | Some l -> l
  in
  match Listing.decode_start line with
  | Ok (start, started_at) -> (start, started_at, String.length line + 1)
  | Error (`Unreadable why) -> failwith (st.path ^ ": " ^ why)
  | Error `Not_a_run -> failwith (st.path ^ ": not a run")

let read_into st ~following ~target =
  let fd, held, stat = open_log st in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let size = stat.Unix.st_size in
  let replaced =
    st.offset > 0
    && (stat.Unix.st_ino <> st.ino || size < st.offset
       || File.pread fd ~pos:(st.offset - 1) ~len:1 <> "\n")
  in
  let st, from =
    if replaced || st.offset = 0 then
      let start, started_at, len = start_of st fd in
      let base = fresh ~keep:st.keep { st.run with start; started_at } in
      ({ base with ino = stat.Unix.st_ino; offset = len }, len)
    else (st, st.offset)
  in
  let st, again = run_fold st fd ~from ~target ~size in
  let st = { st with restarted = following && (replaced || again) } in
  finish_status st ~held:(Some held) ~mtime:stat.Unix.st_mtime

let read ?limit ?keep (run : Listing.t) =
  let st = fresh ~keep run in
  read_into st ~following:false
    ~target:(match limit with Some l -> l | None -> max_int)

let follow ?limit st =
  let before = st.offset in
  read_into st ~following:true
    ~target:(match limit with Some l -> before + l | None -> max_int)

(* [for_resume run ~step] is [run]'s history with a resume at [step] after its
   last line, keeping no samples: what a resumed writer needs. *)
let for_resume (run : Listing.t) ~step =
  let st = read ~keep:[] run in
  try cut st step
  with Reread ->
    let fd = File.open_read st.path in
    Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
    let size = (File.fstat fd).Unix.st_size in
    let known = (size, step) :: st.known in
    let st' = refold st fd ~target:max_int ~size ~known in
    { st' with run = st.run }

(* Accessors *)

(* A key enters [keys] with its first sample. *)
(* A key whose every sample is an orphan has no statistics, and is no metric
   of the run's. *)
let metrics st =
  Smap.fold
    (fun k (key : key) acc ->
      if key.kind = `Metric && Acc.stats key.acc <> None then k :: acc
      else acc)
    st.keys []
  |> List.rev

let readings st =
  Smap.fold
    (fun k (key : key) acc -> if key.kind = `Reading then k :: acc else acc)
    st.keys []
  |> List.rev

let defs st = Smap.bindings st.defs
let goal st k = Option.bind (Smap.find_opt k st.defs) fst
let step_metric st k = Option.bind (Smap.find_opt k st.defs) snd

let kept st k name =
  if not (keeps st k) then
    invalid_arg
      (Printf.sprintf "History.%s: the history does not keep %S's samples" name
         k)

let empty_series =
  {
    Series.steps = [||];
    times = Float.Array.create 0;
    values = Float.Array.create 0;
    length = 0;
    ordered = true;
  }

let series st k =
  kept st k "series";
  match Smap.find_opt k st.keys with
  | Some { col = Some c; _ } -> col_series c
  | _ -> empty_series

let orphans st k =
  kept st k "orphans";
  match Smap.find_opt k st.keys with
  | Some { orph = Some c; _ } -> col_series c
  | _ -> empty_series

let stats st k =
  Option.bind (Smap.find_opt k st.keys) (fun (key : key) -> Acc.stats key.acc)

let summary st = Smap.bindings st.summary
let files st = List.rev st.files
let named_files st = st.named
let uses st = List.rev st.uses
let resumes st = List.rev st.resumes
let damaged st = st.damaged
let restarted st = st.restarted
let run st = st.run
let offset st = st.offset
let ended st = st.last_end <> None

let hint_keys st =
  Smap.fold
    (fun k (key : key) acc ->
      match Acc.stats key.acc with
      | None -> acc
      | Some stats ->
          (k, { Hint.kind = key.kind; goal = goal st k; stats }) :: acc)
    st.keys []
  |> List.rev

let kinds st =
  let m = Smap.map (fun (key : key) -> key.kind) st.keys in
  Smap.fold
    (fun k _ m -> if Smap.mem k m then m else Smap.add k `Metric m)
    st.defs m
  |> Smap.bindings

let accs st =
  Smap.bindings st.keys
  |> List.map (fun (k, (key : key)) -> (k, (key.kind, key.acc)))
