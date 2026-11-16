(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.amd on the machine's AMD GPU 0, skipping without one: its sums as
   nx_amd.mli states them, with its workspace and tickets kept and of their
   own; its copies' bits around a destination; reductions and scans past
   a launch's first round against nx.cpu's bits; the loops' memory; its
   kinds' bits at every float and complex dtype, and which NaN its float
   arithmetic keeps; the device memory its transforms keep and hold, whose
   laws are nx.kernel's (test/kernel/test_fft.ml); the matrix routines'
   products, exact on exact values and in their stated association;
   contractions on the matrix units against their model; and, through the
   harness, what its launches rely on: arguments up to 8 KiB, past the
   most any kernel takes, the ticket protocol's last group, and grids of
   2^24 groups along x and 65,536 along y and z. On any machine: that a
   plan's slots are those its launches name and its bytes those
   nx_amd.mli counts, every kernel's name within the bytes the count
   gives it, the submissions a device keeps against a model, and the
   matrix units' bounds on their model. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module P = Nx_kernel.Prog
module S = Nx_amd_support
module Support = Nx_kernels_support

let strf = Printf.sprintf

let answer =
  Testable.make ~pp:Nx_array_support.pp_answer ~equal:(fun a b -> a = b)

(* Integers of 32 or 64 bits *)

type int_dtype = I : ('v, 's) D.t * int -> int_dtype

let int_dtypes =
  [ I (D.Int32, 32); I (D.Uint32, 32); I (D.Int64, 64); I (D.Uint64, 64) ]

(* The low [bits] bits of [v]. *)
let low bits v = if bits = 64 then v else Int64.logand v 0xFFFF_FFFFL

(* An element's bytes, as {!Support.elements} gives them, as an integer of
   [bits] bits. *)
let integer bits s =
  if bits = 64 then String.get_int64_le s 0
  else low 32 (Int64.of_int32 (String.get_int32_le s 0))

let random_bytes st n =
  String.init n (fun _ -> Char.chr (Random.State.int st 256))

(* A fresh array of [dt] and shape [s] on the GPU, its bytes drawn from
   [st]. *)
let drawn st dt s =
  let x = A.create (S.device ()) dt (Iarray.of_array s) in
  let b = A.buffer x in
  let n = Rig.Buffer.length b in
  Rig.Buffer.blit_from_string (random_bytes st n) 0 b 0 n;
  x

(* Sums *)

(* plan.mli's most groups of a striding grid, the terms of a lane-order
   tile, and kernels.h's work-items of a group (NX_LOOP_THREADS) and most
   elements a work-item holds (NX_LOOP_MOST). *)
let stride_groups = 1024
let tile = 1024
let group_items = 256
let item_most = 8

type view = Plain | Reversed_axes | Every_other | Flipped | Broadcast

let views = [ Plain; Reversed_axes; Every_other; Flipped; Broadcast ]

let view_name = function
  | Plain -> "plain"
  | Reversed_axes -> "reversed axes"
  | Every_other -> "every other"
  | Flipped -> "flipped"
  | Broadcast -> "broadcast"

let move v s =
  let r = Array.length s in
  match v with
  | Plain -> None
  | Reversed_axes -> Some (M.Permute (Iarray.init r (fun i -> r - 1 - i)))
  | Every_other ->
      Some
        (M.Slice
           (Iarray.map
              (fun d -> { M.start = 0; count = (d + 1) / 2; step = 2 })
              (Iarray.of_array s)))
  | Flipped ->
      Some
        (M.Slice
           (Iarray.map
              (fun d -> { M.start = Int.max 0 (d - 1); count = d; step = -1 })
              (Iarray.of_array s)))
  | Broadcast -> Some (M.Broadcast (Iarray.of_array (Array.append [| 3 |] s)))

let viewed v x =
  match move v (Iarray.to_array (A.Layout.shape (A.layout x))) with
  | None -> x
  | Some m -> Option.get (A.move m x)

let shape_of x = A.Layout.shape (A.layout x)

(* [Nx_amd.reduce] of a [Monoid Sum] of [x] along every axis into each of
   [dsts]. *)
let sum ~dsts x =
  let p = P.v ~ins:[| D.Any (A.dtype x) |] [| P.In 0 |] ~outs:[| 0 |] in
  let shape = shape_of x in
  let rs = List.map (fun _ -> (Nx_kernel.Monoid Nx_kernel.Sum, 0)) dsts in
  Nx_amd.reduce p ~shape
    ~axes:(Iarray.init (Iarray.length shape) Fun.id)
    (Iarray.of_list rs)
    ~dsts:(Iarray.of_list (List.map (fun d -> A.Any d) dsts))
    [| Nx_kernel.Plain x |]

(* The sum nx_amd.mli states: the wrapping sum of [x]'s elements, read on
   the host. *)
let expected bits x =
  low bits
    (Array.fold_left
       (fun acc s -> Int64.add acc (integer bits s))
       0L
       (Support.elements (A.Any x)))

(* The elements of an array of 4 a sum stores into: element [at], of rank
   0, the others its neighbours, which it keeps. *)
let destinations = 4

let destination st dt at =
  let whole = drawn st dt [| destinations |] in
  let range = { M.start = at; count = 1; step = 1 } in
  let one = Option.get (A.move (M.Slice [| range |]) whole) in
  (whole, Option.get (A.move (M.Reshape [||]) one))

(* That [sums] sums of [x] in one call store [want] into elements [at],
   [at + 1], ... of arrays whose bytes they find unspecified, and keep
   those arrays' other elements. *)
let check_sums st ~sums ~at bits x want =
  let dt = A.dtype x in
  let place i = (at + i) mod destinations in
  let ds = List.init sums (fun i -> destination st dt (place i)) in
  let was = List.map (fun (w, _) -> Support.elements (A.Any w)) ds in
  equal answer A.Done (sum ~dsts:(List.map snd ds) x);
  List.iteri
    (fun i ((whole, _), was) ->
      let now = Support.elements (A.Any whole) in
      let others a = List.filteri (fun j _ -> j <> place i) (Array.to_list a) in
      equal ~msg:(strf "sum %d" i) int64 want (integer bits now.(place i));
      equal ~msg:(strf "around sum %d" i) (list string) (others was)
        (others now))
    (List.combine ds was)

type case = {
  dtype : int;
  shape : int array;
  view : view;
  sums : int;
  at : int;
  seed : int;
}

let pp_case ppf c =
  let (I (dt, _)) = List.nth int_dtypes c.dtype in
  Format.fprintf ppf "%s [%s] %s, %d sums from %d, seed %d" (D.name dt)
    (String.concat "; " (Array.to_list (Array.map string_of_int c.shape)))
    (view_name c.view) c.sums c.at c.seed

(* Shapes of up to 3 small axes, empty ones among them, or one axis of
   several tiles. *)
let case =
  let open Gen in
  let small =
    let* r = int_range 0 3 in
    array ~size:(constant r)
      (frequency [ (1, constant 0); (6, int_range 1 6) ])
  in
  let long = map (fun n -> [| n |]) (int_range (tile + 1) 5000) in
  with_pp pp_case
    (let+ dtype = int_range 0 3
     and+ shape = frequency [ (3, small); (1, long) ]
     and+ view = of_list views
     and+ sums = int_range 1 3
     and+ at = int_range 0 (destinations - 1)
     and+ seed = nat in
     { dtype; shape; view; sums; at; seed })

let law_sum c =
  let (I (dt, bits)) = List.nth int_dtypes c.dtype in
  let st = Random.State.make [| c.seed |] in
  let x = viewed c.view (drawn st dt c.shape) in
  let n = A.Layout.numel (A.layout x) in
  cover "no term" (n = 0);
  cover "several tiles" (n > tile);
  cover "32 bits" (bits = 32);
  cover "64 bits" (bits = 64);
  cover "several sums" (c.sums > 1);
  cover "a destination past its buffer's start" (c.at > 0);
  List.iter (fun v -> cover (view_name v) (c.view = v)) views;
  check_sums st ~sums:c.sums ~at:c.at bits x (expected bits x)

(* [x], a row of [n] elements broadcast to [rows] rows, and its sum. *)
let rows st dt bits ~rows n =
  let row = drawn st dt [| n |] in
  let x = Option.get (A.move (M.Broadcast [| rows; n |]) row) in
  (x, low bits (Int64.mul (Int64.of_int rows) (expected bits row)))

(* Sums of more terms than the most groups' work-items hold at once, so
   that each group loops over several runs. *)
let test_wide_grid () =
  List.iter
    (fun (I (dt, bits)) ->
      let st = Random.State.make [| bits |] in
      let x, want =
        rows st dt bits ~rows:((stride_groups * item_most) + 3) group_items
      in
      check_sums st ~sums:1 ~at:1 bits x want)
    int_dtypes

(* Sums of 1 tile to twice the most groups' 16 tiles each, three times
   each: the last group to take an output's ticket combines every group's
   state, and leaves the ticket zero for the next; past the most groups, a
   group takes several. *)
let test_groups () =
  List.iter
    (fun tiles ->
      for round = 1 to 3 do
        let st = Random.State.make [| tiles; round |] in
        let x, want = rows st D.Int64 64 ~rows:tiles tile in
        check_sums st ~sums:2 ~at:0 64 x want
      done)
    (let most = 16 * stride_groups in
     [ 1; 2; 4096; most - 1; most; most + 1; (2 * most) + 17 ])

let kept_workspace = 64 * 1024 * 1024
let kept_tickets = 4 * 1024 * 1024

(* Runs [f] with every sum taking a workspace and a ticket of its own. *)
let all_fresh f =
  let restore () =
    Nx_amd.bound_fresh ~workspace:kept_workspace ~tickets:kept_tickets
  in
  Fun.protect ~finally:restore (fun () ->
      Nx_amd.bound_fresh ~workspace:0 ~tickets:0;
      f ())

(* Bounds of 0 bytes make every sum take a workspace and a ticket of its
   own, which its first launch zeroes; the kept ones serve sums after. *)
let test_fresh () =
  let sums () =
    List.iter
      (fun n ->
        List.iter
          (fun (I (dt, bits)) ->
            let st = Random.State.make [| n; bits |] in
            let x = drawn st dt [| n |] in
            check_sums st ~sums:2 ~at:3 bits x (expected bits x))
          int_dtypes)
      [ 0; 1; 300; 70_000 ]
  in
  all_fresh sums;
  sums ()

(* nx_amd.mli: a call past the bytes kept takes a submission of its own
   and leaves the kept ones as they were; the same call, kept, finds the
   submission its first call kept, whatever the device kept before. *)
let test_unkept () =
  let b =
    match Support.amd () with
    | Some b -> b
    | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()
  in
  let st = Random.State.make [| 7 |] in
  let x = drawn st D.Int64 [| 1000 |] in
  let call () = check_sums st ~sums:7 ~at:0 64 x (expected 64 x) in
  let before = Nx_amd.kept_calls b.device in
  Fun.protect
    ~finally:(fun () -> Nx_amd.bound_kept Nx_amd.Plan.kept_submissions)
    (fun () ->
      Nx_amd.bound_kept 0;
      call ();
      call ());
  equal ~msg:"past the bound" int before (Nx_amd.kept_calls b.device);
  call ();
  let first = Nx_amd.kept_calls b.device in
  call ();
  equal ~msg:"within it, again" int first (Nx_amd.kept_calls b.device)

let test_bounds () =
  raises (Invalid_argument "Nx_amd.bound_fresh: workspace") (fun () ->
      Nx_amd.bound_fresh ~workspace:(-1) ~tickets:0);
  raises (Invalid_argument "Nx_amd.bound_fresh: tickets") (fun () ->
      Nx_amd.bound_fresh ~workspace:0 ~tickets:(kept_tickets + 1));
  raises (Invalid_argument "Nx_amd.bound_kept") (fun () ->
      Nx_amd.bound_kept (-1));
  raises (Invalid_argument "Nx_amd.bound_kept") (fun () ->
      Nx_amd.bound_kept (Nx_amd.Plan.kept_submissions + 1));
  raises (Invalid_argument "Plan.bound_grid: groups") (fun () ->
      Nx_amd.Plan.bound_grid ~groups:0)

(* A sum of [domain_terms] int64 terms on its own operand and destination,
   the device's workspace and tickets shared. *)
let domain_terms = 4_096
type int64s = (int64, D.int64_elt) A.t

let summing : (unit, int64s * int64s) abstract = abstract "s"

(* Two domains summing at once each get the sum one call alone gets. *)
let domains_sum_alike =
  let draw () = drawn (Random.State.make [| 5 |]) D.Int64 [| domain_terms |] in
  let want = lazy (expected 64 (draw ())) in
  let make () = (draw (), A.create (S.device ()) D.Int64 [||]) in
  let run (x, dst) =
    equal answer A.Done (sum ~dsts:[ dst ] x);
    integer 64 (Support.elements (A.Any dst)).(0)
  in
  stateful ~domains:2 ~count:4 "domains summing at once get one call's sum"
    [
      command "make" (Gen.unit @-> makes summing) (fun () -> ()) make;
      command "run"
        (summing ^-> returns int64)
        (fun () -> Lazy.force want)
        run;
    ]

(* Plans *)

module Plan = Nx_amd.Plan

(* A call of [slots] slots, the first the workspace if [workspace], the
   next the tickets if [tickets], the others host buffers, and [launches]
   launches: launch [l] names slot [s] in its refs iff [s] is not
   [unread] and [s mod launches = l] or [extra.(l).(s)]. *)
type planned = {
  slots : int;
  workspace : bool;
  tickets : bool;
  launches : int;
  unread : int option;
  extra : bool array array;
}

let pp_planned ppf c =
  let bit b = if b then "1" else "0" in
  let row r = String.concat "" (Array.to_list (Array.map bit r)) in
  Format.fprintf ppf "%d slots%s%s, %d launches, unread %s, extra [%s]"
    c.slots
    (if c.workspace then ", workspace" else "")
    (if c.tickets then ", tickets" else "")
    c.launches
    (match c.unread with None -> "none" | Some s -> string_of_int s)
    (String.concat "; " (Array.to_list (Array.map row c.extra)))

let planned =
  let open Gen in
  with_pp pp_planned
    (let* slots = int_range 1 5 in
     let* launches = int_range 1 3 in
     let+ workspace = bool
     and+ tickets = bool
     and+ unread = option (int_range 0 (slots - 1))
     and+ extra =
       array ~size:(constant launches) (array ~size:(constant slots) bool)
     in
     let tickets = tickets && slots > 1 in
     { slots; workspace; tickets; launches; unread; extra })

let host_buffers = lazy (Array.init 5 (fun _ -> Rig.Buffer.create Rig.host 4))

(* [c]'s builder, its slots named in order. *)
let builder c =
  let b = Plan.make () in
  for s = 0 to c.slots - 1 do
    let i =
      if s = 0 && c.workspace then Plan.workspace b
      else if s = 1 && c.tickets then Plan.tickets_slot b
      else Plan.slot b (Lazy.force host_buffers).(s)
    in
    equal ~msg:"slot index" int s i
  done;
  for l = 0 to c.launches - 1 do
    let names s =
      c.unread <> Some s && (s mod c.launches = l || c.extra.(l).(s))
    in
    let refs =
      List.filter names (List.init c.slots Fun.id)
      |> List.map (fun s -> { Plan.word = s; slot = s; write = true })
    in
    Plan.launch b Nx_amd.Kernels.Loop_map ~groups:(1, 1, 1)
      (Plan.words Nx_amd.Kernels.Loop_map)
      (Array.of_list refs)
  done;
  b

(* Plan.finish leaves out a slot no launch's ref names, so that a call's
   slots are those of its key's submission: each ref names, in the call,
   the slot it named in the plan, and the slots keep the plan's order. *)
let law_slots c =
  cover "a slot no launch reads" (c.unread <> None);
  cover "every slot read" (c.unread = None);
  cover "the workspace unread" (c.workspace && c.unread = Some 0);
  cover "the tickets unread" (c.tickets && c.unread = Some 1);
  let named s =
    if s = 0 && c.workspace then "workspace"
    else if s = 1 && c.tickets then "tickets"
    else strf "buffer %d" s
  in
  let buffers = Lazy.force host_buffers in
  let held : Plan.slot -> string = function
    | Workspace -> "workspace"
    | Tickets -> "tickets"
    | Array x -> (
        let is_x s = buffers.(s) == x in
        match List.find_opt is_x (List.init 5 Fun.id) with
        | Some s -> strf "buffer %d" s
        | None -> "another buffer")
    | Fresh n -> strf "fresh %d" n
    | Constant k -> strf "constant %S" k.key
  in
  let call = Plan.finish (builder c) in
  let read =
    List.filter (fun s -> c.unread <> Some s) (List.init c.slots Fun.id)
  in
  equal ~msg:"slots" (list string) (List.map named read)
    (Array.to_list (Array.map held call.slots));
  let refs f (l : Plan.launch) = Array.to_list (Array.map f l.refs) in
  Array.iter
    (fun l ->
      equal ~msg:"refs" (list string)
        (refs (fun r -> named r.Plan.word) l)
        (refs (fun r -> held call.slots.(r.Plan.slot)) l))
    call.launches

(* The workspace, the tickets and a buffer, each in no launch's refs. *)
let unread_examples =
  let unread ~slots ~kept ~launches s =
    let extra = Array.make_matrix launches slots false in
    let unread = Some s in
    { slots; workspace = kept; tickets = kept; launches; unread; extra }
  in
  [
    unread ~slots:2 ~kept:true ~launches:1 0;
    unread ~slots:3 ~kept:true ~launches:2 1;
    unread ~slots:4 ~kept:false ~launches:3 3;
  ]

(* A call of [launches], each of a kernel that reads the first [used] of
   its words and names the tickets in [refs] of them, which are fresh if
   [fresh], finished under [bound_kept] of its bytes and [slack], or under
   the library's bound for no [slack]. *)
type launched = { kernel : Nx_amd.Kernels.t; refs : int; used : int }
type counted = { launches : launched list; slack : int option; fresh : bool }

let pp_launched ppf d =
  Format.fprintf ppf "%s (%d words, %d used, %d refs)"
    (Nx_amd.Kernels.name d.kernel)
    (Nx_amd.Kernels.words d.kernel)
    d.used d.refs

let pp_counted ppf c =
  let pp_slack ppf = function
    | None -> Format.fprintf ppf "the library's bound"
    | Some s -> Format.fprintf ppf "its bytes and %d" s
  in
  Format.fprintf ppf "%d launches, %a%s: %a" (List.length c.launches) pp_slack
    c.slack
    (if c.fresh then ", fresh tickets" else "")
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_launched)
    c.launches

(* A launch of any kernel, of up to 4 refs, reading all its words or
   fewer. *)
let launched =
  let open Gen in
  let kernels = Array.to_list Nx_amd.Kernels.all in
  bind (of_list kernels) (fun kernel ->
      let words = Nx_amd.Kernels.words kernel in
      let+ refs = int_range 0 (Int.min 4 words)
      and+ unused = frequency [ (1, constant 0); (1, int_range 0 words) ] in
      { kernel; refs; used = Int.max refs (words - unused) })

let counted =
  let open Gen in
  with_pp pp_counted
    (let+ launches = list ~size:(int_range 1 300) launched
     and+ slack =
       frequency
         [
           (1, constant None);
           (1, constant (Some 0));
           (3, map (fun s -> Some s) (int_range (-64) 64));
         ]
     and+ fresh = bool in
     { launches; slack; fresh })

(* nx_amd.mli: the bytes a device counts for a call, 1 KiB and
   32 (9 + ⌈w / 2⌉ + r) a launch of [w] words and [r] refs. *)
let stated_bytes (c : Plan.t) =
  Array.fold_left
    (fun n (l : Plan.launch) ->
      let w = Nx_amd.Kernels.words l.kernel in
      n + (32 * (9 + ((w + 1) / 2) + Array.length l.refs)))
    1024 c.launches

(* Plan.finish counts a call's bytes as nx_amd.mli states, the launch that
   zeroes fresh tickets included, and keeps it iff they are at most the
   bound: 64 MiB, or less under [bound_kept]. *)
let law_kept c =
  let words d = Nx_amd.Kernels.words d.kernel in
  let any f = List.exists f c.launches in
  cover "fresh tickets" c.fresh;
  cover "kept tickets" (not c.fresh);
  cover "more than 256 launches" (List.length c.launches > 256);
  cover "an odd count of words" (any (fun d -> words d mod 2 = 1));
  cover "an even count of words" (any (fun d -> words d mod 2 = 0));
  cover "a launch of no ref" (any (fun d -> d.refs = 0));
  cover "a launch of 4 refs" (any (fun d -> d.refs = 4));
  cover "words left unread" (any (fun d -> d.used < words d));
  cover "the library's bound" (c.slack = None);
  cover "a bound of its bytes" (c.slack = Some 0);
  cover "a bound below its bytes"
    (match c.slack with Some s -> s < 0 | None -> false);
  let plan () =
    let b = Plan.make () in
    let t = lazy (Plan.tickets_slot b) in
    List.iter
      (fun d ->
        let w = Plan.words d.kernel in
        let named i =
          Plan.set w i (Plan.tickets b 1);
          { Plan.word = i; slot = Lazy.force t; write = true }
        in
        Plan.launch b d.kernel ~groups:(1, 1, 1) ~used:d.used w
          (Array.init d.refs named))
      c.launches;
    b
  in
  let finish ~bound =
    let tickets = if c.fresh then 0 else kept_tickets in
    Fun.protect
      ~finally:(fun () ->
        Nx_amd.bound_kept Plan.kept_submissions;
        Nx_amd.bound_fresh ~workspace:kept_workspace ~tickets:kept_tickets)
      (fun () ->
        Nx_amd.bound_kept bound;
        Nx_amd.bound_fresh ~workspace:kept_workspace ~tickets;
        Plan.finish (plan ()))
  in
  let call = finish ~bound:Plan.kept_submissions in
  let bytes = stated_bytes call in
  equal ~msg:"bytes" int bytes call.bytes;
  equal ~msg:"kept, 64 MiB" bool (bytes <= Plan.kept_submissions) call.kept;
  match c.slack with
  | None -> ()
  | Some s ->
      let bound = Int.max 0 (bytes + s) in
      let call = finish ~bound in
      equal ~msg:(strf "kept, %d bytes" bound) bool (bytes <= bound) call.kept

(* The kernel of the most words, and one of an odd count. *)
let widest =
  Array.fold_left
    (fun k j ->
      if Nx_amd.Kernels.words j > Nx_amd.Kernels.words k then j else k)
    Nx_amd.Kernels.all.(0) Nx_amd.Kernels.all

let odd =
  List.find
    (fun k -> Nx_amd.Kernels.words k mod 2 = 1)
    (Array.to_list Nx_amd.Kernels.all)

let kept_examples =
  let all k ~refs = { kernel = k; refs; used = Nx_amd.Kernels.words k } in
  let widest = all widest ~refs:4 and odd = all odd ~refs:0 in
  [
    { launches = List.init 256 (fun _ -> widest); slack = None; fresh = true };
    {
      launches = List.init 300 (fun i -> if i mod 2 = 0 then widest else odd);
      slack = Some (-1);
      fresh = false;
    };
    { launches = [ odd; { odd with used = 1 } ]; slack = Some 0; fresh = true };
    { launches = [ widest ]; slack = Some (-64); fresh = false };
  ]

(* Kept submissions

   Kept against a model: entries newest kept first, each with the clock at
   its last use, a find or its keep. *)

module Kept_model = struct
  type entry = {
    cell : int;
    key : int;
    value : int;
    size : int;
    mutable used : int;
  }

  type t = { count : int; mutable entries : entry list; mutable clock : int }

  let make count = { count; entries = []; clock = 0 }

  let tick m =
    let c = m.clock in
    m.clock <- c + 1;
    c

  let find m (cell, key) =
    match List.find_opt (fun e -> e.cell = cell && e.key = key) m.entries with
    | None -> None
    | Some e ->
        e.used <- tick m;
        Some e.value

  (* The least recently used of [es], if any. *)
  let lru es =
    List.fold_left
      (fun o e -> match o with Some b when b.used < e.used -> o | _ -> Some e)
      None es

  let drop m e = m.entries <- List.filter (fun x -> x != e) m.entries
  let bytes m = List.fold_left (fun n e -> n + e.size) 0 m.entries
  let count m = List.length m.entries
  let clear m = m.entries <- []

  (* The keep, and the count of entries it drops past its cell's count and
     past the bytes. *)
  let keep m (cell, most, size, key, value) =
    if size > most then (0, 0)
    else begin
      let mine = List.filter (fun e -> e.cell = cell) m.entries in
      let by_count =
        if List.length mine < m.count then 0
        else match lru mine with Some e -> drop m e; 1 | None -> 0
      in
      let rec room n =
        if bytes m + size <= most then n
        else
          match lru m.entries with
          | Some e ->
              drop m e;
              room (n + 1)
          | None -> n
      in
      let by_bytes = room 0 in
      m.entries <- { cell; key; value; size; used = tick m } :: m.entries;
      (by_count, by_bytes)
    end
end

let kept_cells = 3

let modelled = abstract "k"
let bare = abstract "k"

let keep_args =
  Gen.(
    with_pp
      (fun ppf (cell, most, size, key, value) ->
        Format.fprintf ppf "cell %d, most %d, %d bytes, key %d, value %d" cell
          most size key value)
      (let+ cell = int_range 0 (kept_cells - 1)
       and+ most =
         frequency
           [ (6, int_range 8 20); (1, int_range 0 7); (1, constant 1000) ]
       and+ size =
         frequency [ (6, int_range 1 6); (2, int_range 7 10); (1, constant 0) ]
       and+ key = int_range 0 1
       and+ value = int_range 0 99 in
       (cell, most, size, key, value)))

let find_args =
  Gen.(
    with_pp
      (fun ppf (cell, key) -> Format.fprintf ppf "cell %d, key %d" cell key)
      (pair (int_range 0 (kept_cells - 1)) (int_range 0 1)))

let kept_find t (cell, key) = Nx_amd.Kept.find_by Int.equal t cell key

let kept_keep t (cell, most, size, key, value) =
  Nx_amd.Kept.keep t cell ~most ~bytes:size key value

(* [keep] with its most bytes replaced by [most]. *)
let keep_most most (cell, _, size, key, value) = (cell, most, size, key, value)

(* kept.mli: a find answers a cell's newest entry of its key; a keep past
   its cell's count drops the cell's least recently used entry, and past
   the bytes it is given the least recently used of every cell. *)
let kept_commands =
  let module M = Kept_model in
  [
    command "make"
      (Gen.with_pp Format.pp_print_int (Gen.int_range 1 3) @-> makes modelled)
      M.make
      (fun count -> Nx_amd.Kept.make ~cells:kept_cells ~count);
    command "find"
      (find_args @-> modelled ^-> returns (option int))
      (fun a m ->
        let r = M.find m a in
        cover "found" (Option.is_some r);
        r)
      (fun a t -> kept_find t a);
    command "keep"
      (keep_args @-> modelled ^-> returns unit)
      (fun ((_, most, size, _, _) as a) m ->
        let by_count, by_bytes = M.keep m a in
        cover "past the most bytes, alone" (size > most);
        cover "dropped past the cell's count" (by_count > 0);
        cover "dropped past the bytes" (by_bytes > 0);
        cover "dropped two past the bytes" (by_bytes > 1))
      (fun a t -> kept_keep t a);
    command "count" (modelled ^-> returns int) M.count Nx_amd.Kept.count;
    command "bytes" (modelled ^-> returns int) M.bytes Nx_amd.Kept.bytes;
    command "clear" (modelled ^-> returns unit) M.clear Nx_amd.Kept.clear;
  ]

(* On two domains, keeps that drop nothing lose none of each other's. *)
let kept_roomy =
  let module M = Kept_model in
  let roomy = keep_most 1000 in
  [
    command "make" (Gen.unit @-> makes modelled)
      (fun () -> M.make 1000)
      (fun () -> Nx_amd.Kept.make ~cells:kept_cells ~count:1000);
    command "find"
      (find_args @-> modelled ^-> returns (option int))
      (fun a m -> M.find m a)
      (fun a t -> kept_find t a);
    command "keep"
      (keep_args @-> modelled ^-> returns unit)
      (fun a m -> ignore (M.keep m (roomy a)))
      (fun a t -> kept_keep t (roomy a));
    command "count" (modelled ^-> returns int) M.count Nx_amd.Kept.count;
    command "bytes" (modelled ^-> returns int) M.bytes Nx_amd.Kept.bytes;
  ]

(* On two domains, the bytes kept stay within the most every keep is
   given, 24, and the entries within their cells' count, 2. *)
let kept_bounded =
  let within what ~most () = function
    | Ok n -> at_most ~msg:what int ~than:most n
    | Error e -> raise e
  in
  [
    command "make" (Gen.unit @-> makes bare) Fun.id (fun () ->
        Nx_amd.Kept.make ~cells:kept_cells ~count:2);
    command "keep"
      (keep_args @-> bare ^-> returns unit)
      (fun _ () -> ())
      (fun a t -> kept_keep t (keep_most 24 a));
    command "find"
      (find_args @-> bare ^-> judges (option int))
      (fun _ () _ -> ())
      (fun a t -> kept_find t a);
    command "bytes" (bare ^-> judges int) (within "bytes" ~most:24)
      Nx_amd.Kept.bytes;
    command "count"
      (bare ^-> judges int)
      (within "entries" ~most:(2 * kept_cells))
      Nx_amd.Kept.count;
  ]

(* Plan.name_bytes: the count of a launch covers rig's copy of its
   kernel's name only up to so many bytes. *)
let test_names () =
  Array.iter
    (fun k ->
      let name = Nx_amd.Kernels.name k in
      at_most ~msg:name int ~than:Plan.name_bytes (String.length name))
    Nx_amd.Kernels.all

let test_kept_args () =
  raises (Invalid_argument "Kept.make: cells") (fun () ->
      Nx_amd.Kept.make ~cells:(-1) ~count:1);
  raises (Invalid_argument "Kept.make: count") (fun () ->
      Nx_amd.Kept.make ~cells:1 ~count:0);
  let t = Nx_amd.Kept.make ~cells:1 ~count:1 in
  raises (Invalid_argument "Kept.find_by: cell") (fun () ->
      Nx_amd.Kept.find_by Int.equal t 1 0);
  raises (Invalid_argument "Kept.keep: cell") (fun () ->
      Nx_amd.Kept.keep t (-1) ~most:1 ~bytes:0 0 0);
  raises (Invalid_argument "Kept.keep: bytes") (fun () ->
      Nx_amd.Kept.keep t 0 ~most:1 ~bytes:(-1) 0 0)

(* plan.mli's zero: one Zero launch over the bytes from [at] of its slot,
   which it writes; a count of bytes not a multiple of 4 raises. *)
let test_zero () =
  let module K = Nx_amd.Kernels in
  let b = Plan.make () in
  let ws = Plan.workspace b in
  Plan.zero b ~slot:ws ~at:256 1024;
  let call = Plan.finish b in
  let l = call.launches.(0) in
  let word i = Int64.to_int (Bytes.get_int64_le l.words (8 * i)) in
  equal ~msg:"launches" int 1 (Array.length call.launches);
  equal ~msg:"first byte" int 256 (word K.zero_dst);
  equal ~msg:"words zeroed" int 256 (word K.zero_count);
  equal ~msg:"refs" (list (pair int int))
    [ (K.zero_dst, ws) ]
    (Array.to_list (Array.map (fun r -> (r.Plan.word, r.slot)) l.refs));
  equal ~msg:"written" (list bool) [ true ]
    (Array.to_list (Array.map (fun r -> r.Plan.write) l.refs));
  raises (Invalid_argument "Plan.zero: bytes") (fun () ->
      Plan.zero (Plan.make ()) ~slot:0 ~at:0 6)

(* Plan.launch: a launch's used words lie among its kernel's, and hold
   its refs. *)
let test_used () =
  let k = Nx_amd.Kernels.Loop_map in
  let all = Nx_amd.Kernels.words k in
  let b = Plan.make () in
  let ws = Plan.workspace b in
  let ref word = { Plan.word; slot = ws; write = true } in
  let launch ~used refs =
    Plan.launch b k ~groups:(1, 1, 1) ~used (Plan.words k) refs
  in
  raises (Invalid_argument "Plan.launch: used") (fun () ->
      launch ~used:(all + 1) [||]);
  raises (Invalid_argument "Plan.launch: used") (fun () ->
      launch ~used:(-1) [||]);
  raises (Invalid_argument "Plan.launch: a ref past the words used")
    (fun () -> launch ~used:3 [| ref 3 |]);
  launch ~used:4 [| ref 3 |];
  launch ~used:all [| ref (all - 1) |];
  let call = Plan.finish b in
  equal ~msg:"used" (list int) [ 4; all ]
    (Array.to_list (Array.map (fun (l : Plan.launch) -> l.used) call.launches))

(* Constants *)

(* Adds to [b] a launch of Move that copies [n] int32 elements from the
   start of slot [src] to slot [dst] from its byte [at], as moves.ml plans
   a copy of one axis. *)
let copy_launch b ~dst ?(at = 0) ~src n =
  let module K = Nx_amd.Kernels in
  let w = Plan.words K.Move in
  let groups = Plan.grid K.Move n in
  Plan.set w K.move_dst at;
  Plan.set w K.move_numel n;
  Plan.set w K.move_rank 1;
  Plan.set w K.move_axis (-1);
  Plan.set w K.move_mode K.move_copy;
  Plan.set w K.move_small 1;
  Plan.set w K.move_unit 32;
  Plan.set w K.move_count 1;
  Plan.set w K.move_groups groups;
  Plan.axis w K.move_ax n;
  Plan.set w (K.move_ax + K.move_axis_sd) 1;
  Plan.set w (K.move_ax + K.move_axis_ss) 1;
  Plan.launch b K.Move ~groups:(groups, 1, 1)
    ~used:(K.move_ax + K.move_axis_words)
    w
    [|
      { Plan.word = K.move_dst; slot = dst; write = true };
      { Plan.word = K.move_src; slot = src; write = false };
    |]

(* A call that copies the constant [key] of [bytes] into [dst], a fresh
   array of int32. *)
let constant_call ~key bytes dst =
  let n = A.Layout.numel (A.layout dst) in
  let b = Plan.make () in
  let c = Plan.constant b ~key ~length:(4 * n) bytes in
  copy_launch b ~dst:(Plan.slot b (A.buffer dst)) ~src:c n;
  Plan.finish b

let constant_n = 1000

(* [constant_n] int32 elements of drawn bytes, and their bytes by
   element. *)
let constant_table seed =
  let st = Random.State.make [| seed |] in
  let module B = Bigarray in
  let t = B.Array1.init B.int8_unsigned B.c_layout (4 * constant_n) (fun _ ->
      Random.State.int st 256)
  in
  let element i = String.init 4 (fun j -> Char.chr t.{(4 * i) + j}) in
  (t, Array.init constant_n element)

(* Copies the constant [key] into a fresh array, checking that planning
   makes no bytes and no buffer, and that the copy holds them. *)
let copy_constant device ~key ~made table want =
  let dst = A.create device D.Int32 [| constant_n |] in
  let was = !made and before = Rig.allocated device in
  let bytes () = incr made; table in
  let c = constant_call ~key bytes dst in
  equal ~msg:"bytes made by the plan" int was !made;
  equal ~msg:"device bytes made by the plan" int before
    (Rig.allocated device);
  let call = Nx_amd.call device ~written:[| A.Any dst |] ~read:[||] in
  equal answer A.Done (call c);
  equal ~msg:"copied" (array string) want (Support.elements (A.Any dst))

let restore_constants () = Nx_amd.bound_constants (64 * 1024 * 1024)

(* nx_amd.mli: a call uploads its constants once the door accepts it, and
   a later call of the same constant takes the one the device kept. *)
let test_constant_kept () =
  let device = S.device () in
  restore_constants ();
  let table, want = constant_table 1 in
  let made = ref 0 in
  let key = "test kept" in
  copy_constant device ~key ~made table want;
  equal ~msg:"bytes made by the first call" int 1 !made;
  copy_constant device ~key ~made table want;
  equal ~msg:"bytes made by the second call" int 1 !made

(* nx_amd.mli: a constant whose kept buffer or made bytes are not of its
   length raises, and the device computes on. *)
let test_constant_length () =
  let device = S.device () in
  restore_constants ();
  let table, want = constant_table 3 in
  (* Two elements, 8 bytes, which a copy reads safely from a longer
     buffer were a length not checked. *)
  let dst = A.create device D.Int32 [| 2 |] in
  let call ~key bytes =
    let c = constant_call ~key bytes dst in
    Nx_amd.call device ~written:[| A.Any dst |] ~read:[||] c
  in
  let made = ref 0 in
  copy_constant device ~key:"test length" ~made table want;
  raises
    (Invalid_argument {|Nx_amd: constant "test length" of 4000 bytes, not 8|})
    (fun () -> call ~key:"test length" (fun () -> table));
  raises
    (Invalid_argument {|Nx_amd: constant "test made" of 4000 bytes, not 8|})
    (fun () -> call ~key:"test made" (fun () -> table));
  copy_constant device ~key:"test length" ~made table want

(* nx_amd.mli: a call's own buffers return once its work is done, and a
   call that raises does no work. Each call below takes a workspace of
   its own, then raises on its constant's length, so calls in a loop hold
   at most one workspace, into which the copy would have written. *)
let test_raise_frees () =
  let b =
    match Support.amd () with
    | Some b -> b
    | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()
  in
  let table, _ = constant_table 5 in
  let fresh = 1024 * 1024 and n = 2 in
  let call () =
    let p = Plan.make () in
    let ws = Plan.workspace p in
    let at = Plan.take p fresh in
    let c =
      Plan.constant p ~key:"test raise" ~length:(4 * n) (fun () -> table)
    in
    copy_launch p ~dst:ws ~at ~src:c n;
    Nx_amd.call b.device ~written:[||] ~read:[||] (Plan.finish p)
  in
  let refused () =
    raises
      (Invalid_argument {|Nx_amd: constant "test raise" of 4000 bytes, not 8|})
      (fun () -> ignore (call ()))
  in
  let restore () =
    Nx_amd.bound_fresh ~workspace:kept_workspace ~tickets:kept_tickets
  in
  Fun.protect ~finally:restore (fun () ->
      Nx_amd.bound_fresh ~workspace:0 ~tickets:0;
      let bound = Rig.allocation b.device fresh in
      Kernels_cases.loop_held ~calls:8 b ~bound refused)

(* nx_amd.mli: a device keeps the 16 constants most recently made. *)
let test_constant_count () =
  let device = S.device () in
  restore_constants ();
  let table, want = constant_table 4 in
  let made = ref 0 in
  for i = 1 to 20 do
    copy_constant device ~key:(strf "test count %d" i) ~made table want
  done;
  equal ~msg:"kept" int 16 (Nx_amd.kept_constants device)

(* nx_amd.mli: a call whose constants exceed what the device keeps uploads
   them for itself, and they return once its work is done, so calls in a
   loop hold one call's. *)
let test_constant_own () =
  let b =
    match Support.amd () with
    | Some b -> b
    | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()
  in
  let table, want = constant_table 2 in
  let made = ref 0 in
  let key = "test own" in
  let kept = Nx_amd.kept_constants b.device in
  Fun.protect ~finally:restore_constants (fun () ->
      Nx_amd.bound_constants 0;
      copy_constant b.device ~key ~made table want;
      copy_constant b.device ~key ~made table want;
      equal ~msg:"bytes made by each call" int 2 !made;
      equal ~msg:"kept" int kept (Nx_amd.kept_constants b.device);
      let dst = A.create b.device D.Int32 [| constant_n |] in
      let bytes () = table in
      Kernels_cases.loop_held ~calls:8 b ~bound:(4 * constant_n) (fun () ->
          let c = constant_call ~key bytes dst in
          let call = Nx_amd.call b.device ~written:[| A.Any dst |] ~read:[||] in
          equal answer A.Done (call c);
          Rig.Buffer.wait (A.buffer dst) Read_write))

let test_constant_bound () =
  raises (Invalid_argument "Nx_amd.bound_constants") (fun () ->
      Nx_amd.bound_constants (-1));
  raises (Invalid_argument "Nx_amd.bound_constants") (fun () ->
      Nx_amd.bound_constants ((64 * 1024 * 1024) + 1))

(* A device that keeps two constants of [constant_table]'s bytes. *)
let two_tables = 2 * 4 * constant_n

(* nx_amd.mli: a device keeps the constants most recently made, and a call
   of one it dropped uploads it again. *)
let test_constant_evicted () =
  let device = S.device () in
  let tables = Array.init 3 (fun i -> constant_table (10 + i)) in
  let made = Array.init 3 (fun _ -> ref 0) in
  let copy i =
    let table, want = tables.(i) in
    let key = strf "test evicted %d" i in
    copy_constant device ~key ~made:made.(i) table want
  in
  let counts () = Array.to_list (Array.map ( ! ) made) in
  Fun.protect ~finally:restore_constants (fun () ->
      Nx_amd.bound_constants two_tables;
      copy 0;
      copy 1;
      copy 0;
      equal ~msg:"0 kept" (list int) [ 1; 1; 0 ] (counts ());
      copy 2;
      copy 0;
      equal ~msg:"0 dropped by 2" (list int) [ 2; 1; 1 ] (counts ());
      copy 1;
      equal ~msg:"1 dropped by 0" (list int) [ 2; 2; 1 ] (counts ());
      equal ~msg:"kept" int 2 (Nx_amd.kept_constants device))

(* Constants across domains *)

(* A run's keys, [prefix] then a table's index, unique to the run, so that
   what earlier runs kept is never found. *)
let next_run = Atomic.make 0

let runs : (int list ref, string * Rig.t) abstract = abstract "r"

(* Bounds the device's constants to two tables, which a stateful test
   cannot restore: every other test of constants sets the bound it needs. *)
let start () =
  Nx_amd.bound_constants two_tables;
  (strf "test domains %d " (Atomic.fetch_and_add next_run 1), S.device ())

let domain_tables = Array.init 3 (fun i -> constant_table (20 + i))

(* Copies table [i] of the run [prefix] on [device]: the call's answer,
   whether it made the table's bytes, and whether the copy holds them. *)
let copy_table i (prefix, device) =
  let table, want = domain_tables.(i) in
  let dst = A.create device D.Int32 [| constant_n |] in
  let made = ref 0 in
  let bytes () = incr made; table in
  let c = constant_call ~key:(prefix ^ string_of_int i) bytes dst in
  let a = Nx_amd.call device ~written:[| A.Any dst |] ~read:[||] c in
  (a, (!made > 0, Support.elements (A.Any dst) = want))

(* The model: the run's tables the device keeps, newest first, two at most,
   a table twice where two domains made it at once (cache.mli). A call may
   make any table, and find only one the model keeps. *)
let judge_copy i model = function
  | Error e -> raise e
  | Ok (a, (made, right)) ->
      equal ~msg:"answer" answer A.Done a;
      equal ~msg:"the copy holds the table" bool true right;
      if made then model := List.filteri (fun j _ -> j < 2) (i :: !model)
      else
        equal ~msg:(strf "table %d found" i) (list int) !model
          (if List.mem i !model then !model else i :: !model)

(* Once the run made a table, the device keeps at most two. *)
let judge_kept model = function
  | Error e -> raise e
  | Ok n -> if !model <> [] then at_most ~msg:"kept" int ~than:2 n

let domains_constants =
  stateful ~domains:2 ~count:4 "constants across domains follow the model"
    [
      command "start" (Gen.unit @-> makes runs) (fun () -> ref []) start;
      command "copy"
        (Gen.int_range 0 2 @-> runs ^-> judges (pair answer (pair bool bool)))
        judge_copy copy_table;
      command "kept"
        (runs ^-> judges int)
        judge_kept
        (fun (_, device) -> Nx_amd.kept_constants device);
    ]

(* Plan.constant names one slot a key, and Plan.finish refuses a launch
   that writes it. *)
let test_constant_slot () =
  let module K = Nx_amd.Kernels in
  let none () = failwith "made" in
  let b = Plan.make () in
  let constant key length = Plan.constant b ~key ~length none in
  let c = constant "test slot" 8 in
  equal ~msg:"same key" int c (constant "test slot" 8);
  equal ~msg:"other key" int (c + 1) (constant "test other" 8);
  raises (Invalid_argument {|Plan.constant: "test slot"'s length|}) (fun () ->
      constant "test slot" 16);
  raises (Invalid_argument "Plan.constant: length") (fun () ->
      constant "test negative" (-1));
  let read slot = { Plan.word = K.move_src; slot; write = false } in
  let written = { Plan.word = K.move_dst; slot = c; write = true } in
  Plan.launch b K.Move ~groups:(1, 1, 1) (Plan.words K.Move)
    [| written; read (c + 1) |];
  raises (Invalid_argument (strf "Plan.finish: slot %d is written" c))
    (fun () -> Plan.finish b)

(* Grids *)

type gridded = { kernel : Nx_amd.Kernels.t; per : int option; units : int }

let pp_gridded ppf c =
  Format.fprintf ppf "%s, %d units%s" (Nx_amd.Kernels.name c.kernel) c.units
    (match c.per with None -> "" | Some p -> strf ", %d a group" p)

(* plan.mli: whether [groups] of a kernel of [threads] work-items a group
   and 16 waves a SIMD, dealt one to each of 4 shader engines, fill the
   busiest engine's 512 wave slots, or twice that, to within 4 waves. *)
let band ~threads groups =
  let waves = (threads + 31) / 32 in
  let busiest = waves * ((groups + 3) / 4) in
  List.exists (fun k -> busiest > (512 * k) - 4 && busiest <= 512 * k) [ 1; 2 ]

let in_band kernel groups =
  let module K = Nx_amd.Kernels in
  K.vgprs kernel <= 96 && band ~threads:(K.threads kernel) groups

(* Group counts near the bands of groups of [threads] and the cap. *)
let near_bands threads =
  let open Gen in
  let waves = (threads + 31) / 32 in
  let edges = [ 2048 / waves; 4096 / waves; stride_groups ] in
  let near e = int_range (e - 6) (e + 2) in
  frequency [ (3, one_of (List.map near edges)); (1, int_range 1 1100) ]

(* Unit counts whose groups lie near the kernel's bands and the cap, of
   kernels that have a band (few VGPRs) more often than not, whatever
   share of the code object they are. *)
let gridded =
  let open Gen in
  let kernels = Array.to_list Nx_amd.Kernels.all in
  let banded = List.filter (fun k -> Nx_amd.Kernels.vgprs k <= 96) kernels in
  let pick ks = one_of (List.map constant ks) in
  with_pp pp_gridded
    (let* kernel = frequency [ (3, pick banded); (1, pick kernels) ] in
     let* per =
       frequency
         [ (3, constant None); (1, map Option.some (int_range 1 512)) ]
     in
     let p = Option.value per ~default:(Nx_amd.Kernels.threads kernel) in
     let* groups = near_bands (Nx_amd.Kernels.threads kernel) in
     let+ less = int_range 0 (p - 1) in
     { kernel; per; units = Int.max 0 ((groups * p) - less) })

(* plan.mli: [g] groups for [need] are never in the band ([in_band g]),
   at most the cap, and step from [need] only across the band, down only
   where the band reaches the cap. *)
let stepped ~in_band ~need g =
  at_least ~msg:"groups" int ~than:1 g;
  at_most ~msg:"groups" int ~than:stride_groups g;
  equal ~msg:"in the band" bool false (in_band g);
  let lo = Int.min g need and hi = Int.max g need in
  for between = lo + 1 to hi - 1 do
    equal ~msg:(strf "%d groups, between" between) bool true (in_band between)
  done;
  if g < need then
    for above = need to stride_groups do
      equal ~msg:(strf "%d groups, at or above the need" above) bool true
        (in_band above)
    done

(* plan.mli: a striding grid is never in the slow band, which in_band
   states. *)
let law_grid c =
  let p = Option.value c.per ~default:(Nx_amd.Kernels.threads c.kernel) in
  let need = Int.max 1 (Int.min stride_groups ((c.units + p - 1) / p)) in
  let g = Plan.grid ?per:c.per c.kernel c.units in
  cover "a count in the band" (in_band c.kernel need);
  cover ~share:0.05 "stepped up" (g > need);
  equal ~msg:"in_band, need" bool (in_band c.kernel need)
    (Plan.in_band c.kernel need);
  equal ~msg:"in_band" bool (in_band c.kernel g) (Plan.in_band c.kernel g);
  stepped ~in_band:(in_band c.kernel) ~need g

(* A group's work-items, among them 64 and 128, whose bands reach the
   cap, and a group count near their bands. *)
let banded =
  let open Gen in
  let pp ppf (threads, need) =
    Format.fprintf ppf "%d groups of %d work-items" need threads
  in
  with_pp pp
    (let* threads =
       frequency
         [ (2, one_of [ constant 64; constant 128 ]); (1, int_range 1 1024) ]
     in
     let+ need = near_bands threads in
     (threads, Int.max 1 (Int.min stride_groups need)))

(* plan.mli: grid steps the groups of a kernel of any group size, of 16
   waves a SIMD, out of the band, through past_band. *)
let law_band (threads, need) =
  let g = Plan.past_band ~threads need in
  cover "a count in the band" (band ~threads need);
  cover ~share:0.05 "stepped up" (g > need);
  cover ~share:0.02 "stepped down" (g < need);
  stepped ~in_band:(band ~threads) ~need g

(* plan.mli's divisor: device.h's [divide], here in 63-bit integers,
   gives [n / d] for every unsigned 32-bit [n] and divisor [d]. Divisors:
   1, powers of two and their neighbours, and any; numerators: 0, the
   most, a multiple of [d] and its neighbours, and any. *)
let most32 = (1 lsl 32) - 1

let divided =
  let open Gen in
  let pp ppf (n, d) = Format.fprintf ppf "%d / %d" n d in
  with_pp pp
    (let* d =
       frequency
         [
           (1, constant 1);
           (1, constant most32);
           ( 3,
             let* k = int_range 1 31 in
             let+ off = int_range (-1) 1 in
             (1 lsl k) + off );
           (3, int_range 1 most32);
         ]
     in
     let+ n =
       frequency
         [
           (1, constant 0);
           (1, constant most32);
           ( 3,
             let* q = int_range 0 (most32 / d) in
             let+ off = int_range (-1) 1 in
             Int.max 0 (Int.min most32 ((q * d) + off)) );
           (3, int_range 0 most32);
         ]
     in
     (n, d))

let law_divisor (n, d) =
  let w = Plan.divisor d in
  let m = Int64.to_int (Int64.logand w 0xffff_ffffL) in
  let s = Int64.to_int (Int64.shift_right_logical w 32) in
  (* ⌊n m / 2³²⌋ in 63 bits, m in two halves of 16 bits *)
  let hi = ((n * (m lsr 16)) + ((n * (m land 0xffff)) lsr 16)) lsr 16 in
  cover "a power of two" (d land (d - 1) = 0);
  cover "an odd divisor past 2^16" (d land 1 = 1 && d > 1 lsl 16);
  cover "the most numerator" (n = most32);
  equal ~msg:(strf "%d / %d" n d) int (n / d) ((hi + n) lsr s)

(* plan.mli: the counts around the band's edges. The rows of 256, 128 and
   64 work-items were measured on the R9700; those of 96 and 32 follow
   from the rule alone. *)
let test_grid_band () =
  let module K = Nx_amd.Kernels in
  let at k ns = List.map (fun n -> Plan.grid k (n * K.threads k)) ns in
  let measured = [ 252; 253; 256; 257; 508; 509; 512; 513; 768; 1024 ] in
  equal ~msg:"256 work-items, measured" (list int)
    [ 252; 257; 257; 257; 508; 513; 513; 513; 768; 1024 ]
    (at K.Zero measured);
  let past threads ns = List.map (Plan.past_band ~threads) ns in
  equal ~msg:"128 work-items, measured" (list int)
    [ 508; 513; 513; 513; 768; 1020; 1020; 1020 ]
    (past 128 [ 508; 509; 512; 513; 768; 1020; 1021; 1024 ]);
  equal ~msg:"64 work-items, measured" (list int)
    [ 505; 512; 516; 1016; 1016; 1016; 1016 ]
    (past 64 [ 505; 512; 516; 1016; 1017; 1020; 1024 ]);
  equal ~msg:"96 work-items, extrapolated" (list int) [ 676; 681; 681; 681 ]
    (past 96 [ 676; 677; 680; 681 ]);
  equal ~msg:"32 work-items, extrapolated" (list int) [ 1024 ]
    (past 32 [ 1024 ])

(* Plan.launch: a group's dynamic LDS and its kernel's static LDS fit the
   64 KiB a group of RDNA4 holds. *)
let test_shared () =
  let group_lds = 64 * 1024 in
  Array.iter
    (fun k ->
      let name = Nx_amd.Kernels.name k in
      let most = group_lds - Nx_amd.Kernels.lds k in
      equal ~msg:name int most (Plan.shared_most k);
      let launch shared =
        let b = Plan.make () in
        let ws = Plan.workspace b in
        Plan.launch b k ~groups:(1, 1, 1) ~shared (Plan.words k)
          [| { Plan.word = 0; slot = ws; write = true } |]
      in
      launch 0;
      launch most;
      raises ~msg:name (Invalid_argument "Plan.launch: shared") (fun () ->
          launch (most + 1));
      raises ~msg:name (Invalid_argument "Plan.launch: shared") (fun () ->
          launch (-1)))
    Nx_amd.Kernels.all

(* Staging *)

(* A launch of the harness's [staged] of [n] words, header included, that
   stage copies but those from [gap] to [gap_end]. *)
type staging = { n : int; gap : int; gap_end : int; seed : int }

(* staged.hip's work-items, words and header: the destination's address,
   [n], [gap] and [gap_end]. *)
let staged_threads = 64
let staged_words = 256
let staged_header = 4

let pp_staging ppf c =
  Format.fprintf ppf "%d words but %d to %d, seed %d" c.n c.gap c.gap_end
    c.seed

(* Counts near the group's work-items and their multiples, gaps empty,
   inside and at either end. *)
let staging =
  let open Gen in
  let clamp m = Int.max staged_header (Int.min staged_words m) in
  let near m = map (fun d -> clamp (m + d)) (int_range (-1) 1) in
  with_pp pp_staging
    (let* n =
       frequency
         [
           (2, int_range staged_header staged_words);
           (2, one_of (List.map near [ 64; 128; 192; 256 ]));
           (1, constant staged_header);
         ]
     in
     let* gap = int_range staged_header n in
     let+ gap_end =
       frequency [ (1, constant gap); (1, constant n); (3, int_range gap n) ]
     and+ seed = nat in
     { n; gap; gap_end; seed })

(* device.h: stage copies a launch's words in rounds of the group's
   work-items, all but a gap's. *)
let law_stage c =
  cover "a second round" (c.n - (c.gap_end - c.gap) > staged_threads);
  cover "a gap" (c.gap < c.gap_end);
  cover "a gap to the end" (c.gap < c.gap_end && c.gap_end = c.n);
  cover "a multiple of the work-items" (c.n mod staged_threads = 0);
  let st = Random.State.make [| c.seed |] in
  let bytes = 8 * staged_words in
  let dst = Rig.Buffer.create (S.device ()) bytes in
  let marker = 0x5A5A5A5A5A5A5A5AL in
  Rig.Buffer.fill dst '\x5A';
  (* Only words past the header are drawn: the count and the gap bound
     the kernel's stores to [staged_words] words of the destination, so a
     drawn one would store past it and fault the device for every process
     on it. *)
  let drawn = Array.init (staged_words - staged_header) (fun _ ->
      Random.State.bits st) in
  S.launch "staged" ~groups:(1, 1, 1) ~threads:staged_threads
    (S.A dst :: S.W c.n :: S.W c.gap :: S.W c.gap_end
     :: List.map (fun w -> S.W w) (Array.to_list drawn));
  let got = Bytes.create bytes in
  Rig.Buffer.blit_to_bytes dst 0 got 0 bytes;
  let got = Array.init staged_words (fun i -> Bytes.get_int64_le got (8 * i)) in
  let word i =
    match i with
    | 0 -> got.(0)
    | 1 -> Int64.of_int c.n
    | 2 -> Int64.of_int c.gap
    | 3 -> Int64.of_int c.gap_end
    | i -> Int64.of_int drawn.(i - staged_header)
  in
  let want i =
    if i < c.n && (i < c.gap || i >= c.gap_end) then word i else marker
  in
  equal (array int64) (Array.init staged_words want) got

(* Copies *)

type around = { dt : int; before : int; count : int; after : int; seed : int }

let pp_around ppf c =
  let (D.Any dt) = List.nth D.all c.dt in
  Format.fprintf ppf "%s %d|%d|%d seed %d" (D.name dt) c.before c.count c.after
    c.seed

let around =
  let open Gen in
  with_pp pp_around
    (let+ dt = int_range 0 (List.length D.all - 1)
     and+ before = int_range 0 9
     and+ count = int_range 0 40
     and+ after = int_range 0 9
     and+ seed = nat in
     { dt; before; count; after; seed })

(* A copy into elements [before, before + count) of an array keeps every
   other element's bits, the bits of a byte or word it shares with them
   included. *)
let law_around c =
  let (D.Any dt) = List.nth D.all c.dt in
  cover "sub-byte" (D.bits dt < 8);
  cover "sub-word" (D.bits dt < 32);
  cover "odd start" (c.before mod 2 = 1);
  let st = Random.State.make [| c.seed |] in
  let n = c.before + c.count + c.after in
  let whole = drawn st dt [| n |] in
  let x = drawn st dt [| c.count |] in
  let range = { M.start = c.before; count = c.count; step = 1 } in
  let dst = Option.get (A.move (M.Slice [| range |]) whole) in
  let was = Support.elements (A.Any whole) in
  equal answer A.Done (Support.map1 (module Nx_amd) P.Copy ~dst x);
  let want =
    Array.concat
      [
        Array.sub was 0 c.before;
        Support.elements (A.Any x);
        Array.sub was (c.before + c.count) c.after;
      ]
  in
  equal (array string) want (Support.elements (A.Any whole))

(* Loops against nx.cpu *)

let amd_backend () =
  match Support.amd () with
  | Some b -> b
  | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()

(* A reduction or a scan [r] of [x] along [axis], [label] naming the path
   of nx.amd's it reaches. *)
(* A case of [r] along [axis], and the kernel it takes, with tickets
   where [parts] says so: an output of several groups' parts. *)
type looped = {
  label : string;
  r : Nx_kernel.reduction;
  scan : bool;
  x : A.any;
  axis : int;
  takes : Nx_amd.Kernels.t;
  parts : bool option;
}

let pp_looped ppf c = Format.pp_print_string ppf c.label

let looped label ?(scan = false) ?(specials = 0) ?parts ~takes r dt s axis =
  let x = (Reduce_cases.example ~specials Sum dt s [| axis |]).x in
  { label; r; scan; x; axis; takes; parts }

(* [c]'s operand on [b] and fresh destinations, and the call into them. *)
let prepared (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x as xa) = Reduce_cases.on b c.x in
  let s = Reduce_cases.shape_of xa in
  let out =
    if c.scan then s
    else Array.of_list (List.filteri (fun i _ -> i <> c.axis) (Array.to_list s))
  in
  let like dt = A.Any (A.create b.device dt (Iarray.of_array out)) in
  let dsts =
    Iarray.of_list
      (match c.r with
      | Arg _ -> [ like (A.dtype x); like D.Int64 ]
      | Moments -> [ like (A.dtype x); like (A.dtype x) ]
      | Monoid _ | Logsumexp -> [ like (A.dtype x) ])
  in
  let p = Reduce_cases.identity (D.Any (A.dtype x)) in
  let loads = Iarray.of_list [ Nx_kernel.Plain x ] in
  let call () =
    if c.scan then K.scan p ~axis:c.axis (c.r, 0) ~dsts loads
    else
      K.reduce p ~shape:(Iarray.of_array s)
        ~axes:(Iarray.of_list [ c.axis ])
        (Iarray.of_list [ (c.r, 0) ])
        ~dsts loads
  in
  (call, dsts)

(* [c] on [b] into fresh destinations: their dtypes and bits, read where
   the host reads them. *)
let run_looped (b : Support.backend) c =
  let call, dsts = prepared b c in
  equal answer A.Done (call ());
  List.map
    (fun (A.Any d) ->
      let h = Support.host d in
      (D.Any (A.dtype h), Kernels_cases.bits_of h))
    (Iarray.to_list dsts)

(* nx.amd's launches run at most 1024 groups of 256 work-items, each
   group over rounds of units: of chunks, 16 or 1 work-items a chunk, of a
   network's slices, a group's run of at most 1024 terms a slice of more
   than 256, else a run's slots of slices of at most 256, and of a
   reduction's outputs' parts, 16 a round where an output's terms fill a
   block. A network's slice of more runs than one carries its levels from
   run to run. A Sum, Max or Min of one load at its carrier, an integer or
   float of 32 or 64 bits, or a bfloat16 or float16 reduced into float32,
   along rows or columns, is plain: a cell of 16 work-items a block, a
   group's values merged 2048 at a time; columns of at most 16 blocks; a
   plain scan's chunk a work-item. Each case reaches a round past the
   first, an output's parts merged by a group, a slice's chunk totals
   folded by a group, or a plain path, and has nx.cpu's bits, any NaN for
   a NaN. *)
let f32 = D.Any D.Float32
let f64 = D.Any D.Float64
let round_units = stride_groups * group_items
let block_round = stride_groups * 16
let part_terms = 16 * tile
let merge_chunk = 2048

let looped_cases =
  let module Kn = Nx_amd.Kernels in
  Nx_kernel.
    [
      looped "a second round of chunks of 16 work-items" ~scan:true
        ~takes:Kn.Loop_scan (Monoid Prod) f32 [| block_round + 52; 16 |] 1;
      looped "a second round of chunks of one work-item" ~scan:true
        ~takes:Kn.Loop_scan (Monoid Prod) f32 [| round_units + 232; 2 |] 1;
      looped "a second round of a network's slices" ~scan:true
        ~takes:Kn.Loop_network Logsumexp f32 [| stride_groups + 232; 300 |] 1;
      looped "a second round of a network's slots" ~scan:true
        ~takes:Kn.Loop_network Logsumexp f32
        [| (stride_groups * 512) + 232; 2 |]
        1;
      looped "a network's slices of many runs" ~scan:true
        ~takes:Kn.Loop_network Logsumexp f32 [| 3; 20000 |] 1;
      looped "a network's Moments of many runs" ~scan:true
        ~takes:Kn.Loop_network Moments f64 [| 2; 9000 |] 1;
      looped "a network's slots with specials" ~scan:true ~specials:200
        ~takes:Kn.Loop_network Logsumexp f64 [| 1000; 37 |] 1;
      looped "a network's slices of one term" ~scan:true
        ~takes:Kn.Loop_network Moments f32 [| 3000; 1 |] 1;
      looped "a second round of a network's Moments slots" ~scan:true
        ~takes:Kn.Loop_network Moments f32 [| 3000; 5 |] 1;
      looped "a network's slices along the first axis" ~scan:true
        ~takes:Kn.Loop_network Logsumexp f32 [| 700; 5 |] 0;
      looped "a second round of an Arg's outputs" ~takes:Kn.Loop_reduce
        ~parts:false (Arg Max) f32 [| block_round + 232; 130 |] 1;
      looped "an output of 3 groups' parts" ~takes:Kn.Loop_reduce ~parts:true
        (Monoid Prod) f64
        [| 3; (2 * 16384) + 1 |]
        1;
      looped "an output of more parts than groups" ~takes:Kn.Loop_reduce
        ~parts:true Logsumexp f32
        [| ((stride_groups + 6) * part_terms) - 5 |]
        0;
      looped "slices of 301 chunks, their totals folded by a group" ~scan:true
        ~takes:Kn.Loop_scan (Arg Max) f32
        [| 2; (300 * 4096) + 1 |]
        1;
      looped "columns too tall for the plain path" ~takes:Kn.Loop_reduce
        (Monoid Sum) f32
        [| (16 * tile) + 1; 33 |]
        0;
      looped "a second round of plain rows" ~specials:20 ~takes:Kn.Loop_plain
        ~parts:false (Monoid Max) f32 [| block_round + 232; 130 |] 1;
      looped "plain rows of 130 groups' values" ~specials:20
        ~takes:Kn.Loop_plain ~parts:true (Monoid Sum) f32
        [| 130 * part_terms |]
        0;
      looped "plain rows of more values than groups" ~takes:Kn.Loop_plain
        ~parts:true (Monoid Sum) f32
        [| ((stride_groups + 6) * part_terms) - 5 |]
        0;
      looped "plain rows of more values than a merge holds"
        ~takes:Kn.Loop_plain ~parts:true (Monoid Sum) f32
        [| ((merge_chunk + 3) * part_terms) - 7 |]
        0;
      looped "plain rows of one output spread over its tiles' groups"
        ~takes:Kn.Loop_plain ~parts:true (Monoid Sum) f32 [| (64 * tile) - 5 |]
        0;
      looped "plain rows of 3 outputs spread over their tiles' groups"
        ~specials:20000 ~takes:Kn.Loop_plain ~parts:true (Monoid Sum) f64
        [| 3; 50 * tile |]
        1;
      looped "plain rows of several outputs a group" ~specials:20
        ~takes:Kn.Loop_plain ~parts:false (Monoid Max) (D.Any D.Int64)
        [| 1000; 37 |] 1;
      looped "plain rows of uint32, spread" ~specials:20 ~takes:Kn.Loop_plain
        ~parts:true (Monoid Sum) (D.Any D.Uint32) [| 7; 3000 |] 1;
      looped "plain rows of bfloat16 into float32, spread" ~specials:20
        ~takes:Kn.Loop_plain ~parts:true (Monoid Sum) (D.Any D.Bfloat16)
        [| 4; 5000 |] 1;
      looped "plain rows of float16 into float32, spread" ~specials:20
        ~takes:Kn.Loop_plain ~parts:true (Monoid Min) (D.Any D.Float16)
        [| 3; 2049 |] 1;
      looped "plain columns" ~specials:20 ~takes:Kn.Loop_plain ~parts:false
        (Monoid Min) f64 [| 5000; 300 |] 0;
      looped "plain columns of int32" ~specials:20 ~takes:Kn.Loop_plain
        ~parts:false (Monoid Sum) (D.Any D.Int32)
        [| 16 * tile; 70 |]
        0;
      looped "plain columns of bfloat16 into float32" ~specials:20
        ~takes:Kn.Loop_plain ~parts:false (Monoid Max) (D.Any D.Bfloat16)
        [| 3000; 70 |] 0;
      looped "a second round of plain scans" ~scan:true ~specials:20
        ~takes:Kn.Loop_plain_scan (Monoid Sum) f32 [| round_units + 232; 2 |] 1;
      looped "a plain scan of 3 chunks a slice" ~scan:true ~specials:20
        ~takes:Kn.Loop_plain_scan (Monoid Sum) f32 [| 3; 10000 |] 1;
      looped "a plain scan of strided slices" ~scan:true ~specials:20
        ~takes:Kn.Loop_plain_scan (Monoid Max) (D.Any D.Int32) [| 5000; 7 |] 0;
      looped "a plain scan of int64" ~scan:true ~specials:20
        ~takes:Kn.Loop_plain_scan (Monoid Sum) (D.Any D.Int64) [| 2; 9000 |] 1;
      looped "a plain scan of uint64" ~scan:true ~specials:20
        ~takes:Kn.Loop_plain_scan (Monoid Min) (D.Any D.Uint64) [| 3; 4097 |] 1;
    ]

(* Network scans of 3 slices of each length around the edges of a slot
   and of a run, 256 to 1024 terms, and around 4096: Logsumexp and
   Moments at float32 and bfloat16, about two terms a slice NaNs,
   infinities or -0. *)
let network_lengths =
  [ 1; 16; 255; 256; 257; 511; 512; 513; 1023; 1024; 1025; 4095; 4096; 4097 ]
  @ [ 8191; 8192; 8193; (3 * 4096) + 1 ]

let network_cases =
  let module Kn = Nx_amd.Kernels in
  let dts = [ (f32, "float32"); (D.Any D.Bfloat16, "bfloat16") ] in
  let rs = Nx_kernel.[ (Logsumexp, "Logsumexp"); (Moments, "Moments") ] in
  List.concat_map
    (fun n ->
      List.concat_map
        (fun (r, rn) ->
          List.map
            (fun (dt, dn) ->
              looped
                (strf "%s of %s, slices of %d" rn dn n)
                ~scan:true
                ~specials:(Int.max 2 (n / 2))
                ~takes:Kn.Loop_network r dt [| 3; n |] 1)
            dts)
        rs)
    network_lengths

(* [c]'s launches take its kernel, with tickets where [c.parts] says. *)
let check_path c (launches : Plan.traced list) =
  let named =
    List.map
      (fun (l : Plan.traced) ->
        Nx_amd.Kernels.name l.kernel ^ if l.tickets then " with tickets" else "")
      launches
  in
  let took =
    List.filter_map
      (fun (l : Plan.traced) ->
        if l.kernel = c.takes then Some l.tickets else None)
      launches
  in
  let msg = String.concat ", " named in
  equal ~msg bool true (took <> []);
  Option.iter
    (fun p -> equal ~msg (list bool) [ p ] (List.sort_uniq compare took))
    c.parts

let law_looped c =
  let b = amd_backend () in
  cover c.label true;
  let want = Support.cpu.around (fun () -> run_looped Support.cpu c) in
  let got, launches = Plan.traced (fun () -> run_looped b c) in
  check_path c launches;
  List.iter2
    (fun (dt, w) (_, g) -> equal (Kernels_cases.same_values dt) w g)
    want got

(* A reduction's or scan's association is fixed by its dtypes and
   extents, never by its launches' groups, which Plan.grid's count also
   draws from the kernel's VGPRs: with at most 1 and 7 groups, each case
   has the bits it has at Plan.grid's count, which is larger for those
   past a launch's first round. *)
let bounded_grids = [ 1; 7 ]

let law_grid_free c =
  let b = amd_backend () in
  cover c.label true;
  let want = run_looped b c in
  List.iter
    (fun groups ->
      let got =
        Fun.protect
          ~finally:(fun () -> Plan.bound_grid ~groups:max_int)
          (fun () ->
            Plan.bound_grid ~groups;
            run_looped b c)
      in
      List.iter2
        (fun (_, w) (_, g) ->
          equal ~msg:(strf "%d groups" groups) (array int) w g)
        want got)
    bounded_grids

(* Float32 maps of a tape of two planes, 8 elements a work-item and 2048
   a group: of 255 and 510 groups' elements, counts in the slow band that
   Plan.grid steps past, and of one past the most groups', a second round.
   Each has nx.cpu's bits, from a grid out of the band. *)
let map_grids = [ 255 * 2048; (510 * 2048) - 7; (stride_groups * 2048) + 1 ]

let law_map_grid n =
  let b = amd_backend () in
  let needed = (n + 2047) / 2048 in
  cover (strf "%d elements" n) true;
  cover "a count in the band" (Plan.in_band Nx_amd.Kernels.Loop_map needed);
  let x = Kernels_cases.seeded D.Float32 [| n |] n in
  let (), launches =
    Plan.traced (fun () ->
        Kernels_cases.law_map1 ~kinds:[ (P.Neg, "neg") ] b (Kernels_cases.Case x))
  in
  List.iter
    (fun (l : Plan.traced) ->
      let g, _, _ = l.groups in
      if l.kernel = Nx_amd.Kernels.Loop_map then
        equal ~msg:(strf "%d groups" g) bool false
          (Plan.in_band Nx_amd.Kernels.Loop_map g))
    launches

(* Plain scans: a Sum, Max or Min of one load at its carrier, the 32- and
   64-bit integers and floats. Each slice falls into chunks of 4096 terms;
   the shapes put runs of 32 chunks across slices, a slice's last chunk
   short, a count of chunks that is no multiple of 32, rows that start off
   a line (an offset of [k] elements a row), terms strided (along axis 0)
   and a strided destination, and up to 262,147 slices. Each term is drawn
   whole, so a chunk's results stored at another chunk's place, or a
   chunk scanned from another's carry, change bits. *)
type scan_layout = Rows | Offset of int | Axis0 | Strided_dst

type plain_scan = {
  m : Nx_kernel.monoid;
  dt : D.any;
  slices : int;
  terms : int;
  layout : scan_layout;
  seed : int;
}

let chunk = 4096

let pp_plain_scan ppf c =
  let layout =
    match c.layout with
    | Rows -> "rows"
    | Offset k -> strf "rows %d elements off" k
    | Axis0 -> "along axis 0"
    | Strided_dst -> "into a strided destination"
  in
  Format.fprintf ppf "%s of %a, %d slices of %d terms, %s (seed %d)"
    (Reduce_cases.monoid_name c.m) Reduce_cases.pp_dtype c.dt c.slices c.terms
    layout c.seed

let scan_monoids : Nx_kernel.monoid list = [ Sum; Max; Min ]

let pp_monoid ppf m =
  Format.pp_print_string ppf (Reduce_cases.monoid_name m)

let plain_dtypes =
  D.[ Any Float32; Any Float64; Any Int32; Any Uint32; Any Int64; Any Uint64 ]

(* [n] elements of [dt] on the host: floats finite and of full mantissas,
   integers of random bits. *)
let scan_terms (type v s) (dt : (v, s) D.t) n seed : A.any =
  match dt with
  | D.Float32 | D.Float64 -> A.Any (Reduce_cases.elements dt n ~specials:0 seed)
  | _ ->
      let st = Random.State.make [| seed |] in
      let b = Rig.Buffer.of_string (random_bytes st (D.bytes dt n)) in
      A.Any (A.v dt (Nx_array.Layout.contiguous [| n |]) b)

let whole e = { M.start = 0; count = e; step = 1 }
let moved m x = Option.get (A.move m x)

(* [c]'s operand on [b], made contiguous there, then viewed, so that its
   layout is the view's on every device. *)
let scan_operand (b : Support.backend) c =
  let k = match c.layout with Offset k -> k | _ -> 0 in
  let s =
    match c.layout with
    | Axis0 -> [| c.terms; c.slices |]
    | _ -> [| c.slices; c.terms + k |]
  in
  let (D.Any dt) = c.dt in
  let (A.Any x) = scan_terms dt (c.slices * (c.terms + k)) c.seed in
  let x = Support.on b (moved (M.Reshape (Iarray.of_array s)) x) in
  if k = 0 then A.Any x
  else
    let cols = { M.start = k; count = c.terms; step = 1 } in
    A.Any (moved (M.Slice [| whole c.slices; cols |]) x)

let scan_axis c = match c.layout with Axis0 -> 0 | _ -> 1

(* [c] by [b]'s kernels: its results' bits, as nx.cpu's elements give
   them. *)
let run_plain_scan (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x as xa) = scan_operand b c in
  let s = Reduce_cases.shape_of xa in
  let dst =
    match c.layout with
    | Strided_dst ->
        let w = A.create b.device (A.dtype x) [| c.slices; 2 * c.terms |] in
        let odd = { M.start = 1; count = c.terms; step = 2 } in
        moved (M.Slice [| whole c.slices; odd |]) w
    | _ -> A.create b.device (A.dtype x) (Iarray.of_array s)
  in
  let p = Reduce_cases.identity (D.Any (A.dtype x)) in
  let r =
    K.scan p ~axis:(scan_axis c) (Monoid c.m, 0) ~dsts:[| A.Any dst |]
      [| Nx_kernel.Plain x |]
  in
  equal answer A.Done r;
  Kernels_cases.bits_of (Support.host (A.copy dst))

let plain_scan_gen =
  let open Gen in
  let pp_int = Format.pp_print_int in
  let* m = of_list ~pp:pp_monoid scan_monoids in
  let* dt = of_list ~pp:Reduce_cases.pp_dtype plain_dtypes in
  let* layout =
    frequency
      [
        (3, constant Rows);
        (2, map (fun k -> Offset k) (of_list ~pp:pp_int [ 1; 3; 31 ]));
        (2, constant Axis0);
        (1, constant Strided_dst);
      ]
  in
  let chunks =
    let* k = int_range 0 40 in
    let* tail = of_list ~pp:pp_int [ 0; 1; 31; 32; 33; 4095 ] in
    let terms = Int.max 1 ((chunk * k) + tail) in
    let+ slices = int_range 1 (Int.max 1 (Int.min 12 ((1 lsl 20) / terms))) in
    (slices, terms)
  in
  let many =
    let* slices = of_list ~pp:pp_int [ 65543; 131077; 262147 ] in
    let+ terms = int_range 1 3 in
    (slices, terms)
  in
  let* slices, terms = frequency [ (4, chunks); (1, many) ] in
  let+ seed = int in
  { m; dt; slices; terms; layout; seed }

(* One case of each regime the law covers, so that every seed reaches
   every label. *)
let plain_scan_examples =
  let c m dt slices terms layout = { m; dt; slices; terms; layout; seed = 3 } in
  let i32 = D.Any D.Int32 and i64 = D.Any D.Int64 in
  let u32 = D.Any D.Uint32 and u64 = D.Any D.Uint64 in
  [
    c Sum f32 3 ((11 * chunk) + 33) Rows;
    c Sum f32 1 (33 * chunk) Rows;
    c Sum f64 5 ((7 * chunk) + 1) (Offset 1);
    c Max i32 40 ((2 * chunk) + 31) Axis0;
    c Min u64 2 ((20 * chunk) + 4095) Strided_dst;
    c Sum i64 65543 2 Rows;
    c Min u32 131077 1 (Offset 3);
    c Max f64 262147 3 Axis0;
    c Sum u32 7 ((3 * chunk) + 32) (Offset 31);
  ]

let law_plain_scan c =
  let b = amd_backend () in
  let chunks = (c.terms + chunk - 1) / chunk in
  let units = c.slices * chunks in
  let (D.Any dt) = c.dt in
  cover "32 chunks across slices" (chunks mod 32 <> 0 && units > 32);
  cover "a slice's last chunk short" (chunks > 1 && c.terms mod chunk <> 0);
  cover "chunks not a multiple of 32" (units mod 32 <> 0);
  cover "65,536 slices or more" (c.slices >= 65536);
  cover "rows off a line" (match c.layout with Offset _ -> true | _ -> false);
  cover "strided terms" (c.layout = Axis0);
  cover "strided terms past a chunk" (c.layout = Axis0 && chunks > 1);
  cover "a strided destination" (c.layout = Strided_dst);
  cover "64 bits" (D.bits dt = 64);
  cover "32 bits" (D.bits dt = 32);
  List.iter
    (fun m -> cover (Reduce_cases.monoid_name m) (c.m = m))
    scan_monoids;
  let want = Support.cpu.around (fun () -> run_plain_scan Support.cpu c) in
  let got, launches = Plan.traced (fun () -> run_plain_scan b c) in
  let took =
    List.exists
      (fun (l : Plan.traced) -> l.kernel = Nx_amd.Kernels.Loop_plain_scan)
      launches
  in
  equal ~msg:"takes loop_plain_scan" bool true took;
  equal (Kernels_cases.same_values c.dt) want got

(* Memory *)

(* nx_amd.mli's memory of the loops: with bounds of 0 bytes, a call's
   workspace and tickets are buffers of its own, which return to the
   device once its work is done, so calls in a loop hold one call's: the
   bytes the section states. *)
let held label c ~bound =
  test label (fun () ->
      let b = amd_backend () in
      let call, dsts = prepared b c in
      all_fresh (fun () ->
          Kernels_cases.loop_held ~calls:4 b ~bound (fun () ->
              equal answer A.Done (call ());
              Iarray.iter
                (fun (A.Any d) -> Rig.Buffer.wait (A.buffer d) Read_write)
                dsts)))

(* A reduction's [32 m] bytes an output, [m = ⌈t / 16384⌉], or 8 bytes a
   tile where fewer than 256 tiles spread, its 8 bytes an output more for
   a Logsumexp, rounded to 16, and a ticket an output; a scan's 32 bytes
   a chunk; a network's none. *)
let memory_laws =
  let module Kn = Nx_amd.Kernels in
  let ticket = 4 and part = 32 in
  [
    held "a sum of 16 groups' parts" ~bound:((part * 16) + ticket)
      (looped "" ~takes:Kn.Loop_plain (Monoid Sum) (D.Any D.Int64)
         [| 256 * tile |] 0);
    held "a sum spread over its 64 tiles' groups" ~bound:((8 * 64) + ticket)
      (looped "" ~takes:Kn.Loop_plain (Monoid Sum) (D.Any D.Int64)
         [| 64 * tile |] 0);
    held "a sum of 130 groups' parts" ~bound:((part * 130) + ticket)
      (looped "" ~takes:Kn.Loop_plain (Monoid Sum) (D.Any D.Int64)
         [| 130 * 16384 |] 0);
    held "a Logsumexp of 3 groups' parts an output"
      ~bound:((2 * part * 3) + 16 + (2 * ticket))
      (looped "" ~takes:Kn.Loop_reduce Logsumexp f32 [| 2; 3 * 16384 |] 1);
    held "a scan of 301 chunks a slice" ~bound:(part * 2 * 301)
      (looped "" ~scan:true ~takes:Kn.Loop_plain_scan (Monoid Sum) f32
         [| 2; (300 * 4096) + 1 |] 1);
    held "a Logsumexp scan of slices of many runs" ~bound:0
      (looped "" ~scan:true ~takes:Kn.Loop_network Logsumexp f32
         [| 3; 20000 |] 1);
  ]

(* Kinds *)

(* nx_amd.mli: a kind has its nx_kinds.h function's bits, nx.cpu's, but
   for a NaN's sign and payload, at every float and complex dtype: each
   kind of one and of two operands over 4096 seeded elements, and over
   specials where the dtype has them. *)
let kind_terms = 4096

let kind_dtypes =
  List.filter (fun (D.Any dt) -> D.is D.Float dt || D.is D.Complex dt) D.all

let kind_cases =
  List.concat_map
    (fun (D.Any dt) ->
      let seeded seed = Kernels_cases.seeded dt [| kind_terms |] seed in
      let specials =
        if Kernels_cases.specials dt = [] then []
        else
          [
            Kernels_cases.Pair
              ( Kernels_cases.specials_only dt kind_terms 1,
                Kernels_cases.specials_only dt kind_terms 2 );
          ]
      in
      Kernels_cases.Pair (seeded 1, seeded 2) :: specials)
    kind_dtypes

let pp_kind_case ppf (Kernels_cases.Pair (x, _)) =
  Format.pp_print_string ppf (D.name (A.dtype x))

let law_kinds (Kernels_cases.Pair (x, _) as p) =
  let b =
    match Support.amd () with
    | Some b -> b
    | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()
  in
  Kernels_cases.law_map1 b (Kernels_cases.Case x);
  Kernels_cases.law_map2 b p

(* Which NaN float32 and float64 arithmetic keeps: [Add], [Mul],
   [Maximum], [Minimum] and [Fma] of a quiet NaN of payload 1 ([q1]), a
   negative quiet one of payload 2 ([-q2]), a signalling one of payload 3
   ([s3]) and [1], each pair of them with a NaN. nx_amd.mli leaves a NaN's
   sign and payload unstated: this is what the GPU does, a baseline. *)
type nan_dtype =
  | N : (float, 's) D.t * string * int64 array * int64 -> nan_dtype

let nan_names = [| "q1"; "-q2"; "s3"; "1" |]

let nan_dtypes =
  [
    N
      ( D.Float32,
        "f32",
        [| 0x7FC0_0001L; 0xFFC0_0002L; 0x7F80_0003L; 0x3F80_0000L |],
        0x40_0000L );
    N
      ( D.Float64,
        "f64",
        [|
          0x7FF8_0000_0000_0001L;
          0xFFF8_0000_0000_0002L;
          0x7FF0_0000_0000_0003L;
          0x3FF0_0000_0000_0000L;
        |],
        0x8_0000_0000_0000L );
  ]

(* The pairs of [nan_names]' indices with a NaN, in order. *)
let nan_pairs =
  List.concat_map
    (fun i ->
      List.filter_map
        (fun j -> if i = 3 && j = 3 then None else Some (i, j))
        [ 0; 1; 2; 3 ])
    [ 0; 1; 2; 3 ]

let le bytes v =
  String.init bytes (fun i ->
      let byte = Int64.shift_right_logical v (8 * i) in
      Char.chr (Int64.to_int (Int64.logand byte 0xFFL)))

(* An array on the GPU of [dt] whose elements have the bits [vs]. *)
let of_bits dt vs =
  let s = String.concat "" (List.map (le (D.bits dt / 8)) vs) in
  let x = A.create (S.device ()) dt [| List.length vs |] in
  Rig.Buffer.blit_from_string s 0 (A.buffer x) 0 (String.length s);
  x

let probe_nans () =
  let lines =
    List.concat_map
      (fun (N (dt, name, v, quiet)) ->
        let bits = D.bits dt in
        let named r =
          match Array.find_index (Int64.equal r) v with
          | Some i -> nan_names.(i)
          | None when r = Int64.logor v.(2) quiet -> "q3"
          | None -> strf "%0*Lx" (bits / 4) r
        in
        let operand pick =
          of_bits dt (List.map (fun p -> v.(pick p)) nan_pairs)
        in
        let x = operand fst and y = operand snd in
        let one = of_bits dt (List.map (fun _ -> v.(3)) nan_pairs) in
        let results run =
          let dst = A.create (S.device ()) dt [| List.length nan_pairs |] in
          equal answer A.Done (run dst);
          Array.map
            (fun s -> named (integer bits s))
            (Support.elements (A.Any dst))
        in
        let map2 op =
          results (fun dst ->
              Support.map2 (module Nx_amd) (P.Binary op) ~dst x y)
        in
        let fma a b c =
          results (fun dst -> Support.map3 (module Nx_amd) P.Fma ~dst a b c)
        in
        let columns =
          [
            ("add", map2 Add);
            ("mul", map2 Mul);
            ("max", map2 Maximum);
            ("min", map2 Minimum);
            ("fma(x,y,1)", fma x y one);
            ("fma(x,1,y)", fma x one y);
            ("fma(1,x,y)", fma one x y);
          ]
        in
        List.mapi
          (fun k (i, j) ->
            strf "%s x=%s y=%s: %s" name nan_names.(i) nan_names.(j)
              (String.concat " "
                 (List.map (fun (c, r) -> strf "%s %s" c r.(k)) columns)))
          nan_pairs)
      nan_dtypes
  in
  expect (String.concat "\n" lines)
  @@ __POS_OF__
       {|
    f32 x=q1 y=q1: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f32 x=q1 y=-q2: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f32 x=q1 y=s3: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f32 x=q1 y=1: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f32 x=-q2 y=q1: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f32 x=-q2 y=-q2: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f32 x=-q2 y=s3: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f32 x=-q2 y=1: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f32 x=s3 y=q1: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f32 x=s3 y=-q2: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f32 x=s3 y=s3: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f32 x=s3 y=1: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f32 x=1 y=q1: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f32 x=1 y=-q2: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f32 x=1 y=s3: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f64 x=q1 y=q1: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f64 x=q1 y=-q2: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f64 x=q1 y=s3: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f64 x=q1 y=1: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f64 x=-q2 y=q1: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f64 x=-q2 y=-q2: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f64 x=-q2 y=s3: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f64 x=-q2 y=1: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f64 x=s3 y=q1: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f64 x=s3 y=-q2: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f64 x=s3 y=s3: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f64 x=s3 y=1: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
    f64 x=1 y=q1: add q1 mul q1 max q1 min q1 fma(x,y,1) q1 fma(x,1,y) q1 fma(1,x,y) q1
    f64 x=1 y=-q2: add -q2 mul -q2 max -q2 min -q2 fma(x,y,1) -q2 fma(x,1,y) -q2 fma(1,x,y) -q2
    f64 x=1 y=s3: add q3 mul q3 max s3 min s3 fma(x,y,1) q3 fma(x,1,y) q3 fma(1,x,y) q3
  |}

(* The harness *)

let read64 b =
  let s = Bytes.create 8 in
  Rig.Buffer.blit_to_bytes b 0 s 0 8;
  Bytes.get_int64_le s 0

let zeroed n =
  let b = Rig.Buffer.create (S.device ()) n in
  Rig.Buffer.fill b '\000';
  b

(* Each args kernel stores the sum of its words past the first. The
   largest holds every kernel's words. *)
let test_arguments () =
  let words k = Nx_amd.Kernels.words k in
  let most = Array.fold_left (fun n k -> Int.max n (words k)) 0 in
  at_most ~msg:"a kernel's words" int ~than:1024 (most Nx_amd.Kernels.all);
  List.iter
    (fun (kernel, n) ->
      let st = Random.State.make [| n |] in
      let ws = List.init (n - 1) (fun _ -> Random.State.bits st) in
      let out = zeroed 8 in
      S.launch kernel ~groups:(1, 1, 1) ~threads:32
        (S.A out :: List.map (fun w -> S.W w) ws);
      let add a w = Int64.add a (Int64.of_int w) in
      let want = List.fold_left add 0L ws in
      equal ~msg:kernel int64 want (read64 out))
    [
      ("args_256", 32); ("args_1k", 128); ("args_4k", 512); ("args_8k", 1024);
    ]

(* Of [n] groups that finish out of order, one takes the ticket last, sees
   every group's part, and leaves the ticket zero. *)
let test_tickets () =
  List.iter
    (fun (n, spin) ->
      for round = 1 to 3 do
        let parts = zeroed (8 * n)
        and ticket = zeroed 4
        and total = zeroed 8
        and lasts = zeroed 4 in
        S.launch "tickets" ~groups:(n, 1, 1) ~threads:256
          [ S.A parts; S.A ticket; S.A total; S.A lasts; S.W n; S.W spin ];
        let what = strf "%d groups, spin %d, round %d" n spin round in
        equal ~msg:what int64 (Int64.of_int (n * (n + 1) / 2)) (read64 total);
        let word b =
          let s = Bytes.create 4 in
          Rig.Buffer.blit_to_bytes b 0 s 0 4;
          Bytes.get_int32_le s 0
        in
        equal ~msg:(what ^ ": last groups") int32 1l (word lasts);
        equal ~msg:(what ^ ": ticket") int32 0l (word ticket)
      done)
    [ (1, 0); (2, 0); (255, 50); (4096, 10); (65_535, 0) ]

(* device.h's [divide] on the device, through the harness's [divided]:
   each drawn numerator's quotient by its divisor's word is [n / d]. *)
let law_device_divide pairs =
  let n = Array.length pairs in
  let num = Bytes.create (4 * n) and words = Bytes.create (8 * n) in
  Array.iteri
    (fun i (x, d) ->
      Bytes.set_int32_le num (4 * i) (Int32.of_int x);
      Bytes.set_int64_le words (8 * i) (Plan.divisor d))
    pairs;
  let upload b =
    let d = zeroed (Bytes.length b) in
    Rig.Buffer.blit_from_string (Bytes.to_string b) 0 d 0 (Bytes.length b);
    d
  in
  let quot = zeroed (4 * n) in
  S.launch "divided" ~groups:(1, 1, 1) ~threads:64
    [ S.A quot; S.A (upload num); S.A (upload words); S.W n ];
  let got = Bytes.create (4 * n) in
  Rig.Buffer.blit_to_bytes quot 0 got 0 (4 * n);
  let quotient i = Int32.to_int (Bytes.get_int32_le got (4 * i)) land most32 in
  equal (array int)
    (Array.map (fun (x, d) -> x / d) pairs)
    (Array.init n quotient)

let divided_pairs = Gen.array ~size:(Gen.constant 4096) divided

(* device.h's [split] on the device, through the harness's [split_axes]:
   along an axis of extent [n], an index [i] has coordinate [i mod n] and
   leaves [i / n], by the head {!Plan.axis} writes and by the extent
   alone. A small space's indices are below 2^32 (by [divided]'s draws);
   another's reach OCaml's [max_int], its extents past 2^32 included. *)
let split_in ~small pairs =
  let n = Array.length pairs in
  let idx = Bytes.create (8 * n) and heads = Bytes.create (16 * n) in
  Array.iteri
    (fun j (i, e) ->
      Bytes.set_int64_le idx (8 * j) (Int64.of_int i);
      Plan.axis heads (2 * j) e)
    pairs;
  let upload b =
    let d = zeroed (Bytes.length b) in
    Rig.Buffer.blit_from_string (Bytes.to_string b) 0 d 0 (Bytes.length b);
    d
  in
  let coord = zeroed (16 * n) and quot = zeroed (16 * n) in
  S.launch "split_axes" ~groups:(1, 1, 1) ~threads:64
    [
      S.A coord;
      S.A quot;
      S.A (upload idx);
      S.A (upload heads);
      S.W n;
      S.W (Bool.to_int small);
    ];
  let read b =
    let got = Bytes.create (16 * n) in
    Rig.Buffer.blit_to_bytes b 0 got 0 (16 * n);
    Array.init (2 * n) (fun j -> Int64.to_int (Bytes.get_int64_le got (8 * j)))
  in
  let twice f = Array.init (2 * n) (fun j -> f pairs.(j / 2)) in
  equal ~msg:"coordinates" (array int)
    (twice (fun (i, e) -> i mod e))
    (read coord);
  equal ~msg:"quotients" (array int) (twice (fun (i, e) -> i / e)) (read quot)

let law_split (smalls, larges) =
  cover "an extent past 2^32" (Array.exists (fun (_, e) -> e > most32) larges);
  cover "an index past 2^32" (Array.exists (fun (i, _) -> i > most32) larges);
  split_in ~small:true smalls;
  split_in ~small:false larges

let split_cases =
  let open Gen in
  let large =
    let* e =
      frequency
        [
          (1, constant 1);
          (2, int_range 2 most32);
          (2, int_range (1 lsl 32) (1 lsl 40));
        ]
    in
    let+ i =
      frequency
        [
          (1, constant 0);
          (1, int_range 0 (1 lsl 34));
          (2, int_range 0 max_int);
        ]
    in
    (i, e)
  in
  let pp ppf (smalls, larges) =
    Format.fprintf ppf "%d small and %d large splits" (Array.length smalls)
      (Array.length larges)
  in
  with_pp pp
    (pair
       (array ~size:(constant 4096) divided)
       (array ~size:(constant 4096) large))

(* Every group of grids past 65,535 groups along x, up to 2^24, and of
   65,536 along y or z adds 1 to its own word. On the R9700 through amdgpu
   a group's index along y or z is its index modulo 65,536, so a grid one
   group longer misses its last group's word and adds to the first's
   twice; along x it is not. *)
let test_grids () =
  let name (x, y, z) = strf "%dx%dx%d" x y z in
  let wrong ((x, y, z) as g) =
    let n = x * y * z in
    let words = zeroed (4 * n) in
    S.launch "grid" ~groups:(x, y, z) ~threads:32
      [ S.A words; S.W x; S.W y; S.W z ];
    let s = Bytes.create (4 * n) in
    Rig.Buffer.blit_to_bytes words 0 s 0 (4 * n);
    let wrong = ref [] in
    for i = n - 1 downto 0 do
      let c = Int32.to_int (Bytes.get_int32_le s (4 * i)) in
      if c <> 1 then wrong := (i, c) :: !wrong
    done;
    (name g, List.filteri (fun i _ -> i < 8) !wrong)
  in
  let grids =
    [
      (65_536, 1, 1);
      (65_537, 1, 1);
      (1 lsl 20, 1, 1);
      (1 lsl 24, 1, 1);
      (1, 65_536, 1);
      (1, 1, 65_536);
      (2, 65_536, 1);
      (3, 2, 65_536);
    ]
  in
  equal
    (list (pair string (list (pair int int))))
    (List.map (fun g -> (name g, [])) grids)
    (List.map wrong grids)

(* The bench's harness *)

(* Element [i] of [s], of [bytes] bytes each, as an integer, sign-extended if
   [signed]. *)
let raw s ~bytes ~signed i =
  let v = ref 0L in
  for k = bytes - 1 downto 0 do
    v :=
      Int64.logor (Int64.shift_left !v 8)
        (Int64.of_int (Char.code s.[(i * bytes) + k]))
  done;
  let shift = 64 - (8 * bytes) in
  if signed then Int64.shift_right (Int64.shift_left !v shift) shift else !v

(* Every dtype a byte or wider, by every draw: elements are a function of the
   seed and their index alone, another seed draws others, and each lies in its
   draw's range, a float neither NaN nor subnormal. *)
let test_draws () =
  let g = S.gpu () in
  let n = 3000 in
  let draws =
    [ ("uniform", S.Uniform); ("wide 4", S.Wide 4); ("small", S.Small) ]
  in
  let check (D.Any dt) (dname, d) =
    let what = D.name dt ^ " " ^ dname in
    let bytes = D.bits dt / 8 in
    let drawn k ~seed =
      let b = S.buffer g (k * bytes) in
      S.generate g b dt d ~seed;
      b
    in
    let b = drawn n ~seed:7 in
    let s = S.read b in
    let prefix = S.read (drawn (n / 3) ~seed:7) in
    let other = S.read (drawn n ~seed:8) in
    let wrong = ref [] in
    let bad i v = if List.length !wrong < 4 then wrong := (i, v) :: !wrong in
    if String.sub s 0 (String.length prefix) <> prefix then bad (-1) "prefix";
    if other = s then bad (-1) "seed";
    (match D.kind dt with
    | Float ->
        let f = D.float_format dt in
        let a =
          A.to_device Rig.host
            (A.v dt (A.Layout.contiguous (Iarray.of_list [ n ])) b)
        in
        for i = 0 to n - 1 do
          let x = A.get a (Iarray.of_list [ i ]) in
          let m = Float.abs x in
          let range =
            match d with
            | S.Wide e -> m = 0. || (m >= ldexp 1. (-e) && m <= ldexp 1. (e + 1))
            | _ -> x >= -1. && x < 1.
          in
          if Float.is_nan x || (m <> 0. && m < f.min_normal) || not range then
            bad i (Printf.sprintf "%h" x)
        done
    | Boolean ->
        for i = 0 to n - 1 do
          if raw s ~bytes ~signed:false i > 1L then bad i "not 0 or 1"
        done
    | Signed | Unsigned ->
        let signed = D.is Signed dt in
        let lo = if signed then -8L else 0L in
        if d = S.Small then
          for i = 0 to n - 1 do
            let v = raw s ~bytes ~signed i in
            if v < lo || v >= Int64.add lo 16L then bad i (Int64.to_string v)
          done
    | Complex -> ());
    (what, List.rev !wrong)
  in
  let dtypes =
    List.filter
      (fun (D.Any dt) -> D.bits dt >= 8 && not (D.is Complex dt))
      D.all
  in
  let cases = List.concat_map (fun dt -> List.map (check dt) draws) dtypes in
  equal
    (list (pair string (list (pair int string))))
    (List.map (fun (w, _) -> (w, [])) cases)
    cases

(* floor_copy writes each vector of its destination as the XOR of the inputs
   that reach it. *)
let test_floor_copy () =
  let g = S.gpu () in
  let vecs = 100_003 and short = 1000 in
  let a = S.buffer g (16 * vecs) and b = S.buffer g (16 * short) in
  S.generate g a D.Int32 S.Uniform ~seed:1;
  S.generate g b D.Int32 S.Uniform ~seed:2;
  let out = S.buffer g (16 * vecs) in
  S.run g (S.record [ S.floor_copy ~ins:[ a; b ] ~out ]);
  let sa = S.read a and sb = S.read b in
  let want =
    String.init (16 * vecs) (fun i ->
        if i < 16 * short then Char.chr (Char.code sa.[i] lxor Char.code sb.[i])
        else sa.[i])
  in
  equal string want (S.read out)

(* The XOR of the 32-bit words of [b], read 64 MiB at a time. *)
let xor_words b =
  let n = Rig.Buffer.length b and chunk = 1 lsl 26 in
  let x = ref 0l in
  let rec go first =
    if first < n then begin
      let length = Int.min chunk (n - first) in
      let s = S.read (Rig.Buffer.view b ~first ~length) in
      for i = 0 to (length / 4) - 1 do
        x := Int32.logxor !x (String.get_int32_le s (4 * i))
      done;
      go (first + length)
    end
  in
  go 0;
  !x

(* floor_read reads every vector once: its groups' words XOR to its input's,
   around a group's 1,024 vectors and past the 65,535 groups' first step. *)
let test_floor_read () =
  let g = S.gpu () in
  let out = S.buffer g (4 * 65_535) in
  let xors vecs =
    let b = S.buffer g (16 * vecs) in
    S.generate g b D.Int32 S.Uniform ~seed:vecs;
    S.write out (String.make (4 * 65_535) '\000');
    S.run g (S.record [ S.floor_read ~out g b ]);
    (vecs, (xor_words b, xor_words out))
  in
  let sizes = [ 1; 1023; 1024; 1025; (3 * 1024) + 7; (65_535 * 1024) + 5 ] in
  let got = List.map xors sizes in
  equal
    (list (pair int (pair int32 int32)))
    (List.map (fun (v, (i, _)) -> (v, (i, i))) got)
    got

(* A copy of [n] int32 elements by nx.amd's map, a call of the support's, and
   its source and destination. *)
let copy_call g n =
  let x = A.create (S.device ()) D.Int32 (Iarray.of_list [ n ]) in
  S.generate g (A.buffer x) D.Int32 S.Uniform ~seed:n;
  let y = A.create (S.device ()) D.Int32 (Iarray.of_list [ n ]) in
  let p =
    P.v
      ~ins:(Iarray.of_list [ D.Any D.Int32 ])
      (Iarray.of_list [ P.In 0 ])
      ~outs:(Iarray.of_list [ 0 ])
  in
  let call () =
    Nx_amd.map p
      ~dsts:(Iarray.of_list [ A.Any y ])
      (Iarray.of_list [ Nx_kernel.Plain x ])
  in
  match
    S.kernel_call ~what:"copy" ~launches:S.loop_launches [ A.Any y; A.Any x ]
      call
  with
  | Some r -> (r, x, y)
  | None -> fail "nx.amd declines a copy"

(* Calls issued behind a hold run once it is released: the destination, zeroed
   first, holds the source's bits after issue and after device_time's rounds of
   holds. *)
let test_hold () =
  let g = S.gpu () in
  let n = 1_000_003 in
  let r, x, y = copy_call g n in
  let source = S.read (A.buffer x) in
  let zero () = S.write (A.buffer y) (String.make (4 * n) '\000') in
  zero ();
  S.issue g ~count:64 r;
  equal ~msg:"after issue" string source (S.read (A.buffer y));
  zero ();
  let t = S.device_time g r ~count:1_000 in
  less ~msg:"device time" float_exact ~than:t 0.;
  equal ~msg:"after device_time" string source (S.read (A.buffer y))

(* [count] runs' device time lies within the host's time of the call that
   measures them, and above 0.8 of it for runs the GPU spends 50 ms or more on:
   the stamps count at the device's clock_hz, which an error by a factor would
   break. Of three tries, the one the host saw shortest. *)
let within_host g r ~count =
  ignore (S.device_time g r ~count);
  let try_ () =
    let t0 = Unix.gettimeofday () in
    let device = S.device_time g r ~count *. Float.of_int count in
    (Unix.gettimeofday () -. t0, device)
  in
  let host, device = List.fold_left min (try_ ()) [ try_ (); try_ () ] in
  less ~msg:"device below host" float_exact ~than:host device;
  less ~msg:"device above 0.8 host" float_exact ~than:device (0.8 *. host)

let test_device_time () =
  let g = S.gpu () in
  let cycles = int_of_float (S.clock g *. 1e6 *. 0.1) in
  let out = S.buffer g 16 in
  let r =
    S.record
      [
        S.dispatch "shader_clock" ~groups:(1, 1, 1) ~threads:32
          [ S.A out; S.W cycles ];
      ]
  in
  within_host g r ~count:1

(* The same of calls, each a submission of its own: 64 copies of 256 MiB. *)
let test_call_time () =
  let g = S.gpu () in
  let r, _, _ = copy_call g (1 lsl 26) in
  within_host g r ~count:64

(* Transforms *)

let amd () =
  match Support.amd () with
  | Some b -> b
  | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()

(* The bytes of device memory nx_amd.mli states a device keeps for later
   transforms: 64 MiB of constants, beside 64 MiB of workspace and 4 MiB
   of tickets. *)
let kept_transforms = (64 + 64 + 4) * 1024 * 1024

(* Lines of primes past 2^19, each by Bluestein's convolution, whose
   tables at complex128, a chirp of 8 MiB and a kernel of 16 MiB or more
   each, take more than the 64 MiB of constants a device keeps. *)
let large_tables = [ 524309; 524341; 524347; 524351; 524353 ]

let[@inline never] transform_line (b : Support.backend) n =
  let x = A.create b.device D.Complex128 [| n |] in
  let y = A.create b.device D.Complex128 [| n |] in
  equal answer A.Done (Nx_amd.c2c Nx_kernel.Forward ~axes:[| 0 |] ~dst:y x);
  Rig.Buffer.wait (A.buffer y) Read_write

let test_kept_tables () =
  let b = amd () in
  (* The first call loads the library's code on the device. *)
  transform_line b 64;
  let before = Support.settled b.device in
  List.iter (transform_line b) large_tables;
  at_most ~msg:"bytes kept" int ~than:kept_transforms
    (Support.settled b.device - before)

(* nx_amd.mli: a device keeps a call's submission whatever its launches,
   within 64 MiB. A transform runs along one axis at a time, so one along
   10 axes takes more than 8. *)
let test_transform_kept () =
  let b = amd () in
  let shape = Iarray.init 10 (fun _ -> 2) in
  let x = A.create b.device D.Complex64 shape in
  let y = A.create b.device D.Complex64 shape in
  let axes = Iarray.init (Iarray.length shape) Fun.id in
  let call () =
    equal answer A.Done (Nx_amd.c2c Nx_kernel.Forward ~axes ~dst:y x)
  in
  let (), launches = Plan.traced call in
  greater ~msg:"launches" int ~than:8 (List.length launches);
  Nx_amd.drop_kept b.device;
  call ();
  call ();
  equal ~msg:"kept calls" int 1 (Nx_amd.kept_calls b.device)

(* Lines of a prime past 2^20, each by Bluestein's convolution of
   [m = 2^21] points, take a workspace past the 64 MiB the device keeps:
   nx_amd.mli states at most [3m] complex points a line, [m = 4n]. *)
let test_transform_workspace () =
  let b = amd () in
  let lines = 4 and n = 1048583 in
  let x = A.create b.device D.Complex64 [| lines; n |] in
  let y = A.create b.device D.Complex64 [| lines; n |] in
  let m = 4 * n in
  Kernels_cases.loop_held b ~bound:(lines * 3 * m * 8) (fun () ->
      equal answer A.Done (Nx_amd.c2c Nx_kernel.Forward ~axes:[| 1 |] ~dst:y x);
      Rig.Buffer.wait (A.buffer y) Read_write)

(* A complex128 line of the prime 1013, by Bluestein's convolution of [m]
   points, 2n - 1 <= m <= 4n, on a device that keeps no constant: each
   call uploads its tables, which nx_amd.mli says return once the call's
   work is done. A call holds at most that constant, [9n] points and
   3 KiB, and its [3m] points of workspace, calls after the first hold no
   more, and the device keeps none of them. *)
let test_own_table () =
  let b = amd () in
  let n = 1013 in
  let m = 4 * n in
  let x = A.create b.device D.Complex128 [| n |] in
  let y = A.create b.device D.Complex128 [| n |] in
  let kept = Nx_amd.kept_constants b.device in
  Fun.protect ~finally:restore_constants (fun () ->
      Nx_amd.bound_constants 0;
      Kernels_cases.loop_held b ~bound:((((9 * n) + (3 * m)) * 16) + 3072)
        (fun () ->
          equal answer A.Done
            (Nx_amd.c2c Nx_kernel.Forward ~axes:[| 0 |] ~dst:y x);
          Rig.Buffer.wait (A.buffer y) Read_write);
      equal ~msg:"kept" int kept (Nx_amd.kept_constants b.device))
(* Products *)

(* Values whose products, and sums of them over the extents drawn below,
   are exact in every float accumulator, so that every association and
   every count of ranges gives the one exact sum. *)
let grid = [| 0.; 0.5; -0.5; 1.; 1.5; 2.; -3.; 4.; -6. |]

type acc = F32 | F64 | C64 | C128

let accs = [ F32; F64; C64; C128 ]

let acc_name = function
  | F32 -> "float32"
  | F64 -> "float64"
  | C64 -> "complex64"
  | C128 -> "complex128"

let acc_dtype = function
  | F32 -> D.Any D.Float32
  | F64 -> D.Any D.Float64
  | C64 -> D.Any D.Complex64
  | C128 -> D.Any D.Complex128

let is_complex = function C64 | C128 -> true | F32 | F64 -> false
let width = function F32 -> 4 | F64 | C64 -> 8 | C128 -> 16

let encode acc (z : Complex.t) =
  let b = Bytes.create (width acc) in
  let f32 at v = Bytes.set_int32_le b at (Int32.bits_of_float v) in
  let f64 at v = Bytes.set_int64_le b at (Int64.bits_of_float v) in
  (match acc with
  | F32 -> f32 0 z.re
  | F64 -> f64 0 z.re
  | C64 ->
      f32 0 z.re;
      f32 4 z.im
  | C128 ->
      f64 0 z.re;
      f64 8 z.im);
  Bytes.to_string b

let decode acc s : Complex.t =
  let f32 at = Int32.float_of_bits (String.get_int32_le s at) in
  let f64 at = Int64.float_of_bits (String.get_int64_le s at) in
  match acc with
  | F32 -> { re = f32 0; im = 0. }
  | F64 -> { re = f64 0; im = 0. }
  | C64 -> { re = f32 0; im = f32 4 }
  | C128 -> { re = f64 0; im = f64 8 }

(* How a matrix [batch; rows; cols] lies: its rows contiguous, its columns
   contiguous, its rows reversed and every other column of a store twice
   as wide, or one matrix broadcast along the batch. *)
type lay = Rows | Columns | Strided | Broadcast_batch

let lays = [ Rows; Columns; Strided; Broadcast_batch ]

let lay_name = function
  | Rows -> "rows"
  | Columns -> "columns"
  | Strided -> "strided"
  | Broadcast_batch -> "broadcast batch"

(* A matrix of [acc] and shape [batch; r; c] lying as [lay] on the GPU, its
   elements' parts drawn by [draw], from [grid] by default, real where [acc]
   is. *)
let matrix ?draw st acc lay (batch, r, c) =
  let (D.Any dt) = acc_dtype acc in
  let store =
    match lay with
    | Rows -> [| batch; r; c |]
    | Columns -> [| batch; c; r |]
    | Strided -> [| batch; r; 2 * c |]
    | Broadcast_batch -> [| 1; r; c |]
  in
  let x = A.create (S.device ()) dt (Iarray.of_array store) in
  let draw =
    match draw with
    | Some d -> d
    | None -> fun () -> grid.(Random.State.int st (Array.length grid))
  in
  let n = Array.fold_left ( * ) 1 store in
  let bytes =
    String.concat ""
      (List.init n (fun _ ->
           let im = if is_complex acc then draw () else 0. in
           encode acc { re = draw (); im }))
  in
  Rig.Buffer.blit_from_string bytes 0 (A.buffer x) 0 (String.length bytes);
  let moved m = A.Any (Option.get (A.move m x)) in
  match lay with
  | Rows -> A.Any x
  | Columns -> moved (M.Permute [| 0; 2; 1 |])
  | Strided ->
      moved
        (M.Slice
           [|
             { M.start = 0; count = batch; step = 1 };
             { M.start = Int.max 0 (r - 1); count = r; step = -1 };
             { M.start = 0; count = c; step = 2 };
           |])
  | Broadcast_batch -> moved (M.Broadcast [| batch; r; c |])

let values acc x = Array.map (decode acc) (Support.elements x)

type product = {
  acc : acc;
  batch : int;
  rows : int;
  cols : int;
  k : int;
  lay_a : lay;
  lay_m : lay;
  lay_y : lay;
  init : bool;
  init_first : bool;
  neg : bool;
  conj_a : bool;
  conj_b : bool;
  tri : [ `Lower | `Upper ] option;
  ranges : int option;
  workspace : bool;
  pseed : int;
}

let tri_name = function
  | None -> "none"
  | Some `Lower -> "lower"
  | Some `Upper -> "upper"

let pp_product ppf c =
  let flag name b = if b then " " ^ name else "" in
  Format.fprintf ppf
    "%s [%d; %d; %d] k %d, a %s, m %s, y %s,%s%s%s%s%s tri %s, ranges %s%s, \
     seed %d"
    (acc_name c.acc) c.batch c.rows c.cols c.k (lay_name c.lay_a)
    (lay_name c.lay_m) (lay_name c.lay_y) (flag "init" c.init)
    (flag "init_first" c.init_first)
    (flag "neg" c.neg) (flag "conj_a" c.conj_a) (flag "conj_b" c.conj_b)
    (tri_name c.tri)
    (match c.ranges with None -> "model's" | Some s -> string_of_int s)
    (flag ", workspace" c.workspace)
    c.pseed

(* Extents of no element, of one, of a few, within a tile of 64 and past
   one of 128; k of no term, of a few spans of 32 and of several. Half the
   cases ask no option, a contraction's product. *)
let product =
  let open Gen in
  let extent =
    frequency
      [
        (1, int_range 0 1);
        (1, int_range 2 8);
        (3, int_range 9 40);
        (2, int_range 65 160);
      ]
  in
  let k =
    frequency [ (1, int_range 0 1); (3, int_range 2 64); (2, int_range 65 400) ]
  in
  with_pp pp_product
    (let+ acc = of_list accs
     and+ batch = frequency [ (1, constant 0); (6, int_range 1 3) ]
     and+ rows = extent
     and+ cols = frequency [ (1, constant 1); (4, extent) ]
     and+ k
     and+ lay_a = of_list lays
     and+ lay_m = of_list lays
     and+ lay_y = of_list [ Rows; Columns; Strided ]
     and+ init = bool
     and+ init_first = bool
     and+ neg = bool
     and+ conj_a = bool
     and+ conj_b = bool
     and+ tri = of_list [ None; Some `Lower; Some `Upper ]
     and+ ranges =
       frequency [ (1, constant None); (2, map Option.some (int_range 1 8)) ]
     and+ workspace = bool
     and+ routine = bool
     and+ pseed = nat in
     let asks b = routine && b in
     let complex = is_complex acc in
     {
       acc;
       batch;
       rows;
       cols;
       k;
       lay_a;
       lay_m;
       lay_y;
       init;
       init_first = asks (init && init_first);
       neg = asks neg;
       conj_a = asks (complex && conj_a);
       conj_b = asks (complex && conj_b);
       tri = (if routine then tri else None);
       ranges = (if routine then ranges else None);
       workspace;
       pseed;
     })

(* [y <- [y +] op(a) op(m)] computed on the host, exact on [grid]'s
   values. *)
let reference c ~y ~a ~m =
  let op ~conj z = if conj then Complex.conj z else z in
  let at b i j ~rows ~cols = (((b * rows) + i) * cols) + j in
  Array.init (Array.length y) (fun e ->
      let j = e mod c.cols and i = e / c.cols mod c.rows in
      let b = e / (c.cols * c.rows) in
      let s = ref (if c.init then y.(e) else Complex.zero) in
      for u = 0 to c.k - 1 do
        let x = op ~conj:c.conj_a a.(at b i u ~rows:c.rows ~cols:c.k) in
        let x = if c.neg then Complex.neg x else x in
        let w = op ~conj:c.conj_b m.(at b u j ~rows:c.k ~cols:c.cols) in
        s := Complex.add !s (Complex.mul x w)
      done;
      !s)

(* Whether output [i, j] lies in [tri]'s triangle, the diagonal
   included. *)
let in_triangle tri i j =
  match tri with None -> true | Some `Lower -> j <= i | Some `Upper -> j >= i

let law_product c =
  let st = Random.State.make [| c.pseed |] in
  let a = matrix st c.acc c.lay_a (c.batch, c.rows, c.k)
  and m = matrix st c.acc c.lay_m (c.batch, c.k, c.cols)
  and y = matrix st c.acc c.lay_y (c.batch, c.rows, c.cols) in
  let was = values c.acc y in
  let want = reference c ~y:was ~a:(values c.acc a) ~m:(values c.acc m) in
  cover "complex" (is_complex c.acc);
  cover "no output" (c.batch * c.rows * c.cols = 0);
  cover "no term" (c.k = 0 && c.batch * c.rows * c.cols > 0);
  cover "one column" (c.cols = 1);
  cover "past one tile of 128" (c.rows > 128 || c.cols > 128);
  cover "several ranges" (c.k > 32 && Option.value c.ranges ~default:1 > 1);
  cover "init_first" c.init_first;
  cover "neg" c.neg;
  cover "conjugated" (c.conj_a || c.conj_b);
  cover "a triangle past one tile of 128"
    (c.tri <> None && (c.rows > 128 || c.cols > 128));
  cover "workspace" c.workspace;
  let plain =
    not
      (c.init_first || c.neg || c.conj_a || c.conj_b || c.tri <> None
     || c.ranges <> None)
  in
  cover "no option" plain;
  cover "no option, few rows or columns" (plain && (c.rows <= 8 || c.cols <= 8));
  List.iter (fun l -> cover ("a by " ^ lay_name l) (c.lay_a = l)) lays;
  List.iter (fun l -> cover ("m by " ^ lay_name l) (c.lay_m = l)) lays;
  equal answer A.Done
    (Nx_amd.product ~init_first:c.init_first ~neg:c.neg ~conj_a:c.conj_a
       ~conj_b:c.conj_b ?tri:c.tri ?ranges:c.ranges ~workspace:c.workspace
       ~acc:(acc_dtype c.acc) ~init:c.init ~y a m);
  let now = values c.acc y in
  (* Outside the triangle an output is its sum where its tile is computed,
     else as it was. Values compare as numbers: a zero's sign follows the
     association, which law_bits states. *)
  let wrong = ref [] in
  Array.iteri
    (fun e got ->
      let i = e / c.cols mod c.rows and j = e mod c.cols in
      let ok =
        got = want.(e) || ((not (in_triangle c.tri i j)) && got = was.(e))
      in
      if (not ok) && List.length !wrong < 8 then
        wrong :=
          strf "output %d (%d, %d): %g%+gi, want %g%+gi" e i j got.re got.im
            want.(e).re want.(e).im
          :: !wrong)
    now;
  equal (list string) [] (List.rev !wrong)

(* Bits *)

(* A part of a value of [acc] whose products float64 holds exactly: for float32
   and complex64 6 significant bits within 2^-7 and 2^6, whose sums of a few
   hundred terms float64 holds too, so that the accumulator's roundings alone
   decide the bits; for float64 and complex128 20 bits within 2^-31 and 2^30,
   whose sums round. A zero, of either sign, with probability [zeros]. *)
let part st acc ~zeros () =
  if Random.State.float st 1. < zeros then
    if Random.State.bool st then 0. else -0.
  else
    let bits, e =
      match acc with F32 | C64 -> (6, 6) | F64 | C128 -> (20, 30)
    in
    let m = (1 lsl (bits - 1)) + Random.State.int st (1 lsl (bits - 1)) in
    let x =
      Float.ldexp (float_of_int m) (Random.State.int st ((2 * e) + 1) - e - bits)
    in
    if Random.State.bool st then -.x else x

(* [x] rounded into [acc]'s parts. *)
let round_part acc x =
  match acc with
  | F32 | C64 -> Int32.float_of_bits (Int32.bits_of_float x)
  | F64 | C128 -> x

(* An output part's sum as contract.mli states it, of the terms [ts] in order,
   each a pair of factors: in [S = min s ⌈k' / 32⌉] ranges of spans of 32 terms,
   each summed by fused multiply-adds from +0, the first from [init] where
   [init_first]; the ranges' sums added in increasing k, then [init] where not
   [init_first]. Without terms, its init or +0. A float32 fused multiply-add is
   exact in float64 on {!part}'s values, then rounded once. *)
let sum_model acc ~ranges ~init_first ~init ts =
  let k' = Array.length ts in
  if k' = 0 then Option.value init ~default:0.
  else
    let fma x w c =
      match acc with
      | F64 | C128 -> Float.fma x w c
      | F32 | C64 -> round_part acc ((x *. w) +. c)
    in
    let r = (k' + 31) / 32 in
    let n = Int.max 1 (Int.min ranges r) in
    let range i =
      let lo = 32 * (i * r / n) and hi = Int.min k' (32 * ((i + 1) * r / n)) in
      let from =
        match init with Some y when i = 0 && init_first -> y | _ -> 0.
      in
      let acc = ref from in
      for t = lo to hi - 1 do
        let x, w = ts.(t) in
        acc := fma x w !acc
      done;
      !acc
    in
    let total = ref (range 0) in
    for i = 1 to n - 1 do
      total := round_part acc (!total +. range i)
    done;
    match init with
    | Some y when not init_first -> round_part acc (!total +. y)
    | _ -> !total

(* Whether output [i, j] lies outside [tri]'s triangle by more than two tiles of
   128, in a tile the triangle skips whatever the model's tile. *)
let far tri i j =
  match tri with
  | None -> false
  | Some `Lower -> j - i > 256
  | Some `Upper -> i - j > 256

type bits = {
  bacc : acc;
  bbatch : int;
  brows : int;
  bcols : int;
  bk : int;
  blay_a : lay;
  blay_m : lay;
  blay_y : lay;
  binit : bool;
  binit_first : bool;
  bneg : bool;
  bconj_a : bool;
  bconj_b : bool;
  btri : [ `Lower | `Upper ] option;
  branges : int;
  bzeros : float;
  bworkspace : bool;
  bseed : int;
}

let pp_bits ppf c =
  let flag name b = if b then " " ^ name else "" in
  Format.fprintf ppf
    "%s [%d; %d; %d] k %d, a %s, m %s, y %s,%s%s%s%s%s tri %s, ranges %d, \
     zeros %g%s, seed %d"
    (acc_name c.bacc) c.bbatch c.brows c.bcols c.bk (lay_name c.blay_a)
    (lay_name c.blay_m) (lay_name c.blay_y) (flag "init" c.binit)
    (flag "init_first" c.binit_first)
    (flag "neg" c.bneg) (flag "conj_a" c.bconj_a) (flag "conj_b" c.bconj_b)
    (tri_name c.btri) c.branges c.bzeros
    (flag ", workspace" c.bworkspace)
    c.bseed

(* Products in ranges a routine asks, of every accumulator, init first or last,
   many zeros of either sign at times; few rows of columns past two tiles of
   128, where a triangle skips tiles whatever the model's tile. *)
let bits =
  let open Gen in
  let extent =
    frequency
      [
        (1, int_range 0 1);
        (2, int_range 2 8);
        (3, int_range 9 40);
        (1, int_range 65 160);
      ]
  in
  with_pp pp_bits
    (let+ bacc = of_list accs
     and+ bbatch = int_range 1 2
     and+ shape =
       frequency
         [
           (4, pair extent extent);
           (1, pair (int_range 1 4) (int_range 257 300));
           (1, pair (int_range 257 300) (int_range 1 4));
         ]
     and+ bk =
       frequency
         [ (1, int_range 0 1); (3, int_range 2 64); (2, int_range 65 300) ]
     and+ blay_a = of_list lays
     and+ blay_m = of_list lays
     and+ blay_y = of_list [ Rows; Columns; Strided ]
     and+ binit = bool
     and+ binit_first = bool
     and+ bneg = bool
     and+ bconj_a = bool
     and+ bconj_b = bool
     and+ btri =
       frequency
         [
           (2, constant None);
           (1, constant (Some `Lower));
           (1, constant (Some `Upper));
         ]
     and+ branges = int_range 1 8
     and+ bzeros = frequency [ (3, constant 0.02); (1, constant 0.6) ]
     and+ bworkspace = bool
     and+ bseed = nat in
     let complex = is_complex bacc in
     {
       bacc;
       bbatch;
       brows = fst shape;
       bcols = snd shape;
       bk;
       blay_a;
       blay_m;
       blay_y;
       binit;
       binit_first = binit && binit_first;
       bneg;
       bconj_a = complex && bconj_a;
       bconj_b = complex && bconj_b;
       btri;
       branges;
       bzeros;
       bworkspace;
       bseed;
     })

(* A complex y the kernels write in the workspace, columns past two tiles of 128
   from its two rows: a lower triangle skips the tiles of the outputs far from
   it, which keep y's values. *)
let skipped_tiles =
  {
    bacc = C64;
    bbatch = 1;
    brows = 2;
    bcols = 300;
    bk = 3;
    blay_a = Rows;
    blay_m = Rows;
    blay_y = Columns;
    binit = false;
    binit_first = false;
    bneg = false;
    bconj_a = false;
    bconj_b = false;
    btri = Some `Lower;
    branges = 1;
    bzeros = 0.;
    bworkspace = false;
    bseed = 0;
  }

(* A product's outputs have the bits of the association contract.mli states, a
   complex one's parts summing the terms Nx_kernel numbers: an output in the
   triangle or near it its sum's, one far outside it y's value, one between
   either. *)
let law_bits c =
  let st = Random.State.make [| c.bseed |] in
  let draw = part st c.bacc ~zeros:c.bzeros in
  let a = matrix ~draw st c.bacc c.blay_a (c.bbatch, c.brows, c.bk)
  and m = matrix ~draw st c.bacc c.blay_m (c.bbatch, c.bk, c.bcols)
  and y = matrix ~draw st c.bacc c.blay_y (c.bbatch, c.brows, c.bcols) in
  let was = values c.bacc y in
  let va = values c.bacc a and vm = values c.bacc m in
  let at b i j ~cols ~rows = (((b * rows) + i) * cols) + j in
  let model e =
    let j = e mod c.bcols and i = e / c.bcols mod c.brows in
    let b = e / (c.bcols * c.brows) in
    let factors u =
      let x = va.(at b i u ~rows:c.brows ~cols:c.bk) in
      let w = vm.(at b u j ~rows:c.bk ~cols:c.bcols) in
      let x = if c.bconj_a then Complex.conj x else x in
      let x = if c.bneg then Complex.neg x else x in
      let w = if c.bconj_b then Complex.conj w else w in
      (x, w)
    in
    let sum init ts =
      sum_model c.bacc ~ranges:c.branges ~init_first:c.binit_first
        ~init:(if c.binit then Some init else None)
        (Array.concat (List.init c.bk ts))
    in
    if not (is_complex c.bacc) then
      {
        Complex.re =
          sum was.(e).re (fun u ->
              let x, w = factors u in
              [| (x.re, w.re) |]);
        im = 0.;
      }
    else
      {
        re =
          sum was.(e).re (fun u ->
              let x, w = factors u in
              [| (x.re, w.re); (x.im, -.w.im) |]);
        im =
          sum was.(e).im (fun u ->
              let x, w = factors u in
              [| (x.re, w.im); (x.im, w.re) |]);
      }
  in
  let has_far =
    c.btri <> None && c.brows > 0
    &&
    match c.btri with
    | Some `Lower -> c.bcols - 1 > 256
    | _ -> c.brows - 1 > 256
  in
  cover "complex" (is_complex c.bacc);
  cover "several ranges" (c.bk > 32 && c.branges > 1);
  cover "init first" c.binit_first;
  cover "init last" (c.binit && not c.binit_first);
  cover "no term, with init" (c.bk = 0 && c.binit);
  cover "many zeros" (c.bzeros > 0.5);
  cover "outputs a triangle skips" has_far;
  cover "workspace" c.bworkspace;
  equal answer A.Done
    (Nx_amd.product ~init_first:c.binit_first ~neg:c.bneg ~conj_a:c.bconj_a
       ~conj_b:c.bconj_b ?tri:c.btri ~ranges:c.branges ~workspace:c.bworkspace
       ~acc:(acc_dtype c.bacc) ~init:c.binit ~y a m);
  let now = values c.bacc y in
  let same (u : Complex.t) (v : Complex.t) =
    Int64.bits_of_float u.re = Int64.bits_of_float v.re
    && Int64.bits_of_float u.im = Int64.bits_of_float v.im
  in
  let wrong = ref [] in
  Array.iteri
    (fun e got ->
      let i = e / c.bcols mod c.brows and j = e mod c.bcols in
      let want = model e in
      let ok =
        if in_triangle c.btri i j then same got want
        else if far c.btri i j then same got was.(e)
        else same got want || same got was.(e)
      in
      if (not ok) && List.length !wrong < 8 then
        wrong :=
          strf "output %d (%d, %d): %h%+hi, want %h%+hi, was %h%+hi" e i j
            got.re got.im want.re want.im was.(e).re was.(e).im
          :: !wrong)
    now;
  equal (list string) [] (List.rev !wrong)

(* The matrix units *)

module W = Wmma_model

(* The harness's wmma kinds, in its numbering: float pairs into float32, then
   integers of [bits] bits, [k] terms, signed where [sa], [sb]. *)
type unit_kind =
  | Float of W.fmt * W.fmt
  | Int of { bits : int; k : int; sa : bool; sb : bool }

let unit_kinds =
  Array.of_list
    (List.map
       (fun (a, b) -> Float (a, b))
       W.
         [
           (F16, F16);
           (Bf16, Bf16);
           (E4m3, E4m3);
           (E4m3, E5m2);
           (E5m2, E4m3);
           (E5m2, E5m2);
         ]
    @ List.concat_map
        (fun (bits, k) ->
          List.map
            (fun (sa, sb) -> Int { bits; k; sa; sb })
            [ (true, true); (true, false); (false, true); (false, false) ])
        [ (8, 16); (4, 16); (4, 32) ])

let fmt_name = function
  | W.F16 -> "float16"
  | W.Bf16 -> "bfloat16"
  | W.E4m3 -> "e4m3"
  | W.E5m2 -> "e5m2"

let unit_kind_name = function
  | Float (a, b) -> fmt_name a ^ " x " ^ fmt_name b
  | Int { bits; k; sa; sb } ->
      let s b = if b then "s" else "u" in
      strf "%s%d x %s%d, k %d" (s sa) bits (s sb) bits k

(* Families of records that tell models of the units apart: sums exact in any
   order; one 1 in A against a B whose values name their slot; products of
   exponents within 2 or 8 of each other, C near them; C the negated sum;
   subnormal factors; signed zeros; NaN and infinities; and integers, C near the
   int32 wrap at times. *)
type family =
  | Exact
  | One_hot
  | Near
  | Wide
  | Cancel
  | Subnormal
  | Zeros
  | Specials
  | Ints

let family_name = function
  | Exact -> "exact"
  | One_hot -> "one-hot"
  | Near -> "near"
  | Wide -> "wide"
  | Cancel -> "cancel"
  | Subnormal -> "subnormal"
  | Zeros -> "zeros"
  | Specials -> "specials"
  | Ints -> "integers"

let families = function
  | Int _ -> [ Ints ]
  | Float (a, _) ->
      [ Exact; Near; Wide; Cancel; Subnormal; Zeros; Specials ]
      @ if W.bits a = 16 then [ One_hot ] else []

(* A record: A's rows and B's columns, [k] codes each, and C's 16 x 16 32-bit
   words. *)
type record = { a : int array array; b : int array array; c : int array array }

let bias = function W.F16 -> 15 | W.Bf16 -> 127 | W.E4m3 -> 7 | W.E5m2 -> 15
let frac = function W.F16 -> 10 | W.Bf16 -> 7 | W.E4m3 -> 3 | W.E5m2 -> 2
let f32_bits x = Int32.to_int (Int32.bits_of_float x) land 0xFFFF_FFFF

(* A finite code of [f], of exponent in [lo, hi] and at most [mbits] leading
   fraction bits, a signed zero with probability [zero], a subnormal with
   probability [sub]. *)
let code st f ?(zero = 0.) ?(sub = 0.) ?mbits (lo, hi) =
  let nb = W.bits f and fr = frac f in
  let sign = Random.State.int st 2 lsl (nb - 1) in
  if Random.State.float st 1. < zero then sign
  else if Random.State.float st 1. < sub then
    sign lor (1 + Random.State.int st ((1 lsl fr) - 1))
  else
    let top = (1 lsl (nb - 1 - fr)) - if f = W.E4m3 then 1 else 2 in
    let rec go () =
      let e =
        Int.min top
          (Int.max 1 (lo + Random.State.int st (hi - lo + 1) + bias f))
      in
      let m = Random.State.int st (1 lsl fr) in
      let m =
        match mbits with
        | Some b when b < fr -> m land (((1 lsl b) - 1) lsl (fr - b))
        | _ -> m
      in
      let c = sign lor (e lsl fr) lor m in
      if Float.is_nan (W.value f c) then go () else c
    in
    go ()

(* float32 bits of exponent in [lo, hi], +-0 with probability [zero]. *)
let f32 st ?(zero = 0.) (lo, hi) =
  let sign = Random.State.int st 2 lsl 31 in
  if Random.State.float st 1. < zero then sign
  else
    let e =
      Int.min 254 (Int.max 1 (lo + Random.State.int st (hi - lo + 1) + 127))
    in
    sign lor (e lsl 23) lor Random.State.int st (1 lsl 23)

let square f = Array.init 16 (fun i -> Array.init 16 (fun j -> f i j))

let record st kind family =
  match (kind, family) with
  | Int { bits; k; _ }, _ ->
      let codes () =
        Array.init 16 (fun _ ->
            Array.init k (fun _ -> Random.State.int st (1 lsl bits)))
      in
      let wrap = Random.State.int st 4 = 0 in
      let c =
        square (fun _ _ ->
            if wrap then 0x7FFF_FFF0 + Random.State.int st 16
            else Random.State.bits st land 0xFFFF_FFFF)
      in
      { a = codes (); b = codes (); c }
  | Float (fa, fb), Exact ->
      let codes f = square (fun _ _ -> code st f ~mbits:2 (-2, 2)) in
      let c =
        square (fun _ _ ->
            f32_bits (float_of_int (Random.State.int st 129 - 64)))
      in
      { a = codes fa; b = codes fb; c }
  | Float (fa, fb), One_hot ->
      let r = Random.State.int st 16 and kk = Random.State.int st 16 in
      let unit = if fa = W.F16 then 0x3C00 else 0x3F80 in
      let a = square (fun i k -> if i = r && k = kk then unit else 0) in
      (* B's column j at k is 2^(k - 7) (1 + j / 16), exact in either. *)
      let enc x =
        if fb = W.F16 then
          (* float16 of a value with 4 fraction bits, exponent -7 to 8. *)
          let e = snd (Float.frexp x) - 1 in
          let m = Float.to_int (Float.ldexp x (10 - e)) land 0x3FF in
          ((e + 15) lsl 10) lor m
        else f32_bits x lsr 16
      in
      let b =
        square (fun j k ->
            enc (Float.ldexp (1. +. (float_of_int j /. 16.)) (k - 7)))
      in
      { a; b; c = square (fun _ _ -> 0) }
  | Float (fa, fb), (Near | Wide | Cancel | Subnormal | Zeros) ->
      let span f = if f = W.Bf16 then 20 else 6 in
      let ea = Random.State.int st ((2 * span fa) + 1) - span fa in
      let eb = Random.State.int st ((2 * span fb) + 1) - span fb in
      let w = match family with Near | Cancel -> 2 | _ -> 8 in
      let sub = if family = Subnormal then 0.15 else 0. in
      let zero = match family with Subnormal | Zeros -> 0.3 | _ -> 0.05 in
      let a = square (fun _ _ -> code st fa ~zero ~sub (ea - w, ea + w)) in
      let b = square (fun _ _ -> code st fb ~zero ~sub (eb - w, eb + w)) in
      let neg_zero = 1 lsl (W.bits fa - 1) in
      let a =
        if family = Zeros then
          Array.map
            (Array.map (fun x ->
                 if Random.State.int st 5 < 4 then
                   Random.State.int st 2 * neg_zero
                 else x))
            a
        else a
      in
      let c =
        match family with
        | Cancel ->
            (* Products of codes 4 binades apart at most, of 22 bits: their sum
               is exact in float64. *)
            square (fun i j ->
                let s = ref 0. in
                for k = 0 to 15 do
                  s := !s +. (W.value fa a.(i).(k) *. W.value fb b.(j).(k))
                done;
                let c = f32_bits (-. !s) in
                if Random.State.bool st then c lxor Random.State.int st 4 else c)
        | Zeros -> square (fun _ _ -> Random.State.int st 2 lsl 31)
        | _ -> square (fun _ _ -> f32 st ~zero:0.1 (ea + eb - 30, ea + eb + 30))
      in
      { a; b; c }
  | Float (fa, fb), Specials ->
      let a = square (fun _ _ -> code st fa (-2, 2)) in
      let b = square (fun _ _ -> code st fb (-2, 2)) in
      a.(0).(3) <-
        (match fa with
        | W.E4m3 -> 0x7F
        | W.E5m2 -> 0x7D
        | W.F16 -> 0x7E01
        | W.Bf16 -> 0x7FC1);
      (match fb with
      | W.E4m3 -> ()
      | _ ->
          let inf =
            match fb with W.E5m2 -> 0x7C | W.F16 -> 0x7C00 | _ -> 0x7F80
          in
          b.(1).(5) <- inf;
          b.(2).(5) <- inf;
          a.(2).(5) <- 0);
      if fa = W.E4m3 then a.(3).(0) <- 0xFF;
      let c = square (fun _ _ -> f32_bits 1.) in
      c.(4).(4) <- 0x7FC0_0123;
      c.(5).(5) <- 0xFF80_0001;
      c.(6).(6) <- 0x7F80_0000;
      c.(7).(7) <- 0xFF80_0000;
      { a; b; c }
  | Float _, Ints -> invalid_arg "record: integers of a float kind"

(* The model's D of a record. *)
let model kind r =
  square (fun i j ->
      match kind with
      | Float (fa, fb) -> W.wmma fa fb r.a.(i) r.b.(j) r.c.(i).(j)
      | Int { bits; sa; sb; _ } ->
          W.iwmma ~bits ~sa ~sb r.a.(i) r.b.(j) r.c.(i).(j))

let operand_bits = function Float (f, _) -> W.bits f | Int { bits; _ } -> bits
let operand_k = function Float _ -> 16 | Int { k; _ } -> k

(* The harness's D of records of kind [n], one group each. *)
let on_units n records =
  let kind = unit_kinds.(n) in
  let bits = operand_bits kind and k = operand_k kind in
  let words ws =
    let b = Buffer.create (4 * Array.length ws) in
    Array.iter (fun w -> Buffer.add_int32_le b (Int32.of_int w)) ws;
    Buffer.contents b
  in
  let buffer f =
    let s = String.concat "" (List.map (fun r -> words (f r)) records) in
    let b = Rig.Buffer.create (S.device ()) (String.length s) in
    Rig.Buffer.blit_from_string s 0 b 0 (String.length s);
    b
  in
  let kinds = buffer (fun _ -> [| n |]) in
  let a = buffer (fun r -> W.image ~bits ~k r.a) in
  let b = buffer (fun r -> W.image ~bits ~k r.b) in
  let c = buffer (fun r -> W.c_image r.c) in
  let count = List.length records in
  let d = zeroed (1024 * count) in
  S.launch "wmma" ~groups:(count, 1, 1) ~threads:32
    [ S.A kinds; S.A a; S.A b; S.A c; S.A d ];
  let s = Bytes.create (1024 * count) in
  Rig.Buffer.blit_to_bytes d 0 s 0 (1024 * count);
  List.init count (fun g ->
      W.c_matrix
        (Array.init 256 (fun w ->
             Int32.to_int (Bytes.get_int32_le s ((1024 * g) + (4 * w)))
             land 0xFFFF_FFFF)))

type unit_case = { kind : int; family : family; seed : int }

let pp_unit_case ppf c =
  Format.fprintf ppf "%s, %s, seed %d"
    (unit_kind_name unit_kinds.(c.kind))
    (family_name c.family) c.seed

let unit_case =
  let open Gen in
  with_pp pp_unit_case
    (bind
       (int_range 0 (Array.length unit_kinds - 1))
       (fun kind ->
         (* One-hot records exist for 16-bit floats alone, 2 kinds of 18: half
            their cases take it, so that every run reaches it. *)
         let fs = families unit_kinds.(kind) in
         let weight f = if f = One_hot then List.length fs - 1 else 1 in
         let+ family = frequency (List.map (fun f -> (weight f, constant f)) fs)
         and+ seed = nat in
         { kind; family; seed }))

let records_a_case = 8

(* Each output of the GPU's matrix instruction is the model's, bit for bit, on
   records that tell its candidates apart: the order of its steps and pairs, its
   truncations, its grid, its NaNs. *)
let law_units c =
  let kind = unit_kinds.(c.kind) in
  List.iter
    (fun f -> cover (family_name f) (c.family = f))
    [ Exact; One_hot; Near; Wide; Cancel; Subnormal; Zeros; Specials; Ints ];
  cover "float8" (match kind with Float (f, _) -> W.bits f = 8 | _ -> false);
  cover "4-bit integers of k 32"
    (match kind with Int { k; _ } -> k = 32 | _ -> false);
  let st = Random.State.make [| c.seed |] in
  let records = List.init records_a_case (fun _ -> record st kind c.family) in
  let got = on_units c.kind records in
  let wrong = ref [] in
  List.iteri
    (fun g (r, d) ->
      let want = model kind r in
      for i = 0 to 15 do
        for j = 0 to 15 do
          if d.(i).(j) <> want.(i).(j) && List.length !wrong < 8 then
            wrong :=
              strf "record %d (%d, %d): %08x, model %08x, c %08x" g i j
                d.(i).(j)
                want.(i).(j)
                r.c.(i).(j)
              :: !wrong
        done
      done)
    (List.combine records got);
  equal (list string) [] (List.rev !wrong)

(* nx_amd.mli: a step rounds its total once, and a total that rounds to zero
   gives +0. Products of bfloat16 subnormals lie below float32's least
   subnormal: one such product, in each of the instruction's steps, leaves a c
   of -0 or +0 at +0 and any other c as it is. *)
let test_step_zero () =
  let a =
    square (fun i k ->
        if k <> (4 * (i mod 4)) + (i / 4) then 0
        else if i < 8 then 0x0040
        else 0x0001)
  in
  let b = square (fun j _ -> if j mod 2 = 0 then 0x8040 else 0x0040) in
  let cs =
    [|
      0; 0x8000_0000; 1; 0x8000_0001; 8; 0x8000_0008; 0x0080_0000; 0x8080_0000;
    |]
  in
  let r = { a; b; c = square (fun _ j -> cs.(j / 2)) } in
  let words d =
    List.concat_map
      (fun i -> List.init 16 (fun j -> strf "(%d, %d) %08x" i j d.(i).(j)))
      (List.init 16 Fun.id)
  in
  let want =
    square (fun _ j -> if cs.(j / 2) land 0x7FFF_FFFF = 0 then 0 else cs.(j / 2))
  in
  equal ~msg:"the units" (list string) (words want)
    (words (List.hd (on_units 1 [ r ])));
  equal ~msg:"the model" (list string) (words want)
    (words (model (Float (W.Bf16, W.Bf16)) r))

(* The matrix units' bound *)

let pairs =
  W.
    [|
      (F16, F16);
      (Bf16, Bf16);
      (E4m3, E4m3);
      (E4m3, E5m2);
      (E5m2, E4m3);
      (E5m2, E5m2);
    |]

let pair_name (fa, fb) =
  if fa = fb then fmt_name fa else fmt_name fa ^ " x " ^ fmt_name fb

(* A step's products and runs: four in runs of 2 for 16-bit floats, eight in
   runs of 4 for float8. *)
let step_terms f = if W.bits f = 16 then 4 else 8
let run_terms f = if W.bits f = 16 then 2 else 4

(* The exponents of [f]'s normal codes the laws draw: within its normal range,
   and for bfloat16 narrower, so that sums of a few hundred products neither
   overflow nor underflow float32. *)
let exponents = function
  | W.F16 -> (-14, 15)
  | W.Bf16 -> (-30, 30)
  | W.E4m3 -> (-6, 8)
  | W.E5m2 -> (-14, 15)

let unit_roundoff = 0x1p-24

let gamma n =
  let x = float_of_int n *. unit_roundoff in
  x /. (1. -. x)

let of_f32 w = Int32.float_of_bits (Int32.of_int w)

(* The sum of [c] and [ps] with an error under 2^-100 of the sum of their
   magnitudes, held in two parts, and that sum. *)
let exact c ps =
  let hi = ref c and lo = ref 0. and mag = ref (Float.abs c) in
  Array.iter
    (fun p ->
      let s = !hi +. p in
      let v = s -. !hi in
      lo := !lo +. (!hi -. (s -. v) +. (p -. v));
      hi := s;
      mag := !mag +. Float.abs p)
    ps;
  (!hi, !lo, !mag)

(* Whether [got] is within [factor] of the exact sum of [c] and [ps]. *)
let within_factor factor ~got c ps =
  let hi, lo, mag = exact c ps in
  let err = Float.abs (got -. hi -. lo) in
  let bound = (factor *. mag) +. (0x1p-100 *. mag) in
  if err <= bound then None
  else Some (strf "%h is %h from %h, past %h" got err (hi +. lo) bound)

(* A normal code of [f] of exponent within [w] of [e], both brought into its
   range. *)
let normal_code st f ~e ~w =
  let lo, hi = exponents f in
  let e = Int.max lo (Int.min hi e) in
  code st f (Int.max lo (e - w), Int.min hi (e + w))

(* float32 bits of a normal value of exponent [e], random fraction and sign. *)
let f32_at st e = f32 st (e, e)

type step_case = {
  pair : int;
  m : int;
  lead : int option;
  spread : int;
  skew : bool;
  seed : int;
}

let pp_step_case ppf c =
  Format.fprintf ppf "%s, %d products, C %s, spread %d,%s seed %d"
    (pair_name pairs.(c.pair))
    c.m
    (match c.lead with
    | None -> "0"
    | Some d -> strf "lead at G's boundary %+d" d)
    c.spread
    (if c.skew then " skewed," else "")
    c.seed

(* Steps of every format pair and count of nonzero products, their exponents
   within [spread] of each other, or [skew]ed: one product and the others
   negative and smaller by 2^-12 or less, whose truncations each err by nearly a
   unit of G; C zero or of a lead [d] binades past the products' grid plus 26,
   where G turns from the products' grid to C's. *)
let step_case =
  let open Gen in
  with_pp pp_step_case
    (bind
       (int_range 0 (Array.length pairs - 1))
       (fun pair ->
         let n = step_terms (fst pairs.(pair)) in
         let+ m = int_range 0 n
         and+ lead =
           frequency
             [
               (1, constant None);
               (2, constant (Some 0));
               (2, map Option.some (int_range (-2) 2));
               (3, map Option.some (int_range (-30) 30));
             ]
         and+ spread = of_list [ 0; 1; 3; 8 ]
         and+ skew = bool
         and+ seed = nat in
         { pair; m; lead; spread; skew; seed }))

(* A step of [m] products of normal factors, as the model computes it, errs by
   at most γ(m + 1) (|C| + Σ|p|), its truncations by fewer than m units of its
   grid G <= u max(max|p|, |C| / 4) and its rounding by u: the bound nx_amd.mli
   derives its error factor from. *)
let law_step c =
  let fa, fb = pairs.(c.pair) in
  let n = step_terms fa in
  let st = Random.State.make [| c.seed |] in
  let mid f =
    let lo, hi = exponents f in
    (lo + hi) / 2
  in
  let ea = mid fa + Random.State.int st 5 - 2
  and eb = mid fb + Random.State.int st 5 - 2 in
  let zero f = Random.State.int st 2 lsl (W.bits f - 1) in
  (* A zero product: of a zero and, at times, a factor of its dtype's largest
     exponents, which takes no part in the step's grid. *)
  let large = ref false in
  let zero_product () =
    let big f = normal_code st f ~e:(snd (exponents f) - 1) ~w:0 in
    match Random.State.int st 3 with
    | 0 ->
        large := true;
        (zero fa, big fb)
    | 1 ->
        large := true;
        (big fa, zero fb)
    | _ -> (zero fa, zero fb)
  in
  let nonzero = Array.make n false in
  let placed = ref 0 in
  while !placed < c.m do
    let k = Random.State.int st n in
    if not nonzero.(k) then begin
      nonzero.(k) <- true;
      incr placed
    end
  done;
  let first = ref true in
  let a, b =
    Array.split
      (Array.init n (fun k ->
           if not nonzero.(k) then zero_product ()
           else if c.skew && not !first then
             let small = normal_code st fa ~e:(ea - 6) ~w:3 in
             ( small lor (1 lsl (W.bits fa - 1)),
               normal_code st fb ~e:(eb - 6) ~w:0
               land lnot (1 lsl (W.bits fb - 1)) )
           else begin
             first := false;
             ( normal_code st fa ~e:ea ~w:c.spread,
               normal_code st fb ~e:eb ~w:c.spread )
           end))
  in
  let lead_of f code = snd (Float.frexp (W.value f code)) - 1 in
  let grid = ref min_int in
  Array.iteri
    (fun k on ->
      if on then grid := Int.max !grid (lead_of fa a.(k) + lead_of fb b.(k) - 24))
    nonzero;
  let cbits =
    match c.lead with
    | None -> zero W.Bf16 lsl 16
    | Some d ->
        let at = if c.m = 0 then 0 else !grid + 26 + d in
        f32_at st (Int.max (-126) (Int.min 127 at))
  in
  cover (strf "%d products" c.m) true;
  cover "a zero times a large factor" !large;
  cover "C zero" (c.lead = None);
  cover "one product larger" (c.skew && c.m > 1);
  cover "G from the products"
    (c.m > 0 && match c.lead with Some d -> d < 0 | None -> true);
  cover "G at both" (c.m > 0 && c.lead = Some 0);
  cover "G from C"
    (c.m > 0 && match c.lead with Some d -> d > 0 | None -> false);
  let ps = Array.init n (fun k -> W.value fa a.(k) *. W.value fb b.(k)) in
  let got = of_f32 (W.dot fa fb ~group:(run_terms fa) a b cbits) in
  equal (option string) None
    (within_factor (gamma (c.m + 1)) ~got (of_f32 cbits) ps)

type sub_case = {
  spair : int;
  sm : int;
  da : int;
  db : int;
  near : int;
  slead : int option;
  sseed : int;
}

let pp_sub_case ppf c =
  Format.fprintf ppf
    "%s, %d products, a %d bits short, b %d, near %+d, C %s, seed %d"
    (pair_name pairs.(c.spair))
    c.sm c.da c.db c.near
    (match c.slead with
    | None -> "0"
    | Some d -> strf "lead at G's boundary %+d" d)
    c.sseed

(* Steps with one product of a subnormal factor [da] bits short of its format's
   least normal, times the largest normal of the law's range or a subnormal [db]
   bits short, where float16 and float8 keep the product clear of float32's
   underflow; the other products near its magnitude, at [near] binades, on finer
   grids. *)
let sub_case =
  let open Gen in
  with_pp pp_sub_case
    (bind
       (int_range 0 (Array.length pairs - 1))
       (fun spair ->
         let fa, fb = pairs.(spair) in
         let+ sm = int_range 1 (step_terms fa)
         and+ da = int_range 1 (frac fa)
         and+ db =
           if fb = W.Bf16 then constant 0
           else frequency [ (3, constant 0); (1, int_range 1 (frac fb)) ]
         and+ near = int_range (-3) 3
         and+ slead =
           frequency
             [
               (1, constant None);
               (2, map Option.some (int_range (-2) 2));
               (2, map Option.some (int_range (-30) 30));
             ]
         and+ sseed = nat in
         { spair; sm; da; db; near; slead; sseed }))

(* A step whose products' factors fall short of their formats' least normals by
   [δ] bits at most, a product's two together, errs by at most (1 + 2^δ m) u
   (|C| + Σ|p|): its grids are up to 2^δ times coarser against the products than
   normal factors' (nx_amd.mli). *)
let law_sub_step c =
  let fa, fb = pairs.(c.spair) in
  let n = step_terms fa in
  let st = Random.State.make [| c.sseed |] in
  let least f = 1 - bias f in
  let sub f d =
    let sign = Random.State.int st 2 lsl (W.bits f - 1) in
    let lead = frac f - d in
    sign lor (1 lsl lead) lor Random.State.int st (1 lsl lead)
  in
  let eb = if c.db > 0 then least fb else snd (exponents fb) in
  let target = least fa - c.da + eb - c.db + c.near in
  let slots = Array.init n Fun.id in
  for k = n - 1 downto 1 do
    let r = Random.State.int st (k + 1) in
    let t = slots.(k) in
    slots.(k) <- slots.(r);
    slots.(r) <- t
  done;
  let a = Array.make n 0 and b = Array.make n 0 in
  for i = 0 to c.sm - 1 do
    let k = slots.(i) in
    if i = 0 then begin
      a.(k) <- sub fa c.da;
      b.(k) <- (if c.db > 0 then sub fb c.db else code st fb (eb, eb))
    end
    else
      let e2 = eb - Random.State.int st 3 in
      let e1 = target - e2 in
      a.(k) <- code st fa (e1, e1);
      b.(k) <- code st fb (e2, e2)
  done;
  let grid_of f x = Int.max (least f) (snd (Float.frexp (W.value f x)) - 1) in
  let grid = ref min_int in
  for i = 0 to c.sm - 1 do
    let k = slots.(i) in
    grid := Int.max !grid (grid_of fa a.(k) + grid_of fb b.(k) - 24)
  done;
  let cbits =
    match c.slead with
    | None -> 0
    | Some d -> f32_at st (Int.max (-126) (Int.min 127 (!grid + 26 + d)))
  in
  cover (pair_name (fa, fb)) true;
  cover "a's factor at its shortest" (c.da = frac fa);
  cover "both factors subnormal" (c.db > 0);
  cover "G from the products"
    (match c.slead with Some d -> d < 0 | None -> true);
  cover "G from C" (match c.slead with Some d -> d > 0 | None -> false);
  let ps = Array.init n (fun k -> W.value fa a.(k) *. W.value fb b.(k)) in
  let got = of_f32 (W.dot fa fb ~group:(run_terms fa) a b cbits) in
  let factor = gamma (1 + ((1 lsl (c.da + c.db)) * c.sm)) in
  equal (option string) None (within_factor factor ~got (of_f32 cbits) ps)

(* An output of [k] products through instructions of 16, in [ranges] ranges of
   whole instructions, each from +0, each instruction from the last's output,
   the ranges' sums added in increasing k, then [init], each addition rounding
   to float32. *)
let chain fa fb ~ranges a b init =
  let k = Array.length a in
  let tiles = (k + 15) / 16 in
  let ranges = Int.max 1 (Int.min ranges tiles) in
  let pad x t =
    Array.init 16 (fun e ->
        let i = (16 * t) + e in
        if i < k then x.(i) else 0)
  in
  let range s =
    let acc = ref 0 in
    for t = s * tiles / ranges to ((s + 1) * tiles / ranges) - 1 do
      acc := W.wmma fa fb (pad a t) (pad b t) !acc
    done;
    of_f32 !acc
  in
  let round x = Int32.float_of_bits (Int32.bits_of_float x) in
  let total = ref (range 0) in
  for s = 1 to ranges - 1 do
    total := round (!total +. range s)
  done;
  (round (!total +. of_f32 init), ranges)

(* The error factor of an output of [k] products in [ranges] ranges: γ(K + ⌈K/s⌉
   + ranges - 1), [s] a step's products. *)
let units_factor f ~k ~ranges =
  let s = step_terms f in
  gamma (k + ((k + s - 1) / s) + ranges - 1)

type chain_case = {
  cpair : int;
  k : int;
  ranges : int;
  init : bool;
  cspread : int;
  cseed : int;
}

let pp_chain_case ppf c =
  Format.fprintf ppf "%s, k %d, %d ranges,%s spread %d, seed %d"
    (pair_name pairs.(c.cpair))
    c.k c.ranges
    (if c.init then " init," else "")
    c.cspread c.cseed

let chain_case =
  let open Gen in
  with_pp pp_chain_case
    (let+ cpair = int_range 0 (Array.length pairs - 1)
     and+ k =
       frequency
         [
           (1, int_range 1 8);
           (2, int_range 9 16);
           (2, int_range 17 48);
           (1, int_range 49 300);
         ]
     and+ ranges = frequency [ (2, constant 1); (1, int_range 2 5) ]
     and+ init = bool
     and+ cspread = of_list [ 0; 2; 8; 20 ]
     and+ cseed = nat in
     { cpair; k; ranges; init; cspread; cseed })

(* An output of normal factors through a chain of instructions is within
   [u|r| + (1 + u) e Σ|p|] of the exact [r = init + Σ p], [e] its error factor,
   which is at most γ(2K + 1), nx_kernel.mli's for one running sum. *)
let law_chain c =
  let fa, fb = pairs.(c.cpair) in
  let st = Random.State.make [| c.cseed |] in
  let mid f =
    let lo, hi = exponents f in
    (lo + hi) / 2
  in
  let draw f =
    Array.init c.k (fun _ ->
        if Random.State.int st 5 = 0 then
          Random.State.int st 2 lsl (W.bits f - 1)
        else normal_code st f ~e:(mid f) ~w:c.cspread)
  in
  let a = draw fa and b = draw fb in
  let init =
    if c.init then f32 st (mid fa + mid fb - 10, mid fa + mid fb + 10) else 0
  in
  let got, ranges = chain fa fb ~ranges:c.ranges a b init in
  let s = step_terms fa in
  cover "a partial step" (c.k mod s <> 0);
  cover "whole steps" (c.k mod s = 0);
  cover "one instruction" (c.k <= 16);
  cover "several instructions" (c.k > 16);
  cover "several ranges" (ranges > 1);
  let factor = units_factor fa ~k:c.k ~ranges in
  at_most ~msg:"within γ(2K + 1)" float_exact
    ~than:(gamma ((2 * c.k) + 1))
    factor;
  let ps = Array.init c.k (fun i -> W.value fa a.(i) *. W.value fb b.(i)) in
  let hi, lo, _ = exact (of_f32 init) ps and _, _, mag = exact 0. ps in
  let u = 0x1p-24 in
  let err = Float.abs (got -. hi -. lo) in
  let bound =
    (u *. Float.abs (hi +. lo))
    +. ((1. +. u) *. factor *. mag)
    +. (0x1p-100 *. (mag +. Float.abs (of_f32 init)))
  in
  equal (option string) None
    (if err <= bound then None
     else Some (strf "%h is %h from %h, past %h" got err (hi +. lo) bound))

(* Contractions on the matrix units *)

(* The operand pairs the units sum: float pairs into float32, a float8 operand
   beside a bfloat16 one, and integers into int32. *)
type upair =
  | Fpair of W.fmt * W.fmt
  | Mixed of W.fmt * W.fmt
  | Ipair of D.any * D.any

(* Every kind of each tile's kernels: the float pairs, a float8 operand beside a
   16-bit one on either side, and 8-bit integers signed or not, a 4-bit one
   among them. *)
let upairs =
  Array.of_list
    (Array.to_list (Array.map (fun (a, b) -> Fpair (a, b)) pairs)
    @ W.[ Mixed (Bf16, E4m3); Mixed (F16, E4m3); Mixed (E5m2, Bf16) ]
    @ D.
        [
          Ipair (Any Int8, Any Int8);
          Ipair (Any Uint8, Any Int8);
          Ipair (Any Uint8, Any Uint8);
          Ipair (Any Int4, Any Uint8);
          Ipair (Any Uint4, Any Int8);
        ])

let upair_name = function
  | Fpair (a, b) -> pair_name (a, b)
  | Mixed (a, b) -> fmt_name a ^ " x " ^ fmt_name b
  | Ipair (D.Any a, D.Any b) -> D.name a ^ " x " ^ D.name b

let fmt_dtype = function
  | W.F16 -> D.Any D.Float16
  | W.Bf16 -> D.Any D.Bfloat16
  | W.E4m3 -> D.Any D.Float8_e4m3fn
  | W.E5m2 -> D.Any D.Float8_e5m2

(* How an operand [batch; rows; k] lies: k contiguous, its rows contiguous,
   every other k of a store twice as long, which a pack reads, or one element
   a row broadcast along k. *)
type ulay = Along_k | Along_rows | Every_other_k | Along_none

let ulays = [ Along_k; Along_rows; Every_other_k; Along_none ]

let ulay_name = function
  | Along_k -> "k"
  | Along_rows -> "rows"
  | Every_other_k -> "every other k"
  | Along_none -> "one element along k"

(* y's dtype: the accumulator's, a 16-bit float, or one the kernels do not
   write, float64 or int8, into which the plan casts. *)
type uy = Y_acc | Y_bf16 | Y_f16 | Y_other

let uy_name = function
  | Y_acc -> "acc"
  | Y_bf16 -> "bfloat16"
  | Y_f16 -> "float16"
  | Y_other -> "not the kernels'"

type ucall = {
  upair : int;
  ub : int;
  um : int;
  un : int;
  uk : int;
  ula : ulay;
  ulb : ulay;
  uinit : bool;
  uy : uy;
  uunsigned : bool;  (** An integer pair into [uint32]. *)
  uspecial : bool;  (** Some codes NaN or infinite. *)
  utile : int option;
  useed : int;
}

let pp_ucall ppf c =
  Format.fprintf ppf
    "%s [%d; %d; %d] k %d, a by %s, b by %s,%s y %s,%s%s%s seed %d"
    (upair_name upairs.(c.upair))
    c.ub c.um c.un c.uk (ulay_name c.ula) (ulay_name c.ulb)
    (if c.uinit then " init," else "")
    (uy_name c.uy)
    (if c.uunsigned then " uint32," else "")
    (if c.uspecial then " specials," else "")
    (match c.utile with Some r -> strf " tile of %d rows," r | None -> "")
    c.useed

(* Extents of one row or column, of the 16-row tile, past one 128-row tile, and
   at times few outputs of many terms, cut into up to 16 ranges of k; each
   tile, forced at times so that small calls reach the 128-row one. *)
let ucall =
  let open Gen in
  let extent = frequency [ (2, int_range 1 16); (3, int_range 17 48) ] in
  with_pp pp_ucall
    (let* long = frequency [ (4, constant false); (1, constant true) ] in
     let+ upair = int_range 0 (Array.length upairs - 1)
     and+ ub = int_range 1 2
     and+ um = if long then int_range 1 3 else extent
     and+ un = if long then int_range 1 3 else extent
     and+ uk =
       if long then int_range 4097 33_000
       else frequency [ (1, int_range 1 16); (2, int_range 17 300) ]
     and+ ula = of_list ulays
     and+ ulb = of_list ulays
     and+ uinit = bool
     and+ uy = of_list [ Y_acc; Y_bf16; Y_f16; Y_other ]
     and+ uunsigned = bool
     and+ uspecial = frequency [ (3, constant false); (1, constant true) ]
     and+ utile = of_list [ None; Some 16; Some 128 ]
     and+ useed = nat in
     {
       upair;
       ub;
       um;
       un;
       uk;
       ula;
       ulb;
       uinit;
       uy;
       uunsigned;
       uspecial;
       utile;
       useed;
     })

(* One batch element past the units' grid, along z. *)
let past_z =
  {
    upair = 0;
    ub = 65_536;
    um = 1;
    un = 1;
    uk = 1;
    ula = Along_k;
    ulb = Along_k;
    uinit = true;
    uy = Y_acc;
    uunsigned = false;
    uspecial = false;
    utile = None;
    useed = 1;
  }

(* bfloat16 subnormals broadcast along k, whose products sum below float32's
   least subnormal: outputs that round to zero from no init, and from inits of
   -0 and +0. *)
let tiny_sums =
  {
    upair = 1;
    ub = 1;
    um = 2;
    un = 17;
    uk = 239;
    ula = Along_none;
    ulb = Along_none;
    uinit = false;
    uy = Y_acc;
    uunsigned = false;
    uspecial = false;
    utile = None;
    useed = 41;
  }

let tiny_sums_init = { tiny_sums with uinit = true }

(* An array of [dt] [batch; rows; k] of the codes [codes] (in C order of that
   shape), lying as [lay] on the GPU. *)
let coded (D.Any dt) lay (batch, rows, k) codes =
  let store =
    match lay with
    | Along_k -> [| batch; rows; k |]
    | Along_rows -> [| batch; k; rows |]
    | Every_other_k -> [| batch; rows; 2 * k |]
    | Along_none -> [| batch; rows; 1 |]
  in
  let n = Array.fold_left ( * ) 1 store in
  let bytes = Int.max 1 (D.bits dt / 8) in
  let s = Bytes.make (n * bytes) '\000' in
  Array.iteri
    (fun e c ->
      let z = e / (rows * k) and r = e / k mod rows and kk = e mod k in
      let at =
        match lay with
        | Along_k -> e
        | Along_rows -> (((z * k) + kk) * rows) + r
        | Every_other_k -> (((z * rows) + r) * 2 * k) + (2 * kk)
        | Along_none -> (z * rows) + r
      in
      if lay = Along_none && kk > 0 then ()
      else if bytes = 2 then Bytes.set_uint16_le s (2 * at) c
      else Bytes.set_uint8 s at (c land 0xFF))
    codes;
  (* A 4-bit dtype lies two elements a byte, low first. *)
  let s =
    if D.bits dt <> 4 then s
    else
      Bytes.init
        ((n + 1) / 2)
        (fun i ->
          let lo = Bytes.get_uint8 s (2 * i) land 0xF in
          let hi =
            if (2 * i) + 1 < n then Bytes.get_uint8 s ((2 * i) + 1) land 0xF
            else 0
          in
          Char.chr (lo lor (hi lsl 4)))
  in
  let x = A.create (S.device ()) dt (Iarray.of_array store) in
  Rig.Buffer.blit_from_string (Bytes.to_string s) 0 (A.buffer x) 0
    (Bytes.length s);
  let moved m = A.Any (Option.get (A.move m x)) in
  match lay with
  | Along_k -> A.Any x
  | Along_rows -> moved (M.Permute [| 0; 2; 1 |])
  | Every_other_k ->
      moved
        (M.Slice
           [|
             { M.start = 0; count = batch; step = 1 };
             { M.start = 0; count = rows; step = 1 };
             { M.start = 0; count = k; step = 2 };
           |])
  | Along_none ->
      A.Any
        (A.v dt
           (A.Layout.v
              ~strides:(Iarray.of_array [| rows; 1; 0 |])
              (Iarray.of_array [| batch; rows; k |]))
           (A.buffer x))

(* bfloat16 of the float32 bits [w], to nearest even, as nx_dtype.h's encoder
   rounds a finite value. *)
let bf16_of w = ((w + 0x7FFF + ((w lsr 16) land 1)) lsr 16) land 0xFFFF

(* A call's tile on the units, as nx_amd.mli states it: its rows (of [m], or
   of [n] where [n <= 16 < m]), its tiles and its ranges [S]. *)
type unit_plan = {
  rows : int;
  cols : int;
  tiles : int;
  ranges : int;
  filled : bool;  (** Whether the resident groups bound the ranges. *)
}

let unit_plan ?rows ~batch ~m ~n ~k () =
  let cdiv a b = (a + b - 1) / b in
  let m, n = if n <= 16 && m > 16 then (n, m) else (m, n) in
  let least =
    if m <= 4 || n <= 4 || batch * m * n <= 1 lsl 18 then 16 else 256
  in
  let plan ~rows ~cols ~resident ~w =
    let tiles = batch * cdiv m rows * cdiv n cols in
    let by_k = cdiv k 128 / least in
    let ranges = Int.max 1 (Int.min (Int.min 64 (resident / tiles)) by_k) in
    let filled = resident / tiles < Int.min 64 by_k in
    ({ rows; cols; tiles; ranges; filled }, cdiv (tiles * ranges) 64 * w)
  in
  let small, small_cost = plan ~rows:16 ~cols:64 ~resident:256 ~w:3072 in
  let large, large_cost = plan ~rows:128 ~cols:128 ~resident:64 ~w:16384 in
  match rows with
  | Some 16 -> small
  | Some _ -> large
  | None ->
      if m <= 16 then small
      else if small_cost < large_cost then small
      else large

(* The ranges of k of an output of [k] terms, as nx_amd.mli states them. *)
let unit_ranges ?rows ~batch ~m ~n ~k () =
  let runs = (k + 127) / 128 in
  let s = (unit_plan ?rows ~batch ~m ~n ~k ()).ranges in
  List.init s (fun r ->
      (r * runs / s * 128, Int.min k ((r + 1) * runs / s * 128)))

(* An output of [a] and [b], codes in k order, and [init], as nx_amd.mli
   associates it: each range from +0 in instructions of 16 k, the ranges' sums
   added in order, then [init] where there is one. *)
let unit_output instr ~add ~ranges a b init =
  let range (k0, k1) =
    let acc = ref 0 in
    let kb = ref k0 in
    while !kb < k1 do
      let part x =
        Array.init 16 (fun e -> if !kb + e < k1 then x.(!kb + e) else 0)
      in
      acc := instr (part a) (part b) !acc;
      kb := !kb + 16
    done;
    !acc
  in
  match (List.map range ranges, init) with
  | [], _ -> Option.value ~default:0 init
  | s :: rest, None -> List.fold_left add s rest
  | s :: rest, Some x -> add (List.fold_left add s rest) x

let add_f32 x y =
  Int32.to_int (Int32.bits_of_float (of_f32 x +. of_f32 y)) land 0xFFFF_FFFF

(* A code of [f] for the units' law: normal, at times subnormal or zero. *)
let unit_code st f =
  let lo, hi = exponents f in
  code st f ~zero:0.1 ~sub:0.05 (Int.max lo (-6), Int.min hi 6)

(* Extents of a units call: rows and columns of the 16-row tile or past it,
   batches of one to 64 or 907, and contracted extents short or long. *)
let uextents =
  let open Gen in
  let extent top = frequency [ (1, int_range 1 16); (2, int_range 17 top) ] in
  let* batch =
    frequency [ (3, int_range 1 4); (1, int_range 5 64); (1, constant 907) ]
  in
  let top = if batch > 64 then 40 else 300 in
  let+ m = extent top
  and+ n = extent top
  and+ k = frequency [ (1, int_range 1 300); (2, int_range 1000 40_000) ] in
  (batch, m, n, k)

let pp_uextents ppf (batch, m, n, k) =
  Format.fprintf ppf "[%d; %d; %d] k %d" batch m n k

(* nx_amd.mli: a call on the units takes the stated tile and ranges, and holds
   at most 4 MiB of the ranges' sums. Planned on the host: operands of one
   element broadcast to their shapes, which planning alone reads. *)
let law_unit_plan (batch, m, n, k) =
  let want = unit_plan ~batch ~m ~n ~k () in
  let broadcast shape =
    let x = A.create Rig.host D.Float16 [| 1 |] in
    A.Any
      (A.v D.Float16
         (A.Layout.v
            ~strides:(Iarray.of_array [| 0; 0; 0 |])
            (Iarray.of_array shape))
         (A.buffer x))
  in
  let dst = A.create Rig.host D.Float32 [| batch; m; n |] in
  let operand x : Nx_kernel.operand = { array = x; scale = None } in
  let plan =
    match
      Nx_amd.contract_plan
        ~batch:(Iarray.of_array [| (0, 0) |])
        ~contracting:(Iarray.of_array [| (2, 1) |])
        ~segments:None ~acc:(D.Any D.Float32) ~dst ~init:None
        (operand (broadcast [| batch; m; k |]))
        (operand (broadcast [| batch; k; n |]))
    with
    | Ok (Some p) -> p
    | Ok None -> failf "no launch"
    | Error e -> failf "answered %a" Nx_array_support.pp_answer e
  in
  let units =
    Array.to_list plan.launches
    |> List.filter_map (fun (l : Plan.launch) ->
        Option.map (fun t -> (t, l.groups)) (Nx_amd.Kernels.wtile l.kernel))
  in
  match units with
  | [ ((t : Nx_amd.Kernels.wtile), (x, y, z)) ] ->
      cover "16 x 64 past 16 rows" (m > 16 && n > 16 && t.bm = 16);
      cover "128 x 128" (t.bm = 128);
      cover "exchanged" (n <= 16 && m > 16);
      cover "several ranges" (y > 1);
      cover "ranges bounded by the resident groups" want.filled;
      equal ~msg:"tile rows" int want.rows t.bm;
      equal ~msg:"tile columns" int want.cols t.bn;
      equal ~msg:"tiles" int want.tiles (x * z);
      equal ~msg:"ranges" int want.ranges y;
      let sums = if y > 1 then x * y * z * t.bm * t.bn * 4 else 0 in
      at_most ~msg:"bytes of sums" int ~than:(4 lsl 20) sums
  | _ -> failf "%d launches on the units" (List.length units)

(* nx.amd's contraction on the matrix units has the bits of the model's
   instructions in the association nx_amd.mli states: its ranges, its steps in
   increasing k, its truncations, rounded once into y, a NaN any NaN. *)

(* The 16-bit code of [f] holding the value [v] of a float8 code exactly, as
   the units read a float8 operand beside a 16-bit one. *)
let half_codes =
  let table f =
    lazy
      (let t = Hashtbl.create 65536 in
       for c = 0xFFFF downto 0 do
         let v = W.value f c in
         if not (Float.is_nan v) then
           Hashtbl.replace t (Int64.bits_of_float v) c
       done;
       t)
  in
  let f16 = table W.F16 and bf16 = table W.Bf16 in
  fun f v ->
    if Float.is_nan v then if f = W.F16 then 0x7E00 else 0x7FC0
    else
      Hashtbl.find
        (Lazy.force (if f = W.F16 then f16 else bf16))
        (Int64.bits_of_float v)

(* float16 of the float32 bits [w], to nearest even, as nx.cpu's cast stores
   it: an infinity as an infinity, a NaN as 0x7E00. *)
let f16_of w =
  let x = of_f32 w in
  if Float.is_nan x then 0x7E00
  else
    let sign = if Float.sign_bit x then 0x8000 else 0 in
    let a = Float.abs x in
    if a = Float.infinity then sign lor 0x7C00
    else
      let e = if a = 0. then -14 else Int.max (-14) (snd (Float.frexp a) - 1) in
      let r = Float.ldexp a (10 - e) in
      let fl = Float.floor r in
      let d = r -. fl in
      let n =
        Float.to_int fl
        + if d > 0.5 || (d = 0.5 && Float.to_int fl land 1 = 1) then 1 else 0
      in
      let code = if n < 1024 then n else ((e + 15) lsl 10) + (n - 1024) in
      sign lor Int.min code 0x7C00

(* A special code of [f]: a NaN, or an infinity where [f] has one. *)
let special st f =
  let sign = Random.State.int st 2 lsl (W.bits f - 1) in
  let nan, inf =
    match f with
    | W.F16 -> (0x7E00, Some 0x7C00)
    | W.Bf16 -> (0x7FC0, Some 0x7F80)
    | W.E4m3 -> (0x7F, None)
    | W.E5m2 -> (0x7E, Some 0x7C)
  in
  match inf with
  | Some i when Random.State.bool st -> sign lor i
  | _ -> sign lor nan

let law_unit_call c =
  let pair = upairs.(c.upair) in
  let st = Random.State.make [| c.useed |] in
  let integer = match pair with Ipair _ -> true | _ -> false in
  (* The operands' formats as drawn, and as the units read them. *)
  let fa, fb, da, db =
    match pair with
    | Fpair (a, b) | Mixed (a, b) -> (a, b, fmt_dtype a, fmt_dtype b)
    | Ipair (a, b) -> (W.F16, W.F16, a, b)
  in
  let ra, rb =
    match pair with
    | Mixed (a, b) when W.bits a = 8 -> (b, b)
    | Mixed (a, _) -> (a, a)
    | _ -> (fa, fb)
  in
  let int_code (D.Any d) = Random.State.int st (1 lsl D.bits d) in
  let draw f d =
    if integer then int_code d
    else if c.uspecial && Random.State.int st 16 = 0 then special st f
    else unit_code st f
  in
  (* Codes [batch; rows; k], one along k for an operand broadcast along it. *)
  let codes f d rows lay =
    let x = Array.init (c.ub * rows * c.uk) (fun _ -> draw f d) in
    if lay = Along_none then
      Array.iteri (fun e _ -> x.(e) <- x.(e - (e mod c.uk))) x;
    x
  in
  let ca = codes fa da c.um c.ula in
  (* b drawn [batch; n; k], laid out as [batch; k; n]. *)
  let cb = codes fb db c.un c.ulb in
  let wanted = unit_plan ?rows:c.utile ~batch:c.ub ~m:c.um ~n:c.un ~k:c.uk () in
  let ranges =
    unit_ranges ?rows:c.utile ~batch:c.ub ~m:c.um ~n:c.un ~k:c.uk ()
  in
  let tile = if wanted.rows = 16 then "16 x 64" else "128 x 128" in
  cover (upair_name pair ^ " on " ^ tile) true;
  cover "exchanged" (c.un <= 16 && c.um > 16);
  cover ("several ranges on " ^ tile) (List.length ranges > 1);
  cover "six ranges or more on 16 x 64"
    (wanted.rows = 16 && List.length ranges >= 6);
  cover "a by rows" (c.ula = Along_rows);
  cover "b by rows" (c.ulb = Along_rows);
  cover "packed" (c.ula = Every_other_k || c.ulb = Every_other_k);
  cover "broadcast along k" (c.ula = Along_none || c.ulb = Along_none);
  cover "a partial instruction" (c.uk mod 16 <> 0);
  cover "a uint32 accumulator" (integer && c.uunsigned);
  cover "a float16 y and init" (c.uy = Y_f16 && c.uinit && not integer);
  cover "a y the kernels do not write" (c.uy = Y_other);
  cover "a NaN or an infinity" (c.uspecial && not integer);
  let a = coded da c.ula (c.ub, c.um, c.uk) ca in
  let bt = coded db c.ulb (c.ub, c.un, c.uk) cb in
  let b =
    let (A.Any x) = bt in
    A.Any (Option.get (A.move (M.Permute [| 0; 2; 1 |]) x))
  in
  let acc =
    if not integer then D.Any D.Float32
    else if c.uunsigned then D.Any D.Uint32
    else D.Any D.Int32
  in
  let yd =
    match (c.uy, integer) with
    | Y_acc, _ -> acc
    | Y_bf16, false -> D.Any D.Bfloat16
    | Y_f16, false -> D.Any D.Float16
    | Y_other, false -> D.Any D.Float64
    | Y_other, true -> D.Any D.Int8
    | (Y_bf16 | Y_f16), true -> acc
  in
  let (D.Any ydt) = yd in
  let wide = Int.max 1 (D.bits ydt / 8) in
  let ny = c.ub * c.um * c.un in
  (* init's elements as y holds them, at times -0 or +0, then as the
     accumulator does: a float32 value, a 16-bit float decoded, an int8
     sign-extended. *)
  let inits_y =
    Array.init ny (fun _ ->
        if not c.uinit then 0
        else
          match yd with
          | D.Any D.Bfloat16 -> bf16_of (f32 st ~zero:0.25 (-4, 8))
          | D.Any D.Float16 -> f16_of (f32 st ~zero:0.25 (-4, 8))
          | D.Any D.Int8 -> Random.State.int st 256
          | D.Any (D.Int32 | D.Uint32) -> Random.State.bits st land 0xFFFF_FFFF
          | _ -> f32 st ~zero:0.25 (-4, 8))
  in
  let inits =
    Array.map
      (fun w ->
        match yd with
        | D.Any D.Bfloat16 -> w lsl 16
        | D.Any D.Float16 -> f32_bits (W.value W.F16 w)
        | D.Any D.Int8 -> if w >= 128 then (w - 256) land 0xFFFF_FFFF else w
        | _ -> w)
      inits_y
  in
  let make () =
    let y =
      A.create (S.device ()) ydt (Iarray.of_array [| c.ub; c.um; c.un |])
    in
    let s = Bytes.create (ny * wide) in
    Array.iteri
      (fun e w ->
        match wide with
        | 1 -> Bytes.set_uint8 s e w
        | 2 -> Bytes.set_uint16_le s (2 * e) w
        | 4 -> Bytes.set_int32_le s (4 * e) (Int32.of_int w)
        | _ -> Bytes.set_int64_le s (8 * e) (Int64.bits_of_float (of_f32 w)))
      inits_y;
    Rig.Buffer.blit_from_string (Bytes.to_string s) 0 (A.buffer y) 0
      (Bytes.length s);
    y
  in
  let dst = make () and init = if c.uinit then Some (make ()) else None in
  let operand x : Nx_kernel.operand = { array = x; scale = None } in
  Fun.protect
    ~finally:(fun () -> Nx_amd.contract_unit_tile None)
    (fun () ->
      Nx_amd.contract_unit_tile c.utile;
      equal answer A.Done
        (Nx_amd.contract
           ~batch:(Iarray.of_array [| (0, 0) |])
           ~contracting:(Iarray.of_array [| (2, 1) |])
           ~segments:None ~acc ~dst ~init (operand a) (operand b)));
  let got = Bytes.create (ny * wide) in
  Rig.Buffer.blit_to_bytes (A.buffer dst) 0 got 0 (ny * wide);
  (* An operand's codes as the units read them. *)
  let read f r x = if f = r then x else half_codes r (W.value f x) in
  let signed (D.Any d) v =
    if D.is D.Signed d && v lsr (D.bits d - 1) = 1 then v - (1 lsl D.bits d)
    else v
  in
  (* y's bits of the accumulator's [w], and whether [have] is them, a NaN any
     NaN. *)
  let store w =
    match yd with
    | D.Any D.Bfloat16 -> bf16_of w
    | D.Any D.Float16 -> f16_of w
    | D.Any D.Int8 -> w land 0xFF
    | _ -> w
  in
  let same have want =
    have = want
    ||
    match yd with
    | D.Any D.Bfloat16 ->
        Float.is_nan (W.value W.Bf16 have) && Float.is_nan (W.value W.Bf16 want)
    | D.Any D.Float16 ->
        Float.is_nan (W.value W.F16 have) && Float.is_nan (W.value W.F16 want)
    | D.Any D.Float32 ->
        Float.is_nan (of_f32 have) && Float.is_nan (of_f32 want)
    | _ -> false
  in
  let wrong = ref [] in
  let underflow = ref false and init_zero = ref false in
  for z = 0 to c.ub - 1 do
    for i = 0 to c.um - 1 do
      for j = 0 to c.un - 1 do
        let e = (((z * c.um) + i) * c.un) + j in
        let xa =
          Array.init c.uk (fun k ->
              read fa ra ca.((((z * c.um) + i) * c.uk) + k))
        in
        let xb =
          Array.init c.uk (fun k ->
              read fb rb cb.((((z * c.un) + j) * c.uk) + k))
        in
        let want =
          if integer then begin
            let s = ref inits.(e) in
            Array.iteri
              (fun k x -> s := !s + (signed da x * signed db xb.(k)))
              xa;
            store (!s land 0xFFFF_FFFF)
          end
          else begin
            let s =
              unit_output (W.ordered ra rb) ~add:add_f32 ~ranges xa xb
                (if c.uinit then Some inits.(e) else None)
            in
            if s = 0 then begin
              let product x y = W.value ra x *. W.value rb y <> 0. in
              if Array.exists2 product xa xb then underflow := true;
              if inits.(e) = 0x8000_0000 then init_zero := true
            end;
            store s
          end
        in
        let have =
          match wide with
          | 1 -> Bytes.get_uint8 got e
          | 2 -> Bytes.get_uint16_le got (2 * e)
          | 4 -> Int32.to_int (Bytes.get_int32_le got (4 * e)) land 0xFFFF_FFFF
          | _ ->
              let x = Int64.float_of_bits (Bytes.get_int64_le got (8 * e)) in
              if Float.is_nan x then 0x7FC0_0000 else f32_bits x
        in
        let want =
          if wide = 8 && Float.is_nan (of_f32 want) then 0x7FC0_0000 else want
        in
        if (not (same have want)) && List.length !wrong < 8 then
          wrong :=
            strf "output (%d, %d, %d): %x, model %x" z i j have want :: !wrong
      done
    done
  done;
  cover "nonzero products that sum to +0" !underflow;
  cover "an init of -0 that sums to +0" !init_zero;
  equal (list string) [] (List.rev !wrong)

(* nx_amd.mli: a call computes its results window by window on the matrix units
   past 65,535 batch elements, and its launches grow with its windows; it is
   kept like any other call, within 64 MiB. *)
let test_windows_kept () =
  let d = S.device () in
  let operands nb ~k =
    let a = A.create d D.Float16 [| nb; 1; k |] in
    let b = A.create d D.Float16 [| nb; k; 1 |] in
    let dst = A.create d D.Float32 [| nb; 1; 1 |] in
    let operand x : Nx_kernel.operand = { array = A.Any x; scale = None } in
    (dst, operand a, operand b)
  in
  let batch = Iarray.of_array [| (0, 0) |] in
  let contracting = Iarray.of_array [| (2, 1) |] in
  let acc = D.Any D.Float32 in
  let call nb ~k =
    let dst, a, b = operands nb ~k in
    equal answer A.Done
      (Nx_amd.contract ~batch ~contracting ~segments:None ~acc ~dst ~init:None a
         b)
  in
  let plan nb ~k =
    let dst, a, b = operands nb ~k in
    match
      Nx_amd.contract_plan ~batch ~contracting ~segments:None ~acc ~dst
        ~init:None a b
    with
    | Ok (Some p) -> p
    | Ok None -> failf "no launch"
    | Error e -> failf "answered %a" Nx_array_support.pp_answer e
  in
  let kept_rule nb ~k =
    let p = plan nb ~k in
    let launches = Array.length p.launches in
    equal ~msg:(strf "%d batch elements, %d launches" nb launches) bool
      (p.bytes <= Plan.kept_submissions) p.kept;
    launches
  in
  let many = (9 * 65_535) + 1 in
  ignore (kept_rule 65_537 ~k:16);
  greater ~msg:"ten windows" int ~than:8 (kept_rule many ~k:16);
  ignore (kept_rule 1_234 ~k:48);
  Nx_amd.drop_kept d;
  call many ~k:16;
  call many ~k:16;
  equal ~msg:"calls past 8 launches" int 1 (Nx_amd.kept_calls d)

let () =
  exit
    (run "nx_amd"
       [
         group "sums"
           [
             prop "sum is its terms' wrapping sum" case law_sum;
             test "sum past the most groups' work-items" test_wide_grid;
             test "sums of 1 tile to twice the most groups' tiles" test_groups;
             test "sum through buffers of its own" test_fresh;
             test "a call past the bytes kept keeps no submission"
               test_unkept;
             test "bound_fresh, bound_kept and bound_grid refuse bounds out of \
                   range"
               test_bounds;
             domains_sum_alike;
           ];
         group "kept submissions"
           [
             stateful ~count:600 ~steps:40
               "keep the least recently used within the budget" kept_commands;
             stateful ~domains:2 ~count:20
               "lose no entry to a keep on another domain" kept_roomy;
             stateful ~domains:2 ~count:20
               "stay within the budget on two domains" kept_bounded;
             test "refuse a cell they lack and negative bytes"
               test_kept_args;
             test "count every kernel's name whole" test_names;
           ];
         group "plans"
           [
             prop ~examples:unread_examples
               "finish leaves out a slot no launch's ref names" planned
               law_slots;
             prop ~examples:kept_examples
               "finish counts a call's bytes and keeps it within the bound"
               counted law_kept;
             test "zero zeroes a slot's bytes from a byte" test_zero;
             test "a launch's used words hold its refs" test_used;
             test "a launch's LDS fits a group" test_shared;
             prop "a striding grid is never in the slow band" gridded
               law_grid;
             prop "past_band steps any group size out of the band" banded
               law_band;
             prop ~count:500
               "divisor's word divides any unsigned 32-bit integer" divided
               law_divisor;
             test "grids step past the measured band" test_grid_band;
           ];
         group "constants"
           [
             test "a constant is uploaded once accepted, then kept"
               test_constant_kept;
             test "a constant past the bound is the call's own"
               test_constant_own;
             test "bound_constants refuses bounds out of range"
               test_constant_bound;
             test "a constant has one slot a key and no writer"
               test_constant_slot;
             test "a constant not of its length raises" test_constant_length;
             test "a call that raises frees its own buffers" test_raise_frees;
             test "a device keeps 16 constants" test_constant_count;
             test "a dropped constant is uploaded again" test_constant_evicted;
             domains_constants;
           ];
         group "staging"
           [ prop "stage copies all but a gap's words" staging law_stage ];
         group "transforms"
           [
             test "keep at most 132 MiB of device memory for later calls"
               test_kept_tables;
             test "in a loop hold one call's workspace"
               test_transform_workspace;
             test "in a loop return each table the device does not keep"
               test_own_table;
             test "of more than 8 launches keep their submission"
               test_transform_kept;
           ];
         group "products"
           [
             prop "a product is its exact sum on exact values" product
               law_product;
             prop ~examples:[ skipped_tiles ]
               "a product has the bits of its stated association" bits law_bits;
           ];
         group "matrix units"
           [
             prop ~count:200 "an instruction's outputs are the model's"
               unit_case law_units;
             test "a step whose total rounds to zero gives +0" test_step_zero;
             prop ~count:1000
               "a step of normal factors is within γ(m + 1) of its exact sum"
               step_case law_step;
             prop ~count:1000
               "a step of subnormal factors is within its derived bound"
               sub_case law_sub_step;
             prop ~count:300
               "an output of normal factors is within its error factor"
               chain_case law_chain;
             prop
               ~examples:[ (64, 17, 17, 8192); (907, 17, 17, 32_768) ]
               "a call on the units takes the stated tile and ranges"
               (Gen.with_pp pp_uextents uextents)
               law_unit_plan;
             prop ~count:600 ~examples:[ past_z; tiny_sums; tiny_sums_init ]
               "a contraction on the units has the model's bits" ucall
               law_unit_call;
             test "a windowed call keeps its submission past 8 launches"
               test_windows_kept;
           ];
         group "copies"
           [
             prop "a copy keeps the bits around its destination" around
               law_around;
           ];
         group "loops against nx.cpu"
           [
             prop ~count:1 ~examples:looped_cases
               "reductions and scans past a launch's first round have \
                nx.cpu's bits"
               (Gen.with_pp pp_looped (Gen.of_list looped_cases))
               law_looped;
             prop ~count:1 ~examples:network_cases
               "network scans of slices around runs' edges have nx.cpu's \
                bits"
               (Gen.with_pp pp_looped (Gen.of_list network_cases))
               law_looped;
             prop ~count:1 ~examples:looped_cases
               "reductions and scans at 1 and 7 groups have the bits of \
                Plan.grid's count"
               (Gen.with_pp pp_looped (Gen.of_list looped_cases))
               law_grid_free;
             prop ~count:1 ~examples:map_grids
               "maps past the slow band and the most groups have nx.cpu's \
                bits"
               (Gen.with_pp Format.pp_print_int (Gen.of_list map_grids))
               law_map_grid;
             prop ~count:24 ~examples:plain_scan_examples
               "plain scans have nx.cpu's bits, chunk by chunk"
               (Gen.with_pp pp_plain_scan plain_scan_gen)
               law_plain_scan;
           ];
         group "loops memory" memory_laws;
         group "kinds"
           [
             prop ~count:1 ~examples:kind_cases
               "every kind has nx_kinds.h's bits at float and complex dtypes"
               (Gen.with_pp pp_kind_case (Gen.of_list kind_cases))
               law_kinds;
             test "which NaN float arithmetic keeps" probe_nans;
           ];
         group "harness"
           [
             test "arguments of 256 B, 1 KiB, 4 KiB and 8 KiB" test_arguments;
             test "the last group to take a ticket sees every part"
               test_tickets;
             test "grids past 65,535 groups along x, y and z" test_grids;
             prop ~count:4 "device.h's divide gives each drawn quotient"
               divided_pairs law_device_divide;
             prop ~count:8 "device.h's split gives each index's coordinate"
               split_cases law_split;
           ];
         group "bench harness"
           [
             test "generate draws by seed and index, in range" test_draws;
             test "floor_copy writes its inputs' XOR" test_floor_copy;
             test "floor_read's words XOR to its input's" test_floor_read;
             test "calls behind a hold run once released" test_hold;
             test "device time lies within the host's" test_device_time;
             test "a call's device time lies within the host's" test_call_time;
           ];
       ])
