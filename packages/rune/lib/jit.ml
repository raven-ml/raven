(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Ops = Tolk.Ops
module Op = Tolk.Op
module Engine = Tolk_engine
module Ptree = Nx.Ptree
module Repr = Nx.Repr
module Placement = Nx.Placement
module View = Nx_array.View
module B = Nx_device.Buffer
module Claim = Nx_device.Buffer.Claim

let debug = Tolk.Setting.bool ~reach:Process "RUNE_JIT_DEBUG" false
let debugging () = Tolk.Setting.value debug

let report fmt =
  if debugging () then Printf.eprintf ("rune.jit: " ^^ fmt ^^ "\n%!")
  else Printf.ifprintf stderr fmt

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ints a = String.concat "; " (List.map string_of_int (Array.to_list a))

(* Leaves *)

(* What a program depends on of a leaf: what [Lower.param] reads. *)
type layout = {
  dtype : string;
  shape : int array;
  at : Placement.t;
  view : View.t option; (* Its view over its run, if it reads one. *)
  phases : int list; (* Where its run starts within 16 bytes, per device. *)
}

(* A leaf as its parameter binds it: the value, read on the host for one on the
   disk, whose pages it borrows; its view, its storage's buffers, the runs of
   them it binds and its layout. *)
type leaf = {
  x : Nx.packed;
  view : View.t;
  buffers : B.t list;
  runs : B.t list;
  layout : layout;
}

let disk = Nx_device.disk

let leaf (Nx.P t) =
  let t =
    match Nx.Placement.devices (Nx.placement t) with
    | [ d ] when Nx_device.equal (Nx.Device.memory d) disk ->
        Nx.place Nx.Placement.host t
    | _ -> t
  in
  let x = Nx.P t in
  let buffers, view = Nx.shards t in
  let layout =
    {
      dtype = Nx_dtype.to_string (Nx.dtype t);
      shape = Nx.shape t;
      at = Nx.placement t;
      view = None;
      phases = [];
    }
  in
  match Lower.dtype (Nx.dtype t) with
  | Some tdt when View.numel view > 0 ->
      let start, _ = Lower.span tdt view in
      let layout =
        {
          layout with
          view = Some (Lower.within tdt view);
          phases = List.map (fun b -> Lower.phase tdt b start) buffers;
        }
      in
      let runs = List.map (Lower.run tdt view) buffers in
      { x; view; buffers; runs; layout }
  | _ -> { x; view; buffers; runs = []; layout }

(* Whether [l]'s view covers its storage, C-contiguous from its first element: a
   result written over the storage is then the whole of it. *)
let covers l =
  let length = match l.buffers with b :: _ -> B.length b | [] -> 0 in
  View.is_c_contiguous l.view
  && View.offset l.view = 0
  && View.numel l.view = length

(* Whether a leaf of layout [l] is its run, C-contiguous from the run's first
   element: a result of its size written over the run is then the whole of it,
   and a copy of the run stands for it. *)
let starts_its_run (l : layout) =
  match l.view with
  | Some v -> View.is_c_contiguous v && View.offset v = 0
  | None -> false

(* Keys *)

(* The settings a program depends on that a caller may change around a call:
   each setting that shapes what tolk compiles ([Tolk.Setting.shaping]), whether
   the program's batches stamp their kernels for the engine's reports
   ([Engine.profile]), and the counters and traces of the profile being taken,
   which a device's batches count and trace. *)
type settings = {
  shaping : (string * string) list;
  profile : Tolk.Hcq2.profile;
  counters : string list;
  traced : bool;
}

(* [shaping beam] reads tolk's [shaping] for a compiled function searched at the
   width [beam]. An explicit width stands for what [BEAM] and [JITBEAM] decide:
   [BEAM]'s entry holds it and [JITBEAM]'s is left out, so neither setting
   retraces. The entries are made again only when tolk's change, which keeps a
   replay from allocating them. *)
let shaping = function
  | None -> Tolk.Setting.shaping
  | Some width ->
      let beam = Tolk.Setting.key Tolk.Setting.beam
      and jitbeam = Tolk.Setting.key Tolk.Setting.jitbeam in
      let entry ((k, _) as e) =
        if String.equal k beam then Some (k, string_of_int width)
        else if String.equal k jitbeam then None
        else Some e
      in
      let last = Atomic.make ([], []) in
      fun () ->
        let entries = Tolk.Setting.shaping () in
        let seen, widened = Atomic.get last in
        if seen == entries then widened
        else
          let widened = List.filter_map entry entries in
          Atomic.set last (entries, widened);
          widened

let settings shaping =
  {
    shaping = shaping ();
    profile = Engine.profile ();
    counters = Nx_device.Profile.counters ();
    traced = Nx_device.Profile.traced ();
  }

(* [s]'s settings by name, each value printed so that two values print alike
   only if they are equal. *)
let entries s =
  s.shaping
  @ [
      ( "profile",
        match s.profile with Stamped -> "stamped" | Unstamped -> "unstamped" );
      ( "counters",
        "["
        ^ String.concat "; " (List.map (Printf.sprintf "%S") s.counters)
        ^ "]" );
      ("traced", string_of_bool s.traced);
    ]

type key = {
  skeleton : Ptree.Skeleton.t;
  settings : settings;
  layouts : layout list;
}

let same_layout a b =
  String.equal a.dtype b.dtype
  && a.shape = b.shape && Placement.equal a.at b.at && a.view = b.view
  && a.phases = b.phases

let same_key k k' =
  Ptree.Skeleton.equal k.skeleton k'.skeleton
  && k.settings = k'.settings
  && List.equal same_layout k.layouts k'.layouts

let hash k =
  Hashtbl.hash_param 256 512
    (Ptree.Skeleton.hash k.skeleton, List.map (fun l -> l.shape) k.layouts)

(* A layout's parts, in the order a retrace names the first that differs. *)
let parts l =
  [
    l.dtype;
    "shape [" ^ ints l.shape ^ "]";
    Format.asprintf "at %a" Placement.pp l.at;
    "strides [" ^ ints (Option.fold ~none:[||] ~some:View.strides l.view) ^ "]";
    (match Option.fold ~none:0 ~some:View.offset l.view with
    | 1 -> "1 element into its run"
    | n -> Printf.sprintf "%d elements into its run" n);
    "a run at [" ^ ints (Array.of_list l.phases) ^ "] bytes past 16";
  ]

(* The first difference between the key [k] and the previous key [k'], whose
   leaves are at [paths]. *)
let difference paths k k' =
  let differ pp a b =
    Format.asprintf "%a here, %a in the previous key" pp a pp b
  in
  match
    Ptree.Skeleton.diff ~this:"here" k.skeleton ~that:"in the previous key"
      k'.skeleton
  with
  | Some m -> m
  | None when k.settings <> k'.settings ->
      let here = entries k.settings and previous = entries k'.settings in
      let value name e =
        Option.value (List.assoc_opt name e) ~default:"unset"
      in
      let differs (name, _) = value name here <> value name previous in
      Option.fold ~none:"other settings than in the previous key"
        ~some:(fun (name, _) ->
          Printf.sprintf "%s=%s here, %s=%s in the previous key" name
            (value name here) name (value name previous))
        (List.find_opt differs (here @ previous))
  | None ->
      let rec first i = function
        | l :: ls, l' :: ls' when same_layout l l' -> first (i + 1) (ls, ls')
        | l :: _, l' :: _ ->
            let differs (a, b) = not (String.equal a b) in
            let a, b =
              Option.value ~default:("a placement", "another")
                (List.find_opt differs (List.combine (parts l) (parts l')))
            in
            paths.(i) ^ ": " ^ differ Format.pp_print_string a b
        | _ -> "an equal key"
      in
      first 0 (k.layouts, k'.layouts)

(* Programs *)

(* Pins. A program reads its captures on every call, so no call lends their
   memory to a result: it copies instead. The pins are the captures of the
   programs alive, held weakly through the list each program keeps. *)
module Pins = struct
  let lock = Mutex.create ()
  let pinned : B.t list Weak.t list ref = ref []

  let add = function
    | [] -> ()
    | captured ->
        let w = Weak.create 1 in
        Weak.set w 0 (Some captured);
        Mutex.protect lock (fun () ->
            pinned := w :: List.filter (fun w -> Weak.check w 0) !pinned)

  let hold buffers =
    let held w =
      match Weak.get w 0 with
      | Some captured ->
          List.exists (fun c -> List.exists (B.overlaps c) buffers) captured
      | None -> false
    in
    Mutex.protect lock (fun () -> List.exists held !pinned)
end

type output = Empty | Fresh of int | Lent of int

(* A result leaf: a tensor of its dtype, its shape, placement and storage. *)
type result = {
  like : Nx.packed;
  shape : int array;
  at : Placement.t;
  out : output;
  name : string;
}

type 'r program = {
  linked : Engine.t option;
  slots : int array; (* By leaf: its parameter's slot, or [-1]. *)
  consumed : bool array; (* By leaf. *)
  paths : string array; (* By leaf. *)
  results : result array; (* The function's, then a scalar per check. *)
  checks : (int array * (int array -> string)) array;
      (* Each check's shape and message. *)
  captured : B.t list; (* The runs its captures bind. *)
  rebuild : Nx.packed list -> 'r;
}

let span phase f = Nx_device.Profile.span ("rune.jit: " ^ phase) f
let numel shape = Array.fold_left ( * ) 1 shape

let id (Nx.P y) =
  match Repr.v y with
  | Traced t -> Repr.Traced.id t
  | Host _ | Placed _ -> max_int

(* [lend ~leaves ~fits ~reads ~writes nodes ys] pairs results with the consumed
   leaves [fits] allows, where writing the result over the leaf cannot change
   what the program still reads of it: first the indexed writes into a value
   that reads the leaf at its own index, then the results that do, then, in the
   order they were traced, those that do not read it. A write is its value with
   some elements replaced, so it reads the leaf as the value it writes into
   does. It is each result's leaf, or [-1]. *)
let lend ~leaves ~fits ~reads ~writes nodes ys =
  let lent = Array.make (Array.length nodes) (-1) in
  let taken = Array.make leaves false in
  let pass order ok =
    List.iter
      (fun j ->
        if lent.(j) < 0 then
          match
            List.find_opt
              (fun i -> (not taken.(i)) && fits i j && ok j i)
              (List.init leaves Fun.id)
          with
          | Some i ->
              lent.(j) <- i;
              taken.(i) <- true
          | None -> ())
      order
  in
  let all = List.init (Array.length nodes) Fun.id in
  let own j i = reads i nodes.(j) = Staged.Own in
  let written j i =
    List.exists
      (fun (w : Lower.write) ->
        w.result == nodes.(j) && reads i w.into = Staged.Own)
      writes
  in
  pass all written;
  pass all own;
  pass
    (List.stable_sort (fun j k -> Int.compare (id ys.(j)) (id ys.(k))) all)
    (fun j i -> reads i nodes.(j) = Staged.Apart);
  lent

(* [rebuilder ()] is [(value, assign)]: [value u] is [u] with each node [assign]
   paired with an image replaced by it, through one memo, so that a node several
   values reach is rebuilt once. Every node is assigned before [value] reaches a
   node above it: [assign] raises [Invalid_argument] for a node a rebuild read
   below another, whose rebuild would keep the node unreplaced. *)
let rebuilder () =
  let images = Ops.Tbl.create 16
  and memo = Ops.Tbl.create 64
  and below = Ops.Tbl.create 64 in
  let rec value u =
    match Ops.Tbl.find_opt images u with
    | Some v -> v
    | None -> (
        match Ops.Tbl.find_opt memo u with
        | Some v -> v
        | None ->
            let src = Ops.src u in
            let src' = List.map value src in
            List.iter (fun s -> Ops.Tbl.replace below s ()) src;
            let v =
              if List.for_all2 ( == ) src src' then u
              else Ops.replace u ~src:src'
            in
            Ops.Tbl.add memo u v;
            v)
  in
  let assign u v =
    if Ops.Tbl.mem below u then
      invalid_arg "Jit: a result is assigned after a rebuild read it";
    Ops.Tbl.replace images u v
  in
  (value, assign)

(* [ordered ~leaves nodes lent] is [lent] with pairs given up until the stores
   have an order, and that order of the results. A lent result is written over
   its leaf as an assignment: another result reads it through its store, so it
   runs after that store, and a result that reads the leaf itself runs before
   it. Pairs whose results must each run before the other's store are given up,
   the latest result first. *)
let ordered ~leaves nodes lent =
  let n = Array.length nodes in
  let owner i = Option.value ~default:(-1) (Array.find_index (( = ) i) lent) in
  let rec settle () =
    let memo = Ops.Tbl.create 64 in
    (* The lent leaves a node reads, and the lent results it reads. *)
    let rec reads root u =
      match Array.find_index (( == ) u) nodes with
      | Some j when u != root && lent.(j) >= 0 -> ([], [ j ])
      | _ -> (
          match Array.find_index (( == ) u) leaves with
          | Some i when owner i >= 0 -> ([ i ], [])
          | _ -> (
              match Ops.Tbl.find_opt memo u with
              | Some r -> r
              | None ->
                  let r =
                    List.fold_left
                      (fun (l, j) s ->
                        let l', j' = reads root s in
                        ( List.sort_uniq Int.compare (l' @ l),
                          List.sort_uniq Int.compare (j' @ j) ))
                      ([], []) (Ops.src u)
                  in
                  if u != root then Ops.Tbl.add memo u r;
                  r))
    in
    (* [before.(k)] are the results [k] runs before. *)
    let before = Array.make n [] in
    Array.iteri
      (fun k u ->
        let old, fresh = reads u u in
        List.iter
          (fun i -> if owner i <> k then before.(k) <- owner i :: before.(k))
          old;
        List.iter (fun j -> before.(j) <- k :: before.(j)) fresh)
      nodes;
    (* A depth-first walk: the results in an order of their stores, or a cycle,
       from a result back to itself. *)
    let state = Array.make n `New and order = ref [] in
    let rec visit k =
      match state.(k) with
      | `Done -> None
      | `Open -> Some [ k ]
      | `New -> (
          state.(k) <- `Open;
          match List.find_map visit before.(k) with
          | Some c -> Some (k :: c)
          | None ->
              state.(k) <- `Done;
              order := k :: !order;
              None)
    in
    match List.find_map visit (List.init n Fun.id) with
    | None -> (lent, !order)
    | Some path ->
        let last = List.nth path (List.length path - 1) in
        let rec from = function
          | k :: rest when k <> last -> from rest
          | c -> c
        in
        let j =
          List.fold_left max (-1)
            (List.filter (fun k -> lent.(k) >= 0) (from path))
        in
        lent.(j) <- -1;
        settle ()
  in
  settle ()

(* [local at shape] is the shape of each device's window of a value of [shape]
   at [at]. *)
let local at shape =
  let d = List.hd (Placement.devices at) in
  Array.map (fun (lo, hi) -> hi - lo) (Placement.window at shape d)

(* The path of each leaf of [args], and whether its argument is consumed: never
   without [roles], as for a function a transformation derives. *)
let paths args_s roles args =
  let at = List.rev (Ptree.fold args_s (fun p _ acc -> p :: acc) args []) in
  let consumed p =
    match (roles, Ptree.Path.segments p) with
    | Some roles, Ptree.Path.Index k :: _ -> List.nth roles k = Ptree.Consumed
    | _ -> false
  in
  ( Array.of_list (List.map Ptree.Path.to_string at),
    Array.of_list (List.map consumed at) )

(* [compile args_s result_s g args leaves ~paths ~consumed] traces [g] at
   [args], whose leaves are [leaves], and compiles and links its program, its
   batches stamping their kernels as [profile] says. *)
let compile ?beam ?parallel ~profile (type a r) (args_s : a Ptree.t)
    (result_s : r Ptree.t) (g : a -> r) (args : a) leaves ~paths ~consumed =
  let s = Lower.scope ~renderer:Engine.renderer in
  let slots = Array.map (fun _ -> Ops.unique_num ()) leaves in
  let params =
    Array.mapi
      (fun i { x = Nx.P t; _ } -> Nx.P (Lower.param s ~slot:slots.(i) t))
      leaves
  in
  let y =
    span "trace" (fun () ->
        Fun.protect
          ~finally:(fun () -> Lower.finish s)
          (fun () ->
            Staged.install s (fun () ->
                g (Ptree.rebuild args_s ~like:args (Array.to_list params)))))
  in
  let named =
    Array.of_list
      (List.rev (Ptree.fold result_s (fun p t acc -> (p, Nx.P t) :: acc) y []))
  in
  (* A check is answered when the program has run, from the index of its first
     failure, which the program returns after the function's results. *)
  let checks = Array.of_list (Lower.checks s) in
  let ys =
    Array.append (Array.map snd named)
      (Array.map (fun (c : Lower.check) -> Nx.P c.first) checks)
  in
  let user = Array.length named in
  let nodes = Array.map (fun (Nx.P t) -> Lower.value s t) ys in
  let fits i j =
    let (Nx.P x) = leaves.(i).x in
    let (Nx.P y) = ys.(j) in
    let l = leaves.(i).layout in
    consumed.(i) && leaves.(i).runs <> [] && starts_its_run l
    && Nx_dtype.equal (Nx.dtype x) (Nx.dtype y)
    && numel l.shape = numel (Nx.shape y)
    && Placement.equal l.at (Nx.placement y)
    && (l.shape = Nx.shape y || List.length (Placement.devices l.at) = 1)
    && List.for_all (( = ) 0) l.phases
  in
  let reads =
    Array.map (fun (Nx.P t) -> lazy (Staged.reach ~from:(Lower.uop s t))) params
  in
  let lent =
    lend ~leaves:(Array.length leaves) ~fits
      ~reads:(fun i -> Lazy.force reads.(i))
      ~writes:(Lower.writes s) nodes ys
  in
  let leaf_nodes = Array.map (fun (Nx.P t) -> Lower.uop s t) params in
  let lent, order = ordered ~leaves:leaf_nodes nodes lent in
  (* A result lent to the leaf its write writes into stores only the regions the
     write writes. *)
  let regions j =
    if lent.(j) < 0 then None
    else
      List.find_map
        (fun (w : Lower.write) ->
          if
            w.result == nodes.(j)
            && w.into == leaf_nodes.(lent.(j))
            && w.regions <> []
          then Some (w.into, w.regions)
          else None)
        (Lower.writes s)
  in
  (* The results are stored in that order, rebuilt through one substitution: a
     result that reads a lent one reads its store, and a node results share
     stays one node. *)
  let value, assign = rebuilder ()
  and stores = ref []
  and outs = Array.make (Array.length ys) Empty in
  (* A fresh result that is all of a buffer the program makes, in order, has the
     program write it in its storage in place of that buffer, and no copy. *)
  let taken = ref [] in
  let made b =
    Ops.op b = Op.Buffer
    && (match Ops.arg b with
      | Ops.Param p -> not (Array.mem p.slot slots)
      | _ -> false)
    && (not (List.exists (fun (c, _) -> c == b) (Lower.captures s)))
    && not (List.mem_assq b !taken)
  in
  let take j target =
    let (Nx.P y) = ys.(j) in
    let n = numel (Nx.shape y) in
    match
      ( Tolk.Prepare.contiguous_view nodes.(j),
        Tolk.Prepare.contiguous_view target )
    with
    | Some (a, 0), Some (t, 0) when Ops.op a = Op.After && Ops.op t = Op.Buffer
      -> (
        let b = List.hd (Ops.src a) in
        (* The result's own node over the buffer, under its views: the view's is
           rebuilt, and the program would run what it waits on twice. *)
        let rec own u =
          if Ops.op u = Op.After then Some u
          else if Op.Set.mem (Ops.op u) Op.Set.movement || Ops.op u = Op.Bitcast
          then own (List.hd (Ops.src u))
          else None
        in
        match own nodes.(j) with
        | Some a
          when List.hd (Ops.src a) == b
               && made b
               && Ops.max_numel b = n
               && Ops.max_numel t = n
               && Tolk.Dtype.equal (Ops.dtype b) (Ops.dtype t)
               && Ops.device b = Ops.device t ->
            taken := (b, t) :: !taken;
            stores := value a :: !stores;
            true
        | _ -> false)
    | _ -> false
  in
  List.iter
    (fun j ->
      let (Nx.P y) = ys.(j) in
      let store slot =
        let target =
          Lower.output s ~slot (Nx.placement y) (Nx.dtype y) (Nx.shape y)
        in
        if target != nodes.(j) then begin
          let written =
            match regions j with
            | None -> [ Ops.store target (value nodes.(j)) ]
            | Some (into, regions) ->
                List.map
                  (fun (r : Lower_index.region) ->
                    Ops.store
                      (value
                         (Ops.substitute ~calls:Skip ~pass:Fixed_point r.dest
                            [ (into, target) ]))
                      (value r.value))
                  regions
          in
          let stored = Ops.after target written in
          stores := stored :: !stores;
          if lent.(j) >= 0 then assign nodes.(j) stored
        end
      in
      if numel (Nx.shape y) = 0 then ()
      else if lent.(j) >= 0 then begin
        store slots.(lent.(j));
        outs.(j) <- Lent lent.(j)
      end
      else
        let k = Ops.unique_num () in
        let target =
          Lower.output s ~slot:k (Nx.placement y) (Nx.dtype y) (Nx.shape y)
        in
        if not (Option.is_none (regions j) && take j target) then store k;
        outs.(j) <- Fresh k)
    order;
  let results =
    Array.mapi
      (fun j (Nx.P y) ->
        let shape = Nx.shape y and at = Nx.placement y and dt = Nx.dtype y in
        let out = outs.(j) in
        let name =
          if j >= user then "a check"
          else
            match Ptree.Path.to_string (fst named.(j)) with
            | "" -> "the result"
            | path -> "result " ^ path
        in
        (* A structure checks its leaves' shapes; a broadcast scalar has the
           shape without the bytes. *)
        let like = Nx.P (Nx.broadcast_to shape (Nx.zeros dt [||])) in
        { like; shape; at; out; name })
      ys
  in
  let sink =
    Ops.substitute ~calls:Skip ~pass:Fixed_point
      (Ops.sink (List.rev !stores))
      !taken
  in
  let buffers = Hashtbl.create 16 in
  List.iter
    (fun u ->
      match Ops.arg u with
      | Ops.Param p when Ops.op u = Op.Buffer ->
          Hashtbl.replace buffers p.slot u
      | _ -> ())
    (Ops.toposort ~calls:Enter sink);
  let fresh =
    List.filter_map
      (fun r -> match r.out with Fresh k -> Some k | _ -> None)
      (Array.to_list results)
  in
  let order = List.filter (Hashtbl.mem buffers) (Array.to_list slots @ fresh) in
  let index k = Option.value ~default:(-1) (List.find_index (( = ) k) order) in
  let slot_of u = match Ops.arg u with Ops.Param p -> p.slot | _ -> -1 in
  let bound =
    List.filter
      (fun (u, _) -> Hashtbl.mem buffers (slot_of u))
      (Lower.captures s)
  in
  let linked =
    if !stores = [] then None
    else
      let devices = Lower.engine s in
      let linear =
        span "schedule" (fun () ->
            fst (Tolk.Schedule.create_linear_with_vars ~capturing:true sink))
      in
      (* Each kernel asks for a search of width [beam], or else of the width
         tolk's settings give, and one of at least 1 is searched, each candidate
         timed on the device its renderer targets, on as many domains as
         [parallel] gives or the [PARALLEL] setting. The candidates, programs of
         one kernel, run on the slots of the first one timed: a search the disk
         cache answers allocates nothing. *)
      let search width k =
        report "searched a kernel at width %d" width;
        let name = (Tolk.Postrange.Scheduler.ren k).target.device in
        let slots = ref None in
        let link prg = Engine.link_program ~devices name prg in
        let time ~vars s =
          let slots =
            match !slots with
            | Some b -> b
            | None ->
                let b = Engine.slots s in
                slots := Some b;
                b
          in
          Engine.time ~vars s slots
        in
        let search () =
          Tolk.Search.beam_search ~link ~time ~clock:Engine.clock width k
        in
        match parallel with
        | None -> search ()
        | Some p -> Tolk.Setting.context [ B (Tolk.Setting.parallel, p) ] search
      in
      let linear =
        span "compile" (fun () ->
            Tolk.Jit.jit_lower ?beam ~search ~profile
              ~devices:(fun n -> (devices n).compiler)
              ~held_bufs:(List.map fst bound)
              ~inputs:(List.map (Hashtbl.find buffers) order)
              linear)
      in
      Some (span "link" (fun () -> Engine.link ~devices ~bound linear))
  in
  let results =
    Array.map
      (fun r ->
        match r.out with Fresh k -> { r with out = Fresh (index k) } | _ -> r)
      results
  in
  let like =
    Ptree.rebuild result_s ~like:y
      (Array.to_list (Array.map (fun r -> r.like) (Array.sub results 0 user)))
  in
  let p =
    {
      linked;
      slots = Array.map index slots;
      consumed;
      paths;
      results;
      checks = Array.map (fun (c : Lower.check) -> (c.shape, c.msg)) checks;
      captured = List.concat_map snd bound;
      rebuild = Ptree.rebuild result_s ~like;
    }
  in
  Pins.add p.captured;
  p

(* Calls *)

let why path =
  Printf.sprintf
    "this value was consumed at %s in a compiled call's arguments; use the \
     value the call returned"
    path

let reaches l l' =
  List.exists (fun r -> List.exists (B.overlaps r) l'.runs) l.runs

(* [check entry consumed paths leaves] refuses a consumed leaf whose storage
   another leaf reaches. *)
let check entry consumed paths leaves =
  Array.iteri
    (fun i l ->
      if consumed.(i) then
        Array.iteri
          (fun k l' ->
            if k <> i && reaches l l' then
              invalid_argf "%s: %s is consumed and %s reaches its storage" entry
                paths.(i) paths.(k))
          leaves)
    leaves

let fresh at dt shape =
  let n = numel (local at shape) in
  List.map
    (fun d -> B.create (Nx.Device.memory d) (Nx_dtype.Scalar.of_dtype dt) n)
    (Placement.devices at)

let value r bufs =
  let (Nx.P t) = r.like in
  Nx.P (Nx.of_shards r.at (Nx.dtype t) (View.create (local r.at r.shape)) bufs)

(* The buffers a call claims for a leaf: the runs it binds, or its storage's
   buffers when it binds none. *)
let claimed l = match l.runs with [] -> l.buffers | runs -> runs

(* [run entry p leaves] runs [p] on [leaves] under claims on their memory. A
   consumed leaf whose buffers span their memory and that no other live program
   captures is consumed. It is lent to its result if it also covers its storage
   and its memory is exclusive, and then consumed before the run, so that no
   work queued on its old handles reads the write. Otherwise its result is a
   copy, and it is consumed after the run. A leaf over a window of its memory,
   or captured by another program, is copied and stays live. Once [p] has run,
   the first of its checks that failed raises. *)
let run entry p leaves =
  let n = Array.length leaves in
  Array.iteri
    (fun i l ->
      if
        p.consumed.(i)
        && List.exists (fun c -> List.exists (B.overlaps c) l.runs) p.captured
      then
        invalid_argf "%s: %s is consumed and the function captures its storage"
          entry p.paths.(i))
    leaves;
  let read, donate =
    Array.fold_right
      (fun (i, l) (read, donate) ->
        if p.consumed.(i) then (read, claimed l :: donate)
        else (claimed l @ read, donate))
      (Array.mapi (fun i l -> (i, l)) leaves)
      ([], [])
  in
  Claim.with_ ~read ~donate @@ fun c ->
  let consumable =
    Array.mapi
      (fun i l ->
        p.consumed.(i)
        && List.for_all B.spans l.buffers
        && not (Pins.hold l.buffers))
      leaves
  in
  let lendable i =
    let l = leaves.(i) in
    consumable.(i) && covers l && List.for_all (Claim.exclusive c) (claimed l)
  in
  let consume i =
    List.map (Claim.consume c ~why:(why p.paths.(i))) leaves.(i).buffers
  in
  let lends = Array.make n false in
  let storage =
    Array.map
      (fun r ->
        let (Nx.P t) = r.like in
        match r.out with
        | Empty | Fresh _ -> fresh r.at (Nx.dtype t) r.shape
        | Lent i when lendable i ->
            lends.(i) <- true;
            []
        | Lent i ->
            let copy = fresh r.at (Nx.dtype t) r.shape in
            List.iter2 (fun src dst -> B.copy ~src ~dst) leaves.(i).runs copy;
            copy)
      p.results
  in
  let bufs =
    Array.mapi (fun i l -> if lends.(i) then consume i else l.runs) leaves
  in
  let results =
    Array.mapi
      (fun j r ->
        match r.out with Lent i when lends.(i) -> bufs.(i) | _ -> storage.(j))
      p.results
  in
  Option.iter
    (fun linked ->
      let slots =
        Array.make (Array.length leaves + Array.length p.results) []
      in
      Array.iteri (fun i k -> if k >= 0 then slots.(k) <- bufs.(i)) p.slots;
      Array.iteri
        (fun j r ->
          match r.out with
          | Fresh k -> slots.(k) <- results.(j)
          | Lent i ->
              (* A leaf returned as it is and read by no kernel has no slot. *)
              let k = p.slots.(i) in
              if k >= 0 then slots.(k) <- results.(j)
          | Empty -> ())
        p.results;
      Engine.run linked slots)
    p.linked;
  Array.iteri
    (fun i _ -> if consumable.(i) && not lends.(i) then ignore (consume i))
    leaves;
  if debugging () then
    Array.iteri
      (fun i _ ->
        if p.consumed.(i) then
          match
            List.find_opt
              (fun j -> p.results.(j).out = Lent i)
              (List.init (Array.length p.results) Fun.id)
          with
          | Some j ->
              report "%s -> %s %s" p.paths.(i) p.results.(j).name
                (if lends.(i) then "reused" else "copied")
          | None -> report "%s consumed, lent to no result" p.paths.(i))
      leaves;
  let values = Array.map2 value p.results results in
  let user = Array.length values - Array.length p.checks in
  Array.iteri
    (fun k (shape, msg) ->
      let first = Nx.item [] (Nx.unpack Nx.int64 values.(user + k)) in
      if Int64.to_int first < numel shape then
        invalid_arg
          (msg (Nx_array.Shape.unravel_index (Int64.to_int first) shape)))
    p.checks;
  p.rebuild (Array.to_list (Array.sub values 0 user))

module Programs = Memo.Make (struct
  type t = key

  let equal = same_key
  let hash = hash
end)

(* What a split depends on of a call: the leaves it tracks, the arguments'
   visits, and each leaf's dtype, shape and placement. The lane counts of the
   maps around the call that the split read select among a key's splits. *)
type dtype = Dtype : ('a, 'b) Nx_dtype.t -> dtype

module Plans = Memo.Make (struct
  type t = bool list * Ptree.Skeleton.t * (dtype * int array * Placement.t) list

  let equal (t, s, l) (t', s', l') =
    List.equal Bool.equal t t' && Ptree.Skeleton.equal s s'
    && List.equal
         (fun (Dtype d, shape, at) (Dtype d', shape', at') ->
           Nx_dtype.equal d d' && shape = shape' && Placement.equal at at')
         l l'

  let hash (t, s, l) =
    Hashtbl.hash_param 256 512
      (t, Ptree.Skeleton.hash s, List.map (fun (_, shape, _) -> shape) l)
end)

(* A compiler keeps the programs of one function, the compiled function's own or
   one a transformation derives from it, by key, and the compilers of the
   functions derived from it, by step. *)
type ('p, 'q) child =
  | Child :
      ('p, 'q, 'a, 'b) Construct.step * ('a, 'b) Construct.compiler
      -> ('p, 'q) child

let rec find : type p q a b.
    (p, q, a, b) Construct.step ->
    (p, q) child list ->
    (a, b) Construct.compiler option =
 fun step -> function
  | [] -> None
  | Child (step', c) :: rest -> (
      match Construct.same_step step' step with
      | Some Equal -> Some c
      | None -> find step rest)

let rec compiler : type a r.
    int option ->
    int option ->
    string ->
    Ptree.role list option ->
    (a, r) Construct.compiler =
 fun beam parallel entry roles ->
  let table = Programs.create () and last = Atomic.make None in
  let plans = Plans.create () and shaping = shaping beam in
  let children = ref [] and lock = Mutex.create () in
  let call (args_s : a Ptree.t) (result_s : r Ptree.t) (g : a -> r) (args : a) :
      r =
    let ts, skeleton = Ptree.flatten args_s args in
    let leaves = Array.of_list (List.map leaf ts) in
    let key =
      {
        skeleton;
        settings = settings shaping;
        layouts = Array.to_list (Array.map (fun l -> l.layout) leaves);
      }
    in
    let retrace () =
      Option.iter
        (fun k ->
          report "retrace: %s"
            (difference (fst (paths args_s roles args)) key k))
        (Atomic.get last)
    in
    let p =
      Programs.find table key ~miss:retrace (fun () ->
          let paths, consumed = paths args_s roles args in
          compile ?beam ?parallel ~profile:key.settings.profile args_s result_s
            g args leaves ~paths ~consumed)
    in
    Atomic.set last (Some key);
    check entry p.consumed p.paths leaves;
    run entry p leaves
  in
  let derive : type a2 r2.
      (a, r, a2, r2) Construct.step -> (a2, r2) Construct.compiler =
   fun step ->
    Mutex.protect lock @@ fun () ->
    match find step !children with
    | Some c -> c
    | None ->
        let c = compiler beam parallel entry None in
        children := Child (step, c) :: !children;
        c
  in
  let split tracked p q vjp args =
    let ts, skeleton = Ptree.flatten p args in
    let leaves =
      List.map
        (fun (Nx.P t) -> (Dtype (Nx.dtype t), Nx.shape t, Nx.placement t))
        ts
    in
    let lock, splits =
      Plans.find plans (tracked, skeleton, leaves) ~miss:ignore (fun () ->
          (Mutex.create (), ref []))
    in
    let current (axis, n) = Construct.perform (Lane_count axis) = n in
    Mutex.protect lock @@ fun () ->
    match
      List.find_opt (fun (counts, _) -> List.for_all current counts) !splits
    with
    | Some (_, split) -> split
    | None ->
        let counts, split =
          span "residuals" (fun () -> Split.plan p q vjp args)
        in
        splits := (counts, split) :: !splits;
        split
  in
  { run = call; derive; split }

let jit ?beam ?parallel entry s f =
  let (Structure.Signature u) = Structure.signature entry s in
  let compiler = compiler beam parallel entry (Some u.roles) in
  let f = u.apply f in
  u.curry (fun args ->
      Construct.perform
        (Compiled { p = u.args; q = u.result; f; args; compiler }))
