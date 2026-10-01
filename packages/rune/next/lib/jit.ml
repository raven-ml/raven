(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Ops = Tolk_next.Ops
module Op = Tolk_next.Op
module Engine = Tolk_next_engine
module Ptree = Nx.Ptree
module Repr = Nx.Repr
module Placement = Nx.Placement
module View = Nx_array.View
module B = Nx_device.Buffer
module Claim = Nx_device.Buffer.Claim

let debug =
  match Sys.getenv_opt "RUNE_JIT_DEBUG" with
  | None | Some ("" | "0") -> false
  | Some _ -> true

let report fmt =
  if debug then Printf.eprintf ("rune.jit: " ^^ fmt ^^ "\n%!")
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

let disk = Nx.Device.of_runtime Nx_device.disk

let leaf (Nx.P t) =
  let t =
    match Nx.Placement.devices (Nx.placement t) with
    | [ d ] when Nx.Device.equal d disk -> Nx.place Nx.Placement.host t
    | _ -> t
  in
  let x = Nx.P t in
  let buffers, view = Lower.storage t in
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

(* The settings of tolk a program depends on that a caller may change around a
   call: the search's width, unoptimised kernels, and profiled batches. *)
type settings = { beam : int; noopt : bool; profiled : bool }

(* [settings ~beam ()] are the settings a call compiles with: [beam], or else
   the width tolk's lowering reads, [JITBEAM]'s or else [BEAM]'s. *)
let settings ?beam () =
  let module H = Tolk_next.Helpers in
  {
    beam =
      (match beam with
      | Some beam -> beam
      | None -> H.getenv "JITBEAM" (H.Context_var.value H.beam));
    noopt = H.Context_var.value H.noopt;
    profiled = H.Context_var.value H.debug >= 2;
  }

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

let pp_settings ppf s =
  Format.fprintf ppf "BEAM=%d NOOPT=%b profiled=%b" s.beam s.noopt s.profiled

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
      differ pp_settings k.settings k'.settings
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
  results : result array;
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
   what the program still reads of it: first the indexed writes that read the
   leaf at their own index, then the results that do, then, in the order they
   were traced, those that do not read it. It is each result's leaf, or [-1]. *)
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
  let written j =
    List.exists (fun (w : Lower.write) -> w.result == nodes.(j)) writes
  in
  pass all (fun j i -> written j && own j i);
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

(* The path of each leaf of [args], and whether its argument is consumed. *)
let paths args_s roles args =
  let at = List.rev (Ptree.fold args_s (fun p _ acc -> p :: acc) args []) in
  let consumed p =
    match Ptree.Path.segments p with
    | Ptree.Path.Index k :: _ -> List.nth roles k = Ptree.Consumed
    | _ -> false
  in
  ( Array.of_list (List.map Ptree.Path.to_string at),
    Array.of_list (List.map consumed at) )

(* [compile args_s result_s g args leaves ~paths ~consumed] traces [g] at
   [args], whose leaves are [leaves], and compiles and links its program. *)
let compile ~beam ?parallel (type a r) (args_s : a Ptree.t)
    (result_s : r Ptree.t) (g : a -> r) (args : a) leaves ~paths ~consumed =
  let s =
    Lower.scope ~renderer:(fun d -> Engine.renderer (Nx.Device.runtime d))
  in
  let slots = Array.map (fun _ -> Ops.unique_num ()) leaves in
  let params =
    Array.mapi
      (fun i { x = Nx.P t; _ } -> Nx.P (Lower.param s ~slot:slots.(i) t))
      leaves
  in
  let y =
    span "trace" (fun () ->
        Staged.install s (fun () ->
            g (Ptree.rebuild args_s ~like:args (Array.to_list params))))
  in
  let named =
    Array.of_list
      (List.rev (Ptree.fold result_s (fun p t acc -> (p, Nx.P t) :: acc) y []))
  in
  let ys = Array.map snd named in
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
    Array.map (fun (Nx.P t) -> lazy (Staged.reach ~from:(Lower.uop t))) params
  in
  let lent =
    lend ~leaves:(Array.length leaves) ~fits
      ~reads:(fun i -> Lazy.force reads.(i))
      ~writes:(Lower.writes s) nodes ys
  in
  let leaf_nodes = Array.map (fun (Nx.P t) -> Lower.uop t) params in
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
          then Some w.regions
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
      ( Tolk_next.Prepare.contiguous_view nodes.(j),
        Tolk_next.Prepare.contiguous_view target )
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
               && Tolk_next.Dtype.equal (Ops.dtype b) (Ops.dtype t)
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
          let sint = function Ops.Sym u -> Ops.Sym (value u) | d -> d in
          let bounds =
            List.map (Option.map (fun (lo, hi) -> (sint lo, sint hi)))
          in
          let written =
            match regions j with
            | None -> [ Ops.store target (value nodes.(j)) ]
            | Some regions ->
                List.map
                  (fun (r : Lower_index.region) ->
                    let padded =
                      match r.padding with
                      | None -> target
                      | Some padding -> Ops.pad target (bounds padding)
                    in
                    Ops.store
                      (Ops.shrink padded (bounds r.bounds))
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
  let sink = Ops.substitute (Ops.sink (List.rev !stores)) !taken in
  let buffers = Hashtbl.create 16 in
  List.iter
    (fun u ->
      match Ops.arg u with
      | Ops.Param p when Ops.op u = Op.Buffer ->
          Hashtbl.replace buffers p.slot u
      | _ -> ())
    (Ops.toposort sink);
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
            fst
              (Tolk_next.Schedule.create_linear_with_vars ~capturing:true sink))
      in
      (* Each kernel asks for a search of width [beam], and one of at least 1 is
         searched, each candidate timed on the device its renderer targets, on
         as many domains as [parallel] gives or the [PARALLEL] setting. *)
      let search width k =
        report "searched a kernel at width %d" width;
        let name = (Tolk_next.Postrange.Scheduler.ren k).target.device in
        let search () =
          Tolk_next.Search.beam_search
            ~measure:(fun ~cold ~vars prg ->
              Engine.measure ~cold ~vars ~devices name prg)
            width k
        in
        match parallel with
        | None -> search ()
        | Some p ->
            Tolk_next.Helpers.context
              [ B (Tolk_next.Helpers.parallel, p) ]
              search
      in
      let linear =
        span "compile" (fun () ->
            Tolk_next.Jit.jit_lower ~beam ~search
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
      (Array.to_list (Array.map (fun r -> r.like) results))
  in
  let p =
    {
      linked;
      slots = Array.map index slots;
      consumed;
      paths;
      results;
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
    (fun d -> B.create (Nx.Device.runtime d) (Nx_dtype.Scalar.of_dtype dt) n)
    (Placement.devices at)

let value r bufs =
  let (Nx.P t) = r.like in
  let dtype = Nx.dtype t in
  if Placement.equal r.at Placement.host then
    Nx.P
      (Repr.host { dtype; view = View.create r.shape; buffer = List.hd bufs })
  else
    Nx.P
      (Repr.Placed.v r.at dtype
         (View.create (local r.at r.shape))
         (Repr.Storage.v r.at bufs))

(* The buffers a call claims for a leaf: the runs it binds, or its storage's
   buffers when it binds none. *)
let claimed l = match l.runs with [] -> l.buffers | runs -> runs

(* [run entry p leaves] runs [p] on [leaves] under claims on their memory. A
   consumed leaf whose buffers span their memory and that no other live program
   captures is consumed. It is lent to its result if it also covers its storage
   and its memory is exclusive, and then consumed before the run, so that no
   work queued on its old handles reads the write. Otherwise its result is a
   copy, and it is consumed after the run. A leaf over a window of its memory,
   or captured by another program, is copied and stays live. *)
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
  if debug then
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
  p.rebuild (Array.to_list (Array.map2 value p.results results))

module Programs = Memo.Make (struct
  type t = key

  let equal = same_key
  let hash = hash
end)

let compiled ?beam ?parallel (type a r) entry (args_s : a Ptree.t)
    (result_s : r Ptree.t) roles (g : a -> r) : a -> r =
  let table = Programs.create () and last = Atomic.make None in
  fun args ->
    if Nx.Op.intercepted () then g args
    else
      let ts, skeleton = Ptree.flatten args_s args in
      let leaves = Array.of_list (List.map leaf ts) in
      let key =
        {
          skeleton;
          settings = settings ?beam ();
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
            compile ~beam:key.settings.beam ?parallel args_s result_s g args
              leaves ~paths ~consumed)
      in
      Atomic.set last (Some key);
      check entry p.consumed p.paths leaves;
      run entry p leaves

let jit ?beam ?parallel entry s f =
  let (Structure.Signature u) = Structure.signature entry s in
  u.curry (compiled ?beam ?parallel entry u.args u.result u.roles (u.apply f))
