(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Ops = Tolk_next.Ops
module Op = Tolk_next.Op
module Engine = Tolk_next_engine
module Ptree = Nx.Ptree
module Repr = Nx.Repr
module Storage = Nx.Repr.Storage
module Placement = Nx.Placement
module View = Nx_array.View
module B = Nx_device.Buffer

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
  strides : int array;
  lead : int; (* Elements from its run's start to its view's first. *)
  phases : int list; (* Where its run starts within 16 bytes, per device. *)
}

(* A leaf as its parameter binds it: the value, read on the host for one on the
   disk, whose pages it borrows; the storage a call claims for a placed one;
   whether it is a host value; its view, its storage's buffers, the runs of them
   it binds and its layout. *)
type leaf = {
  x : Nx.packed;
  cell : Storage.t option;
  host : bool;
  view : View.t;
  buffers : B.t list;
  runs : B.t list;
  layout : layout;
}

let disk = Nx.Device.of_runtime Nx_device.disk

let leaf (Nx.P t) =
  let cell, host =
    match Repr.v t with
    | Host _ -> (None, true)
    | Placed r -> (Some (Repr.Placed.storage r), false)
    | Traced _ -> (None, false)
  in
  let t =
    match Nx.Placement.devices (Nx.placement t) with
    | [ d ] when Nx.Device.equal d disk -> Nx.place Nx.Placement.host t
    | _ -> t
  in
  let x = Nx.P t in
  let view =
    match Repr.v t with
    | Host a -> a.view
    | Placed r -> Repr.Placed.view r
    | Traced _ ->
        invalid_arg
          "a traced tensor has no bytes; it was used outside the trace that \
           made it"
  in
  let buffers = Lower.buffers t and shape = View.shape view in
  let layout =
    {
      dtype = Nx_dtype.to_string (Nx.dtype t);
      shape = Nx.shape t;
      at = Nx.placement t;
      strides = [||];
      lead = 0;
      phases = [];
    }
  in
  match Lower.dtype (Nx.dtype t) with
  | Some tdt when View.numel view > 0 ->
      let start, _ = Lower.span tdt view in
      let strides =
        Array.mapi
          (fun d s -> if shape.(d) = 1 then 0 else s)
          (View.strides view)
      in
      let layout =
        {
          layout with
          strides;
          lead = fst (View.extent view) - start;
          phases = List.map (fun b -> Lower.phase tdt b start) buffers;
        }
      in
      let runs = List.map (Lower.run tdt view) buffers in
      { x; cell; host; view; buffers; runs; layout }
  | _ -> { x; cell; host; view; buffers; runs = []; layout }

(* Keys *)

(* The settings of tolk a program depends on that a caller may change around a
   call: the search's width, unoptimised kernels, and profiled batches. *)
type settings = { beam : int; noopt : bool; profiled : bool }

let settings () =
  let open Tolk_next.Helpers in
  {
    beam = Context_var.value beam;
    noopt = Context_var.value noopt;
    profiled = Context_var.value debug >= 2;
  }

type key = {
  skeleton : Ptree.Skeleton.t;
  settings : settings;
  layouts : layout list;
}

let same_layout a b =
  String.equal a.dtype b.dtype
  && a.shape = b.shape && Placement.equal a.at b.at && a.strides = b.strides
  && a.lead = b.lead && a.phases = b.phases

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
    "strides [" ^ ints l.strides ^ "]";
    Printf.sprintf "%d elements into its run" l.lead;
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

(* The renderer of each device, made once per trace. *)
let renderers () =
  let made = ref [] in
  fun d ->
    match List.assq_opt d !made with
    | Some r -> r
    | None ->
        let t = Engine.target (Nx.Device.runtime d) in
        let r =
          match Tolk_next.Device.renderer ~arch:t.arch t.device with
          | Ok r -> r
          | Error why -> failwith why
        in
        made := (d, r) :: !made;
        r

(* The engine's devices for the names of a trace, and the hosts that submit
   their work. *)
let engine_devices names =
  let named = List.map (fun (n, d) -> (n, Nx.Device.runtime d)) names in
  let hosts =
    List.fold_left
      (fun hosts (_, d) ->
        let h = Nx_device.host_of d
        and n = Nx_device.name (Nx_device.host_of d) in
        if List.mem_assoc n named || List.mem_assoc n hosts then hosts
        else (n, h) :: hosts)
      [] named
  in
  Engine.device (named @ hosts)

let numel shape = Array.fold_left ( * ) 1 shape

let id (Nx.P y) =
  match Repr.v y with
  | Traced t -> Repr.Traced.id t
  | Host _ | Placed _ -> max_int

(* How a node reads a value: not at all, at each element's own index only
   (through elementwise operations, width-preserving casts, reshapes and
   contiguous markers), or otherwise. *)
type reach = Apart | Own | Other

let keeps_index u =
  match Ops.op u with
  | Op.Cast | Op.Bitcast -> Ops.element_size u = Ops.element_size (Ops.nth u 0)
  | Op.Reshape | Op.Stage -> true
  | o -> Op.Set.mem o Op.Set.elementwise

(* [reach ~from u] is how [u] reads [from]. The partial application [reach
   ~from] walks each node once. *)
let reach ~from =
  let memo = Ops.Tbl.create 64 in
  let rec go u =
    if u == from then Own
    else
      match Ops.Tbl.find_opt memo u with
      | Some r -> r
      | None ->
          let srcs = List.map go (Ops.src u) in
          let r =
            if List.for_all (( = ) Apart) srcs then Apart
            else if keeps_index u && not (List.mem Other srcs) then Own
            else Other
          in
          Ops.Tbl.add memo u r;
          r
  in
  go

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
  let own j i = reads i nodes.(j) = Own in
  pass all (fun j i -> List.memq nodes.(j) writes && own j i);
  pass all own;
  pass
    (List.stable_sort (fun j k -> Int.compare (id ys.(j)) (id ys.(k))) all)
    (fun j i -> reads i nodes.(j) = Apart);
  lent

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
let compile (type a r) (args_s : a Ptree.t) (result_s : r Ptree.t) (g : a -> r)
    (args : a) leaves ~paths ~consumed =
  let s = Lower.scope ~renderer:(renderers ()) in
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
    consumed.(i) && leaves.(i).runs <> []
    && Nx_dtype.equal (Nx.dtype x) (Nx.dtype y)
    && numel l.shape = numel (Nx.shape y)
    && Placement.equal l.at (Nx.placement y)
    && (l.shape = Nx.shape y || List.length (Placement.devices l.at) = 1)
    && List.for_all (( = ) 0) l.phases
  in
  let reads =
    Array.map (fun (Nx.P t) -> lazy (reach ~from:(Lower.uop t))) params
  in
  let lent =
    lend ~leaves:(Array.length leaves) ~fits
      ~reads:(fun i -> Lazy.force reads.(i))
      ~writes:(Lower.writes s) nodes ys
  in
  let stores = ref [] in
  let results =
    Array.mapi
      (fun j (Nx.P y) ->
        let shape = Nx.shape y and at = Nx.placement y and dt = Nx.dtype y in
        let store slot =
          let target = Lower.output s ~slot at dt shape in
          if target != nodes.(j) then
            stores := Ops.after target [ Ops.store target nodes.(j) ] :: !stores
        in
        let out =
          if numel shape = 0 then Empty
          else if lent.(j) >= 0 then (
            store slots.(lent.(j));
            Lent lent.(j))
          else
            let k = Ops.unique_num () in
            store k;
            Fresh k
        in
        let name =
          match Ptree.Path.to_string (fst named.(j)) with
          | "" -> "the result"
          | path -> "result " ^ path
        in
        { like = Nx.P (Nx.zeros dt [| 0 |]); shape; at; out; name })
      ys
  in
  let sink = Ops.sink (List.rev !stores) in
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
      let devices = engine_devices (Lower.devices s) in
      let linear =
        span "schedule" (fun () ->
            fst
              (Tolk_next.Schedule.create_linear_with_vars ~capturing:true sink))
      in
      let linear =
        span "compile" (fun () ->
            Tolk_next.Jit.jit_lower
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
  let held = Lower.held s in
  List.iter Storage.pin held;
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
  (* A consumed storage is retired by the collector once nothing reaches it. *)
  Gc.finalise
    (fun _ -> List.iter (fun st -> ignore (Storage.unpin st : bool)) held)
    p;
  p

(* Calls *)

let why path =
  Printf.sprintf
    "this value was consumed at %s in a compiled call's arguments; use the \
     value the call returned"
    path

let reaches l l' =
  List.exists (fun r -> List.exists (B.overlaps r) l'.runs) l.runs

(* [check entry consumed paths leaves] refuses a consumed leaf that does not
   cover its whole storage or whose storage another leaf reaches. *)
let check entry consumed paths leaves =
  Array.iteri
    (fun i l ->
      if consumed.(i) then begin
        let length = match l.buffers with b :: _ -> B.length b | [] -> 0 in
        let whole =
          View.numel l.view = length
          && (length = 0 || View.extent l.view = (0, length))
          && ((not l.host) || List.for_all B.spans l.buffers)
        in
        if not whole then
          invalid_argf
            "%s: %s is consumed and does not cover its whole storage; consume \
             Nx.copy of it"
            entry paths.(i);
        Array.iteri
          (fun k l' ->
            if k <> i && reaches l l' then
              invalid_argf "%s: %s is consumed and %s reaches its storage" entry
                paths.(i) paths.(k))
          leaves
      end)
    leaves

type claim = { storage : Storage.t; mutable exclusive : bool }

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
         (Storage.v r.at bufs))

(* [run entry p leaves] runs [p] on [leaves]: it claims their storage, allocates
   the results', consumes the consumed leaves, and queues [p]. *)
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
  let claims = Array.make n None in
  let finally () =
    Array.iter
      (Option.iter (fun c ->
           if c.exclusive then ignore (Storage.finish c.storage : bool);
           Storage.release c.storage))
      claims
  in
  Fun.protect ~finally @@ fun () ->
  Array.iteri
    (fun i l ->
      Option.iter
        (fun storage ->
          Storage.borrow storage;
          let c = { storage; exclusive = false } in
          claims.(i) <- Some c;
          if p.consumed.(i) then begin
            Storage.upgrade storage;
            c.exclusive <- true
          end)
        l.cell)
    leaves;
  (* Whether the consumed leaf [i] can lend its memory on this call. *)
  let own i =
    let l = leaves.(i) in
    List.for_all (fun b -> B.spans b && not (B.is_borrowed b)) l.buffers
    &&
    match claims.(i) with
    | Some c -> Storage.pins c.storage = 0
    | None -> true
  in
  let lends = Array.make n false in
  let storage =
    Array.map
      (fun r ->
        let (Nx.P t) = r.like in
        match r.out with
        | Empty | Fresh _ -> fresh r.at (Nx.dtype t) r.shape
        | Lent i when own i ->
            lends.(i) <- true;
            []
        | Lent i ->
            let copy = fresh r.at (Nx.dtype t) r.shape in
            List.iter2 (fun src dst -> B.copy ~src ~dst) leaves.(i).buffers copy;
            copy)
      p.results
  in
  let bufs =
    Array.mapi
      (fun i l ->
        if not p.consumed.(i) then l.runs
        else begin
          Option.iter
            (fun c -> Storage.consume c.storage ~path:p.paths.(i))
            claims.(i);
          if lends.(i) || l.host then
            List.map (B.consume ~why:(why p.paths.(i))) l.buffers
          else l.buffers
        end)
      leaves
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
          | Lent i -> slots.(p.slots.(i)) <- results.(j)
          | Empty -> ())
        p.results;
      Engine.run linked slots)
    p.linked;
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

type 'r entry = { latch : Mutex.t; program : 'r program option Atomic.t }

let compiled (type a r) entry (args_s : a Ptree.t) (result_s : r Ptree.t) roles
    (g : a -> r) : a -> r =
  let lock = Mutex.create () and table = Hashtbl.create 8 and last = ref None in
  fun args ->
    if Nx.Op.intercepted () then g args
    else
      let ts, skeleton = Ptree.flatten args_s args in
      let leaves = Array.of_list (List.map leaf ts) in
      let key =
        {
          skeleton;
          settings = settings ();
          layouts = Array.to_list (Array.map (fun l -> l.layout) leaves);
        }
      in
      let h = hash key in
      let e =
        Mutex.protect lock @@ fun () ->
        let e =
          match
            List.find_opt
              (fun (k, _) -> same_key k key)
              (Hashtbl.find_all table h)
          with
          | Some (_, e) -> e
          | None ->
              Option.iter
                (fun k ->
                  report "retrace: %s"
                    (difference (fst (paths args_s roles args)) key k))
                !last;
              let e = { latch = Mutex.create (); program = Atomic.make None } in
              Hashtbl.add table h (key, e);
              e
        in
        last := Some key;
        e
      in
      let p =
        match Atomic.get e.program with
        | Some p -> p
        | None -> (
            Mutex.protect e.latch @@ fun () ->
            match Atomic.get e.program with
            | Some p -> p
            | None ->
                let paths, consumed = paths args_s roles args in
                let p =
                  compile args_s result_s g args leaves ~paths ~consumed
                in
                Atomic.set e.program (Some p);
                p)
      in
      check entry p.consumed p.paths leaves;
      run entry p leaves

let jit entry s f =
  let (Structure.Signature u) = Structure.signature entry s in
  u.curry (compiled entry u.args u.result u.roles (u.apply f))
