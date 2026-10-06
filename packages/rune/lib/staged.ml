(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk

(* Whether [x] is a value of the trace: traced by its lowering, or one it
   captures. *)
let ours x =
  match Nx.Repr.v x with
  | Traced _ -> Lower.is_traced x
  | Host _ | Placed _ -> true

(* The node of [x] if the trace computes it: neither storage it reads, nor a
   constant, nor a value of another transformation. *)
let computed s x =
  if not (ours x) then None
  else
    let u = Lower.value s x in
    match Ops.op (Ops.unsharded_base u) with
    | Op.Buffer | Op.Const -> None
    | _ -> Some u

(* [kept s x] is [x] with its own storage when the trace computes it. *)
let kept s (Nx.P x) =
  match computed s x with
  | Some u ->
      Nx.P (Lower.traced s (Nx.placement x) (Nx.dtype x) (Ops.contiguous u))
  | None -> Nx.P x

(* [after s values deps] is each of [values] the trace computes read through a
   copy stored once [deps] exist. *)
let after s values deps =
  let deps =
    List.filter_map
      (fun (Nx.P d) ->
        if ours d then Some (Ops.contiguous (Lower.value s d)) else None)
      deps
  in
  List.map
    (fun (Nx.P x) ->
      match computed s x with
      | None -> Nx.P x
      | Some u ->
          let at = Nx.placement x and dt = Nx.dtype x in
          let copy = Lower.scratch s at dt (Nx.shape x) in
          Nx.P
            (Lower.traced s at dt
               (Lower.stored copy
                  [ Lower.store copy (Ops.after (Ops.contiguous u) deps) ])))
    values

(* How a node reads a value: not at all, at each element's own index only
   (through elementwise operations, width-preserving casts, reshapes and
   contiguous markers), or otherwise. *)
type reach = Apart | Own | Other

let keeps_index u =
  let sized v = not (List.exists (Dtype.equal (Ops.dtype v)) Dtype.weaks) in
  match Ops.op u with
  | Op.Cast | Op.Bitcast ->
      let v = Ops.nth u 0 in
      sized u && sized v && Ops.element_size u = Ops.element_size v
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

(* Staged loops *)

let numel shape = Array.fold_left ( * ) 1 shape
let ints l = List.map (fun n -> Ops.Int n) l

(* The elements of [u]'s dtype that rows of [m] of them take, apart enough that
   each starts on 16 bytes of memory, as the body's vector accesses take it. *)
let stride u m =
  let per = Int.max 1 (16 / Ops.element_size u) in
  (m + per - 1) / per * per

(* [stage ~here ~inside s r] is the loop [r] in the trace [s]: a range of as
   many trips as [r] has rows, or of at most [max] trips while a flag holds,
   around one call of its step, which [inside] traces once in [s] as the call's
   body. [here] traces where the loop is written. A loop until a stop carries
   the stop's value and the flag that it does not hold yet in two more carries,
   and the engine reads the flag before each trip. *)
let stage ~here ~inside s (r : Trips.request) =
  let xs, n =
    match r.req_trips with
    | Rows { xs; _ } ->
        let (Nx.P x) = List.hd xs in
        (xs, (Nx.shape x).(0))
    | Until { max; _ } -> ([], max)
  in
  (* A loop over rows that cannot stage is written out; a loop until a stop has
     no written-out form. *)
  let decline why =
    match r.req_trips with
    | Rows _ -> raise Trips.Not_staged
    | Until _ ->
        raise
          (Lower.Jit_error ("Rune.jit: Rune.iterate cannot be compiled: " ^ why))
  in
  let leaves = r.req_carry @ xs in
  let at (Nx.P x) = Nx.placement x in
  (* The loop runs where its leaves lie, on one device, host leaves joining
     it. *)
  let p =
    Option.value ~default:Nx.Placement.host
      (List.find_opt
         (fun p -> not Nx.Placement.(equal p host))
         (List.map at leaves))
  in
  if
    List.compare_length_with (Nx.Placement.devices p) 1 <> 0
    || not
         (List.for_all
            (fun l -> Nx.Placement.(equal (at l) p || equal (at l) host))
            leaves)
  then decline "its carry lies on several devices";
  (* A host leaf joins the loop's device, as nx joins a host operand to a device
     one, unless the function moved it to the host: there it stays, and a loop
     on a device cannot run a step on it. *)
  let host =
    Ops.Single
      (Nx_device.name
         (Nx.Device.memory (List.hd (Nx.Placement.devices Nx.Placement.host))))
  in
  let to_host v = Ops.op v = Op.Copy && Ops.device v = Some host in
  let moved_to_host (Nx.P x as l) =
    (not (Nx.Placement.equal (at l) p))
    &&
    match computed s x with
    | Some u -> List.exists to_host (Ops.toposort ~calls:Enter u)
    | None -> false
  in
  if List.exists moved_to_host leaves then
    decline
      "its step runs on a device with command queues and on the host, or on \
       devices of two kinds";
  let device =
    Ops.Single
      (Nx_device.name (Nx.Device.memory (List.hd (Nx.Placement.devices p))))
  in
  (* The node of [x] on the loop's device: placed there, and a constant held
     there, since a call reads storage. *)
  let node (Nx.P x) =
    let u =
      if Nx.Placement.equal (Nx.placement x) p then Lower.value s x
      else Lower.uop s (Lower.op s (Nx.Op.Place (p, x)))
    in
    if Option.is_none (Ops.device u) then Ops.copy_to_device u device else u
  in
  let count = ref 0 in
  let next () =
    let k = !count in
    incr count;
    k
  in
  (* A parameter is traced under a negative number no other parameter has, so
     that it stays apart from an enclosing call's parameters, whose slots are
     those numbers or slots of their own, and takes its slot once the body is
     cut, with the alignment and phase of what the call passes it. *)
  let own = Ops.Tbl.create 8 and held_params = ref [] in
  let held make at dt shape =
    let slot = next () in
    let v = make s ~slot:(-2 - Ops.unique_num ()) at dt shape in
    let u = Ops.buf_uop (Lower.uop s v) in
    Ops.Tbl.replace own u ();
    held_params := (u, slot) :: !held_params;
    (slot, v)
  in
  let parameter at dt shape = held Lower.parameter at dt shape in
  let carry =
    List.map
      (fun (Nx.P c) ->
        let slot, v = parameter p (Nx.dtype c) (Nx.shape c) in
        (slot, Nx.P v))
      r.req_carry
  in
  let rows =
    List.map
      (fun (Nx.P x) ->
        let shape = Nx.shape x in
        let slot, v =
          held Lower.row p (Nx.dtype x)
            (Array.sub shape 1 (Array.length shape - 1))
        in
        (slot, Nx.P v))
      xs
  in
  let index_slot, index = held Lower.row p Nx.int32 [||] in
  (* The stop's value, and whether the loop runs on, at a carry. *)
  let stop c =
    match r.req_trips with
    | Rows _ -> []
    | Until { until; _ } ->
        let u = until c in
        [ Nx.P u; Nx.P (Nx.reshape [| 1 |] (Nx.logical_not (Nx.all u))) ]
  in
  let (carry', ys, stop'), checks =
    Lower.checking s (fun () ->
        inside s (fun () ->
            let c', ys =
              r.req_step
                { index; key = Nx.Rng.next_key }
                (List.map snd carry) (List.map snd rows)
            in
            (c', ys, stop c')))
  in
  let same_shape (_, Nx.P c) (Nx.P c') =
    Nx.shape c = Nx.shape c' && Nx.Placement.equal (Nx.placement c') p
  in
  if
    List.compare_lengths carry carry' <> 0
    || (not (List.for_all2 same_shape carry carry'))
    || not
         (List.for_all
            (fun y -> Nx.Placement.(equal (at y) p || equal (at y) host))
            ys)
  then decline "its step changes the carry's shapes or placements";
  let stops =
    List.map
      (fun (Nx.P u) ->
        let slot, v = parameter p (Nx.dtype u) (Nx.shape u) in
        (slot, Nx.P v))
      stop'
  in
  (* Each check of the step carries the index of its first failure, or its
     element count while none failed, and its data there: the first trip that
     fails keeps its index and data. *)
  let failures =
    List.concat_map
      (fun (c : Lower.check) ->
        let count = numel c.shape in
        let slot, v = parameter p Nx.int64 [||] in
        let u = Lower.uop s v in
        let failed = Ops.lt u (Ops.const_like u (`Int (Bigint.of_int count))) in
        let kept (Nx.P x) =
          let slot, v = parameter p (Nx.dtype x) [||] in
          let next = Ops.where failed (Lower.uop s v) (Lower.uop s x) in
          ( Nx.P (Nx.zeros (Nx.dtype x) [||]),
            (slot, Nx.P v),
            Nx.P (Lower.traced s p (Nx.dtype x) next) )
        in
        ( Nx.P (Nx.scalar Nx.int64 (Int64.of_int count)),
          (slot, Nx.P v),
          Nx.P
            (Lower.traced s p Nx.int64
               (Ops.where failed u (Lower.uop s c.first))) )
        :: List.map kept c.data)
      checks
  in
  let init =
    r.req_carry
    @ List.map (fun (i, _, _) -> i) failures
    @ here s (fun () -> stop r.req_carry)
  and carry = carry @ List.map (fun (_, c, _) -> c) failures @ stops
  and carry' = carry' @ List.map (fun (_, _, n) -> n) failures @ stop' in
  let range = Ops.range ~axis_type:Loop (Ops.Int n) [ Ops.unique_num () ] in
  let trip =
    match r.req_trips with
    | Rows { reverse = true; _ } -> Ops.O.(int (Int.pred n) - range)
    | Rows _ | Until _ -> range
  in
  let window b start m =
    Ops.shrink b [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + int m)) ]
  in
  let args = ref [] and stores = ref [] in
  let pass slot u = args := (slot, u) :: !args in
  (* Each carry is one buffer that each trip updates in place: the schedule
     orders every kernel that reads the carry before the one that stores it, and
     a next carry that reads it elsewhere than at its own index is materialised
     first, since a kernel computing it would read what it overwrites. Carries
     whose next values read each other have no such order: the latest of each
     cycle is copied, each trip, into a buffer of its own first. *)
  let params = Array.of_list (List.map (fun (_, Nx.P c) -> Lower.uop s c) carry)
  and nexts = Array.of_list (List.map node carry') in
  let reads = Array.map (fun u -> reach ~from:u) params in
  let k = Array.length params in
  let edges = Array.make k [] in
  let copied = Array.make k false and seen = Array.make k false in
  let rec reaches j i =
    i = j
    || (not seen.(j))
       && begin
         seen.(j) <- true;
         List.exists (fun l -> reaches l i) edges.(j)
       end
  in
  for i = 0 to k - 1 do
    Array.fill seen 0 k false;
    let others =
      List.filter
        (fun j -> j <> i && reads.(j) nexts.(i) <> Apart)
        (List.init k Fun.id)
    in
    if List.exists (fun j -> reaches j i) others then copied.(i) <- true
    else edges.(i) <- others
  done;
  let carries =
    List.mapi
      (fun i (init, (read, Nx.P c)) ->
        let u = params.(i) and shape = ints (Array.to_list (Nx.shape c)) in
        let b = Ops.new_buffer device (numel (Nx.shape c)) (Ops.dtype u) in
        pass read (Ops.after b [ Ops.store (Ops.reshape b shape) (node init) ]);
        let v = nexts.(i) in
        let v =
          if copied.(i) then begin
            let slot, w = parameter p (Nx.dtype c) (Nx.shape c) in
            pass slot (Ops.new_buffer device (numel (Nx.shape c)) (Ops.dtype u));
            let w = Lower.uop s w in
            Ops.after w [ Ops.store w v ]
          end
          else if reads.(i) v = Other then Ops.contiguous v
          else v
        in
        stores := Ops.store u v :: !stores;
        fun e -> Ops.reshape (Ops.after b [ e ]) shape)
      (List.combine init carry)
  in
  (* Each trip reads its row of each stacked input, from a copy whose rows are
     16 bytes apart when the input's are not. *)
  let read_rows =
    List.iter2 (fun (Nx.P x as xs) (slot, _) ->
        let shape = Nx.shape x in
        let u =
          Lower.stacked s (Nx.dtype x)
            (Array.sub shape 1 (Array.length shape - 1))
            (node xs)
        in
        let m = Ops.max_numel u / n in
        let k = stride u m in
        let flat =
          if k = m then Ops.reshape u [ Ops.Int (n * m) ]
          else
            Ops.reshape
              (Ops.pad
                 (Ops.reshape u (ints [ n; m ]))
                 [ None; Some (Ops.Int 0, Ops.Int (k - m)) ])
              [ Ops.Int (n * k) ]
        in
        pass slot (window (Ops.contiguous flat) Ops.O.(trip * int k) m))
  in
  read_rows xs rows;
  (* Each trip writes its row of each stacked output. *)
  let outputs =
    List.map
      (fun (Nx.P y as packed) ->
        let u = node packed and m = numel (Nx.shape y) in
        let stacked = Array.append [| n |] (Nx.shape y) in
        (* An empty output has nothing to write. *)
        if m = 0 then fun _ ->
          Lower.broadcast
            (Ops.const ~dtype:(Ops.dtype u) (`Int Bigint.zero))
            stacked
        else
          let k = stride u m in
          let b = Ops.new_buffer device (n * k) (Ops.dtype u) in
          let slot, w = parameter p (Nx.dtype y) (Nx.shape y) in
          pass slot (window b Ops.O.(trip * int k) m);
          stores := Ops.store (Lower.uop s w) u :: !stores;
          fun e ->
            Ops.reshape
              (Ops.shrink
                 (Ops.reshape (Ops.after b [ e ]) (ints [ n; k ]))
                 [ None; Some (Ops.Int 0, Ops.Int m) ])
              (ints (Array.to_list stacked)))
      ys
  in
  (* Each trip reads its index from a row of the [n] indices, made only for a
     step that reads it. *)
  let param = Ops.buf_uop (Lower.uop s index) in
  if reach ~from:param (Ops.sink !stores) <> Apart then
    read_rows
      (here s (fun () -> [ Nx.P (Nx.arange Nx.int32 0 n 1) ]))
      [ (index_slot, Nx.P index) ]
  else pass index_slot (Ops.new_buffer device 1 (Ops.dtype param));
  let renumbered =
    List.map
      (fun (u, slot) ->
        match Ops.arg u with
        | Ops.Param a ->
            let align, phase =
              match List.assoc_opt slot !args with
              | Some arg when a.addrspace = Some Tolk.Dtype.Global ->
                  Ops.storage_phase arg
              | _ -> (a.align, a.phase)
            in
            (u, Ops.replace u ~arg:(Ops.Param { a with slot; align; phase }))
        | _ -> (u, u))
      !held_params
  in
  let body =
    Ops.substitute ~calls:Skip ~pass:Fixed_point
      (Loop.cut ~own next args (Ops.sink (List.rev !stores)))
      renumbered
  in
  (* A loop until a stop reads its flag, the last carry, from the storage each
     trip updates. *)
  let call args =
    let c =
      Ops.call ~precompile:true body (List.map snd (List.sort compare args))
    in
    match stops with
    | [ _; (flag, _) ] ->
        Ops.backedge c ~loop:range ~cond:(List.assoc flag args)
    | _ -> Ops.end_ c [ range ]
  in
  (* Before answering, the loop must run: as one batch, or trip by trip, which a
     probe of its schedule tells. A loop of one trip is its calls, which run. *)
  let probe =
    List.map
      (fun (slot, u) ->
        ( slot,
          Ops.new_buffer
            (Option.get (Ops.device u))
            (Ops.max_numel u) (Ops.dtype u) ))
      !args
  in
  let linear, _ =
    Tolk.Schedule.create_linear_with_vars ~capturing:true
      (Ops.sink [ Ops.after (snd (List.hd probe)) [ call probe ] ])
  in
  let loop e = Ops.op e = Op.End || Ops.op e = Op.Backedge in
  if
    List.exists
      (fun e ->
        loop e
        && not (Hcq2.runs ~devices:(fun d -> (Lower.engine s d).compiler) e))
      (Ops.src linear)
  then
    decline
      "its step runs on a device with command queues and on the host, or on \
       devices of two kinds";
  let e = call !args in
  let finals =
    List.map2
      (fun (Nx.P c) final -> Nx.P (Lower.traced s p (Nx.dtype c) (final e)))
      init carries
  in
  let r_carry, rest = Trips.split (List.length r.req_carry) finals in
  let carried, stopped = Trips.split (List.length failures) rest in
  (* Once the trips are done, the stop holds or the loop fails. *)
  (match (r.req_trips, stopped) with
  | Until { failure; _ }, [ Nx.P u; _ ] ->
      Lower.op s
        (Nx.Op.Check
           {
             ok = Nx.unpack Nx.bool (Nx.P u);
             data = [];
             fail = (fun i _ -> Invalid_argument (failure i));
           })
  | _ -> ());
  (* A step's check holds where the loop's first failure is not, and its data
     there is what that trip read: each check carried its index, then its
     data. *)
  let rec answer checks carried =
    match (checks, carried) with
    | [], [] -> ()
    | (c : Lower.check) :: checks, Nx.P first :: rest ->
        let count = numel c.shape and shape = ints (Array.to_list c.shape) in
        let spread (Nx.P x) =
          let u = Lower.broadcast (Lower.uop s x) [| count |] in
          Nx.P (Lower.traced s p (Nx.dtype x) (Ops.reshape u shape))
        in
        let data, rest = Trips.split (List.length c.data) rest in
        let ok =
          Ops.ne
            (Ops.arange ~dtype:Int64 count)
            (Lower.broadcast (Lower.uop s first) [| count |])
        in
        Lower.op s
          (Nx.Op.Check
             {
               ok = Lower.traced s p Nx.bool (Ops.reshape ok shape);
               data = List.map spread data;
               fail = c.fail;
             });
        answer checks rest
    | _ -> assert false
  in
  answer checks carried;
  {
    Trips.r_carry;
    r_ys =
      (* An output on the host is written on the loop's device, and its rows
         copied to the host once the loop ran. *)
      List.map2
        (fun (Nx.P y) final ->
          let stacked =
            Lower.traced s
              (Nx.Placement.with_leading_axis p)
              (Nx.dtype y) (final e)
          in
          if Nx.Placement.equal (Nx.placement y) p then Nx.P stacked
          else
            Nx.P
              (Lower.op s
                 (Nx.Op.Place
                    (Nx.Placement.with_leading_axis (Nx.placement y), stacked))))
        ys outputs;
  }

(* [reads_body memo u] is [true] iff [u] reads a parameter of a staged body,
   memoised in [memo]: a value a step computes. A called body's own nodes are
   left out. *)
let reads_body memo =
  let rec go u =
    match Ops.Tbl.find_opt memo u with
    | Some r -> r
    | None ->
        let r =
          Ops.op u = Op.Param || List.exists go (Ops.src_without_body u)
        in
        Ops.Tbl.add memo u r;
        r
  in
  go

let escaped () =
  invalid_arg
    "Rune.jit: a value computed inside a loop's step escaped it; return it in \
     the carry or add it to a Rune.Total"

let rec trace :
    'a. body:bool -> bool Ops.Tbl.t -> Lower.scope -> (unit -> 'a) -> 'a =
 fun ~body memo s f ->
  let value = Construct.value and here = Construct.here in
  let call : type r. r Construct.t -> r Construct.answer option =
   fun c ->
    match[@warning "@4@8"] c with
    | Detach x -> Some (value (fun () -> x))
    | Compiled { f; args; _ } ->
        Some (here (fun () -> trace ~body memo s (fun () -> f args)))
    | Loop ({ req_trips = Rows _; _ } as r) ->
        Some
          (here (fun () ->
               stage ~here:(trace ~body memo) ~inside:(trace ~body:true memo) s
                 r))
    | Loop ({ req_trips = Until { until; max = 0; failure }; _ } as r) ->
        (* A loop of no trip checks its stop. *)
        Some
          (here (fun () ->
               trace ~body memo s (fun () ->
                   Nx.check Nx.Ptree.unit (until r.req_carry) () (fun i () ->
                       Invalid_argument (failure i)));
               { Trips.r_carry = r.req_carry; r_ys = [] }))
    | Loop ({ req_trips = Until _; _ } as r) ->
        Some
          (here (fun () ->
               stage ~here:(trace ~body memo) ~inside:(trace ~body:true memo) s
                 r))
    | Remat { recomputed = true; p; f; args; _ } when not body ->
        Some
          (here (fun () ->
               let leaves, _ = Nx.Ptree.flatten p args in
               let args =
                 Nx.Ptree.rebuild p ~like:args (List.map (kept s) leaves)
               in
               trace ~body memo s (fun () -> f args)))
    | Barrier { values; after = deps } when not body ->
        Some (value (fun () -> after s values deps))
    | Remat _ | Barrier _ | Custom _ | Root _ | At_map _ | Lanes _
    | Lane_index _ | Lane_count _ | Add _ ->
        None
  in
  let in_body x = reads_body memo (Lower.value s x) in
  (* Outside every body, an operation reads no value a step computed: one
     reaches it only through a handler around the loop, which runs outside the
     step. *)
  let escapes (Nx.P x) = Lower.traces s x && in_body x in
  let run : type r. r Nx.Op.t -> r =
   fun o ->
    if (not body) && List.exists escapes (Nx.Op.operands o) then escaped ();
    Lower.op s o
  in
  let op = { Nx.Op.run; claims = (fun _ -> true) } in
  Construct.install { op = Some op; call } f

let install s f = trace ~body:false (Ops.Tbl.create 64) s f
