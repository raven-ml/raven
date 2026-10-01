(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_next

(* Whether [x] is a value of the trace: traced by its lowering, or one it
   captures. *)
let ours x =
  match Nx.Repr.v x with
  | Traced t -> (
      match Nx.Repr.Traced.node t with Lower.Uop _ -> true | _ -> false)
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
      Nx.P (Lower.traced (Nx.placement x) (Nx.dtype x) (Ops.contiguous u))
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
          let copy =
            Lower.output s ~slot:(Ops.unique_num ()) at dt (Nx.shape x)
          in
          Nx.P
            (Lower.traced at dt
               (Ops.after copy
                  [ Ops.store copy (Ops.after (Ops.contiguous u) deps) ])))
    values

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

(* Staged scans *)

let numel shape = Array.fold_left ( * ) 1 shape
let ints l = List.map (fun n -> Ops.Int n) l

(* The elements of [u]'s dtype that rows of [m] of them take, apart enough that
   each starts on 16 bytes of memory, as the body's vector accesses take it. *)
let stride u m =
  let per = Int.max 1 (16 / Ops.element_size u) in
  (m + per - 1) / per * per

(* [cut ~own next args body] is [body] with each part of its graph that reaches
   none of the call's own parameters [own], runs inside no range that is open
   there, and reads storage replaced by a parameter of slot [next ()], which the
   call binds to the part, added to [args]: the part is computed once, before
   the loop. A movement stays in the body and the part is what it moves: a view
   costs nothing where it is read, while computing it once would store a
   transposed or broadcast copy that the loop then reads. A range open at a part
   is one of the body's own loops, which the part moves with. Another call's
   parameter, such as an enclosing step's carry, is storage the call binds like
   any other. *)
let cut ~own next args body =
  let reading = Ops.Tbl.create 64
  and storing = Ops.Tbl.create 64
  and rebuilt = Ops.Tbl.create 64 in
  let sources u =
    if Ops.op u = Op.Call then Ops.src_without_body u else Ops.src u
  in
  let memo table f =
    let rec go u =
      match Ops.Tbl.find_opt table u with
      | Some v -> v
      | None ->
          let v = f u || List.exists go (sources u) in
          Ops.Tbl.add table u v;
          v
    in
    go
  in
  let reads_param =
    memo reading (fun u -> Ops.op u = Op.Param && Ops.Tbl.mem own u)
  in
  let reads_storage =
    memo storing (fun u ->
        match Ops.op u with
        | Op.Buffer | Op.After -> true
        | Op.Param -> Ops.addrspace u = Some Tolk_next.Dtype.Global
        | _ -> false)
  in
  let rec storage u =
    match (Ops.op u, Ops.src u) with
    | (Op.Param | Op.Buffer | Op.Alloc), _ | _, [] -> u
    | _, x :: _ -> storage x
  in
  let writes c =
    let slots =
      List.filter_map
        (fun u ->
          if Ops.op u <> Op.Store then None
          else
            match Ops.arg (storage (Ops.nth u 0)) with
            | Ops.Param p -> Some p.slot
            | _ -> None)
        (Ops.toposort ~enter_calls:false (Ops.body c))
    in
    List.filteri (fun k _ -> List.mem k slots) (Ops.src_without_body c)
    |> List.map storage
  in
  let calls =
    List.filter_map
      (fun u -> if Ops.op u = Op.Call then Some (u, writes u) else None)
      (Ops.toposort ~enter_calls:false body)
  in
  let open_ranges u = Ops.Nodes.cardinal (Ops.ranges u) > 0 in
  (* The storage a call of the body that runs each trip writes, such as a loop's
     carry it updates in place: what reads it, its first value included, runs
     each trip too. A call runs each trip when it varies, or reads storage such
     a call writes. *)
  let written = Ops.Tbl.create 8 in
  let rec settle () =
    let reads_written =
      memo (Ops.Tbl.create 64) (fun u -> Ops.Tbl.mem written u)
    in
    let runs c = reads_param c || open_ranges c || reads_written c in
    match
      List.concat_map
        (fun (c, bs) ->
          if runs c then List.filter (fun b -> not (Ops.Tbl.mem written b)) bs
          else [])
        calls
    with
    | [] -> reads_written
    | fresh ->
        List.iter (fun b -> Ops.Tbl.replace written b ()) fresh;
        settle ()
  in
  let reads_written = settle () in
  let varies u = reads_param u || reads_written u || open_ranges u in
  let rec rebuild u =
    match Ops.Tbl.find_opt rebuilt u with
    | Some v -> v
    | None ->
        let v =
          if
            varies u
            || (reads_storage u && Op.Set.mem (Ops.op u) Op.Set.movement)
          then
            let src = List.map rebuild (sources u) in
            Ops.replace u
              ~src:(if Ops.op u = Op.Call then Ops.body u :: src else src)
          else if not (reads_storage u) then u
          else
            let slot = next () in
            args := (slot, Ops.contiguous u) :: !args;
            Ops.param_like u slot
        in
        Ops.Tbl.add rebuilt u v;
        v
  in
  rebuild body

(* [stage trace s r] is the scan [r] in the trace [s]: a range of as many trips
   as [r] has steps around one call of its step, which [trace] traces once in
   [s] as the call's body. *)
let stage trace s (r : Scan.request) =
  let leaves = r.req_carry @ r.req_xs in
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
  then raise Scan.Not_staged;
  let device = Ops.Single (Nx.Device.name (List.hd (Nx.Placement.devices p))) in
  (* The node of [x] on the loop's device: placed there, and a constant held
     there, since a call reads storage. *)
  let node (Nx.P x) =
    let u =
      if Nx.Placement.equal (Nx.placement x) p then Lower.value s x
      else Lower.uop (Lower.op s (Nx.Op.Place (p, x)))
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
     cut. *)
  let own = Ops.Tbl.create 8 and renumbered = ref [] in
  let parameter at dt shape =
    let slot = next () in
    let v = Lower.parameter s ~slot:(-2 - Ops.unique_num ()) at dt shape in
    let u = Ops.buf_uop (Lower.uop v) in
    Ops.Tbl.replace own u ();
    let arg =
      match Ops.arg u with Ops.Param a -> Ops.Param { a with slot } | a -> a
    in
    renumbered := (u, Ops.replace u ~arg) :: !renumbered;
    (slot, v)
  in
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
          parameter p (Nx.dtype x) (Array.sub shape 1 (Array.length shape - 1))
        in
        (slot, Nx.P v))
      r.req_xs
  in
  let carry', ys =
    trace s (fun () -> r.req_step (List.map snd carry) (List.map snd rows))
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
  then raise Scan.Not_staged;
  let (Nx.P x) = List.hd r.req_xs in
  let n = (Nx.shape x).(0) in
  let range = Ops.range ~axis_type:Loop (Ops.Int n) [ Ops.unique_num () ] in
  let trip =
    if r.req_reverse then Ops.O.(int (Int.pred n) - range) else range
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
  let params = Array.of_list (List.map (fun (_, Nx.P c) -> Lower.uop c) carry)
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
            let w = Lower.uop w in
            Ops.after w [ Ops.store w v ]
          end
          else if reads.(i) v = Other then Ops.contiguous v
          else v
        in
        stores := Ops.store u v :: !stores;
        fun e -> Ops.reshape (Ops.after b [ e ]) shape)
      (List.combine r.req_carry carry)
  in
  (* Each trip reads its row of each stacked input, from a copy whose rows are
     16 bytes apart when the input's are not. *)
  List.iter2
    (fun (Nx.P x as xs) (slot, _) ->
      let u = node xs and m = numel (Nx.shape x) / n in
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
    r.req_xs rows;
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
          stores := Ops.store (Lower.uop w) u :: !stores;
          fun e ->
            Ops.reshape
              (Ops.shrink
                 (Ops.reshape (Ops.after b [ e ]) (ints [ n; k ]))
                 [ None; Some (Ops.Int 0, Ops.Int m) ])
              (ints (Array.to_list stacked)))
      ys
  in
  let body =
    Ops.substitute ~enter_calls:false
      (cut ~own next args (Ops.sink (List.rev !stores)))
      !renumbered
  in
  let call args =
    Ops.end_
      (Ops.call ~precompile:true body (List.map snd (List.sort compare args)))
      [ range ]
  in
  (* Before answering, the loop must run: as one batch, or trip by trip on the
     host, which a probe of its schedule tells. *)
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
    Tolk_next.Schedule.create_linear_with_vars ~capturing:true
      (Ops.sink [ Ops.after (snd (List.hd probe)) [ call probe ] ])
  in
  if
    not
      (List.exists
         (Hcq2.runs ~devices:(fun d -> (Lower.engine s d).compiler))
         (Ops.src linear))
  then raise Scan.Not_staged;
  let e = call !args in
  {
    Scan.r_carry =
      List.map2
        (fun (Nx.P c) final -> Nx.P (Lower.traced p (Nx.dtype c) (final e)))
        r.req_carry carries;
    r_ys =
      (* An output on the host is written on the loop's device, and its rows
         copied to the host once the loop ran. *)
      List.map2
        (fun (Nx.P y) final ->
          let stacked =
            Lower.traced
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

let rec trace : 'a. body:bool -> Lower.scope -> (unit -> 'a) -> 'a =
 fun ~body s f ->
  let call : type r. r Construct.t -> (unit -> r) option =
   fun c ->
    match[@warning "@4@8"] c with
    | Detach x -> Some (fun () -> x)
    | Scan r -> Some (fun () -> stage (trace ~body:true) s r)
    | Remat { recomputed = true; p; f; args; _ } when not body ->
        Some
          (fun () ->
            let leaves, _ = Nx.Ptree.flatten p args in
            let args =
              Nx.Ptree.rebuild p ~like:args (List.map (kept s) leaves)
            in
            trace ~body s (fun () -> f args))
    | Barrier { values; after = deps } when not body ->
        Some (fun () -> after s values deps)
    | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _ | Lane_count _
    | Add _ ->
        None
  in
  (* A body runs outside the key scopes the function opened, and once for every
     trip: a draw it cannot vary is drawn where the scan is written. *)
  let run : type r. r Nx.Op.t -> r = function
    | Threefry _ as o when body -> (
        try Lower.op s o with Lower.Jit_error _ -> raise Scan.Not_staged)
    | o -> Lower.op s o
  in
  let op = { Nx.Op.run; claims = (fun _ -> true) } in
  Construct.install { op = Some op; call } f

let install s f = trace ~body:false s f
