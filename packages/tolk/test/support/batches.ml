open Tolk

let queues batch =
  List.map
    (fun submission ->
      let linear = Ops.nth (Ops.without_after submission) 0 in
      match Ops.arg linear with
      | Queue { devices; queue } -> ((List.hd devices, queue), Ops.src linear)
      | _ -> invalid_arg "a submission encodes one queue")
    (Ops.src (Ops.nth batch 0))

let calls batch =
  List.concat_map
    (fun (_, cmds) -> List.filter (fun c -> Ops.op c = Call) cmds)
    (queues batch)

let submitted = 5

type event = Call of int | Signaled of string

let pp_event ppf = function
  | Call i -> Format.fprintf ppf "call %d" i
  | Signaled d -> Format.fprintf ppf "signaled %s" d

type outcome = {
  events : event list;
  left : ((string * string) * int) list;
  signal_words : (string * int) list;
}

let instruction c =
  match (Ops.op c, Ops.arg c) with
  | Ins, Code { code; _ } -> code
  | Call, _ -> "call"
  | _ -> invalid_arg (Format.asprintf "no command: %a" Ops.pp c)

(* A word of memory: its storage and byte offset. *)
let word u =
  if Ops.op u = Index then
    let base, off = Hcq2.unwrap_view (Ops.nth u 0) in
    ( base,
      off
      + (Bigint.to_int (Shape.to_z (Ops.nth u 1)) * Dtype.itemsize (Ops.dtype u))
    )
  else Hcq2.unwrap_view u

let devices batch =
  match Ops.arg batch with
  | Call { aux = Some info; _ } -> info.device
  | _ -> invalid_arg "a batch holds its queues' devices"

let run ?(finished = []) ?order ?capacity batch =
  let devices = devices batch in
  let queues = queues batch in
  let calls = calls batch in
  let memory = ref [] in
  let same (b0, o0) (b1, o1) = b0 == b1 && o0 = o1 in
  let get w =
    Option.value ~default:0
      (List.find_map
         (fun (w', v) -> if same w w' then Some v else None)
         !memory)
  in
  let set w v =
    memory := (w, v) :: List.filter (fun (w', _) -> not (same w w')) !memory
  in
  List.iter
    (fun d ->
      set
        (word (Hcq2.signal_word d))
        (Option.value ~default:submitted (List.assoc_opt d finished)))
    devices;
  let named name what =
    List.exists (fun d -> Ops.expr (what d) = name) devices
  in
  let rec value u =
    match Ops.op u with
    | Const -> Bigint.to_int (Shape.to_z u)
    | Load -> get (word (Ops.nth u 0))
    | Param when named (Ops.expr u) Hcq2.submitted -> submitted
    | Param when named (Ops.expr u) Hcq2.value -> submitted + 1
    | _ -> List.fold_left (fun acc s -> acc + value s) 0 (Ops.src u)
  in
  (* The commands each queue holds, and those the host has yet to hand it. *)
  let pending =
    List.map
      (fun (q, cmds) -> (q, ref (if capacity = None then cmds else [])))
      queues
  in
  let host =
    ref
      (match capacity with
      | None -> []
      | Some _ ->
          List.concat_map
            (fun (q, cmds) -> List.map (fun c -> (q, c)) cmds)
            queues)
  in
  let order = Option.value order ~default:(List.map fst queues) in
  let events = ref [] in
  let index c =
    let rec find i = function
      | x :: _ when x == c -> i
      | _ :: r -> find (i + 1) r
      | [] -> -1
    in
    find 0 calls
  in
  let runs c =
    match instruction c with
    | "wait" -> get (word (Ops.nth c 0)) >= value (Ops.nth c 1)
    | "store" ->
        let w = word (Ops.nth c 0) in
        set w (value (Ops.nth c 1));
        (match Ops.tag (fst w) with
        | Some (String "timeline") ->
            let d =
              List.find
                (fun d -> fst (word (Hcq2.signal_word d)) == fst w)
                devices
            in
            events := Signaled d :: !events
        | _ -> ());
        true
    | "call" ->
        events := Call (index c) :: !events;
        true
    | "timestamp" | "barrier" -> true
    | i -> invalid_arg ("no instruction " ^ i)
  in
  let hand () =
    match (!host, capacity) with
    | (q, c) :: rest, Some n ->
        let cmds = List.assoc q pending in
        List.length !cmds < n
        && begin
          cmds := !cmds @ [ c ];
          host := rest;
          true
        end
    | _ -> false
  in
  let step () =
    List.exists
      (fun q ->
        let cmds = List.assoc q pending in
        match !cmds with
        | c :: rest when runs c ->
            cmds := rest;
            true
        | _ -> false)
      order
    || hand ()
  in
  while step () do
    ()
  done;
  {
    events = List.rev !events;
    left =
      List.map
        (fun (q, cmds) ->
          let unhanded = List.filter (fun (q', _) -> q' = q) !host in
          (q, List.length !cmds + List.length unhanded))
        pending;
    signal_words =
      List.map (fun d -> (d, get (word (Hcq2.signal_word d)))) devices;
  }

let rotations batch =
  let qs = List.map fst (queues batch) in
  List.mapi
    (fun i _ ->
      List.filteri (fun j _ -> j >= i) qs @ List.filteri (fun j _ -> j < i) qs)
    qs
