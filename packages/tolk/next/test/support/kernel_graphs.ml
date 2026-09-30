open Tolk_next

let fail fmt = Format.kasprintf invalid_arg fmt

(* Storage: whether it is call-local, and its slot and type. *)
let rec storage_of u =
  match (Ops.op u, Ops.arg u) with
  | Op.After, _ -> storage_of (Ops.nth u 0)
  | ( (Param | Buffer | Alloc),
      Param ({ device = Some (Single _) | None; _ } as p) )
    when p.addrspace <> Some Alu ->
      Some (Ops.op u = Alloc, p)
  | _ -> None

let same (s0, (p0 : Ops.param_arg)) (s1, (p1 : Ops.param_arg)) =
  s0 = s1 && p0.slot = p1.slot

let untouched dtype : Dtype.value =
  if Dtype.is_float dtype then `Float Float.nan
  else if Dtype.equal dtype Bool then `Bool false
  else Dtype.min dtype

(* A state of storage: its elements, and the indices written to reach it. *)
type state = { values : Dtype.value array; written : int list }

let is_kernel_call u = Ops.op u = Call && Ops.op (Ops.nth u 0) = Sink

let initial buffers (scratch, (p : Ops.param_arg)) =
  let given = if scratch then None else List.assoc_opt p.slot buffers in
  match given with
  | Some a -> Array.copy a
  | None -> Array.make (Option.value p.size ~default:1) (untouched p.dtype)

(* [kernel_writes ~vars ~params values_of call] is what the kernel [call] calls
   writes, each storage argument [a] holding [values_of a], each variable its
   value, bound or in [vars], and each scalar parameter no argument binds the
   value [params] gives its slot. *)
let kernel_writes ~vars ~params:enclosing values_of call =
  let bind (buffers, params) (k, a) =
    match (storage_of a, Ops.arg a) with
    | Some _, _ -> ((k, values_of a) :: buffers, params)
    | None, Param { bound = Some v; _ } -> (buffers, (k, v) :: params)
    | None, Param { name = Some n; _ } when List.mem_assoc n vars ->
        (buffers, (k, List.assoc n vars) :: params)
    | None, _ ->
        fail "a call's argument is a %a, not storage on one device" Op.pp
          (Ops.op a)
  in
  let buffers, params =
    List.fold_left bind ([], [])
      (List.mapi (fun k a -> (k, a)) (List.tl (Ops.src call)))
  in
  Interpreter.writes ~vars ~params:(params @ enclosing) ~buffers
    (Ops.nth call 0)

let sorted_writes written =
  Hashtbl.fold (fun (s, i) v acc -> (s, i, v) :: acc) written []
  |> List.sort (fun (s0, i0, _) (s1, i1, _) -> compare (s0, i0) (s1, i1))

let writes ?(vars = []) ?(params = []) ?(buffers = []) sink =
  let initial st = { values = initial buffers st; written = [] } in
  let states = Ops.Tbl.create 16 and results = Ops.Tbl.create 16 in
  let rec state u =
    match Ops.Tbl.find_opt states u with
    | Some s -> s
    | None ->
        let st =
          match storage_of u with
          | Some st -> st
          | None ->
              fail "a call's argument is a %a, not storage" Op.pp (Ops.op u)
        in
        let s =
          if Ops.op u = After then
            List.fold_left (apply st)
              (state (Ops.nth u 0))
              (List.tl (Ops.src u))
          else initial st
        in
        Ops.Tbl.replace states u s;
        s
  (* [apply st s dep] is the state [dep] leaves storage [st] in, from [s]. *)
  and apply st s dep =
    if not (is_kernel_call dep) then s
    else
      let args = List.tl (Ops.src dep) in
      let values = Array.copy s.values and written = ref s.written in
      List.iter
        (fun (k, i, v) ->
          match storage_of (List.nth args k) with
          | Some st' when same st st' ->
              values.(i) <- v;
              written := i :: !written
          | _ -> ())
        (run dep);
      { values; written = !written }
  and run call =
    match Ops.Tbl.find_opt results call with
    | Some w -> w
    | None ->
        let w = kernel_writes ~vars ~params (fun a -> (state a).values) call in
        Ops.Tbl.replace results call w;
        w
  in
  let final = Hashtbl.create 64 in
  List.iter
    (fun u ->
      match storage_of u with
      | Some (false, p) ->
          let s = state u in
          List.iter
            (fun i -> Hashtbl.replace final (p.slot, i) s.values.(i))
            s.written
      | Some (true, _) | None -> ())
    (Ops.src sink);
  sorted_writes final

let linear_writes ?(vars = []) ?(params = []) ?(buffers = []) linear =
  let memory = Hashtbl.create 16 and written = Hashtbl.create 64 in
  let storage a = Option.get (storage_of a) in
  let values_of a =
    let ((scratch, p) as st) = storage a in
    match Hashtbl.find_opt memory (scratch, p.slot) with
    | Some v -> v
    | None ->
        let v = initial buffers st in
        Hashtbl.replace memory (scratch, p.slot) v;
        v
  in
  let run call =
    if not (is_kernel_call call) then
      fail "a schedule's call is a call of a %a, not a kernel" Op.pp
        (Ops.op (Ops.nth call 0));
    let args = List.tl (Ops.src call) in
    List.iter
      (fun (k, i, v) ->
        let a = List.nth args k in
        (values_of a).(i) <- v;
        match storage a with
        | false, p -> Hashtbl.replace written (p.slot, i) v
        | true, _ -> ())
      (kernel_writes ~vars ~params values_of call)
  in
  List.iter run (Ops.src linear);
  sorted_writes written
