open Tolk_next

let pp_uop ppf u =
  let lines = String.split_on_char '\n' (String.trim (Graph.to_string u)) in
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list Format.pp_print_string)
    lines

let uop =
  Windtrap.Testable.with_compare Ops.compare
    (Windtrap.Testable.make ~pp:pp_uop ~equal:Ops.equal)

let numbered_like like u =
  let made op g = List.filter (fun n -> Ops.op n = op) (Ops.toposort g) in
  let renumbered mine theirs =
    match (Ops.arg mine, Ops.arg theirs) with
    | Param p, Param q ->
        (mine, Ops.replace ~arg:(Param { p with slot = q.slot }) mine)
    | _ -> (mine, mine)
  in
  match
    List.concat_map
      (fun op -> List.map2 renumbered (made op u) (made op like))
      Op.[ Alloc; Buffer ]
  with
  | subs -> Ops.substitute ~enter_calls:true u subs
  | exception Invalid_argument _ -> u

let binaries_as_sources u =
  let recorded prg =
    match List.rev (Ops.src prg) with
    | binary :: source :: _ when Ops.op binary = Binary -> (
        match Ops.arg source with
        | String text -> Some (binary, Ops.v Binary ~arg:(Bytes text))
        | _ -> None)
    | _ -> None
  in
  let programs = List.filter (fun n -> Ops.op n = Program) (Ops.toposort u) in
  Ops.substitute ~enter_calls:true u (List.filter_map recorded programs)

let placeholders_like like u =
  let placeholders g =
    List.filter
      (fun n -> Ops.op n = Param && Option.is_some (Ops.tag n))
      (Ops.toposort g)
  in
  let renumbered mine theirs =
    match (Ops.arg mine, Ops.arg theirs) with
    | Param p, Param q -> (mine, Ops.replace ~arg:(Param { p with slot = q.slot }) mine)
    | _ -> (mine, mine)
  in
  match List.map2 renumbered (placeholders u) (placeholders like) with
  | subs -> Ops.substitute ~enter_calls:true u subs
  | exception Invalid_argument _ -> u
