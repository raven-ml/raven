open Tolk

let pp_uop ppf u =
  let lines = String.split_on_char '\n' (String.trim (Graph.to_string u)) in
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list Format.pp_print_string)
    lines

let uop =
  Windtrap.Testable.with_compare Ops.compare
    (Windtrap.Testable.make ~pp:pp_uop ~equal:Ops.equal)

let numbered_like like u =
  let made op g =
    List.filter (fun n -> Ops.op n = op) (Ops.toposort ~calls:Enter g)
  in
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
  (* Once: a slot in one graph can be another's in the other, as a chain or a
     cycle, and each node takes its pair's slot at most once. *)
  | subs -> Ops.substitute ~calls:Enter ~pass:Once u subs
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
  let programs =
    List.filter (fun n -> Ops.op n = Program) (Ops.toposort ~calls:Enter u)
  in
  Ops.substitute ~calls:Enter ~pass:Fixed_point u
    (List.filter_map recorded programs)

let binary_of prg =
  match Ops.arg (List.nth (Ops.src prg) (List.length (Ops.src prg) - 1)) with
  | Bytes lib -> lib
  | _ -> invalid_arg "a program of no binary"
  | exception (Failure _ | Invalid_argument _) ->
      invalid_arg "a program of no binary"

let entry_of prg =
  match Ops.arg (Ops.nth prg 0) with
  | Kernel k -> Ops.function_name k
  | _ -> invalid_arg "a program of no kernel"
