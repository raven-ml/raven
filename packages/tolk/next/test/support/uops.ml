open Tolk_next

let pp_uop ppf u =
  let lines = String.split_on_char '\n' (String.trim (Graph.to_string u)) in
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list Format.pp_print_string)
    lines

let uop =
  Windtrap.Testable.with_compare Ops.compare
    (Windtrap.Testable.make ~pp:pp_uop ~equal:Ops.equal)
