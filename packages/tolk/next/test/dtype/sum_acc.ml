(* Prints the data type that sums of each data type named on the command line
   accumulate in, then again after SUM_DTYPE changes, or "rejected" if SUM_DTYPE
   names no data type. *)

open Tolk_next

let print names =
  List.iter
    (fun name ->
      let dt = Result.get_ok (Dtype.of_string name) in
      Format.printf "%s: %a@." name Dtype.pp (Dtype.sum_acc dt))
    names

let () =
  let names = List.tl (Array.to_list Sys.argv) in
  match print names with
  | () ->
      Unix.putenv "SUM_DTYPE" "float16";
      print names
  | exception Invalid_argument _ ->
      print_endline "rejected";
      exit 1
