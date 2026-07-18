(* Threaded overlap-fold race witness. Public value semantics are covered by the
   backend contract; this scaled fixture exists only to force the C worker
   path. *)

open Windtrap
module B = Nx_backend
module F = Nx_core.Make_frontend (B)

let ctx = B.create_context ()

let test_threaded_overlap () =
  let len = 131072 and kernel = 4 and stride = 1 in
  let values = Array.init len (fun i -> float_of_int ((i mod 13) + 1)) in
  let x = F.create ctx F.float64 [| 1; len |] values in
  let windows =
    B.unfold x ~kernel_size:[| kernel |] ~stride:[| stride |] ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let folded =
    B.fold windows ~output_size:[| len |] ~kernel_size:[| kernel |]
      ~stride:[| stride |] ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let window_count = ((len - kernel) / stride) + 1 in
  let coverage index =
    let first = Int.max 0 (index - kernel + 1) in
    let last = Int.min index (window_count - 1) in
    Int.max 0 (last - first + 1)
  in
  let expected =
    Array.init len (fun i -> values.(i) *. float_of_int (coverage i))
  in
  equal ~msg:"overlapping windows have exclusive output ownership"
    (array (float 0.))
    expected (F.to_array folded)

let () =
  Windtrap.run "nx C backend fold stress"
    [ group "threading" [ test "overlap race witness" test_threaded_overlap ] ]
