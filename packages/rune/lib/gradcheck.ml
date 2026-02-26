(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module T = Tensor

type gradient_check_result = {
  max_abs_error : float;
  max_rel_error : float;
  mean_abs_error : float;
  mean_rel_error : float;
  failed_indices : (int array * float * float * float) list;
  passed : bool;
  num_checked : int;
  num_failed : int;
}

let default_rtol = 2e-3 (* JAX default for float32 *)
let default_atol = 2e-3 (* JAX default for float32 *)
let to_float_value t = T.item [] t

let check_gradient ?(eps = Finite_diff.default_eps) ?(rtol = default_rtol)
    ?(atol = default_atol) ?(verbose = false) ?(check_indices = None)
    ?(method_ = `Central) f x =
  let autodiff_grad = Autodiff.grad f x in

  let finite_diff_grad = Finite_diff.finite_diff ~eps ~method_ f x in

  let shape = T.shape x in
  let numel = Array.fold_left ( * ) 1 shape in

  let autodiff_flat = T.reshape [| numel |] autodiff_grad in
  let finite_diff_flat = T.reshape [| numel |] finite_diff_grad in

  let indices_to_check =
    match check_indices with
    | None -> List.init numel Fun.id
    | Some indices -> indices
  in

  let failed_indices = ref [] in
  let abs_errors = ref [] in
  let rel_errors = ref [] in

  List.iter
    (fun i ->
      let auto_val = to_float_value (T.get [ i ] autodiff_flat) in
      let finite_val = to_float_value (T.get [ i ] finite_diff_flat) in

      let abs_error = abs_float (auto_val -. finite_val) in
      let rel_error =
        if abs_float auto_val > 1e-12 || abs_float finite_val > 1e-12 then
          abs_error /. max (abs_float auto_val) (abs_float finite_val)
        else 0.0
      in

      abs_errors := abs_error :: !abs_errors;
      rel_errors := rel_error :: !rel_errors;

      let passed_check = abs_error <= atol || rel_error <= rtol in

      if not passed_check then (
        let nd_index =
          let flat_idx = i in
          let nd_idx = Array.make (Array.length shape) 0 in
          let mutable_idx = ref flat_idx in
          for dim = Array.length shape - 1 downto 0 do
            nd_idx.(dim) <- !mutable_idx mod shape.(dim);
            mutable_idx := !mutable_idx / shape.(dim)
          done;
          nd_idx
        in
        failed_indices :=
          (nd_index, auto_val, finite_val, abs_error) :: !failed_indices;

        if verbose then
          Printf.printf
            "Failed at index %s: autodiff=%.6e, finite_diff=%.6e, \
             abs_error=%.6e, rel_error=%.6e\n"
            (nd_index |> Array.to_list |> List.map string_of_int
           |> String.concat ", " |> Printf.sprintf "[%s]")
            auto_val finite_val abs_error rel_error))
    indices_to_check;

  let max_abs_error = List.fold_left max 0.0 !abs_errors in
  let max_rel_error = List.fold_left max 0.0 !rel_errors in
  let mean_abs_error =
    if !abs_errors = [] then 0.0
    else
      List.fold_left ( +. ) 0.0 !abs_errors
      /. float_of_int (List.length !abs_errors)
  in
  let mean_rel_error =
    if !rel_errors = [] then 0.0
    else
      List.fold_left ( +. ) 0.0 !rel_errors
      /. float_of_int (List.length !rel_errors)
  in

  let num_checked = List.length indices_to_check in
  let num_failed = List.length !failed_indices in
  let passed = num_failed = 0 in

  if verbose then (
    Printf.printf "\nGradient check summary:\n";
    Printf.printf "  Checked: %d elements\n" num_checked;
    Printf.printf "  Failed: %d elements\n" num_failed;
    Printf.printf "  Max absolute error: %.6e\n" max_abs_error;
    Printf.printf "  Max relative error: %.6e\n" max_rel_error;
    Printf.printf "  Mean absolute error: %.6e\n" mean_abs_error;
    Printf.printf "  Mean relative error: %.6e\n" mean_rel_error;
    Printf.printf "  Status: %s\n" (if passed then "PASSED" else "FAILED"));

  let result =
    {
      max_abs_error;
      max_rel_error;
      mean_abs_error;
      mean_rel_error;
      failed_indices = List.rev !failed_indices;
      passed;
      num_checked;
      num_failed;
    }
  in

  if passed then `Pass result else `Fail result

let check_gradients ?(eps = Finite_diff.default_eps) ?(rtol = default_rtol)
    ?(atol = default_atol) ?(verbose = false) ?(method_ = `Central) f xs =
  let autodiff_grads = Autodiff.grads f xs in

  let results =
    List.mapi
      (fun idx (x, autodiff_grad) ->
        let f_single x_i =
          let xs_copy = List.mapi (fun i x -> if i = idx then x_i else x) xs in
          f xs_copy
        in

        let finite_diff_grad =
          Finite_diff.finite_diff ~eps ~method_ f_single x
        in

        let shape = T.shape x in
        let numel = Array.fold_left ( * ) 1 shape in

        let autodiff_flat = T.reshape [| numel |] autodiff_grad in
        let finite_diff_flat = T.reshape [| numel |] finite_diff_grad in

        let failed_indices = ref [] in
        let abs_errors = ref [] in
        let rel_errors = ref [] in

        for i = 0 to numel - 1 do
          let auto_val = to_float_value (T.get [ i ] autodiff_flat) in
          let finite_val = to_float_value (T.get [ i ] finite_diff_flat) in

          let abs_error = abs_float (auto_val -. finite_val) in
          let rel_error =
            if abs_float auto_val > 1e-12 || abs_float finite_val > 1e-12 then
              abs_error /. max (abs_float auto_val) (abs_float finite_val)
            else 0.0
          in

          abs_errors := abs_error :: !abs_errors;
          rel_errors := rel_error :: !rel_errors;

          let passed_check = abs_error <= atol || rel_error <= rtol in

          if not passed_check then (
            let nd_index =
              let flat_idx = i in
              let nd_idx = Array.make (Array.length shape) 0 in
              let mutable_idx = ref flat_idx in
              for dim = Array.length shape - 1 downto 0 do
                nd_idx.(dim) <- !mutable_idx mod shape.(dim);
                mutable_idx := !mutable_idx / shape.(dim)
              done;
              nd_idx
            in
            failed_indices :=
              (nd_index, auto_val, finite_val, abs_error) :: !failed_indices;

            if verbose then
              Printf.printf
                "Input %d failed at index %s: autodiff=%.6e, finite_diff=%.6e, \
                 abs_error=%.6e, rel_error=%.6e\n"
                idx
                (nd_index |> Array.to_list |> List.map string_of_int
               |> String.concat ", " |> Printf.sprintf "[%s]")
                auto_val finite_val abs_error rel_error)
        done;

        let max_abs_error =
          if !abs_errors = [] then 0.0 else List.fold_left max 0.0 !abs_errors
        in
        let max_rel_error =
          if !rel_errors = [] then 0.0 else List.fold_left max 0.0 !rel_errors
        in
        let mean_abs_error =
          if !abs_errors = [] then 0.0
          else
            List.fold_left ( +. ) 0.0 !abs_errors
            /. float_of_int (List.length !abs_errors)
        in
        let mean_rel_error =
          if !rel_errors = [] then 0.0
          else
            List.fold_left ( +. ) 0.0 !rel_errors
            /. float_of_int (List.length !rel_errors)
        in

        let num_checked = numel in
        let num_failed = List.length !failed_indices in
        let passed = num_failed = 0 in

        if verbose then (
          Printf.printf "\nGradient check summary for input %d:\n" idx;
          Printf.printf "  Checked: %d elements\n" num_checked;
          Printf.printf "  Failed: %d elements\n" num_failed;
          Printf.printf "  Max absolute error: %.6e\n" max_abs_error;
          Printf.printf "  Max relative error: %.6e\n" max_rel_error;
          Printf.printf "  Mean absolute error: %.6e\n" mean_abs_error;
          Printf.printf "  Mean relative error: %.6e\n" mean_rel_error;
          Printf.printf "  Status: %s\n" (if passed then "PASSED" else "FAILED"));

        {
          max_abs_error;
          max_rel_error;
          mean_abs_error;
          mean_rel_error;
          failed_indices = List.rev !failed_indices;
          passed;
          num_checked;
          num_failed;
        })
      (List.combine xs autodiff_grads)
  in

  let all_passed = List.for_all (fun r -> r.passed) results in
  if all_passed then `Pass results else `Fail results
