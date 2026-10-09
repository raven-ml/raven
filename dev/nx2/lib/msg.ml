(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let fail ~by fmt =
  Format.kasprintf (fun reason -> invalid_arg (by ^ ": " ^ reason)) fmt

let pp_list pp ppf l =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
    pp ppf l

let shape ppf s =
  Format.fprintf ppf "[%a]" (pp_list Format.pp_print_int) (Array.to_list s)

let operand ppf (dt, s, p) =
  Format.fprintf ppf "%s %a" (Nx_array.Dtype.name dt) shape s;
  let set = Devices.set p in
  if Devices.number set <> 0 then Format.fprintf ppf " on %a" Devices.pp set

let declined ~by ~kind ~kernels d dts =
  let dtypes =
    String.concat ", "
      (List.map (fun (Nx_array.Dtype.Any dt) -> Nx_array.Dtype.name dt) dts)
  in
  fail ~by "%s does not compute %s on %s (%s)" kernels kind dtypes (Rig.name d)

let no_kernels ~by ~op s =
  fail ~by
    "%a has no kernels to compute %s; place the value on a set with kernels, \
     mint the set with kernels, or apply the function where a compiler stages \
     it"
    Devices.pp s op
