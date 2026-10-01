(* Witnesses, builders and golden cells shared by the suite's groups. *)

open Windtrap
open Tolk_next

(* Witnesses *)

let uop =
  Testable.with_compare Ops.compare (Testable.make ~pp:Ops.pp ~equal:Ops.equal)

let uops = list uop

let op =
  Testable.with_compare Op.compare (Testable.make ~pp:Op.pp ~equal:Op.equal)

let dtype = Dtypes.dtype
let value = Dtypes.value
let const = Dtypes.const
let z = Dtypes.z

let pp_sint ppf (s : Ops.sint) =
  match s with Int n -> Format.pp_print_int ppf n | Sym u -> Ops.pp ppf u

let equal_sint (s0 : Ops.sint) (s1 : Ops.sint) =
  match (s0, s1) with
  | Int n0, Int n1 -> Int.equal n0 n1
  | Sym u0, Sym u1 -> u0 == u1
  | _ -> false

let sint = Testable.make ~pp:pp_sint ~equal:equal_sint
let shape = list sint
let device = Testable.make ~pp:Ops.pp_device ~equal:Ops.equal_device
let addr_space = Testable.make ~pp:Dtype.pp_addr_space ~equal:( = )

let rejects ?__POS__ f =
  raises_match ?__POS__ (Exn.invalid_arg ?substring:None) f

(* Integers and values, as tests write them *)

let i n = `Int (Bigint.of_int n)
let f x = `Float x
let ints l = List.map (fun n : Ops.sint -> Int n) l

(* [var name lo hi] is the variable [name] of type [dtype] (default
   {!Dtype.Int32}) over the integers from [lo] to [hi]. *)
let var ?(dtype = Dtype.Int32) ?multiple_of name lo hi =
  Ops.variable ~dtype ?multiple_of name (i lo) (i hi)

let weak_var ?multiple_of name lo hi =
  var ~dtype:Dtype.Weak_int ?multiple_of name lo hi

let fvar ?(dtype = Dtype.Float32) name =
  Ops.variable ~dtype name (f (-10.)) (f 10.)

let flag name = Ops.variable ~dtype:Dtype.Bool name (`Bool false) (`Bool true)
let bounds u = (Ops.vmin u, Ops.vmax u)
let int_bounds lo hi = (i lo, i hi)

(* [check_bounds u (lo, hi)] is that [u]'s bounds are [lo] and [hi]. *)
let check_bounds ?__POS__ ?msg u expected =
  equal ?__POS__ ?msg (pair value value) expected (bounds u)

(* Golden cells *)

let after prefix s =
  let n = String.length prefix in
  if String.length s >= n && String.sub s 0 n = prefix then
    String.sub s n (String.length s - n)
  else invalid_arg (Printf.sprintf "%S does not start with %S" s prefix)

let op_of_cell s = Result.get_ok (Op.of_string (after "Ops." s))
let dtype_of_cell = Dtypes.dtype_of_cell
let value_of_cell = Dtypes.value_of_cell
let const_of_cell = Dtypes.const_of_cell
let consts_of_cell = Dtypes.consts_of_cell
let raised cell = String.starts_with ~prefix:"raises " cell

(* [expect w read cell f] is that [f ()] is what [cell] reads as, or that it
   rejects its arguments where tinygrad raises. *)
let expect w read cell f =
  if raised cell then rejects f else equal w (read cell) (f ())
