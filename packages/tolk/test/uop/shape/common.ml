(* Witnesses and builders shared by the suite's groups. *)

open Windtrap
open Tolk

(* Witnesses *)

let uop =
  Testable.with_compare Ops.compare (Testable.make ~pp:Ops.pp ~equal:Ops.equal)

let uops = list uop

let op =
  Testable.with_compare Op.compare (Testable.make ~pp:Op.pp ~equal:Op.equal)

let dtype = Dtypes.dtype
let value = Dtypes.value
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
let device = Testable.make ~pp:Ops.pp_device ~equal:String.equal
let rejects ?__POS__ f = raises_match ?__POS__ (Exn.invalid_arg ?sub:None) f

(* Integers and values, as tests write them *)

let i n = `Int (Bigint.of_int n)
let f x = `Float x
let ints l = List.map (fun n : Ops.sint -> Int n) l

(* [var name lo hi] is the variable [name] of type [dtype] (default
   {!Dtype.Int32}) over the integers from [lo] to [hi]. *)
let var ?(dtype = Dtype.Int32) name lo hi = Ops.variable ~dtype name (i lo) (i hi)

let weak_var name lo hi = var ~dtype:Dtype.Weak_int name lo hi

let flag name = Ops.variable ~dtype:Dtype.Bool name (`Bool false) (`Bool true)
let bounds u = (Ops.vmin u, Ops.vmax u)
let int_bounds lo hi = (i lo, i hi)
