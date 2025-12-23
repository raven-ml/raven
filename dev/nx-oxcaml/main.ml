(* open Nx;;             Now the module is available *)


(* Create a 2x3 tensor *)
let a = create float32 [|2;3|] [|1.; 2.; 3.; 4.; 5.; 6.|]

(* Fill a tensor with ones *)
let b = full float32 [|2;3|] 1.0

(* Element-wise addition *)
let c = add a b
