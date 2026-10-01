open Talon_next

let w = Window.rows ~before:6 ~after:0
let e = Expr.rolling w (Col.float "x")
