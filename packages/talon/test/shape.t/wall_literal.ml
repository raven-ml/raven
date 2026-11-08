open Talon

let stale = Expr.(Col.wall "logged" > Expr.instant (Time.of_ns 0L))
