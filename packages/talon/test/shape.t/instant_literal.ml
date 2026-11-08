open Talon

let stale = Expr.(Col.instant "at" > Expr.wall (Time.of_ns 0L))
