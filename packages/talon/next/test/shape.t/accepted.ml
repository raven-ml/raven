open Talon_next

let cutoff = Expr.float 15.
let late = Expr.(Col.float "dep_delay" > cutoff)
let late_mean = Expr.(mean (Col.float "dep_delay") > cutoff)
let hour = Expr.Temporal.field `Hour (Col.instant "ts")
let year = Expr.Temporal.field `Year (Col.date "orderdate")
let e = Expr.(nx { f = Nx.exp } (Col.float "x"))

let null_counts : Expr.agg Expr.out =
  Expr.(each Sel.all { column = (fun n x -> n := rows - count x) })
