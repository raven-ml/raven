open Talon

let delay = Col.float "dep_delay"
let outs : Expr.agg Expr.out list = Expr.[ "d" := delay ]
