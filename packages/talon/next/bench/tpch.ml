(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next

let scales = [ "0.1"; "1"; "10" ]

let tables =
  [
    "customer";
    "lineitem";
    "nation";
    "orders";
    "part";
    "partsupp";
    "region";
    "supplier";
  ]

let day y m d = Expr.date (Option.get (Time.Date.of_civil (y, m, d)))
let year x = Expr.Temporal.field `Year x
let top n keys q = Kit.top_k n keys q

(* [during x start stop] holds when the date [x] is in \[[start];[stop]). *)
let during x start stop = Expr.(x >= start && x < stop)
let between x low high = Expr.(x >= low && x <= high)
let on l r = Join.eq l r
let extendedprice = Col.float "l_extendedprice"
let discount = Col.float "l_discount"
let quantity = Col.float "l_quantity"
let shipdate = Col.date "l_shipdate"
let orderdate = Col.date "o_orderdate"
let revenue = Expr.(extendedprice *. (float 1. -. discount))

(* Queries *)

let q01 t _ =
  Query.(
    t "lineitem"
    |> filter Expr.(shipdate <= day 1998 9 2)
    |> aggregate
         ~by:[ "l_returnflag"; "l_linestatus" ]
         Expr.
           [
             "sum_qty" := sum quantity;
             "sum_base_price" := sum extendedprice;
             "sum_disc_price" := sum revenue;
             "sum_charge" := sum (revenue *. (float 1. +. Col.float "l_tax"));
             "avg_qty" := mean quantity;
             "avg_price" := mean extendedprice;
             "avg_disc" := mean discount;
             "count_order" := rows;
           ]
    |> sort [ Order.asc "l_returnflag"; Order.asc "l_linestatus" ])

let q02 t _ =
  let europe =
    Query.(
      t "partsupp"
      |> join ~on:(on "ps_suppkey" "s_suppkey") (t "supplier")
      |> join ~on:(on "s_nationkey" "n_nationkey") (t "nation")
      |> join
           ~on:(on "n_regionkey" "r_regionkey")
           (t "region" |> filter Expr.(Col.string "r_name" = string "EUROPE")))
  in
  let cost = Col.float "ps_supplycost" in
  Query.(
    t "part"
    |> filter
         Expr.(
           Col.int "p_size" = int 15
           && Str.(matches (suffix "BRASS")) (Col.string "p_type"))
    |> join ~on:(on "p_partkey" "ps_partkey") europe
    |> filter Expr.(cost = over ~by:[ "p_partkey" ] (min cost))
    |> select
         Expr.
           [
             keep
               (Sel.names
                  [
                    "s_acctbal";
                    "s_name";
                    "n_name";
                    "p_partkey";
                    "p_mfgr";
                    "s_address";
                    "s_phone";
                    "s_comment";
                  ]);
           ]
    |> top 100
         [
           Order.desc "s_acctbal";
           Order.asc "n_name";
           Order.asc "s_name";
           Order.asc "p_partkey";
         ])

let q03 t _ =
  Query.(
    t "customer"
    |> filter Expr.(Col.string "c_mktsegment" = string "BUILDING")
    |> join
         ~on:(on "c_custkey" "o_custkey")
         (t "orders" |> filter Expr.(orderdate < day 1995 3 15))
    |> join
         ~on:(on "o_orderkey" "l_orderkey")
         (t "lineitem" |> filter Expr.(shipdate > day 1995 3 15))
    |> aggregate
         ~by:[ "o_orderkey"; "o_orderdate"; "o_shippriority" ]
         Expr.[ "revenue" := sum revenue ]
    |> select
         Expr.
           [
             "l_orderkey" := Col.int "o_orderkey";
             keep (Sel.names [ "revenue"; "o_orderdate"; "o_shippriority" ]);
           ]
    |> top 10 [ Order.desc "revenue"; Order.asc "o_orderdate" ])

let q04 t _ =
  Query.(
    t "orders"
    |> filter (during orderdate (day 1993 7 1) (day 1993 10 1))
    |> join ~kind:Semi
         ~on:(on "o_orderkey" "l_orderkey")
         (t "lineitem"
         |> filter Expr.(Col.date "l_commitdate" < Col.date "l_receiptdate"))
    |> aggregate ~by:[ "o_orderpriority" ] Expr.[ "order_count" := rows ]
    |> sort [ Order.asc "o_orderpriority" ])

let q05 t _ =
  Query.(
    t "customer"
    |> join
         ~on:(on "c_custkey" "o_custkey")
         (t "orders" |> filter (during orderdate (day 1994 1 1) (day 1995 1 1)))
    |> join ~on:(on "o_orderkey" "l_orderkey") (t "lineitem")
    |> join
         ~on:Join.(on "l_suppkey" "s_suppkey" && on "c_nationkey" "s_nationkey")
         (t "supplier")
    |> join ~on:(on "c_nationkey" "n_nationkey") (t "nation")
    |> join
         ~on:(on "n_regionkey" "r_regionkey")
         (t "region" |> filter Expr.(Col.string "r_name" = string "ASIA"))
    |> aggregate ~by:[ "n_name" ] Expr.[ "revenue" := sum revenue ]
    |> sort [ Order.desc "revenue" ])

let q06 t _ =
  Query.(
    t "lineitem"
    |> filter
         Expr.(
           during shipdate (day 1994 1 1) (day 1995 1 1)
           && between discount (float 0.05) (float 0.07)
           && quantity < float 24.)
    |> aggregate ~by:[] Expr.[ "revenue" := sum (extendedprice *. discount) ])

(* [nation t name] is the nations' keys, with their names as the column
   [name]. *)
let nation t name =
  Query.select
    Expr.[ keep (Sel.names [ "n_nationkey" ]); name := Col.string "n_name" ]
    (t "nation")

let q07 t _ =
  let supp = Col.string "supp_nation" and cust = Col.string "cust_nation" in
  Query.(
    t "lineitem"
    |> filter (between shipdate (day 1995 1 1) (day 1996 12 31))
    |> join ~on:(on "l_suppkey" "s_suppkey") (t "supplier")
    |> join ~on:(on "l_orderkey" "o_orderkey") (t "orders")
    |> join ~on:(on "o_custkey" "c_custkey") (t "customer")
    |> join ~on:(on "s_nationkey" "n_nationkey") (nation t "supp_nation")
    |> join ~on:(on "c_nationkey" "n_nationkey") (nation t "cust_nation")
    |> filter
         Expr.(
           (supp = string "FRANCE" && cust = string "GERMANY")
           || (supp = string "GERMANY" && cust = string "FRANCE"))
    |> derive Expr.[ "l_year" := year shipdate ]
    |> aggregate
         ~by:[ "supp_nation"; "cust_nation"; "l_year" ]
         Expr.[ "revenue" := sum revenue ]
    |> sort
         [
           Order.asc "supp_nation"; Order.asc "cust_nation"; Order.asc "l_year";
         ])

let q08 t _ =
  let keys names = Query.select Expr.[ keep (Sel.names names) ] (t "nation") in
  Query.(
    t "part"
    |> filter Expr.(Col.string "p_type" = string "ECONOMY ANODIZED STEEL")
    |> join ~on:(on "p_partkey" "l_partkey") (t "lineitem")
    |> join ~on:(on "l_suppkey" "s_suppkey") (t "supplier")
    |> join
         ~on:(on "l_orderkey" "o_orderkey")
         (t "orders"
         |> filter (between orderdate (day 1995 1 1) (day 1996 12 31)))
    |> join ~on:(on "o_custkey" "c_custkey") (t "customer")
    |> join
         ~on:(on "c_nationkey" "n_nationkey")
         (keys [ "n_nationkey"; "n_regionkey" ])
    |> join
         ~on:(on "n_regionkey" "r_regionkey")
         (t "region" |> filter Expr.(Col.string "r_name" = string "AMERICA"))
    |> join
         ~on:(on "s_nationkey" "n_nationkey")
         (keys [ "n_nationkey"; "n_name" ])
    |> derive Expr.[ "o_year" := year orderdate ]
    |> aggregate ~by:[ "o_year" ]
         Expr.
           [
             "mkt_share" :=
               sum
                 (if_
                    (Col.string "n_name" = string "BRAZIL")
                    revenue (float 0.))
               /. sum revenue;
           ]
    |> sort [ Order.asc "o_year" ])

let q09 t _ =
  Query.(
    t "part"
    |> filter Expr.(Str.(matches (literal "green")) (Col.string "p_name"))
    |> join ~on:(on "p_partkey" "l_partkey") (t "lineitem")
    |> join ~on:(on "l_suppkey" "s_suppkey") (t "supplier")
    |> join
         ~on:Join.(on "p_partkey" "ps_partkey" && on "l_suppkey" "ps_suppkey")
         (t "partsupp")
    |> join ~on:(on "l_orderkey" "o_orderkey") (t "orders")
    |> join ~on:(on "s_nationkey" "n_nationkey") (t "nation")
    |> derive
         Expr.[ "nation" := Col.string "n_name"; "o_year" := year orderdate ]
    |> aggregate ~by:[ "nation"; "o_year" ]
         Expr.
           [
             "sum_profit" :=
               sum (revenue -. (Col.float "ps_supplycost" *. quantity));
           ]
    |> sort [ Order.asc "nation"; Order.desc "o_year" ])

let q10 t _ =
  Query.(
    t "customer"
    |> join
         ~on:(on "c_custkey" "o_custkey")
         (t "orders" |> filter (during orderdate (day 1993 10 1) (day 1994 1 1)))
    |> join
         ~on:(on "o_orderkey" "l_orderkey")
         (t "lineitem" |> filter Expr.(Col.string "l_returnflag" = string "R"))
    |> join ~on:(on "c_nationkey" "n_nationkey") (t "nation")
    |> aggregate
         ~by:
           [
             "c_custkey";
             "c_name";
             "c_acctbal";
             "c_phone";
             "n_name";
             "c_address";
             "c_comment";
           ]
         Expr.[ "revenue" := sum revenue ]
    |> select
         Expr.
           [
             keep
               (Sel.names
                  [
                    "c_custkey";
                    "c_name";
                    "revenue";
                    "c_acctbal";
                    "n_name";
                    "c_address";
                    "c_phone";
                    "c_comment";
                  ]);
           ]
    |> top 20 [ Order.desc "revenue" ])

let q11 t sf =
  let value = Col.float "value" and fraction = 0.0001 /. sf in
  let germany =
    Query.(
      t "partsupp"
      |> join ~on:(on "ps_suppkey" "s_suppkey") (t "supplier")
      |> join
           ~on:(on "s_nationkey" "n_nationkey")
           (t "nation" |> filter Expr.(Col.string "n_name" = string "GERMANY"))
      |> select
           Expr.
             [
               keep (Sel.names [ "ps_partkey" ]);
               "value" :=
                 Col.float "ps_supplycost"
                 *. cast Type.float64 (Col.int "ps_availqty");
             ])
  in
  Query.(
    germany
    |> aggregate ~by:[ "ps_partkey" ] Expr.[ "value" := sum value ]
    |> join ~on:Join.all
         (germany
         |> aggregate ~by:[] Expr.[ "threshold" := sum value *. float fraction ]
         )
    |> filter Expr.(value > Col.float "threshold")
    |> select Expr.[ keep (Sel.names [ "ps_partkey"; "value" ]) ]
    |> sort [ Order.desc "value" ])

let q12 t _ =
  let high =
    Expr.(is_in [ "1-URGENT"; "2-HIGH" ] (Col.string "o_orderpriority"))
  in
  let tally p = Expr.(sum (if_ p (int 1) (int 0))) in
  let commit = Col.date "l_commitdate" and receipt = Col.date "l_receiptdate" in
  Query.(
    t "orders"
    |> join
         ~on:(on "o_orderkey" "l_orderkey")
         (t "lineitem"
         |> filter
              Expr.(
                is_in [ "MAIL"; "SHIP" ] (Col.string "l_shipmode")
                && commit < receipt && shipdate < commit
                && during receipt (day 1994 1 1) (day 1995 1 1)))
    |> aggregate ~by:[ "l_shipmode" ]
         Expr.
           [
             "high_line_count" := tally high;
             "low_line_count" := tally (not high);
           ]
    |> sort [ Order.asc "l_shipmode" ])

let q13 t _ =
  Query.(
    t "customer"
    |> join ~kind:Left
         ~on:(on "c_custkey" "o_custkey")
         (t "orders"
         |> filter
              Expr.(
                not
                  (Str.(matches (pieces [ "special"; "requests" ]))
                     (Col.string "o_comment"))))
    |> aggregate ~by:[ "c_custkey" ]
         Expr.[ "c_count" := count (Col.int "o_orderkey") ]
    |> aggregate ~by:[ "c_count" ] Expr.[ "custdist" := rows ]
    |> sort [ Order.desc "custdist"; Order.desc "c_count" ])

let q14 t _ =
  let promo = Expr.(Str.(matches (prefix "PROMO")) (Col.string "p_type")) in
  Query.(
    t "lineitem"
    |> filter (during shipdate (day 1995 9 1) (day 1995 10 1))
    |> join ~on:(on "l_partkey" "p_partkey") (t "part")
    |> aggregate ~by:[]
         Expr.
           [
             "promo_revenue" :=
               float 100. *. sum (if_ promo revenue (float 0.)) /. sum revenue;
           ])

let q15 t _ =
  let total = Col.float "total_revenue" in
  Query.(
    t "supplier"
    |> join
         ~on:(on "s_suppkey" "l_suppkey")
         (t "lineitem"
         |> filter (during shipdate (day 1996 1 1) (day 1996 4 1))
         |> aggregate ~by:[ "l_suppkey" ]
              Expr.[ "total_revenue" := sum revenue ])
    |> filter Expr.(total = over (max total))
    |> select
         Expr.
           [
             keep
               (Sel.names
                  [
                    "s_suppkey";
                    "s_name";
                    "s_address";
                    "s_phone";
                    "total_revenue";
                  ]);
           ]
    |> sort [ Order.asc "s_suppkey" ])

let q16 t _ =
  Query.(
    t "part"
    |> filter
         Expr.(
           Col.string "p_brand" <> string "Brand#45"
           && (not
                 (Str.(matches (prefix "MEDIUM POLISHED"))
                    (Col.string "p_type")))
           && is_in [ 49; 14; 23; 45; 19; 3; 36; 9 ] (Col.int "p_size"))
    |> join ~on:(on "p_partkey" "ps_partkey") (t "partsupp")
    |> join ~kind:Anti
         ~on:(on "ps_suppkey" "s_suppkey")
         (t "supplier"
         |> filter
              Expr.(
                Str.(matches (pieces [ "Customer"; "Complaints" ]))
                  (Col.string "s_comment")))
    |> aggregate
         ~by:[ "p_brand"; "p_type"; "p_size" ]
         Expr.[ "supplier_cnt" := n_unique (Col.int "ps_suppkey") ]
    |> sort
         [
           Order.desc "supplier_cnt";
           Order.asc "p_brand";
           Order.asc "p_type";
           Order.asc "p_size";
         ])

let q17 t _ =
  Query.(
    t "part"
    |> filter
         Expr.(
           Col.string "p_brand" = string "Brand#23"
           && Col.string "p_container" = string "MED BOX")
    |> join ~on:(on "p_partkey" "l_partkey") (t "lineitem")
    |> filter
         Expr.(quantity < float 0.2 *. over ~by:[ "p_partkey" ] (mean quantity))
    |> aggregate ~by:[] Expr.[ "avg_yearly" := sum extendedprice /. float 7. ])

let q18 t _ =
  Query.(
    t "orders"
    |> join ~kind:Semi
         ~on:(on "o_orderkey" "l_orderkey")
         (t "lineitem"
         |> aggregate ~by:[ "l_orderkey" ] Expr.[ "l_quantity" := sum quantity ]
         |> filter Expr.(quantity > float 300.))
    |> join ~on:(on "o_custkey" "c_custkey") (t "customer")
    |> join ~on:(on "o_orderkey" "l_orderkey") (t "lineitem")
    |> aggregate
         ~by:
           [
             "c_name"; "o_custkey"; "o_orderkey"; "o_orderdate"; "o_totalprice";
           ]
         Expr.[ "sum(l_quantity)" := sum quantity ]
    |> select
         Expr.
           [
             keep (Sel.names [ "c_name" ]);
             "c_custkey" := Col.int "o_custkey";
             keep
               (Sel.names
                  [
                    "o_orderkey";
                    "o_orderdate";
                    "o_totalprice";
                    "sum(l_quantity)";
                  ]);
           ]
    |> top 100 [ Order.desc "o_totalprice"; Order.asc "o_orderdate" ])

let q19 t _ =
  let branch brand containers low size =
    let high = low +. 10. in
    Expr.(
      Col.string "p_brand" = string brand
      && is_in containers (Col.string "p_container")
      && between quantity (float low) (float high)
      && between (Col.int "p_size") (int 1) (int size))
  in
  Query.(
    t "lineitem"
    |> filter
         Expr.(
           is_in [ "AIR"; "AIR REG" ] (Col.string "l_shipmode")
           && Col.string "l_shipinstruct" = string "DELIVER IN PERSON")
    |> join ~on:(on "l_partkey" "p_partkey") (t "part")
    |> filter
         Expr.(
           branch "Brand#12" [ "SM CASE"; "SM BOX"; "SM PACK"; "SM PKG" ] 1. 5
           || branch "Brand#23"
                [ "MED BAG"; "MED BOX"; "MED PKG"; "MED PACK" ]
                10. 10
           || branch "Brand#34"
                [ "LG CASE"; "LG BOX"; "LG PACK"; "LG PKG" ]
                20. 15)
    |> aggregate ~by:[] Expr.[ "revenue" := sum revenue ])

let q20 t _ =
  let shipped =
    Query.(
      t "lineitem"
      |> filter (during shipdate (day 1994 1 1) (day 1995 1 1))
      |> aggregate
           ~by:[ "l_partkey"; "l_suppkey" ]
           Expr.[ "half" := float 0.5 *. sum quantity ])
  in
  let stocked =
    Query.(
      t "partsupp"
      |> join ~kind:Semi
           ~on:(on "ps_partkey" "p_partkey")
           (t "part"
           |> filter
                Expr.(Str.(matches (prefix "forest")) (Col.string "p_name")))
      |> join
           ~on:Join.(on "ps_partkey" "l_partkey" && on "ps_suppkey" "l_suppkey")
           shipped
      |> filter
           Expr.(cast Type.float64 (Col.int "ps_availqty") > Col.float "half"))
  in
  Query.(
    t "supplier"
    |> join ~kind:Semi ~on:(on "s_suppkey" "ps_suppkey") stocked
    |> join ~kind:Semi
         ~on:(on "s_nationkey" "n_nationkey")
         (t "nation" |> filter Expr.(Col.string "n_name" = string "CANADA"))
    |> select Expr.[ keep (Sel.names [ "s_name"; "s_address" ]) ]
    |> sort [ Order.asc "s_name" ])

(* A late line qualifies when its order has another supplier, and no other
   supplier of the order is late: the order's late lines have one supplier, the
   least and the greatest of their suppliers being one. *)
let q21 t _ =
  let supplier = Col.int "l_suppkey" in
  let late = Expr.(Col.date "l_receiptdate" > Col.date "l_commitdate") in
  let late_supplier = Expr.(if_ late supplier null) in
  let lines =
    Query.select
      Expr.[ keep (Sel.names [ "l_orderkey"; "l_suppkey" ]); "late" := late ]
      (t "lineitem")
  in
  let orders =
    Query.(
      t "lineitem"
      |> aggregate ~by:[ "l_orderkey" ]
           Expr.
             [
               "suppliers" := n_unique supplier;
               "first_late" := min late_supplier;
               "last_late" := max late_supplier;
             ]
      |> filter
           Expr.(
             Col.int "suppliers" > int 1
             && Col.int "first_late" = Col.int "last_late"))
  in
  Query.(
    lines
    |> filter (Col.bool "late")
    |> join ~kind:Semi ~on:(Join.keys [ "l_orderkey" ]) orders
    |> join ~kind:Semi
         ~on:(on "l_orderkey" "o_orderkey")
         (t "orders" |> filter Expr.(Col.string "o_orderstatus" = string "F"))
    |> join ~on:(on "l_suppkey" "s_suppkey") (t "supplier")
    |> join ~kind:Semi
         ~on:(on "s_nationkey" "n_nationkey")
         (t "nation"
         |> filter Expr.(Col.string "n_name" = string "SAUDI ARABIA"))
    |> aggregate ~by:[ "s_name" ] Expr.[ "numwait" := rows ]
    |> top 100 [ Order.desc "numwait"; Order.asc "s_name" ])

let q22 t _ =
  let balance = Col.float "c_acctbal" in
  let customers =
    Query.(
      t "customer"
      |> derive
           Expr.
             [
               "cntrycode" :=
                 Str.slice ~offset:0 ~length:2 (Col.string "c_phone");
             ]
      |> filter
           Expr.(
             is_in
               [ "13"; "31"; "23"; "29"; "30"; "18"; "17" ]
               (Col.string "cntrycode")))
  in
  Query.(
    customers
    |> join ~kind:Anti ~on:(on "c_custkey" "o_custkey") (t "orders")
    |> join ~on:Join.all
         (customers
         |> filter Expr.(balance > float 0.)
         |> aggregate ~by:[] Expr.[ "average" := mean balance ])
    |> filter Expr.(balance > Col.float "average")
    |> aggregate ~by:[ "cntrycode" ]
         Expr.[ "numcust" := rows; "totacctbal" := sum balance ]
    |> sort [ Order.asc "cntrycode" ])

let queries =
  [
    q01;
    q02;
    q03;
    q04;
    q05;
    q06;
    q07;
    q08;
    q09;
    q10;
    q11;
    q12;
    q13;
    q14;
    q15;
    q16;
    q17;
    q18;
    q19;
    q20;
    q21;
    q22;
  ]

let workload ~data size =
  let sf =
    match String.split_on_char 'f' size with
    | [ "s"; sf ] when List.mem sf scales -> sf
    | _ ->
        invalid_arg
          (Printf.sprintf "unknown TPC-H size %S; expected one of %s" size
             (String.concat ", " (List.map (( ^ ) "sf") scales)))
  in
  let root = Filename.concat data ("tpch-sf" ^ sf) in
  let scale = float_of_string sf in
  {
    Workload.id = "tpch/" ^ size;
    tables =
      List.map
        (fun name ->
          (name, Workload.Parquet (Filename.concat root (name ^ ".parquet"))))
        tables;
    questions =
      List.mapi
        (fun i q ->
          {
            Workload.name = Printf.sprintf "q%02d" (i + 1);
            query = (fun t -> q t scale);
          })
        queries;
    ordered = true;
  }
