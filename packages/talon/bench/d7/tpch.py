"""The 22 TPC-H queries, with the specification's validation parameters.

The SQL is the text of DuckDB's tpch extension, except query 11, whose
fraction is the specification's 0.0001 / SF instead of the scale-factor-1
constant. Every engine scans the same Parquet files inside the timed region.
"""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from workload import Question, Table, Workload

SCALES = ["0.1", "1", "10"]

TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]


def scale(sf: str) -> str:
    if sf not in SCALES:
        raise SystemExit(f"unknown TPC-H scale factor {sf!r}; expected one of {SCALES}")
    return sf


def directory(data: Path, sf: str) -> Path:
    return data / f"tpch-sf{sf}"


c = pl.col
revenue = c("l_extendedprice") * (1 - c("l_discount"))


def during(column: str, start: date, stop: date) -> pl.Expr:
    """[during column start stop] holds when [column] is in [[start, stop)]."""
    return c(column).is_between(start, stop, closed="left")


# Queries


def q01(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.lineitem.filter(c("l_shipdate") <= date(1998, 9, 2))
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            sum_qty=c("l_quantity").sum(),
            sum_base_price=c("l_extendedprice").sum(),
            sum_disc_price=revenue.sum(),
            sum_charge=(revenue * (1 + c("l_tax"))).sum(),
            avg_qty=c("l_quantity").mean(),
            avg_price=c("l_extendedprice").mean(),
            avg_disc=c("l_discount").mean(),
            count_order=pl.len(),
        )
        .sort("l_returnflag", "l_linestatus")
    )


def q02(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    europe = (
        t.partsupp.join(t.supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(t.nation, left_on="s_nationkey", right_on="n_nationkey")
        .join(
            t.region.filter(c("r_name") == "EUROPE"),
            left_on="n_regionkey",
            right_on="r_regionkey",
        )
    )
    return (
        t.part.filter((c("p_size") == 15) & c("p_type").str.ends_with("BRASS"))
        .join(europe, left_on="p_partkey", right_on="ps_partkey")
        .filter(c("ps_supplycost") == c("ps_supplycost").min().over("p_partkey"))
        .select(
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        )
        .sort(
            "s_acctbal",
            "n_name",
            "s_name",
            "p_partkey",
            descending=[True, False, False, False],
        )
        .head(100)
    )


def q03(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.customer.filter(c("c_mktsegment") == "BUILDING")
        .join(
            t.orders.filter(c("o_orderdate") < date(1995, 3, 15)),
            left_on="c_custkey",
            right_on="o_custkey",
        )
        .join(
            t.lineitem.filter(c("l_shipdate") > date(1995, 3, 15)),
            left_on="o_orderkey",
            right_on="l_orderkey",
        )
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(revenue=revenue.sum())
        .select(
            c("o_orderkey").alias("l_orderkey"),
            "revenue",
            "o_orderdate",
            "o_shippriority",
        )
        .sort("revenue", "o_orderdate", descending=[True, False])
        .head(10)
    )


def q04(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.orders.filter(during("o_orderdate", date(1993, 7, 1), date(1993, 10, 1)))
        .join(
            t.lineitem.filter(c("l_commitdate") < c("l_receiptdate")),
            left_on="o_orderkey",
            right_on="l_orderkey",
            how="semi",
        )
        .group_by("o_orderpriority")
        .agg(order_count=pl.len())
        .sort("o_orderpriority")
    )


def q05(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.customer.join(
            t.orders.filter(during("o_orderdate", date(1994, 1, 1), date(1995, 1, 1))),
            left_on="c_custkey",
            right_on="o_custkey",
        )
        .join(t.lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            t.supplier,
            left_on=["l_suppkey", "c_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .join(t.nation, left_on="c_nationkey", right_on="n_nationkey")
        .join(
            t.region.filter(c("r_name") == "ASIA"),
            left_on="n_regionkey",
            right_on="r_regionkey",
        )
        .group_by("n_name")
        .agg(revenue=revenue.sum())
        .sort("revenue", descending=True)
    )


def q06(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return t.lineitem.filter(
        during("l_shipdate", date(1994, 1, 1), date(1995, 1, 1))
        & c("l_discount").is_between(0.05, 0.07)
        & (c("l_quantity") < 24)
    ).select(revenue=(c("l_extendedprice") * c("l_discount")).sum())


def q07(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    def nation(alias: str) -> pl.LazyFrame:
        return t.nation.select("n_nationkey", c("n_name").alias(alias))

    return (
        t.lineitem.filter(
            c("l_shipdate").is_between(date(1995, 1, 1), date(1996, 12, 31))
        )
        .join(t.supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(t.orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(t.customer, left_on="o_custkey", right_on="c_custkey")
        .join(nation("supp_nation"), left_on="s_nationkey", right_on="n_nationkey")
        .join(nation("cust_nation"), left_on="c_nationkey", right_on="n_nationkey")
        .filter(
            ((c("supp_nation") == "FRANCE") & (c("cust_nation") == "GERMANY"))
            | ((c("supp_nation") == "GERMANY") & (c("cust_nation") == "FRANCE"))
        )
        .group_by("supp_nation", "cust_nation", l_year=c("l_shipdate").dt.year())
        .agg(revenue=revenue.sum())
        .sort("supp_nation", "cust_nation", "l_year")
    )


def q08(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.part.filter(c("p_type") == "ECONOMY ANODIZED STEEL")
        .join(t.lineitem, left_on="p_partkey", right_on="l_partkey")
        .join(t.supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(
            t.orders.filter(
                c("o_orderdate").is_between(date(1995, 1, 1), date(1996, 12, 31))
            ),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .join(t.customer, left_on="o_custkey", right_on="c_custkey")
        .join(
            t.nation.select("n_nationkey", "n_regionkey"),
            left_on="c_nationkey",
            right_on="n_nationkey",
        )
        .join(
            t.region.filter(c("r_name") == "AMERICA"),
            left_on="n_regionkey",
            right_on="r_regionkey",
        )
        .join(
            t.nation.select("n_nationkey", "n_name"),
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
        .group_by(o_year=c("o_orderdate").dt.year())
        .agg(
            mkt_share=pl.when(c("n_name") == "BRAZIL").then(revenue).otherwise(0).sum()
            / revenue.sum()
        )
        .sort("o_year")
    )


def q09(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.part.filter(c("p_name").str.contains("green", literal=True))
        .join(t.lineitem, left_on="p_partkey", right_on="l_partkey")
        .join(t.supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(
            t.partsupp,
            left_on=["p_partkey", "l_suppkey"],
            right_on=["ps_partkey", "ps_suppkey"],
        )
        .join(t.orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(t.nation, left_on="s_nationkey", right_on="n_nationkey")
        .group_by(nation=c("n_name"), o_year=c("o_orderdate").dt.year())
        .agg(sum_profit=(revenue - c("ps_supplycost") * c("l_quantity")).sum())
        .sort("nation", "o_year", descending=[False, True])
    )


def q10(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.customer.join(
            t.orders.filter(during("o_orderdate", date(1993, 10, 1), date(1994, 1, 1))),
            left_on="c_custkey",
            right_on="o_custkey",
        )
        .join(
            t.lineitem.filter(c("l_returnflag") == "R"),
            left_on="o_orderkey",
            right_on="l_orderkey",
        )
        .join(t.nation, left_on="c_nationkey", right_on="n_nationkey")
        .group_by(
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        )
        .agg(revenue=revenue.sum())
        .select(
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        )
        .sort("revenue", descending=True)
        .head(20)
    )


def q11(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    germany = (
        t.partsupp.join(t.supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(
            t.nation.filter(c("n_name") == "GERMANY"),
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
        .select("ps_partkey", value=c("ps_supplycost") * c("ps_availqty"))
    )
    return (
        germany.group_by("ps_partkey")
        .agg(c("value").sum())
        .join(germany.select(threshold=c("value").sum() * 0.0001 / sf), how="cross")
        .filter(c("value") > c("threshold"))
        .select("ps_partkey", "value")
        .sort("value", descending=True)
    )


def q12(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    high = c("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])
    return (
        t.orders.join(
            t.lineitem.filter(
                c("l_shipmode").is_in(["MAIL", "SHIP"])
                & (c("l_commitdate") < c("l_receiptdate"))
                & (c("l_shipdate") < c("l_commitdate"))
                & during("l_receiptdate", date(1994, 1, 1), date(1995, 1, 1))
            ),
            left_on="o_orderkey",
            right_on="l_orderkey",
        )
        .group_by("l_shipmode")
        .agg(high_line_count=high.sum(), low_line_count=high.not_().sum())
        .sort("l_shipmode")
    )


def q13(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.customer.join(
            t.orders.filter(c("o_comment").str.contains("special.*requests").not_()),
            left_on="c_custkey",
            right_on="o_custkey",
            how="left",
        )
        .group_by("c_custkey")
        .agg(c_count=c("o_orderkey").count())
        .group_by("c_count")
        .agg(custdist=pl.len())
        .sort("custdist", "c_count", descending=True)
    )


def q14(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    promo = pl.when(c("p_type").str.starts_with("PROMO")).then(revenue).otherwise(0)
    return (
        t.lineitem.filter(during("l_shipdate", date(1995, 9, 1), date(1995, 10, 1)))
        .join(t.part, left_on="l_partkey", right_on="p_partkey")
        .select(promo_revenue=100.0 * promo.sum() / revenue.sum())
    )


def q15(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    totals = (
        t.lineitem.filter(during("l_shipdate", date(1996, 1, 1), date(1996, 4, 1)))
        .group_by("l_suppkey")
        .agg(total_revenue=revenue.sum())
    )
    return (
        t.supplier.join(totals, left_on="s_suppkey", right_on="l_suppkey")
        .filter(c("total_revenue") == c("total_revenue").max())
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("s_suppkey")
    )


def q16(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.part.filter(
            (c("p_brand") != "Brand#45")
            & c("p_type").str.starts_with("MEDIUM POLISHED").not_()
            & c("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9])
        )
        .join(t.partsupp, left_on="p_partkey", right_on="ps_partkey")
        .join(
            t.supplier.filter(c("s_comment").str.contains("Customer.*Complaints")),
            left_on="ps_suppkey",
            right_on="s_suppkey",
            how="anti",
        )
        .group_by("p_brand", "p_type", "p_size")
        .agg(supplier_cnt=c("ps_suppkey").n_unique())
        .sort(
            "supplier_cnt",
            "p_brand",
            "p_type",
            "p_size",
            descending=[True, False, False, False],
        )
    )


def q17(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    return (
        t.part.filter((c("p_brand") == "Brand#23") & (c("p_container") == "MED BOX"))
        .join(t.lineitem, left_on="p_partkey", right_on="l_partkey")
        .filter(c("l_quantity") < 0.2 * c("l_quantity").mean().over("p_partkey"))
        .select(avg_yearly=c("l_extendedprice").sum() / 7.0)
    )


def q18(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    large = (
        t.lineitem.group_by("l_orderkey")
        .agg(c("l_quantity").sum())
        .filter(c("l_quantity") > 300)
    )
    return (
        t.orders.join(large, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .join(t.customer, left_on="o_custkey", right_on="c_custkey")
        .join(t.lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
        .agg(c("l_quantity").sum().alias("sum(l_quantity)"))
        .select(
            "c_name",
            c("o_custkey").alias("c_custkey"),
            "o_orderkey",
            "o_orderdate",
            "o_totalprice",
            "sum(l_quantity)",
        )
        .sort("o_totalprice", "o_orderdate", descending=[True, False])
        .head(100)
    )


def q19(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    def branch(brand: str, containers: list[str], low: int, size: int) -> pl.Expr:
        return (
            (c("p_brand") == brand)
            & c("p_container").is_in(containers)
            & c("l_quantity").is_between(low, low + 10)
            & c("p_size").is_between(1, size)
        )

    return (
        t.lineitem.filter(
            c("l_shipmode").is_in(["AIR", "AIR REG"])
            & (c("l_shipinstruct") == "DELIVER IN PERSON")
        )
        .join(t.part, left_on="l_partkey", right_on="p_partkey")
        .filter(
            branch("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 5)
            | branch("Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10, 10)
            | branch("Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20, 15)
        )
        .select(revenue=revenue.sum())
    )


def q20(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    shipped = (
        t.lineitem.filter(during("l_shipdate", date(1994, 1, 1), date(1995, 1, 1)))
        .group_by("l_partkey", "l_suppkey")
        .agg(half=0.5 * c("l_quantity").sum())
    )
    stocked = (
        t.partsupp.join(
            t.part.filter(c("p_name").str.starts_with("forest")),
            left_on="ps_partkey",
            right_on="p_partkey",
            how="semi",
        )
        .join(
            shipped,
            left_on=["ps_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .filter(c("ps_availqty") > c("half"))
    )
    return (
        t.supplier.join(stocked, left_on="s_suppkey", right_on="ps_suppkey", how="semi")
        .join(
            t.nation.filter(c("n_name") == "CANADA"),
            left_on="s_nationkey",
            right_on="n_nationkey",
            how="semi",
        )
        .select("s_name", "s_address")
        .sort("s_name")
    )


def q21(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    lines = t.lineitem.select(
        "l_orderkey", "l_suppkey", late=c("l_receiptdate") > c("l_commitdate")
    )
    # A late line qualifies when its order has another supplier, and no other
    # supplier of the order is late: the order's late suppliers are its own.
    orders = (
        lines.group_by("l_orderkey")
        .agg(
            suppliers=c("l_suppkey").n_unique(),
            late_suppliers=c("l_suppkey").filter(c("late")).n_unique(),
        )
        .filter((c("suppliers") > 1) & (c("late_suppliers") == 1))
    )
    return (
        lines.filter(c("late"))
        .join(orders, on="l_orderkey", how="semi")
        .join(
            t.orders.filter(c("o_orderstatus") == "F"),
            left_on="l_orderkey",
            right_on="o_orderkey",
            how="semi",
        )
        .join(t.supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(
            t.nation.filter(c("n_name") == "SAUDI ARABIA"),
            left_on="s_nationkey",
            right_on="n_nationkey",
            how="semi",
        )
        .group_by("s_name")
        .agg(numwait=pl.len())
        .sort("numwait", "s_name", descending=[True, False])
        .head(100)
    )


def q22(t: SimpleNamespace, sf: float) -> pl.LazyFrame:
    customers = t.customer.with_columns(cntrycode=c("c_phone").str.slice(0, 2)).filter(
        c("cntrycode").is_in(["13", "31", "23", "29", "30", "18", "17"])
    )
    return (
        customers.join(t.orders, left_on="c_custkey", right_on="o_custkey", how="anti")
        .join(
            customers.filter(c("c_acctbal") > 0.0).select(
                average=c("c_acctbal").mean()
            ),
            how="cross",
        )
        .filter(c("c_acctbal") > c("average"))
        .group_by("cntrycode")
        .agg(numcust=pl.len(), totacctbal=c("c_acctbal").sum())
        .sort("cntrycode")
    )


POLARS = [
    q01,
    q02,
    q03,
    q04,
    q05,
    q06,
    q07,
    q08,
    q09,
    q10,
    q11,
    q12,
    q13,
    q14,
    q15,
    q16,
    q17,
    q18,
    q19,
    q20,
    q21,
    q22,
]


# SQL

SQL = [
    """
SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity) AS sum_qty,
    sum(l_extendedprice) AS sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    avg(l_quantity) AS avg_qty,
    avg(l_extendedprice) AS avg_price,
    avg(l_discount) AS avg_disc,
    count(*) AS count_order
FROM
    lineitem
WHERE
    l_shipdate <= CAST('1998-09-02' AS date)
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus
""",
    """
SELECT
    s_acctbal,
    s_name,
    n_name,
    p_partkey,
    p_mfgr,
    s_address,
    s_phone,
    s_comment
FROM
    part,
    supplier,
    partsupp,
    nation,
    region
WHERE
    p_partkey = ps_partkey
    AND s_suppkey = ps_suppkey
    AND p_size = 15
    AND p_type LIKE '%BRASS'
    AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey
    AND r_name = 'EUROPE'
    AND ps_supplycost = (
        SELECT
            min(ps_supplycost)
        FROM
            partsupp,
            supplier,
            nation,
            region
        WHERE
            p_partkey = ps_partkey
            AND s_suppkey = ps_suppkey
            AND s_nationkey = n_nationkey
            AND n_regionkey = r_regionkey
            AND r_name = 'EUROPE')
ORDER BY
    s_acctbal DESC,
    n_name,
    s_name,
    p_partkey
LIMIT 100
""",
    """
SELECT
    l_orderkey,
    sum(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM
    customer,
    orders,
    lineitem
WHERE
    c_mktsegment = 'BUILDING'
    AND c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate < CAST('1995-03-15' AS date)
    AND l_shipdate > CAST('1995-03-15' AS date)
GROUP BY
    l_orderkey,
    o_orderdate,
    o_shippriority
ORDER BY
    revenue DESC,
    o_orderdate
LIMIT 10
""",
    """
SELECT
    o_orderpriority,
    count(*) AS order_count
FROM
    orders
WHERE
    o_orderdate >= CAST('1993-07-01' AS date)
    AND o_orderdate < CAST('1993-10-01' AS date)
    AND EXISTS (
        SELECT
            *
        FROM
            lineitem
        WHERE
            l_orderkey = o_orderkey
            AND l_commitdate < l_receiptdate)
GROUP BY
    o_orderpriority
ORDER BY
    o_orderpriority
""",
    """
SELECT
    n_name,
    sum(l_extendedprice * (1 - l_discount)) AS revenue
FROM
    customer,
    orders,
    lineitem,
    supplier,
    nation,
    region
WHERE
    c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND l_suppkey = s_suppkey
    AND c_nationkey = s_nationkey
    AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey
    AND r_name = 'ASIA'
    AND o_orderdate >= CAST('1994-01-01' AS date)
    AND o_orderdate < CAST('1995-01-01' AS date)
GROUP BY
    n_name
ORDER BY
    revenue DESC
""",
    """
SELECT
    sum(l_extendedprice * l_discount) AS revenue
FROM
    lineitem
WHERE
    l_shipdate >= CAST('1994-01-01' AS date)
    AND l_shipdate < CAST('1995-01-01' AS date)
    AND l_discount BETWEEN 0.05
    AND 0.07
    AND l_quantity < 24
""",
    """
SELECT
    supp_nation,
    cust_nation,
    l_year,
    sum(volume) AS revenue
FROM (
    SELECT
        n1.n_name AS supp_nation,
        n2.n_name AS cust_nation,
        extract(year FROM l_shipdate) AS l_year,
        l_extendedprice * (1 - l_discount) AS volume
    FROM
        supplier,
        lineitem,
        orders,
        customer,
        nation n1,
        nation n2
    WHERE
        s_suppkey = l_suppkey
        AND o_orderkey = l_orderkey
        AND c_custkey = o_custkey
        AND s_nationkey = n1.n_nationkey
        AND c_nationkey = n2.n_nationkey
        AND ((n1.n_name = 'FRANCE'
                AND n2.n_name = 'GERMANY')
            OR (n1.n_name = 'GERMANY'
                AND n2.n_name = 'FRANCE'))
        AND l_shipdate BETWEEN CAST('1995-01-01' AS date)
        AND CAST('1996-12-31' AS date)) AS shipping
GROUP BY
    supp_nation,
    cust_nation,
    l_year
ORDER BY
    supp_nation,
    cust_nation,
    l_year
""",
    """
SELECT
    o_year,
    sum(
        CASE WHEN nation = 'BRAZIL' THEN
            volume
        ELSE
            0
        END) / sum(volume) AS mkt_share
FROM (
    SELECT
        extract(year FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) AS volume,
        n2.n_name AS nation
    FROM
        part,
        supplier,
        lineitem,
        orders,
        customer,
        nation n1,
        nation n2,
        region
    WHERE
        p_partkey = l_partkey
        AND s_suppkey = l_suppkey
        AND l_orderkey = o_orderkey
        AND o_custkey = c_custkey
        AND c_nationkey = n1.n_nationkey
        AND n1.n_regionkey = r_regionkey
        AND r_name = 'AMERICA'
        AND s_nationkey = n2.n_nationkey
        AND o_orderdate BETWEEN CAST('1995-01-01' AS date)
        AND CAST('1996-12-31' AS date)
        AND p_type = 'ECONOMY ANODIZED STEEL') AS all_nations
GROUP BY
    o_year
ORDER BY
    o_year
""",
    """
SELECT
    nation,
    o_year,
    sum(amount) AS sum_profit
FROM (
    SELECT
        n_name AS nation,
        extract(year FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM
        part,
        supplier,
        lineitem,
        partsupp,
        orders,
        nation
    WHERE
        s_suppkey = l_suppkey
        AND ps_suppkey = l_suppkey
        AND ps_partkey = l_partkey
        AND p_partkey = l_partkey
        AND o_orderkey = l_orderkey
        AND s_nationkey = n_nationkey
        AND p_name LIKE '%green%') AS profit
GROUP BY
    nation,
    o_year
ORDER BY
    nation,
    o_year DESC
""",
    """
SELECT
    c_custkey,
    c_name,
    sum(l_extendedprice * (1 - l_discount)) AS revenue,
    c_acctbal,
    n_name,
    c_address,
    c_phone,
    c_comment
FROM
    customer,
    orders,
    lineitem,
    nation
WHERE
    c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate >= CAST('1993-10-01' AS date)
    AND o_orderdate < CAST('1994-01-01' AS date)
    AND l_returnflag = 'R'
    AND c_nationkey = n_nationkey
GROUP BY
    c_custkey,
    c_name,
    c_acctbal,
    c_phone,
    n_name,
    c_address,
    c_comment
ORDER BY
    revenue DESC
LIMIT 20
""",
    """
SELECT
    ps_partkey,
    sum(ps_supplycost * ps_availqty) AS value
FROM
    partsupp,
    supplier,
    nation
WHERE
    ps_suppkey = s_suppkey
    AND s_nationkey = n_nationkey
    AND n_name = 'GERMANY'
GROUP BY
    ps_partkey
HAVING
    sum(ps_supplycost * ps_availqty) > (
        SELECT
            sum(ps_supplycost * ps_availqty) * 0.0001 / {sf}
        FROM
            partsupp,
            supplier,
            nation
        WHERE
            ps_suppkey = s_suppkey
            AND s_nationkey = n_nationkey
            AND n_name = 'GERMANY')
ORDER BY
    value DESC
""",
    """
SELECT
    l_shipmode,
    sum(
        CASE WHEN o_orderpriority = '1-URGENT'
            OR o_orderpriority = '2-HIGH' THEN
            1
        ELSE
            0
        END) AS high_line_count,
    sum(
        CASE WHEN o_orderpriority <> '1-URGENT'
            AND o_orderpriority <> '2-HIGH' THEN
            1
        ELSE
            0
        END) AS low_line_count
FROM
    orders,
    lineitem
WHERE
    o_orderkey = l_orderkey
    AND l_shipmode IN ('MAIL', 'SHIP')
    AND l_commitdate < l_receiptdate
    AND l_shipdate < l_commitdate
    AND l_receiptdate >= CAST('1994-01-01' AS date)
    AND l_receiptdate < CAST('1995-01-01' AS date)
GROUP BY
    l_shipmode
ORDER BY
    l_shipmode
""",
    """
SELECT
    c_count,
    count(*) AS custdist
FROM (
    SELECT
        c_custkey,
        count(o_orderkey)
    FROM
        customer
    LEFT OUTER JOIN orders ON c_custkey = o_custkey
    AND o_comment NOT LIKE '%special%requests%'
GROUP BY
    c_custkey) AS c_orders (c_custkey,
        c_count)
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC
""",
    """
SELECT
    100.00 * sum(
        CASE WHEN p_type LIKE 'PROMO%' THEN
            l_extendedprice * (1 - l_discount)
        ELSE
            0
        END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM
    lineitem,
    part
WHERE
    l_partkey = p_partkey
    AND l_shipdate >= date '1995-09-01'
    AND l_shipdate < CAST('1995-10-01' AS date)
""",
    """
WITH revenue AS (
    SELECT
        l_suppkey AS supplier_no,
        sum(l_extendedprice * (1 - l_discount)) AS total_revenue
    FROM
        lineitem
    WHERE
        l_shipdate >= CAST('1996-01-01' AS date)
      AND l_shipdate < CAST('1996-04-01' AS date)
    GROUP BY
        supplier_no
)
SELECT
    s_suppkey,
    s_name,
    s_address,
    s_phone,
    total_revenue
FROM
    supplier,
    revenue
WHERE
    s_suppkey = supplier_no
    AND total_revenue = (
        SELECT
            max(total_revenue)
        FROM revenue)
ORDER BY
    s_suppkey
""",
    """
SELECT
    p_brand,
    p_type,
    p_size,
    count(DISTINCT ps_suppkey) AS supplier_cnt
FROM
    partsupp,
    part
WHERE
    p_partkey = ps_partkey
    AND p_brand <> 'Brand#45'
    AND p_type NOT LIKE 'MEDIUM POLISHED%'
    AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
    AND ps_suppkey NOT IN (
        SELECT
            s_suppkey
        FROM
            supplier
        WHERE
            s_comment LIKE '%Customer%Complaints%')
GROUP BY
    p_brand,
    p_type,
    p_size
ORDER BY
    supplier_cnt DESC,
    p_brand,
    p_type,
    p_size
""",
    """
SELECT
    sum(l_extendedprice) / 7.0 AS avg_yearly
FROM
    lineitem,
    part
WHERE
    p_partkey = l_partkey
    AND p_brand = 'Brand#23'
    AND p_container = 'MED BOX'
    AND l_quantity < (
        SELECT
            0.2 * avg(l_quantity)
        FROM
            lineitem
        WHERE
            l_partkey = p_partkey)
""",
    """
SELECT
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice,
    sum(l_quantity)
FROM
    customer,
    orders,
    lineitem
WHERE
    o_orderkey IN (
        SELECT
            l_orderkey
        FROM
            lineitem
        GROUP BY
            l_orderkey
        HAVING
            sum(l_quantity) > 300)
    AND c_custkey = o_custkey
    AND o_orderkey = l_orderkey
GROUP BY
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice
ORDER BY
    o_totalprice DESC,
    o_orderdate
LIMIT 100
""",
    """
SELECT
    sum(l_extendedprice * (1 - l_discount)) AS revenue
FROM
    lineitem,
    part
WHERE (p_partkey = l_partkey
    AND p_brand = 'Brand#12'
    AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
    AND l_quantity >= 1
    AND l_quantity <= 1 + 10
    AND p_size BETWEEN 1 AND 5
    AND l_shipmode IN ('AIR', 'AIR REG')
    AND l_shipinstruct = 'DELIVER IN PERSON')
    OR (p_partkey = l_partkey
        AND p_brand = 'Brand#23'
        AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        AND l_quantity >= 10
        AND l_quantity <= 10 + 10
        AND p_size BETWEEN 1 AND 10
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON')
    OR (p_partkey = l_partkey
        AND p_brand = 'Brand#34'
        AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        AND l_quantity >= 20
        AND l_quantity <= 20 + 10
        AND p_size BETWEEN 1 AND 15
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON')
""",
    """
SELECT
    s_name,
    s_address
FROM
    supplier,
    nation
WHERE
    s_suppkey IN (
        SELECT
            ps_suppkey
        FROM
            partsupp
        WHERE
            ps_partkey IN (
                SELECT
                    p_partkey
                FROM
                    part
                WHERE
                    p_name LIKE 'forest%')
                AND ps_availqty > (
                    SELECT
                        0.5 * sum(l_quantity)
                    FROM
                        lineitem
                    WHERE
                        l_partkey = ps_partkey
                        AND l_suppkey = ps_suppkey
                        AND l_shipdate >= CAST('1994-01-01' AS date)
                        AND l_shipdate < CAST('1995-01-01' AS date)))
            AND s_nationkey = n_nationkey
            AND n_name = 'CANADA'
        ORDER BY
            s_name
""",
    """
SELECT
    s_name,
    count(*) AS numwait
FROM
    supplier,
    lineitem l1,
    orders,
    nation
WHERE
    s_suppkey = l1.l_suppkey
    AND o_orderkey = l1.l_orderkey
    AND o_orderstatus = 'F'
    AND l1.l_receiptdate > l1.l_commitdate
    AND EXISTS (
        SELECT
            *
        FROM
            lineitem l2
        WHERE
            l2.l_orderkey = l1.l_orderkey
            AND l2.l_suppkey <> l1.l_suppkey)
    AND NOT EXISTS (
        SELECT
            *
        FROM
            lineitem l3
        WHERE
            l3.l_orderkey = l1.l_orderkey
            AND l3.l_suppkey <> l1.l_suppkey
            AND l3.l_receiptdate > l3.l_commitdate)
    AND s_nationkey = n_nationkey
    AND n_name = 'SAUDI ARABIA'
GROUP BY
    s_name
ORDER BY
    numwait DESC,
    s_name
LIMIT 100
""",
    """
SELECT
    cntrycode,
    count(*) AS numcust,
    sum(c_acctbal) AS totacctbal
FROM (
    SELECT
        substring(c_phone FROM 1 FOR 2) AS cntrycode,
        c_acctbal
    FROM
        customer
    WHERE
        substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND c_acctbal > (
            SELECT
                avg(c_acctbal)
            FROM
                customer
            WHERE
                c_acctbal > 0.00
                AND substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17'))
            AND NOT EXISTS (
                SELECT
                    *
                FROM
                    orders
                WHERE
                    o_custkey = c_custkey)) AS custsale
GROUP BY
    cntrycode
ORDER BY
    cntrycode
""",
]


def workload(data: Path, sf: str) -> Workload:
    root = directory(data, scale(sf))
    queries = zip(SQL, POLARS)
    return Workload(
        id=f"tpch/sf{sf}",
        tables={name: Table(root / f"{name}.parquet", None) for name in TABLES},
        questions=[
            Question(
                f"q{i:02d}",
                sql.replace("{sf}", sf),
                lambda t, query=query: query(t, float(sf)),
            )
            for i, (sql, query) in enumerate(queries, 1)
        ],
        ordered=True,
        spill=data / "duckdb-tmp",
    )
