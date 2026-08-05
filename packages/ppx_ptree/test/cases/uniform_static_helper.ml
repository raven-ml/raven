type 'a t = { w : 'a; meta : 'a meta }
and 'a meta = { name : string } [@@deriving ptree]
