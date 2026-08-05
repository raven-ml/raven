type 'a layer = { lw : 'a; lb : 'a option }

and 'a t = {
  layers : 'a layer list;
  head : 'a layer;
  pair : 'a * 'a;
  tag : string;
}
[@@deriving ptree]
