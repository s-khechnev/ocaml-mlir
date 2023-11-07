type shape = int array
type var = string

type expr =
  | Num of float
  | Literal of shape * expr list
  | Var of var
  | VarDecl of string * shape * expr
  | Return of expr option
  | BinOp of char * expr * expr
  | Call of string * expr list
  | Print of expr

type proto = Prototype of string * var list
type func = Function of proto * expr list
type modul = func list
