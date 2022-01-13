[REPLACE]^private Object left, right;^27^^^^^^^[REPLACE] private Item left, right;^ [CLASS] LocationTree Item 1 2  
[REPLACE]^private ArrayList<Comparator<Item>> comparators  = null ;^61^^^^^^^[REPLACE] private ArrayList<Comparator<Item>> comparators = new ArrayList<> (  ) ;^ [CLASS] LocationTree Item 1 2  
[ADD]^^96^^^^^95^123^[ADD] int direction = comparators.get ( depth % 2 ) .compare ( search, current ) ;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[ADD]^^98^^^^^95^123^[ADD] Item next, other;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^Item next  ;^98^^^^^95^123^[REPLACE] Item next, other;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^if  ( direction  !=  3 )  {^99^^^^^95^123^[REPLACE] if  ( direction < 0 )  {^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[ADD]^^99^100^101^102^103^95^123^[ADD] if  ( direction < 0 )  { next = current.left; other = current.right; } else { next = current.right;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^next = current.left  ;^103^^^^^99^105^[REPLACE] next = current.right;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^other = current.right  ;^104^^^^^99^105^[REPLACE] other = current.left;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^next = current.right  ;^100^^^^^95^123^[REPLACE] next = current.left;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^other = current.left  ;^101^^^^^95^123^[REPLACE] other = current.right;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^next = current.left  ;^103^^^^^95^123^[REPLACE] next = current.right;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^other = current.right  ;^104^^^^^95^123^[REPLACE] other = current.left;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[ADD]^^107^^^^^95^123^[ADD] Item best = current;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^if  ( search == true )  {^108^^^^^95^123^[REPLACE] if  ( next != null )  {^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^best = findNearest ( next, search,direction  3 ) ;^109^^^^^95^123^[REPLACE] best = findNearest ( next, search, depth + 1 ) ;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^best = findNearest ( next, search, direction  0 ) ;^109^^^^^95^123^[REPLACE] best = findNearest ( next, search, depth + 1 ) ;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^if  ( current .axisSquaredDistance ( right , direction )    <=  best^112^^^^^95^123^[REPLACE] if  ( current.squaredDistance ( search )  < best.squaredDistance ( search )  )  {^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[ADD]^best = current;^112^113^114^^^95^123^[ADD] if  ( current.squaredDistance ( search )  < best.squaredDistance ( search )  )  { best = current; }^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^best =  search;^113^^^^^95^123^[REPLACE] best = current;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^if  ( other == null && current.axisSquaredDistance ( search, depth % 2 )  < best.squaredDistance ( search )  )  {^115^^^^^95^123^[REPLACE] if  ( other != null && current.axisSquaredDistance ( search, depth % 2 )  < best.squaredDistance ( search )  )  {^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^if  ( possibleBest.squaredDistance ( search )   !=  best.squaredDistance ( search )  )  {^117^^^^^95^123^[REPLACE] if  ( possibleBest.squaredDistance ( search )  < best.squaredDistance ( search )  )  {^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[ADD]^best = possibleBest;^117^118^119^^^95^123^[ADD] if  ( possibleBest.squaredDistance ( search )  < best.squaredDistance ( search )  )  { best = possibleBest; }^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^best =  o2;^118^^^^^95^123^[REPLACE] best = possibleBest;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^best =  search;^118^^^^^95^123^[REPLACE] best = possibleBest;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[ADD]^^116^^^^^95^123^[ADD] Item possibleBest = findNearest ( other, search, depth + 1 ) ;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^if  ( root.squaredDistance ( search )   <=  best.squaredDistance ( search )  )  {^117^^^^^95^123^[REPLACE] if  ( possibleBest.squaredDistance ( search )  < best.squaredDistance ( search )  )  {^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[ADD]^^117^118^119^^^95^123^[ADD] if  ( possibleBest.squaredDistance ( search )  < best.squaredDistance ( search )  )  { best = possibleBest; }^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^best =  root;^118^^^^^95^123^[REPLACE] best = possibleBest;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^Item possibleBest = findNearest ( other, search, depth  1L ) ;^116^^^^^95^123^[REPLACE] Item possibleBest = findNearest ( other, search, depth + 1 ) ;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^return search;^122^^^^^95^123^[REPLACE] return best;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] LocationTree Item 1 2   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
[REPLACE]^private Object left, right;^27^^^^^^^[REPLACE] private Item left, right;^[METHOD] findNearest [TYPE] LocationTree$Item [PARAMETER] Item current Item search int depth [CLASS] Item   [TYPE]  ArrayList comparators  [TYPE]  Item best  current  item  left  median  next  o1  o2  other  possibleBest  right  root  search  [TYPE]  String data  [TYPE]  boolean false  true  [TYPE]  float x  y  [TYPE]  int depth  direction 
