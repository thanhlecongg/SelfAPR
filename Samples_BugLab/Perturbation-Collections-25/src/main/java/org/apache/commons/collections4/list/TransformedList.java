[buglab_swap_variables]^super ( transformer, list ) ;^105^^^^^104^106^super ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] <init> [RETURN_TYPE] Transformer)   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] List  list  Transformer  transformer  boolean  long  serialVersionUID  
[buglab_swap_variables]^super (  transformer ) ;^105^^^^^104^106^super ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] <init> [RETURN_TYPE] Transformer)   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] List  list  Transformer  transformer  boolean  long  serialVersionUID  
[buglab_swap_variables]^super ( list ) ;^105^^^^^104^106^super ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] <init> [RETURN_TYPE] Transformer)   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] List  list  Transformer  transformer  boolean  long  serialVersionUID  
[buglab_swap_variables]^return new TransformedList<E> ( transformer, list ) ;^61^^^^^59^62^return new TransformedList<E> ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] transformingList [RETURN_TYPE] <E>   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] List  list  Transformer  transformer  boolean  long  serialVersionUID  
[buglab_swap_variables]^return new TransformedList<E> (  transformer ) ;^61^^^^^59^62^return new TransformedList<E> ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] transformingList [RETURN_TYPE] <E>   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] List  list  Transformer  transformer  boolean  long  serialVersionUID  
[buglab_swap_variables]^return new TransformedList<E> ( list ) ;^61^^^^^59^62^return new TransformedList<E> ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] transformingList [RETURN_TYPE] <E>   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] List  list  Transformer  transformer  boolean  long  serialVersionUID  
[buglab_swap_variables]^final TransformedList<E> decorated = new TransformedList<E> ( transformer, list ) ;^81^^^^^79^91^final TransformedList<E> decorated = new TransformedList<E> ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] transformedList [RETURN_TYPE] <E>   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  List  list  TransformedList  decorated  long  serialVersionUID  
[buglab_swap_variables]^final TransformedList<E> decorated = new TransformedList<E> (  transformer ) ;^81^^^^^79^91^final TransformedList<E> decorated = new TransformedList<E> ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] transformedList [RETURN_TYPE] <E>   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  List  list  TransformedList  decorated  long  serialVersionUID  
[buglab_swap_variables]^final TransformedList<E> decorated = new TransformedList<E> ( list ) ;^81^^^^^79^91^final TransformedList<E> decorated = new TransformedList<E> ( list, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] transformedList [RETURN_TYPE] <E>   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  List  list  TransformedList  decorated  long  serialVersionUID  
[buglab_swap_variables]^if  ( list != null && transformer != null && list.size (  )  > 0 )  {^82^^^^^79^91^if  ( transformer != null && list != null && list.size (  )  > 0 )  {^[CLASS] TransformedList TransformedListIterator  [METHOD] transformedList [RETURN_TYPE] <E>   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  List  list  TransformedList  decorated  long  serialVersionUID  
[buglab_swap_variables]^decorated.decorated (  ) .add ( value.transform ( transformer )  ) ;^87^^^^^79^91^decorated.decorated (  ) .add ( transformer.transform ( value )  ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] transformedList [RETURN_TYPE] <E>   List<E> list Transformer<? super E, ? extends E> transformer [VARIABLES] Transformer  transformer  boolean  E  value  E[]  values  List  list  TransformedList  decorated  long  serialVersionUID  
[buglab_swap_variables]^getList (  ) .add ( object, index ) ;^149^^^^^147^150^getList (  ) .add ( index, object ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] add [RETURN_TYPE] void   final int index E object [VARIABLES] boolean  E  object  long  serialVersionUID  int  index  
[buglab_swap_variables]^getList (  ) .add (  object ) ;^149^^^^^147^150^getList (  ) .add ( index, object ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] add [RETURN_TYPE] void   final int index E object [VARIABLES] boolean  E  object  long  serialVersionUID  int  index  
[buglab_swap_variables]^getList (  ) .add ( index ) ;^149^^^^^147^150^getList (  ) .add ( index, object ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] add [RETURN_TYPE] void   final int index E object [VARIABLES] boolean  E  object  long  serialVersionUID  int  index  
[buglab_swap_variables]^return getList (  ) .addAll ( coll, index ) ;^154^^^^^152^155^return getList (  ) .addAll ( index, coll ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] addAll [RETURN_TYPE] boolean   final int index Collection<? extends E> coll [VARIABLES] Collection  coll  boolean  long  serialVersionUID  int  index  
[buglab_swap_variables]^return getList (  ) .addAll (  coll ) ;^154^^^^^152^155^return getList (  ) .addAll ( index, coll ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] addAll [RETURN_TYPE] boolean   final int index Collection<? extends E> coll [VARIABLES] Collection  coll  boolean  long  serialVersionUID  int  index  
[buglab_swap_variables]^return getList (  ) .addAll ( index ) ;^154^^^^^152^155^return getList (  ) .addAll ( index, coll ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] addAll [RETURN_TYPE] boolean   final int index Collection<? extends E> coll [VARIABLES] Collection  coll  boolean  long  serialVersionUID  int  index  
[buglab_swap_variables]^return getList (  ) .set ( object, index ) ;^167^^^^^165^168^return getList (  ) .set ( index, object ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] set [RETURN_TYPE] E   final int index E object [VARIABLES] boolean  E  object  long  serialVersionUID  int  index  
[buglab_swap_variables]^return getList (  ) .set (  object ) ;^167^^^^^165^168^return getList (  ) .set ( index, object ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] set [RETURN_TYPE] E   final int index E object [VARIABLES] boolean  E  object  long  serialVersionUID  int  index  
[buglab_swap_variables]^return getList (  ) .set ( index ) ;^167^^^^^165^168^return getList (  ) .set ( index, object ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] set [RETURN_TYPE] E   final int index E object [VARIABLES] boolean  E  object  long  serialVersionUID  int  index  
[buglab_swap_variables]^final List<E> sub = getList (  ) .subList ( toIndex, fromIndex ) ;^171^^^^^170^173^final List<E> sub = getList (  ) .subList ( fromIndex, toIndex ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] subList [RETURN_TYPE] List   final int fromIndex final int toIndex [VARIABLES] List  sub  boolean  long  serialVersionUID  int  fromIndex  toIndex  
[buglab_swap_variables]^final List<E> sub = getList (  ) .subList (  toIndex ) ;^171^^^^^170^173^final List<E> sub = getList (  ) .subList ( fromIndex, toIndex ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] subList [RETURN_TYPE] List   final int fromIndex final int toIndex [VARIABLES] List  sub  boolean  long  serialVersionUID  int  fromIndex  toIndex  
[buglab_swap_variables]^final List<E> sub = getList (  ) .subList ( fromIndex ) ;^171^^^^^170^173^final List<E> sub = getList (  ) .subList ( fromIndex, toIndex ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] subList [RETURN_TYPE] List   final int fromIndex final int toIndex [VARIABLES] List  sub  boolean  long  serialVersionUID  int  fromIndex  toIndex  
[buglab_swap_variables]^return new TransformedList<E> ( transformer, sub ) ;^172^^^^^170^173^return new TransformedList<E> ( sub, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] subList [RETURN_TYPE] List   final int fromIndex final int toIndex [VARIABLES] List  sub  boolean  long  serialVersionUID  int  fromIndex  toIndex  
[buglab_swap_variables]^return new TransformedList<E> (  transformer ) ;^172^^^^^170^173^return new TransformedList<E> ( sub, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] subList [RETURN_TYPE] List   final int fromIndex final int toIndex [VARIABLES] List  sub  boolean  long  serialVersionUID  int  fromIndex  toIndex  
[buglab_swap_variables]^return new TransformedList<E> ( sub ) ;^172^^^^^170^173^return new TransformedList<E> ( sub, transformer ) ;^[CLASS] TransformedList TransformedListIterator  [METHOD] subList [RETURN_TYPE] List   final int fromIndex final int toIndex [VARIABLES] List  sub  boolean  long  serialVersionUID  int  fromIndex  toIndex  