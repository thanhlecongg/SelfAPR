[buglab_swap_variables]^super ( predicate, bag ) ;^77^^^^^76^78^super ( bag, predicate ) ;^[CLASS] PredicatedBag  [METHOD] <init> [RETURN_TYPE] Predicate)   Bag<E> bag Predicate<? super E> predicate [VARIABLES] Bag  bag  boolean  long  serialVersionUID  Predicate  predicate  
[buglab_swap_variables]^super (  predicate ) ;^77^^^^^76^78^super ( bag, predicate ) ;^[CLASS] PredicatedBag  [METHOD] <init> [RETURN_TYPE] Predicate)   Bag<E> bag Predicate<? super E> predicate [VARIABLES] Bag  bag  boolean  long  serialVersionUID  Predicate  predicate  
[buglab_swap_variables]^super ( bag ) ;^77^^^^^76^78^super ( bag, predicate ) ;^[CLASS] PredicatedBag  [METHOD] <init> [RETURN_TYPE] Predicate)   Bag<E> bag Predicate<? super E> predicate [VARIABLES] Bag  bag  boolean  long  serialVersionUID  Predicate  predicate  
[buglab_swap_variables]^return new PredicatedBag<E> ( predicate, bag ) ;^61^^^^^60^62^return new PredicatedBag<E> ( bag, predicate ) ;^[CLASS] PredicatedBag  [METHOD] predicatedBag [RETURN_TYPE] <E>   Bag<E> bag Predicate<? super E> predicate [VARIABLES] Bag  bag  boolean  long  serialVersionUID  Predicate  predicate  
[buglab_swap_variables]^return new PredicatedBag<E> (  predicate ) ;^61^^^^^60^62^return new PredicatedBag<E> ( bag, predicate ) ;^[CLASS] PredicatedBag  [METHOD] predicatedBag [RETURN_TYPE] <E>   Bag<E> bag Predicate<? super E> predicate [VARIABLES] Bag  bag  boolean  long  serialVersionUID  Predicate  predicate  
[buglab_swap_variables]^return new PredicatedBag<E> ( bag ) ;^61^^^^^60^62^return new PredicatedBag<E> ( bag, predicate ) ;^[CLASS] PredicatedBag  [METHOD] predicatedBag [RETURN_TYPE] <E>   Bag<E> bag Predicate<? super E> predicate [VARIABLES] Bag  bag  boolean  long  serialVersionUID  Predicate  predicate  
[buglab_swap_variables]^return decorated (  ) .add ( count, object ) ;^104^^^^^102^105^return decorated (  ) .add ( object, count ) ;^[CLASS] PredicatedBag  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[buglab_swap_variables]^return decorated (  ) .add (  count ) ;^104^^^^^102^105^return decorated (  ) .add ( object, count ) ;^[CLASS] PredicatedBag  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[buglab_swap_variables]^return decorated (  ) .add ( object ) ;^104^^^^^102^105^return decorated (  ) .add ( object, count ) ;^[CLASS] PredicatedBag  [METHOD] add [RETURN_TYPE] boolean   final E object final int count [VARIABLES] boolean  E  object  long  serialVersionUID  int  count  
[buglab_swap_variables]^return decorated (  ) .remove ( count, object ) ;^108^^^^^107^109^return decorated (  ) .remove ( object, count ) ;^[CLASS] PredicatedBag  [METHOD] remove [RETURN_TYPE] boolean   Object object final int count [VARIABLES] Object  object  boolean  long  serialVersionUID  int  count  
[buglab_swap_variables]^return decorated (  ) .remove (  count ) ;^108^^^^^107^109^return decorated (  ) .remove ( object, count ) ;^[CLASS] PredicatedBag  [METHOD] remove [RETURN_TYPE] boolean   Object object final int count [VARIABLES] Object  object  boolean  long  serialVersionUID  int  count  
[buglab_swap_variables]^return decorated (  ) .remove ( object ) ;^108^^^^^107^109^return decorated (  ) .remove ( object, count ) ;^[CLASS] PredicatedBag  [METHOD] remove [RETURN_TYPE] boolean   Object object final int count [VARIABLES] Object  object  boolean  long  serialVersionUID  int  count  