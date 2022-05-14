[buglab_swap_variables]^return  ( Paint )  key.get ( this.store ) ;^98^^^^^94^99^return  ( Paint )  this.store.get ( key ) ;^[CLASS] PaintMap  [METHOD] getPaint [RETURN_TYPE] Paint   Comparable key [VARIABLES] Map  store  Comparable  key  boolean  
[buglab_swap_variables]^return key.containsKey ( this.store ) ;^111^^^^^110^112^return this.store.containsKey ( key ) ;^[CLASS] PaintMap  [METHOD] containsKey [RETURN_TYPE] boolean   Comparable key [VARIABLES] Map  store  Comparable  key  boolean  
[buglab_swap_variables]^this.store.put (  paint ) ;^128^^^^^124^129^this.store.put ( key, paint ) ;^[CLASS] PaintMap  [METHOD] put [RETURN_TYPE] void   Comparable key Paint paint [VARIABLES] Comparable  key  Paint  paint  boolean  Map  store  
[buglab_swap_variables]^this.store.put ( paint, key ) ;^128^^^^^124^129^this.store.put ( key, paint ) ;^[CLASS] PaintMap  [METHOD] put [RETURN_TYPE] void   Comparable key Paint paint [VARIABLES] Comparable  key  Paint  paint  boolean  Map  store  
[buglab_swap_variables]^this.store.put ( key ) ;^128^^^^^124^129^this.store.put ( key, paint ) ;^[CLASS] PaintMap  [METHOD] put [RETURN_TYPE] void   Comparable key Paint paint [VARIABLES] Comparable  key  Paint  paint  boolean  Map  store  
[buglab_swap_variables]^if  ( that.size (  )  != this.store.store.size (  )  )  {^153^^^^^145^167^if  ( this.store.size (  )  != that.store.size (  )  )  {^[CLASS] PaintMap  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Set  keys  boolean  PaintMap  that  Iterator  iterator  Object  obj  Paint  p1  p2  Map  store  
[buglab_swap_variables]^if  ( that.store.size (  )  != this.store.size (  )  )  {^153^^^^^145^167^if  ( this.store.size (  )  != that.store.size (  )  )  {^[CLASS] PaintMap  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Set  keys  boolean  PaintMap  that  Iterator  iterator  Object  obj  Paint  p1  p2  Map  store  
[buglab_swap_variables]^if  ( this.store.size (  )  != that.store.store.size (  )  )  {^153^^^^^145^167^if  ( this.store.size (  )  != that.store.size (  )  )  {^[CLASS] PaintMap  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Set  keys  boolean  PaintMap  that  Iterator  iterator  Object  obj  Paint  p1  p2  Map  store  
[buglab_swap_variables]^if  ( this.store.size (  )  != that.size (  )  )  {^153^^^^^145^167^if  ( this.store.size (  )  != that.store.size (  )  )  {^[CLASS] PaintMap  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Set  keys  boolean  PaintMap  that  Iterator  iterator  Object  obj  Paint  p1  p2  Map  store  
[buglab_swap_variables]^if  ( !PaintUtilities.equal ( p2, p1 )  )  {^162^^^^^145^167^if  ( !PaintUtilities.equal ( p1, p2 )  )  {^[CLASS] PaintMap  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Set  keys  boolean  PaintMap  that  Iterator  iterator  Object  obj  Paint  p1  p2  Map  store  
[buglab_swap_variables]^if  ( !PaintUtilities.equal (  p2 )  )  {^162^^^^^145^167^if  ( !PaintUtilities.equal ( p1, p2 )  )  {^[CLASS] PaintMap  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Set  keys  boolean  PaintMap  that  Iterator  iterator  Object  obj  Paint  p1  p2  Map  store  
[buglab_swap_variables]^if  ( !PaintUtilities.equal ( p1 )  )  {^162^^^^^145^167^if  ( !PaintUtilities.equal ( p1, p2 )  )  {^[CLASS] PaintMap  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Set  keys  boolean  PaintMap  that  Iterator  iterator  Object  obj  Paint  p1  p2  Map  store  
[buglab_swap_variables]^Paint p2 = key.getPaint ( that ) ;^161^^^^^145^167^Paint p2 = that.getPaint ( key ) ;^[CLASS] PaintMap  [METHOD] equals [RETURN_TYPE] boolean   Object obj [VARIABLES] Comparable  key  Set  keys  boolean  PaintMap  that  Iterator  iterator  Object  obj  Paint  p1  p2  Map  store  
[buglab_swap_variables]^SerialUtilities.writePaint ( stream, paint ) ;^198^^^^^189^200^SerialUtilities.writePaint ( paint, stream ) ;^[CLASS] PaintMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream stream [VARIABLES] Comparable  key  Set  keys  boolean  Iterator  iterator  Paint  paint  ObjectOutputStream  stream  Map  store  
[buglab_swap_variables]^SerialUtilities.writePaint (  stream ) ;^198^^^^^189^200^SerialUtilities.writePaint ( paint, stream ) ;^[CLASS] PaintMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream stream [VARIABLES] Comparable  key  Set  keys  boolean  Iterator  iterator  Paint  paint  ObjectOutputStream  stream  Map  store  
[buglab_swap_variables]^SerialUtilities.writePaint ( paint ) ;^198^^^^^189^200^SerialUtilities.writePaint ( paint, stream ) ;^[CLASS] PaintMap  [METHOD] writeObject [RETURN_TYPE] void   ObjectOutputStream stream [VARIABLES] Comparable  key  Set  keys  boolean  Iterator  iterator  Paint  paint  ObjectOutputStream  stream  Map  store  
[buglab_swap_variables]^for  ( keyCountnt i = 0; i < i; i++ )  {^215^^^^^210^220^for  ( int i = 0; i < keyCount; i++ )  {^[CLASS] PaintMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream stream [VARIABLES] Comparable  key  Paint  paint  boolean  Map  store  int  i  keyCount  ObjectInputStream  stream  
[buglab_swap_variables]^this.store.put (  paint ) ;^218^^^^^210^220^this.store.put ( key, paint ) ;^[CLASS] PaintMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream stream [VARIABLES] Comparable  key  Paint  paint  boolean  Map  store  int  i  keyCount  ObjectInputStream  stream  
[buglab_swap_variables]^this.store.put ( paint, key ) ;^218^^^^^210^220^this.store.put ( key, paint ) ;^[CLASS] PaintMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream stream [VARIABLES] Comparable  key  Paint  paint  boolean  Map  store  int  i  keyCount  ObjectInputStream  stream  
[buglab_swap_variables]^this.store.put ( key ) ;^218^^^^^210^220^this.store.put ( key, paint ) ;^[CLASS] PaintMap  [METHOD] readObject [RETURN_TYPE] void   ObjectInputStream stream [VARIABLES] Comparable  key  Paint  paint  boolean  Map  store  int  i  keyCount  ObjectInputStream  stream  