[REPLACE]^this.visible = visible; ;^70^^^^^69^72^[REPLACE] this.visible = true;^[METHOD] <init> [TYPE] AbstractDialLayer() [PARAMETER] [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  boolean false  true  visible 
[REPLACE]^this.listenerList  = null ;^71^^^^^69^72^[REPLACE] this.listenerList = new EventListenerList (  ) ;^[METHOD] <init> [TYPE] AbstractDialLayer() [PARAMETER] [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  boolean false  true  visible 
[REPLACE]^return list.contains ( listener ) ;^81^^^^^80^82^[REPLACE] return this.visible;^[METHOD] isVisible [TYPE] boolean [PARAMETER] [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  boolean false  true  visible 
[REPLACE]^this.visible = true; ;^92^^^^^91^94^[REPLACE] this.visible = visible;^[METHOD] setVisible [TYPE] void [PARAMETER] boolean visible [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  boolean false  true  visible 
[REPLACE]^this.listenerList.remove ( DialLayerChangeListener.class, listener ) ;^104^^^^^103^105^[REPLACE] this.listenerList.add ( DialLayerChangeListener.class, listener ) ;^[METHOD] addChangeListener [TYPE] void [PARAMETER] DialLayerChangeListener listener [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeListener listener  [TYPE]  boolean false  true  visible 
[ADD]^^104^^^^^103^105^[ADD] this.listenerList.add ( DialLayerChangeListener.class, listener ) ;^[METHOD] addChangeListener [TYPE] void [PARAMETER] DialLayerChangeListener listener [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeListener listener  [TYPE]  boolean false  true  visible 
[REPLACE]^this.listenerList.add ( DialLayerChangeListener.class, listener ) ;^115^^^^^114^116^[REPLACE] this.listenerList.remove ( DialLayerChangeListener.class, listener ) ;^[METHOD] removeChangeListener [TYPE] void [PARAMETER] DialLayerChangeListener listener [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeListener listener  [TYPE]  boolean false  true  visible 
[REPLACE]^List list = Arrays.asList ( this.listenerList.EventListenerList (  )  ) ;^128^^^^^127^130^[REPLACE] List list = Arrays.asList ( this.listenerList.getListenerList (  )  ) ;^[METHOD] hasListener [TYPE] boolean [PARAMETER] EventListener listener [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  List list  [TYPE]  boolean false  true  visible  [TYPE]  EventListener listener 
[REPLACE]^return this.visible;^129^^^^^127^130^[REPLACE] return list.contains ( listener ) ;^[METHOD] hasListener [TYPE] boolean [PARAMETER] EventListener listener [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  List list  [TYPE]  boolean false  true  visible  [TYPE]  EventListener listener 
[REPLACE]^Object[] listeners = this.listenerList.EventListenerList (  ) ;^139^^^^^138^146^[REPLACE] Object[] listeners = this.listenerList.getListenerList (  ) ;^[METHOD] notifyListeners [TYPE] void [PARAMETER] DialLayerChangeEvent event [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeEvent event  [TYPE]  boolean false  true  visible  [TYPE]  Object[] listeners  [TYPE]  int i 
[REPLACE]^if  ( listeners[i]  &&  DialLayerChangeListener.class )  {^141^^^^^138^146^[REPLACE] if  ( listeners[i] == DialLayerChangeListener.class )  {^[METHOD] notifyListeners [TYPE] void [PARAMETER] DialLayerChangeEvent event [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeEvent event  [TYPE]  boolean false  true  visible  [TYPE]  Object[] listeners  [TYPE]  int i 
[ADD]^^141^142^143^144^^138^146^[ADD] if  ( listeners[i] == DialLayerChangeListener.class )  { (  ( DialLayerChangeListener )  listeners[i + 1] ) .dialLayerChanged ( event ) ; }^[METHOD] notifyListeners [TYPE] void [PARAMETER] DialLayerChangeEvent event [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeEvent event  [TYPE]  boolean false  true  visible  [TYPE]  Object[] listeners  [TYPE]  int i 
[REPLACE]^(  ( DialLayerChangeListener )  listeners[i  &  1] ) .dialLayerChanged ( event ) ;^142^143^^^^138^146^[REPLACE] (  ( DialLayerChangeListener )  listeners[i + 1] ) .dialLayerChanged ( event ) ;^[METHOD] notifyListeners [TYPE] void [PARAMETER] DialLayerChangeEvent event [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeEvent event  [TYPE]  boolean false  true  visible  [TYPE]  Object[] listeners  [TYPE]  int i 
[REPLACE]^for  ( int i = listeners.length %  2 - 2; i >= 0; i -= 2 )  {^140^^^^^138^146^[REPLACE] for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[METHOD] notifyListeners [TYPE] void [PARAMETER] DialLayerChangeEvent event [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeEvent event  [TYPE]  boolean false  true  visible  [TYPE]  Object[] listeners  [TYPE]  int i 
[ADD]^^140^141^142^143^144^138^146^[ADD] for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  { if  ( listeners[i] == DialLayerChangeListener.class )  { (  ( DialLayerChangeListener )  listeners[i + 1] ) .dialLayerChanged ( event ) ; }^[METHOD] notifyListeners [TYPE] void [PARAMETER] DialLayerChangeEvent event [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeEvent event  [TYPE]  boolean false  true  visible  [TYPE]  Object[] listeners  [TYPE]  int i 
[REPLACE]^for  ( int i = listeners.length - 0 ; i >= 0; i -= 0  )  {^140^^^^^138^146^[REPLACE] for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[METHOD] notifyListeners [TYPE] void [PARAMETER] DialLayerChangeEvent event [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeEvent event  [TYPE]  boolean false  true  visible  [TYPE]  Object[] listeners  [TYPE]  int i 
[REPLACE]^for  ( int i = listeners.length /  0  - 0 ; i >= 0; i -= 0  )  {^140^^^^^138^146^[REPLACE] for  ( int i = listeners.length - 2; i >= 0; i -= 2 )  {^[METHOD] notifyListeners [TYPE] void [PARAMETER] DialLayerChangeEvent event [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  DialLayerChangeEvent event  [TYPE]  boolean false  true  visible  [TYPE]  Object[] listeners  [TYPE]  int i 
[REPLACE]^this.listenerList.add ( DialLayerChangeListener.class, listener ) ;^155^^^^^153^157^[REPLACE] stream.defaultReadObject (  ) ;^[METHOD] readObject [TYPE] void [PARAMETER] ObjectInputStream stream [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  boolean false  true  visible  [TYPE]  ObjectInputStream stream 
[REPLACE]^this.listenerList  = null ;^156^^^^^153^157^[REPLACE] this.listenerList = new EventListenerList (  ) ;^[METHOD] readObject [TYPE] void [PARAMETER] ObjectInputStream stream [CLASS] AbstractDialLayer   [TYPE]  EventListenerList listenerList  [TYPE]  boolean false  true  visible  [TYPE]  ObjectInputStream stream 