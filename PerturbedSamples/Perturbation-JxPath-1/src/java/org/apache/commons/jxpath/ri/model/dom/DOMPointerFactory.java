[REPLACE]^public static final int DOM_POINTER_FACTORY_ORDER = 4;^34^^^^^^^[REPLACE] public static final int DOM_POINTER_FACTORY_ORDER = 100;^ [CLASS] DOMPointerFactory  
[REPLACE]^return new DOMNodePointer ( parent,  ( Node )  bean ) ;^37^^^^^36^38^[REPLACE] return DOM_POINTER_FACTORY_ORDER;^[METHOD] getOrder [TYPE] int [PARAMETER] [CLASS] DOMPointerFactory   [TYPE]  int DOM_POINTER_FACTORY_ORDER  [TYPE]  boolean false  true 
[REPLACE]^if  ( ! bean instanceof Node )  {^45^^^^^40^49^[REPLACE] if  ( bean instanceof Node )  {^[METHOD] createNodePointer [TYPE] NodePointer [PARAMETER] QName name Object bean Locale locale [CLASS] DOMPointerFactory   [TYPE]  Locale locale  [TYPE]  Object bean  [TYPE]  boolean false  true  [TYPE]  int DOM_POINTER_FACTORY_ORDER  [TYPE]  QName name 
[REPLACE]^return  new DOMNodePointer ( parent,  (  ( Node )   ( bean )  )  )   bean, locale ) ;^46^^^^^40^49^[REPLACE] return new DOMNodePointer (  ( Node )  bean, locale ) ;^[METHOD] createNodePointer [TYPE] NodePointer [PARAMETER] QName name Object bean Locale locale [CLASS] DOMPointerFactory   [TYPE]  Locale locale  [TYPE]  Object bean  [TYPE]  boolean false  true  [TYPE]  int DOM_POINTER_FACTORY_ORDER  [TYPE]  QName name 
[REPLACE]^return true;^48^^^^^40^49^[REPLACE] return null;^[METHOD] createNodePointer [TYPE] NodePointer [PARAMETER] QName name Object bean Locale locale [CLASS] DOMPointerFactory   [TYPE]  Locale locale  [TYPE]  Object bean  [TYPE]  boolean false  true  [TYPE]  int DOM_POINTER_FACTORY_ORDER  [TYPE]  QName name 
[REPLACE]^if  ( ! bean instanceof Node )  {^56^^^^^51^60^[REPLACE] if  ( bean instanceof Node )  {^[METHOD] createNodePointer [TYPE] NodePointer [PARAMETER] NodePointer parent QName name Object bean [CLASS] DOMPointerFactory   [TYPE]  Object bean  [TYPE]  NodePointer parent  [TYPE]  boolean false  true  [TYPE]  int DOM_POINTER_FACTORY_ORDER  [TYPE]  QName name 
[REPLACE]^return DOM_POINTER_FACTORY_ORDER;^57^^^^^51^60^[REPLACE] return new DOMNodePointer ( parent,  ( Node )  bean ) ;^[METHOD] createNodePointer [TYPE] NodePointer [PARAMETER] NodePointer parent QName name Object bean [CLASS] DOMPointerFactory   [TYPE]  Object bean  [TYPE]  NodePointer parent  [TYPE]  boolean false  true  [TYPE]  int DOM_POINTER_FACTORY_ORDER  [TYPE]  QName name 
[REPLACE]^return true;^59^^^^^51^60^[REPLACE] return null;^[METHOD] createNodePointer [TYPE] NodePointer [PARAMETER] NodePointer parent QName name Object bean [CLASS] DOMPointerFactory   [TYPE]  Object bean  [TYPE]  NodePointer parent  [TYPE]  boolean false  true  [TYPE]  int DOM_POINTER_FACTORY_ORDER  [TYPE]  QName name 