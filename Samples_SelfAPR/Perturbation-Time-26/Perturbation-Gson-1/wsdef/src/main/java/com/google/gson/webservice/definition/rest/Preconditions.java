[P8_Replace_Mix]^if  ( condition )  {^21^^^^^20^24^if  ( !condition )  {^[CLASS] Preconditions  [METHOD] checkArgument [RETURN_TYPE] void   boolean condition [VARIABLES] boolean  condition  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException();^21^22^23^^^20^24^if  ( !condition )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] Preconditions  [METHOD] checkArgument [RETURN_TYPE] void   boolean condition [VARIABLES] boolean  condition  
[P16_Remove_Block]^^21^22^23^^^20^24^if  ( !condition )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] Preconditions  [METHOD] checkArgument [RETURN_TYPE] void   boolean condition [VARIABLES] boolean  condition  
[P13_Insert_Block]^if  ( obj == null )  {     throw new IllegalArgumentException (  ) ; }^21^^^^^20^24^[Delete]^[CLASS] Preconditions  [METHOD] checkArgument [RETURN_TYPE] void   boolean condition [VARIABLES] boolean  condition  
[P13_Insert_Block]^if  ( obj == null )  {     throw new IllegalArgumentException (  ) ; }^22^^^^^20^24^[Delete]^[CLASS] Preconditions  [METHOD] checkArgument [RETURN_TYPE] void   boolean condition [VARIABLES] boolean  condition  
[P13_Insert_Block]^if  ( !condition )  {     throw new IllegalArgumentException (  ) ; }^22^^^^^20^24^[Delete]^[CLASS] Preconditions  [METHOD] checkArgument [RETURN_TYPE] void   boolean condition [VARIABLES] boolean  condition  
[P2_Replace_Operator]^if  ( obj != null )  {^27^^^^^26^30^if  ( obj == null )  {^[CLASS] Preconditions  [METHOD] checkNotNull [RETURN_TYPE] void   Object obj [VARIABLES] boolean  Object  obj  
[P8_Replace_Mix]^if  ( obj == false )  {^27^^^^^26^30^if  ( obj == null )  {^[CLASS] Preconditions  [METHOD] checkNotNull [RETURN_TYPE] void   Object obj [VARIABLES] boolean  Object  obj  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException();^27^28^29^^^26^30^if  ( obj == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] Preconditions  [METHOD] checkNotNull [RETURN_TYPE] void   Object obj [VARIABLES] boolean  Object  obj  
[P16_Remove_Block]^^27^28^29^^^26^30^if  ( obj == null )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] Preconditions  [METHOD] checkNotNull [RETURN_TYPE] void   Object obj [VARIABLES] boolean  Object  obj  
[P13_Insert_Block]^if  ( !condition )  {     throw new IllegalArgumentException (  ) ; }^27^^^^^26^30^[Delete]^[CLASS] Preconditions  [METHOD] checkNotNull [RETURN_TYPE] void   Object obj [VARIABLES] boolean  Object  obj  
[P13_Insert_Block]^if  ( obj == null )  {     throw new IllegalArgumentException (  ) ; }^28^^^^^26^30^[Delete]^[CLASS] Preconditions  [METHOD] checkNotNull [RETURN_TYPE] void   Object obj [VARIABLES] boolean  Object  obj  
[P13_Insert_Block]^if  ( !condition )  {     throw new IllegalArgumentException (  ) ; }^28^^^^^26^30^[Delete]^[CLASS] Preconditions  [METHOD] checkNotNull [RETURN_TYPE] void   Object obj [VARIABLES] boolean  Object  obj  
[P8_Replace_Mix]^return ;^28^^^^^26^30^throw new IllegalArgumentException  (" ")  ;^[CLASS] Preconditions  [METHOD] checkNotNull [RETURN_TYPE] void   Object obj [VARIABLES] boolean  Object  obj  