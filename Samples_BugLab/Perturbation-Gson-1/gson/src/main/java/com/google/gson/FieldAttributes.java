[buglab_swap_variables]^String propertyValue = System.getProperty ( defaultMaxCacheSize, String.valueOf ( MAX_CACHE_PROPERTY_NAME )  ) ;^72^73^^^^69^78^String propertyValue = System.getProperty ( MAX_CACHE_PROPERTY_NAME, String.valueOf ( defaultMaxCacheSize )  ) ;^[CLASS] FieldAttributes  [METHOD] getMaxCacheSize [RETURN_TYPE] int   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  NumberFormatException  e  Class  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  
[buglab_swap_variables]^String propertyValue = System.getProperty (  String.valueOf ( defaultMaxCacheSize )  ) ;^72^73^^^^69^78^String propertyValue = System.getProperty ( MAX_CACHE_PROPERTY_NAME, String.valueOf ( defaultMaxCacheSize )  ) ;^[CLASS] FieldAttributes  [METHOD] getMaxCacheSize [RETURN_TYPE] int   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  NumberFormatException  e  Class  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  
[buglab_swap_variables]^ANNOTATION_CACHE.addElement ( annotations, key ) ;^161^^^^^154^165^ANNOTATION_CACHE.addElement ( key, annotations ) ;^[CLASS] FieldAttributes  [METHOD] getAnnotations [RETURN_TYPE] Collection   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  Pair  key  
[buglab_swap_variables]^ANNOTATION_CACHE.addElement (  annotations ) ;^161^^^^^154^165^ANNOTATION_CACHE.addElement ( key, annotations ) ;^[CLASS] FieldAttributes  [METHOD] getAnnotations [RETURN_TYPE] Collection   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  Pair  key  
[buglab_swap_variables]^ANNOTATION_CACHE.addElement ( key ) ;^161^^^^^154^165^ANNOTATION_CACHE.addElement ( key, annotations ) ;^[CLASS] FieldAttributes  [METHOD] getAnnotations [RETURN_TYPE] Collection   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  Pair  key  
[buglab_swap_variables]^annotations = key.getElement ( ANNOTATION_CACHE ) ;^157^^^^^154^165^annotations = ANNOTATION_CACHE.getElement ( key ) ;^[CLASS] FieldAttributes  [METHOD] getAnnotations [RETURN_TYPE] Collection   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  Pair  key  
[buglab_swap_variables]^Pair<Class<?>, String> key = new Pair<Class<?>, String> ( name, declaringClazz ) ;^156^^^^^154^165^Pair<Class<?>, String> key = new Pair<Class<?>, String> ( declaringClazz, name ) ;^[CLASS] FieldAttributes  [METHOD] getAnnotations [RETURN_TYPE] Collection   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  Pair  key  
[buglab_swap_variables]^Pair<Class<?>, String> key = new Pair<Class<?>, String> (  name ) ;^156^^^^^154^165^Pair<Class<?>, String> key = new Pair<Class<?>, String> ( declaringClazz, name ) ;^[CLASS] FieldAttributes  [METHOD] getAnnotations [RETURN_TYPE] Collection   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  Pair  key  
[buglab_swap_variables]^Pair<Class<?>, String> key = new Pair<Class<?>, String> ( declaringClazz ) ;^156^^^^^154^165^Pair<Class<?>, String> key = new Pair<Class<?>, String> ( declaringClazz, name ) ;^[CLASS] FieldAttributes  [METHOD] getAnnotations [RETURN_TYPE] Collection   [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifiers  Cache  ANNOTATION_CACHE  Pair  key  
[buglab_swap_variables]^return  ( modifierss & modifier )  != 0;^178^^^^^177^179^return  ( modifiers & modifier )  != 0;^[CLASS] FieldAttributes  [METHOD] hasModifier [RETURN_TYPE] boolean   int modifier [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifier  modifiers  Cache  ANNOTATION_CACHE  
[buglab_swap_variables]^return  ( modifier & modifiers )  != 0;^178^^^^^177^179^return  ( modifiers & modifier )  != 0;^[CLASS] FieldAttributes  [METHOD] hasModifier [RETURN_TYPE] boolean   int modifier [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifier  modifiers  Cache  ANNOTATION_CACHE  
[buglab_swap_variables]^field.set (  value ) ;^189^^^^^188^190^field.set ( instance, value ) ;^[CLASS] FieldAttributes  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  Object  instance  value  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifier  modifiers  Cache  ANNOTATION_CACHE  
[buglab_swap_variables]^field.set ( value, instance ) ;^189^^^^^188^190^field.set ( instance, value ) ;^[CLASS] FieldAttributes  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  Object  instance  value  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifier  modifiers  Cache  ANNOTATION_CACHE  
[buglab_swap_variables]^field.set ( instance ) ;^189^^^^^188^190^field.set ( instance, value ) ;^[CLASS] FieldAttributes  [METHOD] set [RETURN_TYPE] void   Object instance Object value [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  Object  instance  value  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifier  modifiers  Cache  ANNOTATION_CACHE  
[buglab_swap_variables]^return instance.get ( field ) ;^200^^^^^199^201^return field.get ( instance ) ;^[CLASS] FieldAttributes  [METHOD] get [RETURN_TYPE] Object   Object instance [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  Object  instance  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifier  modifiers  Cache  ANNOTATION_CACHE  
[buglab_swap_variables]^if  ( annotation.annotationType (  )  == a )  {^224^^^^^221^229^if  ( a.annotationType (  )  == annotation )  {^[CLASS] FieldAttributes  [METHOD] getAnnotationFromArray [RETURN_TYPE] <T   Annotation> annotations Class<T> annotation [VARIABLES] Field  f  field  Type  genericType  boolean  isSynthetic  Collection  annotations  Class  annotation  declaredType  declaringClazz  String  MAX_CACHE_PROPERTY_NAME  name  propertyValue  int  defaultMaxCacheSize  modifier  modifiers  Annotation  a  Cache  ANNOTATION_CACHE  