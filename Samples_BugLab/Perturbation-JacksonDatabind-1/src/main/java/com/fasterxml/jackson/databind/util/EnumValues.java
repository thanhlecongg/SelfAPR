[buglab_swap_variables]^return constructFromName ( intr, enumClass ) ;^33^^^^^31^34^return constructFromName ( enumClass, intr ) ;^[CLASS] EnumValues  [METHOD] construct [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] AnnotationIntrospector  intr  Class  _enumClass  enumClass  boolean  EnumMap  _values  
[buglab_swap_variables]^return constructFromName (  intr ) ;^33^^^^^31^34^return constructFromName ( enumClass, intr ) ;^[CLASS] EnumValues  [METHOD] construct [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] AnnotationIntrospector  intr  Class  _enumClass  enumClass  boolean  EnumMap  _values  
[buglab_swap_variables]^return constructFromName ( enumClass ) ;^33^^^^^31^34^return constructFromName ( enumClass, intr ) ;^[CLASS] EnumValues  [METHOD] construct [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] AnnotationIntrospector  intr  Class  _enumClass  enumClass  boolean  EnumMap  _values  
[buglab_swap_variables]^return new EnumValues ( map, enumClass ) ;^50^^^^^36^53^return new EnumValues ( enumClass, map ) ;^[CLASS] EnumValues  [METHOD] constructFromName [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  String  value  EnumMap  _values  Map  map  
[buglab_swap_variables]^return new EnumValues (  map ) ;^50^^^^^36^53^return new EnumValues ( enumClass, map ) ;^[CLASS] EnumValues  [METHOD] constructFromName [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  String  value  EnumMap  _values  Map  map  
[buglab_swap_variables]^return new EnumValues ( enumClass ) ;^50^^^^^36^53^return new EnumValues ( enumClass, map ) ;^[CLASS] EnumValues  [METHOD] constructFromName [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  String  value  EnumMap  _values  Map  map  
[buglab_swap_variables]^String value = en.findEnumValue ( intr ) ;^47^^^^^36^53^String value = intr.findEnumValue ( en ) ;^[CLASS] EnumValues  [METHOD] constructFromName [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  String  value  EnumMap  _values  Map  map  
[buglab_swap_variables]^map.put ( value, new SerializedString ( en )  ) ;^48^^^^^36^53^map.put ( en, new SerializedString ( value )  ) ;^[CLASS] EnumValues  [METHOD] constructFromName [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  String  value  EnumMap  _values  Map  map  
[buglab_swap_variables]^map.put (  new SerializedString ( value )  ) ;^48^^^^^36^53^map.put ( en, new SerializedString ( value )  ) ;^[CLASS] EnumValues  [METHOD] constructFromName [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  String  value  EnumMap  _values  Map  map  
[buglab_swap_variables]^return new EnumValues ( map, enumClass ) ;^65^^^^^55^68^return new EnumValues ( enumClass, map ) ;^[CLASS] EnumValues  [METHOD] constructFromToString [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  EnumMap  _values  Map  map  
[buglab_swap_variables]^return new EnumValues (  map ) ;^65^^^^^55^68^return new EnumValues ( enumClass, map ) ;^[CLASS] EnumValues  [METHOD] constructFromToString [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  EnumMap  _values  Map  map  
[buglab_swap_variables]^return new EnumValues ( enumClass ) ;^65^^^^^55^68^return new EnumValues ( enumClass, map ) ;^[CLASS] EnumValues  [METHOD] constructFromToString [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  EnumMap  _values  Map  map  
[buglab_swap_variables]^map.put (  new SerializedString ( en.toString (  )  )  ) ;^63^^^^^55^68^map.put ( en, new SerializedString ( en.toString (  )  )  ) ;^[CLASS] EnumValues  [METHOD] constructFromToString [RETURN_TYPE] EnumValues   Enum<?>> enumClass AnnotationIntrospector intr [VARIABLES] Enum[]  values  Enum  en  boolean  AnnotationIntrospector  intr  Class  _enumClass  cls  enumClass  EnumMap  _values  Map  map  
[buglab_swap_variables]^return key.get ( _values ) ;^72^^^^^70^73^return _values.get ( key ) ;^[CLASS] EnumValues  [METHOD] serializedValueFor [RETURN_TYPE] SerializedString   Enum<?> key [VARIABLES] Enum  key  Class  _enumClass  cls  enumClass  boolean  EnumMap  _values  