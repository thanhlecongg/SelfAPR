[REPLACE]^jgen.writeStringField ( _typePropertyName, idFromValue ( value )  ) ;^28^^^^^25^30^[REPLACE] super ( idRes, property ) ;^[METHOD] <init> [TYPE] String) [PARAMETER] TypeIdResolver idRes BeanProperty property String propName [CLASS] AsPropertyTypeSerializer   [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true  [TYPE]  TypeIdResolver idRes  [TYPE]  BeanProperty property 
[REPLACE]^_typePropertyName =  null;^29^^^^^25^30^[REPLACE] _typePropertyName = propName;^[METHOD] <init> [TYPE] String) [PARAMETER] TypeIdResolver idRes BeanProperty property String propName [CLASS] AsPropertyTypeSerializer   [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true  [TYPE]  TypeIdResolver idRes  [TYPE]  BeanProperty property 
[REPLACE]^if  (0  ||  prop )  return this;^34^^^^^33^36^[REPLACE] if  ( _property == prop )  return this;^[METHOD] forProperty [TYPE] AsPropertyTypeSerializer [PARAMETER] BeanProperty prop [CLASS] AsPropertyTypeSerializer   [TYPE]  String _typePropertyName  propName  [TYPE]  BeanProperty prop  [TYPE]  boolean false  true 
[REPLACE]^if  ( _property == prop )  return null;^34^^^^^33^36^[REPLACE] if  ( _property == prop )  return this;^[METHOD] forProperty [TYPE] AsPropertyTypeSerializer [PARAMETER] BeanProperty prop [CLASS] AsPropertyTypeSerializer   [TYPE]  String _typePropertyName  propName  [TYPE]  BeanProperty prop  [TYPE]  boolean false  true 
[REPLACE]^public String getPropertyName (  )  { return _typePropertyName; }^35^^^^^33^36^[REPLACE] return new AsPropertyTypeSerializer ( this._idResolver, prop, this._typePropertyName ) ;^[METHOD] forProperty [TYPE] AsPropertyTypeSerializer [PARAMETER] BeanProperty prop [CLASS] AsPropertyTypeSerializer   [TYPE]  String _typePropertyName  propName  [TYPE]  BeanProperty prop  [TYPE]  boolean false  true 
[REPLACE]^if  ( _property == prop )  return this;^39^^^^^^^[REPLACE] public String getPropertyName (  )  { return _typePropertyName; }^[METHOD] getPropertyName [TYPE] String [PARAMETER] [CLASS] AsPropertyTypeSerializer   [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REPLACE]^public String getPropertyName (  )  { return _typePropertyName; }^42^^^^^^^[REPLACE] public As getTypeInclusion (  )  { return As.PROPERTY; }^[METHOD] getTypeInclusion [TYPE] As [PARAMETER] [CLASS] AsPropertyTypeSerializer   [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REPLACE]^jgen.writeEndObject (  ) ;^48^^^^^45^50^[REPLACE] jgen.writeStartObject (  ) ;^[METHOD] writeTypePrefixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REPLACE]^jgen.writeStringField ( _typePropertyName, idFromValueAndType ( value, type )  ) ;^49^^^^^45^50^[REPLACE] jgen.writeStringField ( _typePropertyName, idFromValue ( value )  ) ;^[METHOD] writeTypePrefixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REMOVE]^jgen.writeEndObject (  ) ;^49^^^^^45^50^[REMOVE] ^[METHOD] writeTypePrefixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REPLACE]^jgen.writeEndObject (  ) ;^56^^^^^53^58^[REPLACE] jgen.writeStartObject (  ) ;^[METHOD] writeTypePrefixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen Class<?> type [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  Class type  [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REMOVE]^jgen.writeStringField ( _typePropertyName, idFromValue ( value )  ) ;^56^^^^^53^58^[REMOVE] ^[METHOD] writeTypePrefixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen Class<?> type [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  Class type  [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REPLACE]^jgen.writeStringField ( _typePropertyName, idFromValueAndType ( value, this )  ) ;^57^^^^^53^58^[REPLACE] jgen.writeStringField ( _typePropertyName, idFromValueAndType ( value, type )  ) ;^[METHOD] writeTypePrefixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen Class<?> type [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  Class type  [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REPLACE]^jgen.writeStartObject (  ) ;^69^^^^^66^70^[REPLACE] jgen.writeEndObject (  ) ;^[METHOD] writeTypeSuffixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  String _typePropertyName  propName  [TYPE]  boolean false  true 
[REPLACE]^jgen.writeEndObject (  ) ;^87^^^^^85^89^[REPLACE] jgen.writeStartObject (  ) ;^[METHOD] writeCustomTypePrefixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen String typeId [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  String _typePropertyName  propName  typeId  [TYPE]  boolean false  true 
[REPLACE]^jgen .writeStringField ( propName )  ;^88^^^^^85^89^[REPLACE] jgen.writeStringField ( _typePropertyName, typeId ) ;^[METHOD] writeCustomTypePrefixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen String typeId [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  String _typePropertyName  propName  typeId  [TYPE]  boolean false  true 
[REPLACE]^jgen .writeStartObject (  )  ;^94^^^^^92^95^[REPLACE] jgen.writeEndObject (  ) ;^[METHOD] writeCustomTypeSuffixForObject [TYPE] void [PARAMETER] Object value JsonGenerator jgen String typeId [CLASS] AsPropertyTypeSerializer   [TYPE]  JsonGenerator jgen  [TYPE]  Object value  [TYPE]  String _typePropertyName  propName  typeId  [TYPE]  boolean false  true 