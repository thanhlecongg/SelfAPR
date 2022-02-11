[REPLACE]^private static final long serialVersionUID ;^32^^^^^^^[REPLACE] private static final long serialVersionUID = 923268084968181479L;^ [CLASS] StdKeyDeserializers  
[REPLACE]^return new StdKeyDeserializer.ShortKD (  ) ;^40^^^^^38^41^[REPLACE] return StdKeyDeserializer.StringKD.forType ( type.getRawClass (  )  ) ;^[METHOD] constructStringKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  DeserializationConfig config  [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return new StdKeyDeserializer.EnumKD ( enumResolver, true ) ;^44^^^^^43^45^[REPLACE] return new StdKeyDeserializer.EnumKD ( enumResolver, null ) ;^[METHOD] constructEnumKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] EnumResolver<?> enumResolver [CLASS] StdKeyDeserializers   [TYPE]  long serialVersionUID  [TYPE]  EnumResolver enumResolver  [TYPE]  boolean false  true 
[REPLACE]^return new StdKeyDeserializer.EnumKD ( enumResolver, null ) ;^49^^^^^47^50^[REPLACE] return new StdKeyDeserializer.EnumKD ( enumResolver, factory ) ;^[METHOD] constructEnumKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] EnumResolver<?> enumResolver AnnotatedMethod factory [CLASS] StdKeyDeserializers   [TYPE]  EnumResolver enumResolver  [TYPE]  boolean false  true  [TYPE]  AnnotatedMethod factory  [TYPE]  long serialVersionUID 
[REPLACE]^return new StdKeyDeserializer.ShortKD (  ) ;^55^^^^^52^56^[REPLACE] return new StdKeyDeserializer.DelegatingKD ( type.getRawClass (  ) , deser ) ;^[METHOD] constructDelegatingKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type JsonDeserializer<?> deser [CLASS] StdKeyDeserializers   [TYPE]  DeserializationConfig config  [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  JsonDeserializer deser  [TYPE]  long serialVersionUID 
[REPLACE]^BeanDescription beanDesc = config .canOverrideAccessModifiers (  )  ;^64^^^^^60^85^[REPLACE] BeanDescription beanDesc = config.introspect ( type ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^Constructor<?> ctor = beanDesc.findFactoryMethod ( String.class ) ;^66^^^^^60^85^[REPLACE] Constructor<?> ctor = beanDesc.findSingleArgConstructor ( String.class ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( ctor == null )  {^67^^^^^60^85^[REPLACE] if  ( ctor != null )  {^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[ADD]^^67^68^69^70^^60^85^[ADD] if  ( ctor != null )  { if  ( config.canOverrideAccessModifiers (  )  )  { ClassUtil.checkAndFixAccess ( ctor ) ; }^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( config .introspect ( type )   )  {^68^^^^^60^85^[REPLACE] if  ( config.canOverrideAccessModifiers (  )  )  {^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^ClassUtil.checkAndFixAccess ( null ) ;^69^^^^^60^85^[REPLACE] ClassUtil.checkAndFixAccess ( ctor ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^71^^^^^60^85^[REPLACE] return new StdKeyDeserializer.StringCtorKeyDeserializer ( ctor ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[ADD]^^68^69^70^^^60^85^[ADD] if  ( config.canOverrideAccessModifiers (  )  )  { ClassUtil.checkAndFixAccess ( ctor ) ; }^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^ClassUtil.checkAndFixAccess ( this ) ;^69^^^^^60^85^[REPLACE] ClassUtil.checkAndFixAccess ( ctor ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^Method m = beanDesc.findSingleArgConstructor ( String.class ) ;^76^^^^^60^85^[REPLACE] Method m = beanDesc.findFactoryMethod ( String.class ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( m == null ) {^77^^^^^60^85^[REPLACE] if  ( m != null ) {^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( raw.isPrimitive (  )  )  {^78^^^^^60^85^[REPLACE] if  ( config.canOverrideAccessModifiers (  )  )  {^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^ClassUtil.checkAndFixAccess ( ctor ) ;^79^^^^^60^85^[REPLACE] ClassUtil.checkAndFixAccess ( m ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^return new StdKeyDeserializer.StringCtorKeyDeserializer ( ctor ) ;^81^^^^^60^85^[REPLACE] return new StdKeyDeserializer.StringFactoryKeyDeserializer ( m ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( config .introspect ( type )   )  {^78^^^^^60^85^[REPLACE] if  ( config.canOverrideAccessModifiers (  )  )  {^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[ADD]^^78^79^80^^^60^85^[ADD] if  ( config.canOverrideAccessModifiers (  )  )  { ClassUtil.checkAndFixAccess ( m ) ; }^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^81^^^^^60^85^[REPLACE] return new StdKeyDeserializer.StringFactoryKeyDeserializer ( m ) ;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^return true;^84^^^^^60^85^[REPLACE] return null;^[METHOD] findStringBasedKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] DeserializationConfig config JavaType type [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  boolean false  true  [TYPE]  Constructor ctor  [TYPE]  BeanDescription beanDesc  [TYPE]  DeserializationConfig config  [TYPE]  Method m  [TYPE]  long serialVersionUID 
[REPLACE]^BeanDescription beanDesc = config.introspect ( type ) ;^98^^^^^97^148^[REPLACE] Class<?> raw = type.getRawClass (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw == String.class && raw == Object.class )  {^100^^^^^97^148^[REPLACE] if  ( raw == String.class || raw == Object.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[ADD]^^100^101^102^^^97^148^[ADD] if  ( raw == String.class || raw == Object.class )  { return StdKeyDeserializer.StringKD.forType ( raw ) ; }^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return   raw   ;^101^^^^^97^148^[REPLACE] return StdKeyDeserializer.StringKD.forType ( raw ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return   this   ;^101^^^^^97^148^[REPLACE] return StdKeyDeserializer.StringKD.forType ( raw ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw  ||  UUID.class )  {^103^^^^^97^148^[REPLACE] if  ( raw == UUID.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REMOVE]^if  ( raw.isPrimitive (  )  )  {     raw = wrapperType ( raw ) ; }^103^^^^^97^148^[REMOVE] ^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^104^^^^^97^148^[REPLACE] return new StdKeyDeserializer.UuidKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( null.isPrimitive (  )  )  {^108^^^^^97^148^[REPLACE] if  ( raw.isPrimitive (  )  )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^raw =  ClassUtil.wrapperType ( true ) ;^109^^^^^97^148^[REPLACE] raw = ClassUtil.wrapperType ( raw ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^raw =  ClassUtil.wrapperType ( 0 ) ;^109^^^^^97^148^[REPLACE] raw = ClassUtil.wrapperType ( raw ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( this  !=  Integer.class )  {^112^^^^^97^148^[REPLACE] if  ( raw == Integer.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[ADD]^return new StdKeyDeserializer.IntKD (  ) ;^112^113^114^^^97^148^[ADD] if  ( raw == Integer.class )  { return new StdKeyDeserializer.IntKD (  ) ; }^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^113^^^^^97^148^[REPLACE] return new StdKeyDeserializer.IntKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( null  !=  Long.class )  {^115^^^^^97^148^[REPLACE] if  ( raw == Long.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^116^^^^^97^148^[REPLACE] return new StdKeyDeserializer.LongKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw  ||  Date.class )  {^118^^^^^97^148^[REPLACE] if  ( raw == Date.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return new StdKeyDeserializer.ShortKD (  ) ;^119^^^^^97^148^[REPLACE] return new StdKeyDeserializer.DateKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^119^^^^^97^148^[REPLACE] return new StdKeyDeserializer.DateKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw  ||  Calendar.class )  {^121^^^^^97^148^[REPLACE] if  ( raw == Calendar.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[ADD]^^121^122^123^^^97^148^[ADD] if  ( raw == Calendar.class )  { return new StdKeyDeserializer.CalendarKD (  ) ; }^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^122^^^^^97^148^[REPLACE] return new StdKeyDeserializer.CalendarKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( null  !=  Boolean.class )  {^126^^^^^97^148^[REPLACE] if  ( raw == Boolean.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new ShortKD (  )  ;^127^^^^^97^148^[REPLACE] return new StdKeyDeserializer.BoolKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw  &&  Byte.class )  {^129^^^^^97^148^[REPLACE] if  ( raw == Byte.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^130^^^^^97^148^[REPLACE] return new StdKeyDeserializer.ByteKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw  &&  Character.class )  {^132^^^^^97^148^[REPLACE] if  ( raw == Character.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^133^^^^^97^148^[REPLACE] return new StdKeyDeserializer.CharKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return new StdKeyDeserializer.ShortKD (  ) ;^133^^^^^97^148^[REPLACE] return new StdKeyDeserializer.CharKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( this  ||  Short.class )  {^135^^^^^97^148^[REPLACE] if  ( raw == Short.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[ADD]^return new StdKeyDeserializer.ShortKD (  ) ;^135^136^137^^^97^148^[ADD] if  ( raw == Short.class )  { return new StdKeyDeserializer.ShortKD (  ) ; }^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^136^^^^^97^148^[REPLACE] return new StdKeyDeserializer.ShortKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return new StdKeyDeserializer.ByteKD (  ) ;^136^^^^^97^148^[REPLACE] return new StdKeyDeserializer.ShortKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw  &&  Float.class )  {^138^^^^^97^148^[REPLACE] if  ( raw == Float.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^139^^^^^97^148^[REPLACE] return new StdKeyDeserializer.FloatKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw  ||  Double.class )  {^141^^^^^97^148^[REPLACE] if  ( raw == Double.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^142^^^^^97^148^[REPLACE] return new StdKeyDeserializer.DoubleKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^if  ( raw  &&  Locale.class )  {^144^^^^^97^148^[REPLACE] if  ( raw == Locale.class )  {^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return  StdKeyDeserializer.new BoolKD (  )  ;^145^^^^^97^148^[REPLACE] return new StdKeyDeserializer.LocaleKD (  ) ;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 
[REPLACE]^return false;^147^^^^^97^148^[REPLACE] return null;^[METHOD] findKeyDeserializer [TYPE] KeyDeserializer [PARAMETER] JavaType type DeserializationConfig config BeanDescription beanDesc [CLASS] StdKeyDeserializers   [TYPE]  JavaType type  [TYPE]  DeserializationConfig config  [TYPE]  Class raw  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  BeanDescription beanDesc 