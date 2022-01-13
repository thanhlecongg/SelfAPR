[REPLACE]^private final Iterator<? extends Object parameters;^53^^^^^^^[REPLACE] private final Iterator<? extends Parameter> parameters;^ [CLASS] ParameterMetadataProvider ParameterMetadata  
[REPLACE]^private  List<Object expressions;^54^^^^^^^[REPLACE] private final List<ParameterMetadata<?>> expressions;^ [CLASS] ParameterMetadataProvider ParameterMetadata  
[REPLACE]^private final @Object Iterator<Object> bindableParameterValues;^55^^^^^^^[REPLACE] private final @Nullable Iterator<Object> bindableParameterValues;^ [CLASS] ParameterMetadataProvider ParameterMetadata  
[REPLACE]^private final Object persistenceProvider;^56^^^^^^^[REPLACE] private final PersistenceProvider persistenceProvider;^ [CLASS] ParameterMetadataProvider ParameterMetadata  
[REPLACE]^static final Object PLACEHOLDER ;^186^^^^^^^[REPLACE] static final Object PLACEHOLDER = new Object (  ) ;^ [CLASS] ParameterMetadataProvider ParameterMetadata  
[REPLACE]^private final Object expression;^189^^^^^^^[REPLACE] private final ParameterExpression<T> expression;^ [CLASS] ParameterMetadataProvider ParameterMetadata  
[REPLACE]^Assert.notNull ( type, "Type must not be this!" ) ;^156^^^^^154^177^[REPLACE] Assert.notNull ( type, "Type must not be null!" ) ;^[METHOD] next [TYPE] <T> [PARAMETER] Part part Class<T> type Parameter parameter [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterMetadata metadata  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  Part part  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  Parameter parameter  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class reifiedType  type  [TYPE]  Supplier name 
[ADD]^^162^163^164^165^^154^177^[ADD] Class<T> reifiedType = Expression.class.equals ( type )  ?  ( Class<T> )  Object.class : type;  Supplier<String> name =  (  )  -> parameter.getName (  ) .orElseThrow (  (  )  -> new IllegalArgumentException ( "o_O Parameter needs to be named" )  ) ;^[METHOD] next [TYPE] <T> [PARAMETER] Part part Class<T> type Parameter parameter [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterMetadata metadata  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  Part part  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  Parameter parameter  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class reifiedType  type  [TYPE]  Supplier name 
[ADD]^^164^165^^^^154^177^[ADD] Supplier<String> name =  (  )  -> parameter.getName (  ) .orElseThrow (  (  )  -> new IllegalArgumentException ( "o_O Parameter needs to be named" )  ) ;^[METHOD] next [TYPE] <T> [PARAMETER] Part part Class<T> type Parameter parameter [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterMetadata metadata  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  Part part  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  Parameter parameter  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class reifiedType  type  [TYPE]  Supplier name 
[REPLACE]^ParameterExpression<T> expression = parameter.getName (  ) ? builder.parameter ( this, name.get (  )  ) : builder.parameter ( reifiedType ) ;^167^168^169^^^154^177^[REPLACE] ParameterExpression<T> expression = parameter.isExplicitlyNamed (  ) ? builder.parameter ( reifiedType, name.get (  )  ) : builder.parameter ( reifiedType ) ;^[METHOD] next [TYPE] <T> [PARAMETER] Part part Class<T> type Parameter parameter [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterMetadata metadata  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  Part part  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  Parameter parameter  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class reifiedType  type  [TYPE]  Supplier name 
[REPLACE]^Object value  =  bindableParameterValues.next (  ) ;^171^^^^^154^177^[REPLACE] Object value = bindableParameterValues == null ? ParameterMetadata.PLACEHOLDER : bindableParameterValues.next (  ) ;^[METHOD] next [TYPE] <T> [PARAMETER] Part part Class<T> type Parameter parameter [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterMetadata metadata  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  Part part  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  Parameter parameter  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class reifiedType  type  [TYPE]  Supplier name 
[REPLACE]^ParameterMetadata<T> metadata = new ParameterMetadata<> ( expression, part .getName (  )  , PLACEHOLDER, persistenceProvider ) ;^173^^^^^154^177^[REPLACE] ParameterMetadata<T> metadata = new ParameterMetadata<> ( expression, part.getType (  ) , value, persistenceProvider ) ;^[METHOD] next [TYPE] <T> [PARAMETER] Part part Class<T> type Parameter parameter [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterMetadata metadata  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  Part part  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  Parameter parameter  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class reifiedType  type  [TYPE]  Supplier name 
[REPLACE]^expressions.add ( null ) ;^174^^^^^154^177^[REPLACE] expressions.add ( metadata ) ;^[METHOD] next [TYPE] <T> [PARAMETER] Part part Class<T> type Parameter parameter [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterMetadata metadata  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  Part part  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  Parameter parameter  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class reifiedType  type  [TYPE]  Supplier name 
[REPLACE]^return null;^176^^^^^154^177^[REPLACE] return metadata;^[METHOD] next [TYPE] <T> [PARAMETER] Part part Class<T> type Parameter parameter [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterMetadata metadata  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  Part part  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  Parameter parameter  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class reifiedType  type  [TYPE]  Supplier name 
[REPLACE]^notNull ( type, "Type must not be null!" )  ;^227^^^^^225^249^[REPLACE] Assert.notNull ( value, "Value must not be null!" ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType 
[REPLACE]^return String.format ( "%s%%", PLACEHOLDER.toString (  )  ) ;^235^^^^^225^249^[REPLACE] return String.format ( "%s%%", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType 
[REPLACE]^return String.format ( "%%%s", value.Object (  )  ) ;^237^^^^^225^249^[REPLACE] return String.format ( "%%%s", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType 
[REPLACE]^return String.format ( "%%%s%%", PLACEHOLDER.toString (  )  ) ;^240^^^^^225^249^[REPLACE] return String.format ( "%%%s%%", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType 
[REPLACE]^return PLACEHOLDER;^242^^^^^225^249^[REPLACE] return value;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType 
[REPLACE]^return Collection.class.isAssignableFrom ( expressionType ) ? persistenceProvider.potentiallyConvertEmptyCollection ( toCollection ( PLACEHOLDER )  ) : value;^246^247^248^^^225^249^[REPLACE] return Collection.class.isAssignableFrom ( expressionType ) ? persistenceProvider.potentiallyConvertEmptyCollection ( toCollection ( value )  ) : value;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadataProvider ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType 
[REPLACE]^static  Object PLACEHOLDER = new Object (  ) ;^186^^^^^^^[REPLACE] static final Object PLACEHOLDER = new Object (  ) ;^[METHOD] toCollection [TYPE] Collection [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value 
[REPLACE]^private final ParameterExpression type;^188^^^^^^^[REPLACE] private final Type type;^[METHOD] toCollection [TYPE] Collection [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value 
[REPLACE]^private final Object expression;^189^^^^^^^[REPLACE] private final ParameterExpression<T> expression;^[METHOD] toCollection [TYPE] Collection [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value 
[REPLACE]^private final Object persistenceProvider;^190^^^^^^^[REPLACE] private final PersistenceProvider persistenceProvider;^[METHOD] toCollection [TYPE] Collection [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  CriteriaBuilder builder  [TYPE]  Type type  [TYPE]  boolean false  true  [TYPE]  PersistenceProvider persistenceProvider  provider  [TYPE]  ParameterExpression expression  [TYPE]  Iterator bindableParameterValues  parameters  [TYPE]  List expressions  [TYPE]  Object PLACEHOLDER  value 
[REPLACE]^Assert.notNull ( value, "Value must not be true!" ) ;^227^^^^^225^249^[REPLACE] Assert.notNull ( value, "Value must not be null!" ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[REPLACE]^if  ( String.class.equals ( null )  )  {^231^^^^^225^249^[REPLACE] if  ( String.class.equals ( expressionType )  )  {^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[ADD]^^231^232^233^234^235^225^249^[ADD] if  ( String.class.equals ( expressionType )  )  {  switch  ( type )  { case STARTING_WITH: return String.format ( "%s%%", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[REPLACE]^return format ( "%%%s%%", value.toString (  )  )  ;^235^^^^^225^249^[REPLACE] return String.format ( "%s%%", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[REPLACE]^return String.format ( "%%%s", PLACEHOLDER.toString (  )  ) ;^237^^^^^225^249^[REPLACE] return String.format ( "%%%s", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[REPLACE]^return String.format ( "%%%s%%", PLACEHOLDER.toString (  )  ) ;^240^^^^^225^249^[REPLACE] return String.format ( "%%%s%%", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[REPLACE]^return PLACEHOLDER;^242^^^^^225^249^[REPLACE] return value;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[REPLACE]^return String.format ( "%s%%", PLACEHOLDER.toString (  )  ) ;^235^^^^^225^249^[REPLACE] return String.format ( "%s%%", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[REPLACE]^return format ( "%%%s%%", value.toString (  )  )  ;^237^^^^^225^249^[REPLACE] return String.format ( "%%%s", value.toString (  )  ) ;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
[REPLACE]^return Collection.class.isAssignableFrom ( expressionType ) ? persistenceProvider.potentiallyConvertEmptyCollection ( toCollection ( PLACEHOLDER )  ) : value;^246^247^248^^^225^249^[REPLACE] return Collection.class.isAssignableFrom ( expressionType ) ? persistenceProvider.potentiallyConvertEmptyCollection ( toCollection ( value )  ) : value;^[METHOD] prepare [TYPE] Object [PARAMETER] Object value [CLASS] ParameterMetadata   [TYPE]  Type type  [TYPE]  Object PLACEHOLDER  value  [TYPE]  Class expressionType  [TYPE]  boolean false  true  [TYPE]  ParameterExpression expression  [TYPE]  PersistenceProvider persistenceProvider  provider 
