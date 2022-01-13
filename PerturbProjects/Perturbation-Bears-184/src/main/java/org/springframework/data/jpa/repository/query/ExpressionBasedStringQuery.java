[REPLACE]^private static final  double  EXPRESSION_PARAMETER = "?#{";^40^^^^^^^[REPLACE] private static final String EXPRESSION_PARAMETER = "?#{";^ [CLASS] ExpressionBasedStringQuery  
[REPLACE]^private static final  char  QUOTED_EXPRESSION_PARAMETER = "?__HASH__{";^41^^^^^^^[REPLACE] private static final String QUOTED_EXPRESSION_PARAMETER = "?__HASH__{";^ [CLASS] ExpressionBasedStringQuery  
[REPLACE]^private static final Object EXPRESSION_PARAMETER_QUOTING = Pattern.compile ( Pattern.quote ( EXPRESSION_PARAMETER )  ) ;^43^^^^^^^[REPLACE] private static final Pattern EXPRESSION_PARAMETER_QUOTING = Pattern.compile ( Pattern.quote ( EXPRESSION_PARAMETER )  ) ;^ [CLASS] ExpressionBasedStringQuery  
[REPLACE]^private static final Pattern EXPRESSION_PARAMETER_UNQUOTING ;^44^45^^^^44^45^[REPLACE] private static final Pattern EXPRESSION_PARAMETER_UNQUOTING = Pattern.compile ( Pattern .quote ( QUOTED_EXPRESSION_PARAMETER )  ) ;^ [CLASS] ExpressionBasedStringQuery  
[REPLACE]^private static final String ENTITY_NAME ;^47^^^^^^^[REPLACE] private static final String ENTITY_NAME = "entityName";^ [CLASS] ExpressionBasedStringQuery  
[REPLACE]^private static  String ENTITY_NAME_VARIABLE = "#" + ENTITY_NAME;^48^^^^^^^[REPLACE] private static final String ENTITY_NAME_VARIABLE = "#" + ENTITY_NAME;^ [CLASS] ExpressionBasedStringQuery  
[REPLACE]^private static final String ENTITY_NAME_VARIABLE_EXPRESSION ;^49^^^^^^^[REPLACE] private static final String ENTITY_NAME_VARIABLE_EXPRESSION = "#{" + ENTITY_NAME_VARIABLE + "}";^ [CLASS] ExpressionBasedStringQuery  
[REPLACE]^notNull ( parser, "parser must not be null!" )  ;^71^^^^^69^93^[REPLACE] Assert.notNull ( query, "query must not be null!" ) ;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^notNull ( parser, "parser must not be null!" )  ;^72^^^^^69^93^[REPLACE] Assert.notNull ( metadata, "metadata must not be null!" ) ;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^Assert.notNull ( parser, "parser must not be true!" ) ;^73^^^^^69^93^[REPLACE] Assert.notNull ( parser, "parser must not be null!" ) ;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^if  ( !containsExpression ( EXPRESSION_PARAMETER )  )  {^75^^^^^69^93^[REPLACE] if  ( !containsExpression ( query )  )  {^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^return QUOTED_EXPRESSION_PARAMETER;^76^^^^^69^93^[REPLACE] return query;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^query.contains ( ENTITY_NAME_VARIABLE_EXPRESSION )  ;^80^^^^^69^93^[REPLACE] evalContext.setVariable ( ENTITY_NAME, metadata.getEntityName (  )  ) ;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^evalContext.setVariable ( ENTITY_NAME, this.getEntityName (  )  ) ;^80^^^^^69^93^[REPLACE] evalContext.setVariable ( ENTITY_NAME, metadata.getEntityName (  )  ) ;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^query =  potentiallyQuoteExpressionsParameter ( QUOTED_EXPRESSION_PARAMETER ) ;^82^^^^^69^93^[REPLACE] query = potentiallyQuoteExpressionsParameter ( query ) ;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[ADD]^^84^^^^^69^93^[ADD] Expression expr = parser.parseExpression ( query, ParserContext.TEMPLATE_EXPRESSION ) ;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^if  ( result != null )  {^88^^^^^69^93^[REPLACE] if  ( result == null )  {^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^return QUOTED_EXPRESSION_PARAMETER;^89^^^^^69^93^[REPLACE] return query;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
[REPLACE]^return potentiallyUnquoteParameterExpressions ( QUOTED_EXPRESSION_PARAMETER ) ;^92^^^^^69^93^[REPLACE] return potentiallyUnquoteParameterExpressions ( result ) ;^[METHOD] renderQueryIfExpressionOrReturnQuery [TYPE] String [PARAMETER] String query JpaEntityMetadata<?> metadata SpelExpressionParser parser [CLASS] ExpressionBasedStringQuery   [TYPE]  Pattern EXPRESSION_PARAMETER_QUOTING  EXPRESSION_PARAMETER_UNQUOTING  [TYPE]  SpelExpressionParser parser  [TYPE]  StandardEvaluationContext evalContext  [TYPE]  boolean false  true  [TYPE]  Expression expr  [TYPE]  String ENTITY_NAME  ENTITY_NAME_VARIABLE  ENTITY_NAME_VARIABLE_EXPRESSION  EXPRESSION_PARAMETER  QUOTED_EXPRESSION_PARAMETER  query  result  [TYPE]  JpaEntityMetadata metadata 
