[buglab_swap_variables]^if  ( ! ( second instanceof JsonParserSequence || first instanceof JsonParserSequence )  )  {^53^^^^^51^69^if  ( ! ( first instanceof JsonParserSequence || second instanceof JsonParserSequence )  )  {^[CLASS] JsonParserSequence  [METHOD] createFlattened [RETURN_TYPE] JsonParserSequence   JsonParser first JsonParser second [VARIABLES] ArrayList  p  boolean  JsonParser[]  _parsers  parsers  int  _nextParser  JsonParser  first  second  
[buglab_swap_variables]^return new JsonParserSequence ( new JsonParser[] { second, first } ) ;^55^^^^^51^69^return new JsonParserSequence ( new JsonParser[] { first, second } ) ;^[CLASS] JsonParserSequence  [METHOD] createFlattened [RETURN_TYPE] JsonParserSequence   JsonParser first JsonParser second [VARIABLES] ArrayList  p  boolean  JsonParser[]  _parsers  parsers  int  _nextParser  JsonParser  first  second  
[buglab_swap_variables]^return new JsonParserSequence ( new JsonParser[] {  second } ) ;^55^^^^^51^69^return new JsonParserSequence ( new JsonParser[] { first, second } ) ;^[CLASS] JsonParserSequence  [METHOD] createFlattened [RETURN_TYPE] JsonParserSequence   JsonParser first JsonParser second [VARIABLES] ArrayList  p  boolean  JsonParser[]  _parsers  parsers  int  _nextParser  JsonParser  first  second  
[buglab_swap_variables]^return new JsonParserSequence ( new JsonParser[] { first } ) ;^55^^^^^51^69^return new JsonParserSequence ( new JsonParser[] { first, second } ) ;^[CLASS] JsonParserSequence  [METHOD] createFlattened [RETURN_TYPE] JsonParserSequence   JsonParser first JsonParser second [VARIABLES] ArrayList  p  boolean  JsonParser[]  _parsers  parsers  int  _nextParser  JsonParser  first  second  