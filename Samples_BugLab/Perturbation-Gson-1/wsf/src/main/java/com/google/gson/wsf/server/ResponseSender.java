[buglab_swap_variables]^sendHeaders ( response, conn.getHeaders (  )  ) ;^47^^^^^45^52^sendHeaders ( conn, response.getHeaders (  )  ) ;^[CLASS] ResponseSender  [METHOD] send [RETURN_TYPE] void   HttpServletResponse conn WebServiceResponse response [VARIABLES] Gson  gson  HttpServletResponse  conn  WebServiceResponse  response  IOException  e  boolean  Logger  logger  
[buglab_swap_variables]^sendHeaders (  response.getHeaders (  )  ) ;^47^^^^^45^52^sendHeaders ( conn, response.getHeaders (  )  ) ;^[CLASS] ResponseSender  [METHOD] send [RETURN_TYPE] void   HttpServletResponse conn WebServiceResponse response [VARIABLES] Gson  gson  HttpServletResponse  conn  WebServiceResponse  response  IOException  e  boolean  Logger  logger  
[buglab_swap_variables]^sendHeaders ( conn.getHeaders (  )  ) ;^47^^^^^45^52^sendHeaders ( conn, response.getHeaders (  )  ) ;^[CLASS] ResponseSender  [METHOD] send [RETURN_TYPE] void   HttpServletResponse conn WebServiceResponse response [VARIABLES] Gson  gson  HttpServletResponse  conn  WebServiceResponse  response  IOException  e  boolean  Logger  logger  
[buglab_swap_variables]^sendBody ( response, conn.getBody (  )  ) ;^48^^^^^45^52^sendBody ( conn, response.getBody (  )  ) ;^[CLASS] ResponseSender  [METHOD] send [RETURN_TYPE] void   HttpServletResponse conn WebServiceResponse response [VARIABLES] Gson  gson  HttpServletResponse  conn  WebServiceResponse  response  IOException  e  boolean  Logger  logger  
[buglab_swap_variables]^sendBody (  response.getBody (  )  ) ;^48^^^^^45^52^sendBody ( conn, response.getBody (  )  ) ;^[CLASS] ResponseSender  [METHOD] send [RETURN_TYPE] void   HttpServletResponse conn WebServiceResponse response [VARIABLES] Gson  gson  HttpServletResponse  conn  WebServiceResponse  response  IOException  e  boolean  Logger  logger  
[buglab_swap_variables]^sendBody ( conn.getBody (  )  ) ;^48^^^^^45^52^sendBody ( conn, response.getBody (  )  ) ;^[CLASS] ResponseSender  [METHOD] send [RETURN_TYPE] void   HttpServletResponse conn WebServiceResponse response [VARIABLES] Gson  gson  HttpServletResponse  conn  WebServiceResponse  response  IOException  e  boolean  Logger  logger  
[buglab_swap_variables]^Type paramType = paramName.getTypeFor ( spec ) ;^59^^^^^54^64^Type paramType = spec.getTypeFor ( paramName ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^String json = paramValue.toJson ( gson, paramType ) ;^60^^^^^54^64^String json = gson.toJson ( paramValue, paramType ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^String json = gson.toJson (  paramType ) ;^60^^^^^54^64^String json = gson.toJson ( paramValue, paramType ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^String json = paramType.toJson ( paramValue, gson ) ;^60^^^^^54^64^String json = gson.toJson ( paramValue, paramType ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^String json = gson.toJson ( paramValue ) ;^60^^^^^54^64^String json = gson.toJson ( paramValue, paramType ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^String json = gson.toJson ( paramType, paramValue ) ;^60^^^^^54^64^String json = gson.toJson ( paramValue, paramType ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^logger.fine ( "RESPONSE HEADER:{" + json + ", " + paramName + "}" ) ;^61^^^^^54^64^logger.fine ( "RESPONSE HEADER:{" + paramName + ", " + json + "}" ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^conn.addHeader ( json, paramName ) ;^62^^^^^54^64^conn.addHeader ( paramName, json ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^conn.addHeader (  json ) ;^62^^^^^54^64^conn.addHeader ( paramName, json ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^conn.addHeader ( paramName ) ;^62^^^^^54^64^conn.addHeader ( paramName, json ) ;^[CLASS] ResponseSender  [METHOD] sendHeaders [RETURN_TYPE] void   HttpServletResponse conn HeaderMap responseParams [VARIABLES] Entry  param  Type  paramType  boolean  HeaderMap  responseParams  Gson  gson  HttpServletResponse  conn  Object  paramValue  String  json  paramName  Logger  logger  HeaderMapSpec  spec  
[buglab_swap_variables]^String json = responseBody.toJson ( gson ) ;^69^^^^^66^72^String json = gson.toJson ( responseBody ) ;^[CLASS] ResponseSender  [METHOD] sendBody [RETURN_TYPE] void   HttpServletResponse conn ResponseBody responseBody [VARIABLES] Gson  gson  HttpServletResponse  conn  String  json  boolean  Logger  logger  ResponseBody  responseBody  