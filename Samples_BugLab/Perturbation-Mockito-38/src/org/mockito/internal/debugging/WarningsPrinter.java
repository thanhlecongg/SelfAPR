[buglab_swap_variables]^while ( unused.hasNext (  )  )  {^29^^^^^27^49^while ( unusedIterator.hasNext (  )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[buglab_swap_variables]^while ( unstubbed.hasNext (  )  )  {^32^^^^^27^49^while ( unstubbedIterator.hasNext (  )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[buglab_swap_variables]^if ( unused.hasSimilarMethod ( unstubbed )  )  {^34^^^^^27^49^if ( unstubbed.hasSimilarMethod ( unused )  )  {^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[buglab_swap_variables]^logger.log ( stubbedMethodCalledWithDifferentArguments ( unstubbed, unused )  ) ;^35^^^^^27^49^logger.log ( stubbedMethodCalledWithDifferentArguments ( unused, unstubbed )  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[buglab_swap_variables]^logger.log ( stubbedMethodCalledWithDifferentArguments (  unstubbed )  ) ;^35^^^^^27^49^logger.log ( stubbedMethodCalledWithDifferentArguments ( unused, unstubbed )  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[buglab_swap_variables]^logger.log ( stubbedMethodCalledWithDifferentArguments ( unused )  ) ;^35^^^^^27^49^logger.log ( stubbedMethodCalledWithDifferentArguments ( unused, unstubbed )  ) ;^[CLASS] WarningsPrinter  [METHOD] print [RETURN_TYPE] void   MockitoLogger logger [VARIABLES] Iterator  unstubbedIterator  unusedIterator  InvocationMatcher  i1  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  i  unused  MockitoLogger  logger  
[buglab_swap_variables]^return join ( "[Mockito] Warning - stubbed method called with different arguments.", "Stubbed this way:",^68^69^70^71^^67^78^return join ( "[Mockito] Warning - stubbed method called with different arguments.", "Stubbed this way:", unused,^[CLASS] WarningsPrinter  [METHOD] stubbedMethodCalledWithDifferentArguments [RETURN_TYPE] String   Invocation unused InvocationMatcher unstubbed [VARIABLES] InvocationMatcher  unstubbed  List  unstubbedInvocations  unusedStubs  boolean  Invocation  unused  