[buglab_swap_variables]^return getCost (  Integer.MAX_VALUE ) ;^40^^^^^39^41^return getCost ( root, Integer.MAX_VALUE ) ;^[CLASS] InlineCostEstimator CompiledSizeEstimator  [METHOD] getCost [RETURN_TYPE] int   Node root [VARIABLES] char  last  boolean  continueProcessing  String  ESTIMATED_IDENTIFIER  int  ESTIMATED_IDENTIFIER_COST  cost  costThreshhold  maxCost  Node  root  
[buglab_swap_variables]^if  ( cost <= maxCost )  {^89^^^^^86^92^if  ( maxCost <= cost )  {^[CLASS] InlineCostEstimator CompiledSizeEstimator  [METHOD] append [RETURN_TYPE] void   String str [VARIABLES] char  last  boolean  continueProcessing  String  ESTIMATED_IDENTIFIER  str  int  ESTIMATED_IDENTIFIER_COST  cost  costThreshhold  maxCost  
[buglab_swap_variables]^if  ( cost <= maxCost )  {^89^^^^^86^92^if  ( maxCost <= cost )  {^[CLASS] CompiledSizeEstimator  [METHOD] append [RETURN_TYPE] void   String str [VARIABLES] char  last  boolean  continueProcessing  String  str  int  cost  costThreshhold  maxCost  