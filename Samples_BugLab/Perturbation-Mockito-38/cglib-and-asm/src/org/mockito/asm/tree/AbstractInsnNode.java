[buglab_swap_variables]^return  ( LabelNode )  label.get ( map ) ;^216^^^^^215^217^return  ( LabelNode )  map.get ( label ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode   LabelNode label Map map [VARIABLES] LabelNode  label  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  index  opcode  
[buglab_swap_variables]^clones[i] =  ( LabelNode )  map.get ( i.get ( labels )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[buglab_swap_variables]^clones[i] =  ( LabelNode )  labels.get ( map.get ( i )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  
[buglab_swap_variables]^clones[i] =  ( LabelNode )  i.get ( labels.get ( map )  ) ;^229^^^^^226^232^clones[i] =  ( LabelNode )  map.get ( labels.get ( i )  ) ;^[CLASS] AbstractInsnNode  [METHOD] clone [RETURN_TYPE] LabelNode[]   List labels Map map [VARIABLES] LabelNode[]  clones  List  labels  boolean  AbstractInsnNode  next  prev  Map  map  int  FIELD_INSN  FRAME  IINC_INSN  INSN  INT_INSN  JUMP_INSN  LABEL  LDC_INSN  LINE  LOOKUPSWITCH_INSN  METHOD_INSN  MULTIANEWARRAY_INSN  TABLESWITCH_INSN  TYPE_INSN  VAR_INSN  i  index  opcode  