[buglab_swap_variables]^mv.visitIntInsn (  operand ) ;^75^^^^^74^76^mv.visitIntInsn ( opcode, operand ) ;^[CLASS] IntInsnNode  [METHOD] accept [RETURN_TYPE] void   MethodVisitor mv [VARIABLES] int  opcode  operand  MethodVisitor  mv  boolean  
[buglab_swap_variables]^mv.visitIntInsn ( opcode ) ;^75^^^^^74^76^mv.visitIntInsn ( opcode, operand ) ;^[CLASS] IntInsnNode  [METHOD] accept [RETURN_TYPE] void   MethodVisitor mv [VARIABLES] int  opcode  operand  MethodVisitor  mv  boolean  
[buglab_swap_variables]^return new IntInsnNode ( operand, opcode ) ;^79^^^^^78^80^return new IntInsnNode ( opcode, operand ) ;^[CLASS] IntInsnNode  [METHOD] clone [RETURN_TYPE] AbstractInsnNode   Map labels [VARIABLES] Map  labels  int  opcode  operand  boolean  
[buglab_swap_variables]^return new IntInsnNode (  operand ) ;^79^^^^^78^80^return new IntInsnNode ( opcode, operand ) ;^[CLASS] IntInsnNode  [METHOD] clone [RETURN_TYPE] AbstractInsnNode   Map labels [VARIABLES] Map  labels  int  opcode  operand  boolean  
[buglab_swap_variables]^return new IntInsnNode ( opcode ) ;^79^^^^^78^80^return new IntInsnNode ( opcode, operand ) ;^[CLASS] IntInsnNode  [METHOD] clone [RETURN_TYPE] AbstractInsnNode   Map labels [VARIABLES] Map  labels  int  opcode  operand  boolean  