[REPLACE]^private final Map<Long, Set<AttributeAlias>> deviceAliases ;^35^^^^^^^[REPLACE] private final Map<Long, Set<AttributeAlias>> deviceAliases = new ConcurrentHashMap<> (  ) ;^ [CLASS] AliasesManager  
[REPLACE]^private  Map<Long, AttributeAlias> aliasesById = new ConcurrentHashMap<> (  ) ;^36^^^^^^^[REPLACE] private final Map<Long, AttributeAlias> aliasesById = new ConcurrentHashMap<> (  ) ;^ [CLASS] AliasesManager  
