我想加一个feature，现在MemSaver的arena_virtual mode只会有虚拟地址，不会绑定句柄。allocator的BlockPool的type 3还没有使用。我想给type 3分给arena virtual mode专用，然后就是动态激活显存并分配。比如最开始申请一个1GB的tensor并释放，这样就有了arena virtual。之后的所有申请，比如申请20MB的tensor，就动态创建并绑定一个句柄。你看看可行性

做个约束，对于type 3的BlockPool，第一次分配没有block，所以只能走创建Block的路径。而这个路径就会走memsaver来进行创建。而后续的所有tensor都从其中动态激活，也就是find_free_block里面找到了就激活。这就要求第一次必须是申请比较大的tensor来创建虚拟地址供后续使用。此外，虚拟地址一定远大于可用的物理显存，这就保证即使出现了碎片，也不会影响分配。同时做一个隐形的假设，显存分配一定是一轮一轮的（比如LLM的step推理），即每一轮分配后，下一轮一定会所有都释放，然后就能从头开始继续分配。

相当于第一次分配是初始化，之后的分配才是动态激活的分配。每次分配显存的句柄大小最小是20MB，如果分配大小超出20MB，则按2MB对齐。

在Block内记录offset，表示绑定了句柄的范围，（offset左边是绑定了句柄的，右边是没有绑定句柄的）。这样进行Block分配的时候，如果offset满足，就可以直接进行Block的拆分。如果offset满足，就直接拆分Block，回收时也不需要调用DeactivateArenaOffsets进行句柄的释放，只是进行合并Block，然后合并offset。分配的时候如果offset不满足，则创建句柄并调用ActivateArenaOffsets绑定上去。

也就是说
Block* block = find_free_block(pool, rounded, stream);
if (block == nullptr) {
    block = allocate_from_cuda(rounded, stream, pool);
}

下面的if只有在第一次分配时有用，后续分配就是修改find_free_block的函数逻辑，来进行动态激活。