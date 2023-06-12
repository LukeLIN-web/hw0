Public repository and stub/testing code for Homework of 10-714.s

## 笔记

#### lec7

TensorFlow has a declaration API. What are the advantages?

You can get the middle result without calculating all the computation graphs. So it has more chance to optimize. However, you don't have value when constructing the computation graphs, so debugging is cumbersome.

Google is good at declarative API. It can easily scale.

Pytorch and needle, imperative API.  It can execute computation as we construct the computational graph. "Define by run"  Dynamic graph. 

#### lec8

needle. detach, create a new tensor, which doesn't contain inputs/op, but shares the underlying memory. 

eager mode vs. lazy mode.  the eager mode will call tensor.detach() if all input is not requiregrad.

When creating some computation, always ask yourself whether you need gradients. Otherwise, it costs too much memory.











