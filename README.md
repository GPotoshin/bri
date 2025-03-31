# Bri

What if we train an LLM to translate proofs from human language to Coq in order
to formalize them? This way, we can add another layer of rigor to verify the
reasoning of LLMs or have a powerful tool to verify proofs.

If you want, you can join me on this journey or take your own path toward a brighter future!

However, to train an LLM for this purpose, we need suitable training data.

## State
Unusable

In the future the tokeniser will be factorised out and will be managed by a
special tool. Most functionality is done, but I am debugging the attentinon block
as it's hard to get all the calculations right and clean.

## Matrices
Every matrix is represented as consequent in memry rows that can also represent an
array of vectors and the matrix multiplication is done with the convention line
by line instead of classical line by column. I suspect that this approach should
run as fast as block by block multiplication on GPU and CPU and we can avoid
usage of transpositions.

## Structure
The code is devided in compute blocks, that are structures with weights, memory
for caculus and methodes. Each compute block either `computes` by writing the
result matrix to out field or `applys` by overiting the field. As the prototype
is designed for CPU, I favoire `apply` methode where it's possible. Never the
less sometimes we can set the out field to the input value in such case there
will be 2 methods.


## Acheivements
### Tokeniser
We have a tokeniser. As the example a text
```
The infinite is the answer to all questions. All questions have one answer. And therefore there are not many questions, but only one question. This question is: what is the infinite?
```
is tokenised into
```
 the| in|fini|t|e| is| the| answer| to| all| quest|ion|s|.| all| quest|ion|s| have| on|e| answer|.| and| there|for|e| there| are| not| many| quest|ion|s|,| but| only| on|e| quest|ion|.| this| quest|ion| is|:| what| is| the| in|fini|t|e|?|
```
Words are tokenised by the longest prefix principle, which works well in most
cases. There are 2 types of morphemes, ones that starts with white space and
once that do not. It's done to note to have white spaces, but the price is
that the same root at the start and in the middle is treated as deferent token,
which is probably fine.

You can take the tokeniser from this porject and repurpose it for your use.
It has a hash map and an odered list of correspondens of tokens and their ids,
ordinal numbers.

### Attention
We have a prototype of attention algorithm

### MHAttention
We have a prototype of mhattention algorithm

## Tests
Most funcitons are covered by tests, so you can see how they work by looking at
those examples.

## Problems
+ hashes of stored files
+ we need to agree on a common style of proofs
+ we need a table of proofs in English -> in Coq
+ we need a working LLM
+ we need a Cuda/Metal support
+ we need a json library reader

To contact me, feel free write to <george.potoshin@gmail.com> or submit a suggestion
on GitHub.
