# Bri

What if we train an LLM to translate proofs from human language to Coq in order to formalize them?
This way, we can add another layer of rigor to verify the reasoning of LLMs or have a powerful tool to verify proofs.

If you want, you can join me on this journey or take your own path toward a brighter future!

However, to train an LLM, we need some data.

## State
Unusable

In the future the tokenizer will be factorised out and will be managed by a
special tool. Most functionality is done, but I am debuging the attetinon block
as it's hard to get all the calculus right and clean.

## Acheivements
### Tokenizer
We have the tokenizer. As the example the text
```
The infinite is the answer to all questions. All questions have one answer. And therefore there are not many questions, but only one question. This question is: what is the infinite?
```
is tokenized into
```
 the| in|fini|t|e| is| the| answer| to| all| quest|ion|s|.| all| quest|ion|s| have| on|e| answer|.| and| there|for|e| there| are| not| many| quest|ion|s|,| but| only| on|e| quest|ion|.| this| quest|ion| is|:| what| is| the| in|fini|t|e|?|
```
Words are tokenised by the longest prefix princip, which works well in most
cases. There are 2 types of morphemes, ones that starts with white space and
once that do not. It's done to note to have white spaces, but the price is
that the same root at the start and in the middle is treated as deferent token,
which is probably fine.

You can take the tokenizer from this porject and repurpose it for your use.
It has a hash map and odered list of correspondens of tokens and their ids,
ordinal numbers.

## Tests
Most funciton are covered by tests, so you can see how they work by looking at
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
